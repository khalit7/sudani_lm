"""Gate for the model rewrite (Part I, Step 3).

The important test here is test_sdpa_attention_matches_reference: it pins the new SDPA attention
against the hand-written implementation in decoder_deprecated.py, so adopting the fast path costs
nothing in correctness and the original stays a usable reference.
"""

import math

import pytest
import torch

from src.models.decoder import (
    Attention,
    DecoderLMHeadModel,
    DecoderModel,
    apply_rope,
    swiglu_hidden_dim,
)
from src.models.decoder_deprecated import MaskedMultiHeadAttn

CONFIG = {
    "vocab_size": 256,
    "d_model": 64,
    "num_layers": 3,
    "num_heads": 4,
    "max_seq_len": 32,
    "dropout": 0.0,
}


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DecoderLMHeadModel(dict(CONFIG))


# --- equivalence with the deprecated hand-written attention ---------------------------------

def test_sdpa_attention_matches_reference():
    """The deprecated module computes K @ Qᵀ, which is standard attention with the q and k
    projections swapped. Swap them back and the two must agree exactly."""
    torch.manual_seed(0)
    cfg = {"d_model": 64, "num_heads": 4, "max_seq_len": 16, "dropout": 0.0}

    reference = MaskedMultiHeadAttn(dict(cfg))
    new = Attention(dict(cfg))
    with torch.no_grad():
        # note the deliberate q<->k swap
        new.q_proj.weight.copy_(reference.k_proj.weight)
        new.k_proj.weight.copy_(reference.q_proj.weight)
        new.v_proj.weight.copy_(reference.v_proj.weight)
        new.out_proj.weight.copy_(reference.out_proj.weight)
        for linear in (reference.q_proj, reference.k_proj, reference.v_proj, reference.out_proj):
            linear.bias.zero_()

    reference.eval()
    new.eval()
    x = torch.randn(2, 16, 64)
    with torch.no_grad():
        expected = reference(x, torch.ones(2, 16, dtype=torch.long))
        got = new(x)  # attn_mask=None -> is_causal fast path

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)


def test_masked_path_matches_reference_with_padding():
    """Same equivalence, but through the explicit key-padding mask instead of is_causal."""
    torch.manual_seed(1)
    cfg = {"d_model": 32, "num_heads": 4, "max_seq_len": 12, "dropout": 0.0}
    reference = MaskedMultiHeadAttn(dict(cfg))
    new = Attention(dict(cfg))
    with torch.no_grad():
        new.q_proj.weight.copy_(reference.k_proj.weight)
        new.k_proj.weight.copy_(reference.q_proj.weight)
        new.v_proj.weight.copy_(reference.v_proj.weight)
        new.out_proj.weight.copy_(reference.out_proj.weight)
        for linear in (reference.q_proj, reference.k_proj, reference.v_proj, reference.out_proj):
            linear.bias.zero_()
    reference.eval(); new.eval()

    attention_mask = torch.tensor([[1] * 8 + [0] * 4, [1] * 12], dtype=torch.long)
    x = torch.randn(2, 12, 32)
    holder = DecoderModel({**CONFIG, "d_model": 32, "num_heads": 4, "max_seq_len": 12})
    attn_mask = holder.build_attn_mask(attention_mask, 12, x.device, x.dtype)

    with torch.no_grad():
        expected = reference(x, attention_mask)
        got = new(x, attn_mask=attn_mask)

    # compare only the real (non-padding) rows; padded rows are discarded downstream
    torch.testing.assert_close(got[0, :8], expected[0, :8], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(got[1], expected[1], rtol=1e-4, atol=1e-5)


# --- causality -------------------------------------------------------------------------------

def test_future_tokens_cannot_change_the_past(model):
    model.eval()
    ids = torch.randint(0, CONFIG["vocab_size"], (1, 16))
    with torch.no_grad():
        base = model(ids)
        perturbed_ids = ids.clone()
        perturbed_ids[0, 10] = (perturbed_ids[0, 10] + 7) % CONFIG["vocab_size"]
        perturbed = model(perturbed_ids)

    torch.testing.assert_close(base[0, :10], perturbed[0, :10], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(base[0, 10], perturbed[0, 10]), "position 10 should have changed"


def test_padding_rows_do_not_produce_nan(model):
    """A fully-masked row would make softmax return NaN, which then spreads through the batch."""
    model.eval()
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 12))
    attention_mask = torch.tensor([[1] * 3 + [0] * 9, [1] * 12], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, attention_mask)
    assert torch.isfinite(out).all()


# --- architecture properties -----------------------------------------------------------------

def test_embeddings_are_tied_by_default(model):
    assert model.head.weight is model.token_embedding.weight


def test_untied_when_configured():
    m = DecoderLMHeadModel({**CONFIG, "tie_embeddings": False})
    assert m.head.weight is not m.token_embedding.weight


def test_final_norm_exists_and_is_applied(model):
    """Absent in the deprecated model; mandatory under pre-LN."""
    assert hasattr(model, "final_norm")
    model.eval()
    ids = torch.randint(0, CONFIG["vocab_size"], (1, 8))
    with torch.no_grad():
        hidden = DecoderModel.forward(model, ids)
    # RMSNorm output should have roughly unit RMS
    rms = hidden.pow(2).mean(dim=-1).sqrt()
    assert (rms > 0.1).all() and (rms < 10.0).all()


def test_residual_projections_are_downscaled():
    """out_proj / down_proj initialise at 0.02/sqrt(2L) so depth does not inflate the stream."""
    layers = 8
    m = DecoderLMHeadModel({**CONFIG, "num_layers": layers})
    expected = 0.02 / math.sqrt(2 * layers)
    for name, param in m.named_parameters():
        if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
            assert param.std().item() < 0.02, f"{name} was not downscaled"
            assert abs(param.std().item() - expected) < expected * 0.5


def test_swiglu_hidden_is_multiple_of_64():
    for d_model in (512, 768, 1024):
        hidden = swiglu_hidden_dim(d_model)
        assert hidden % 64 == 0
        assert hidden >= 8 * d_model / 3
    assert swiglu_hidden_dim(768) == 2048       # the Stage-B config
    assert swiglu_hidden_dim(512) == 1408       # the Stage-A config


def test_rope_is_a_rotation():
    """RoPE must preserve vector norms — it rotates, it does not rescale."""
    torch.manual_seed(0)
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    from src.models.decoder import RotaryEmbedding

    rotary = RotaryEmbedding(16, 32)
    cos, sin = rotary(8)
    q2, k2 = apply_rope(q, k, cos, sin)
    torch.testing.assert_close(q.norm(dim=-1), q2.norm(dim=-1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k.norm(dim=-1), k2.norm(dim=-1), rtol=1e-4, atol=1e-5)


def test_rope_depends_on_relative_position():
    """The dot product after RoPE should depend on the offset between positions, not absolutes."""
    from src.models.decoder import RotaryEmbedding

    torch.manual_seed(0)
    rotary = RotaryEmbedding(16, 64)
    cos, sin = rotary(64)
    q = torch.randn(1, 1, 1, 16).expand(1, 1, 64, 16).clone()
    k = q.clone()
    q2, k2 = apply_rope(q, k, cos, sin)
    # same offset -> same score, regardless of absolute position
    score_a = (q2[0, 0, 10] * k2[0, 0, 5]).sum()
    score_b = (q2[0, 0, 40] * k2[0, 0, 35]).sum()
    torch.testing.assert_close(score_a, score_b, rtol=1e-3, atol=1e-4)


# --- plumbing the trainer relies on ------------------------------------------------------------

def test_forward_shapes(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (3, 16))
    assert model(ids).shape == (3, 16, CONFIG["vocab_size"])
    assert model(ids, torch.ones(3, 16, dtype=torch.long)).shape == (3, 16, CONFIG["vocab_size"])


def test_grad_norms_available_after_backward(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 8))
    model(ids).sum().backward()
    norms = model.calc_grad_norms()
    assert "model_grad" in norms and norms["model_grad"] > 0
    assert all(f"layer{i}_grad" in norms for i in range(1, CONFIG["num_layers"] + 1))


def test_model_stats_reports_non_embedding_params(model):
    stats = model.get_model_stats(verbose=False)
    assert stats["num_params"] > stats["num_params_non_embedding"] > 0


@pytest.mark.parametrize("d_model,layers,heads,expected", [(512, 8, 8, 42_000_000),
                                                           (768, 12, 12, 110_000_000)])
def test_planned_configs_hit_their_parameter_budget(d_model, layers, heads, expected):
    """Stage A is specified as ~42M and Stage B as ~110M in plan.md."""
    m = DecoderLMHeadModel({
        "vocab_size": 32_000, "d_model": d_model, "num_layers": layers,
        "num_heads": heads, "max_seq_len": 1024,
    })
    total = m.get_model_stats(verbose=False)["num_params"]
    assert abs(total - expected) / expected < 0.05, f"got {total:,}, expected ~{expected:,}"
