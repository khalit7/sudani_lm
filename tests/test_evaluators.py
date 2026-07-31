"""Gate for the eval rewrite (Part I, Step 5).

Two things are being pinned: that the evaluators are correct on a model whose answers are known,
and that the KV cache used by generation produces exactly what an uncached forward would.
"""

import json

import pytest
import torch

from src.evaluator import (
    Evaluator,
    FloresPerplexityEvaluator,
    GenerationEvaluator,
    MMLULoglikelihoodEvaluator,
)
from src.models.decoder import DecoderLMHeadModel
from src.tokenizer.utils import get_tokenizer

CONFIG = {
    "vocab_size": 32000, "d_model": 64, "num_layers": 2,
    "num_heads": 4, "max_seq_len": 256, "dropout": 0.0,
}


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DecoderLMHeadModel(dict(CONFIG)).eval()


# --- scheduling --------------------------------------------------------------------------------

class Dummy(Evaluator):
    name = "dummy"

    def evaluate(self, model, device, tokenizer):
        return {"dummy/value": 1.0}


def test_run_at_0_controls_the_baseline():
    assert Dummy(frequency=500, run_at_0=True).should_run(0) is True
    assert Dummy(frequency=500, run_at_0=False).should_run(0) is False


def test_frequency_gating():
    ev = Dummy(frequency=100, run_at_0=False)
    assert [s for s in range(1, 301) if ev.should_run(s)] == [100, 200, 300]


def test_zero_frequency_disables():
    ev = Dummy(frequency=0, run_at_0=False)
    assert not any(ev.should_run(s) for s in range(1, 50))


# --- KV cache (generation depends on it) --------------------------------------------------------

def test_cached_decode_matches_uncached_forward(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 12))
    full = model(ids)

    past, outs = None, []
    for t in range(12):
        logits, past = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        outs.append(logits)

    torch.testing.assert_close(full, torch.cat(outs, dim=1), rtol=1e-3, atol=1e-4)


def test_prefill_then_decode_matches_uncached(model):
    """The path generate() actually takes: one big prefill, then single tokens."""
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 12))
    full = model(ids)

    logits, past = model(ids[:, :8], use_cache=True)
    outs = [logits]
    for t in range(8, 12):
        logits, past = model(ids[:, t : t + 1], past_key_values=past, use_cache=True)
        outs.append(logits)

    torch.testing.assert_close(full, torch.cat(outs, dim=1), rtol=1e-3, atol=1e-4)


def test_cache_grows_by_one_per_step(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (1, 5))
    _, past = model(ids, use_cache=True)
    assert past[0][0].shape[2] == 5
    _, past = model(ids[:, :1], past_key_values=past, use_cache=True)
    assert past[0][0].shape[2] == 6


def test_generate_returns_prompt_plus_new_tokens(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (2, 4))
    out = model.generate(ids, max_new_tokens=6, temperature=0.0)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], ids), "the prompt must be preserved"


def test_greedy_generation_is_deterministic(model):
    ids = torch.randint(0, CONFIG["vocab_size"], (1, 4))
    a = model.generate(ids, max_new_tokens=8, temperature=0.0)
    b = model.generate(ids, max_new_tokens=8, temperature=0.0)
    assert torch.equal(a, b)


def test_generation_stops_at_eos(model, tokenizer):
    """max_new_tokens counts *new* tokens; the old loop compared against total length, so a long
    prompt generated nothing at all."""
    ids = torch.tensor([[tokenizer.bos_token_id]])
    out = model.generate(ids, max_new_tokens=5, temperature=0.0,
                         eos_token_id=tokenizer.eos_token_id)
    assert out.shape[1] <= 6


# --- MMLU loglikelihood ---------------------------------------------------------------------------

class ScriptedModel(torch.nn.Module):
    """Puts probability mass on a chosen token id, so scoring can be checked exactly."""

    def __init__(self, vocab_size, favoured):
        super().__init__()
        self.vocab_size = vocab_size
        self.favoured = favoured

    def forward(self, input_ids, attention_mask=None, **kwargs):
        b, s = input_ids.shape
        out = torch.full((b, s, self.vocab_size), -10.0)
        out[..., self.favoured] = 10.0
        return out


def test_loglikelihood_prefers_the_option_it_scores_highest(tokenizer):
    """A model that loves one token should pick whichever option contains it."""
    rows = [{
        "Question": "سؤال",
        "Context": "",
        "Answer Key": "B",
        "Option 1": "خيار اول",
        "Option 2": "نعم",
        "Option 3": None, "Option 4": None, "Option 5": None,
    }]

    class Wrapper:
        dataset = rows

    favoured = tokenizer.encode(" نعم", add_special_tokens=False)[0]
    ev = MMLULoglikelihoodEvaluator(Wrapper(), max_examples=1, batch_size=1)
    metrics = ev.evaluate(ScriptedModel(len(tokenizer), favoured), "cpu", tokenizer)
    assert metrics["mmlu/loglikelihood_acc"] == 1.0


def test_loglikelihood_chance_accounts_for_variable_option_counts(tokenizer):
    rows = [
        {"Question": "q", "Context": "", "Answer Key": "A",
         "Option 1": "a", "Option 2": "b", "Option 3": None, "Option 4": None, "Option 5": None},
        {"Question": "q", "Context": "", "Answer Key": "A",
         "Option 1": "a", "Option 2": "b", "Option 3": "c", "Option 4": "d", "Option 5": None},
    ]

    class Wrapper:
        dataset = rows

    ev = MMLULoglikelihoodEvaluator(Wrapper(), max_examples=2, batch_size=2)
    metrics = ev.evaluate(ScriptedModel(len(tokenizer), 5), "cpu", tokenizer)
    # (1/2 + 1/4) / 2 = 0.375, not 0.2
    assert metrics["mmlu/loglikelihood_chance"] == pytest.approx(0.375)


# --- Flores ----------------------------------------------------------------------------------------

def test_flores_reports_both_sides_and_the_gap(tmp_path, tokenizer, model, monkeypatch):
    rows = [{"translation": {"Arb": "هذه جملة عربية فصحى", "Sud": "دي جملة سودانية"}} for _ in range(4)]
    path = tmp_path / "raw" / "sudanese_flores"
    path.mkdir(parents=True)
    (path / "DEV.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    monkeypatch.setattr("src.evaluator.data_root", tmp_path)

    ev = FloresPerplexityEvaluator(split="DEV", batch_size=2)
    metrics = ev.evaluate(model, "cpu", tokenizer)

    assert metrics["flores/sudanese_ppl"] > 1
    assert metrics["flores/msa_ppl"] > 1
    assert metrics["flores/sud_minus_msa_ppl"] == pytest.approx(
        metrics["flores/sudanese_ppl"] - metrics["flores/msa_ppl"]
    )


def test_random_init_perplexity_is_near_vocab_size(tokenizer, model, tmp_path, monkeypatch):
    """An untrained model assigns roughly uniform probability, so ppl should land near |vocab|.
    Far from it means the eval plumbing is wrong, not the model."""
    rows = [{"translation": {"Arb": "جملة عربية للاختبار", "Sud": "جملة سودانية للاختبار"}}] * 8
    path = tmp_path / "raw" / "sudanese_flores"
    path.mkdir(parents=True)
    (path / "DEV.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    monkeypatch.setattr("src.evaluator.data_root", tmp_path)

    metrics = FloresPerplexityEvaluator(split="DEV").evaluate(model, "cpu", tokenizer)
    assert 0.2 * len(tokenizer) < metrics["flores/msa_ppl"] < 5 * len(tokenizer)


# --- generation evaluator --------------------------------------------------------------------------

def test_generation_evaluator_returns_one_sample_per_prompt_and_temperature(model, tokenizer):
    ev = GenerationEvaluator(prompts=["<s>", "<s>الخرطوم"], temperatures=[0.0, 0.7],
                             max_new_tokens=4)
    samples = ev.evaluate(model, "cpu", tokenizer)["generation/samples"]
    assert len(samples) == 4
    assert {s["temperature"] for s in samples} == {0.0, 0.7}
    assert all(isinstance(s["text"], str) and s["text"] for s in samples)
