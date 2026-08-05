"""Decoder for the Sudanese chat model.

Rewritten from src/models/decoder_deprecated.py, which remains as the reference implementation
and correctness oracle. What changed and why:

  pre-LN instead of post-LN     post-LN degrades as layers are added and needs careful warmup;
                                this is what makes the 4 -> 12 layer jump trainable
  final norm before the head    absent before. Harmless under post-LN, wrong under pre-LN
  RoPE instead of additive      better length behaviour, and drops the fixed max_seq_len buffer
  SDPA instead of a manual      measured 25.7 ms / 2.45 GB -> 0.40 ms / 0.88 GB at Stage-B
  score matrix                  shapes; the freed memory is what allows the planned batch size
  SwiGLU instead of GELU        same parameter count at 2/3 the hidden width
  tied embeddings               saves 24.6M parameters at 110M — 22% of the model
  scaled residual init          keeps residual-stream variance stable with depth

Config keys (all optional except vocab_size/d_model/num_layers/num_heads):
    vocab_size, d_model, num_layers, num_heads, max_seq_len,
    dropout (0.0), rope_base (10000.0), tie_embeddings (True), mlp_hidden (derived)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu_hidden_dim(d_model: int, multiple_of: int = 64) -> int:
    """8/3 * d_model rounded up to a multiple of 64.

    SwiGLU uses three projections instead of two, so 8/3 rather than 4 keeps the parameter
    count level with a standard GELU MLP.
    """
    hidden = int(8 * d_model / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class RotaryEmbedding(nn.Module):
    """Precomputed RoPE tables.

    Positions are encoded by rotating (q, k) rather than by adding a vector to the embedding,
    so position information reaches every layer's attention rather than only the input.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        position = torch.arange(max_seq_len).float()
        angles = torch.outer(position, inv_freq)          # (max_seq_len, head_dim/2)
        # Duplicated so the table lines up with the rotate_half split below.
        emb = torch.cat([angles, angles], dim=-1)         # (max_seq_len, head_dim)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos[:seq_len], self.sin[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k are (batch, heads, seq, head_dim); cos/sin are (seq, head_dim)
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    """Multi-head causal self-attention via F.scaled_dot_product_attention."""

    def __init__(self, config) -> None:
        super().__init__()
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = self.d_model // self.num_heads
        self.dropout = config.get("dropout", 0.0)

        # bias=False throughout: it buys nothing measurable and costs parameters.
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def forward(self, x, cos=None, sin=None, attn_mask=None, past_kv=None, use_cache=False):
        batch, seq_len, _ = x.shape

        def split(proj):
            return proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        if cos is not None:
            q, k = apply_rope(q, k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        # Three cases:
        #   prefill / training  q_len == k_len  -> is_causal handles the mask, no tensor built
        #   cached decode       q_len == 1      -> the single query may attend to *every* cached
        #                                          key, so no mask at all. is_causal would be
        #                                          wrong here: SDPA aligns a causal mask to the
        #                                          top-left, so it would let the token see only
        #                                          position 0.
        #   explicit mask       padding present -> use it as given
        if attn_mask is None and seq_len == k.shape[2]:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0, is_causal=False,
            )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.out_proj(out)
        return (out, present) if use_cache else out


class SwiGLU(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        d_model = config["d_model"]
        hidden = config.get("mlp_hidden") or swiglu_hidden_dim(d_model)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Pre-LN block: x + sublayer(norm(x)).

    The residual stream is never normalized in place, so gradients reach the embedding through
    an unbroken identity path. The deprecated implementation used post-LN — norm(sublayer + x) —
    which degrades with depth.
    """

    def __init__(self, config) -> None:
        super().__init__()
        d_model = config["d_model"]
        dropout = config.get("dropout", 0.0)
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = Attention(config)
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos=None, sin=None, attn_mask=None, past_kv=None, use_cache=False):
        attn_out = self.attn(self.norm1(x), cos, sin, attn_mask, past_kv, use_cache)
        if use_cache:
            attn_out, present = attn_out
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return (x, present) if use_cache else x


class DecoderModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config["vocab_size"]
        self.num_layers = config["num_layers"]
        self.d_model = config["d_model"]
        self.num_heads = config["num_heads"]
        self.max_seq_len = config.get("max_seq_len", 1024)
        self.head_dim = self.d_model // self.num_heads

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.embed_dropout = nn.Dropout(config.get("dropout", 0.0))
        self.rotary = RotaryEmbedding(
            self.head_dim, self.max_seq_len, config.get("rope_base", 10000.0)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_layers)
        )
        # Missing entirely in the deprecated model. Under pre-LN the residual stream reaches the
        # head unnormalized without it, and logit scale drifts with depth.
        self.final_norm = nn.RMSNorm(self.d_model)

    def build_attn_mask(self, attention_mask, seq_len, device, dtype):
        """Combine key padding with causality into a boolean SDPA mask.

        Returns None when there is nothing to mask, which lets SDPA take the flash path.
        """
        if attention_mask is None:
            return None
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
        key_pad = attention_mask.bool()[:, None, None, :]          # (batch,1,1,seq)
        mask = causal[None, None, :, :] & key_pad
        # Always allow the diagonal. A padding row would otherwise be fully masked, and softmax
        # over all -inf yields NaN which then propagates through the whole batch. Those rows are
        # discarded downstream, so letting them attend to themselves is free.
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None, :, :]
        return mask | eye

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        _, seq_len = input_ids.shape
        x = self.embed_dropout(self.token_embedding(input_ids))

        # With a cache the new tokens sit *after* everything already cached, so RoPE has to be
        # read at the absolute positions rather than from 0 — otherwise every generated token
        # would be encoded as if it were at the start of the sequence.
        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        cos, sin = self.rotary(past_len + seq_len)
        cos, sin = cos[past_len:], sin[past_len:]

        attn_mask = self.build_attn_mask(attention_mask, seq_len, x.device, x.dtype)
        presents = [] if use_cache else None
        for i, layer in enumerate(self.decoder_layers):
            past = past_key_values[i] if past_key_values else None
            out = layer(x, cos, sin, attn_mask, past, use_cache)
            if use_cache:
                x, present = out
                presents.append(present)
            else:
                x = out
        x = self.final_norm(x)
        return (x, presents) if use_cache else x

    def get_model_stats(self, verbose=True):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        num_params = sum(p.nelement() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        num_buffers = sum(b.nelement() for b in self.buffers())
        embedding_params = self.token_embedding.weight.nelement()

        stats = {
            "num_params": num_params,
            "num_params_non_embedding": num_params - embedding_params,
            "num_buffers": num_buffers,
            "param size (MB)": param_size / 1024**2,
            "buffer size (MB)": buffer_size / 1024**2,
        }
        if verbose:
            print("-" * 20)
            for key, value in stats.items():
                print(f"{key:<26}: {value:,.2f}" if isinstance(value, float) else f"{key:<26}: {value:,}")
            print("-" * 20)
        return stats


class DecoderLMHeadModel(DecoderModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if config.get("tie_embeddings", True):
            # One matrix serving both directions. Saves 22% of the parameters at Stage-B size,
            # and small models generally benefit from the shared representation.
            self.head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Residual projections are scaled down so that summing num_layers residual branches does
        # not inflate the variance of the residual stream.
        residual_scale = 1.0 / math.sqrt(2 * self.num_layers)
        for name, param in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 * residual_scale)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        out = super().forward(input_ids, attention_mask, past_key_values, use_cache)
        if use_cache:
            hidden_states, presents = out
            return self.head(hidden_states), presents
        return self.head(out)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, temperature=1.0, top_k=None, top_p=None,
                 eos_token_id=None, stop_token_ids=None):
        """Sample a continuation, reusing the KV cache.

        Without the cache each new token re-runs attention over the whole prefix, making
        generation quadratic — which is why sampling used to be too expensive to log often.

        `stop_token_ids` accepts several terminators. A chat turn ends at `<|end|>`, but the
        model may also emit `</s>` to close the conversation, and stopping on only one of them
        leaves the other running to max_new_tokens.
        """
        stops = set()
        if eos_token_id is not None:
            stops.add(int(eos_token_id))
        if stop_token_ids:
            stops.update(int(t) for t in stop_token_ids)
        self.eval()
        past = None
        generated = input_ids
        step_input = input_ids

        for _ in range(max_new_tokens):
            logits, past = self(step_input, past_key_values=past, use_cache=True)
            logits = logits[:, -1, :].float()

            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                if top_p is not None:
                    ordered, order = logits.sort(dim=-1, descending=True)
                    cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
                    # keep the first token that crosses the threshold, drop the rest
                    remove = cumulative - ordered.softmax(dim=-1) > top_p
                    ordered = ordered.masked_fill(remove, float("-inf"))
                    logits = ordered.gather(1, order.argsort(dim=-1))
                next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)
            step_input = next_token          # only the new token goes through the model
            if stops and all(int(t) in stops for t in next_token.flatten()):
                break

        return generated

    def calc_grad_norms(self):
        grad_norms = {}
        total_sq_norm = 0.0
        for i, layer in enumerate(self.decoder_layers, 1):
            layer_sq_norm = sum(
                p.grad.detach().pow(2).sum().item()
                for p in layer.parameters() if p.grad is not None
            )
            grad_norms[f"layer{i}_grad"] = layer_sq_norm ** 0.5
            total_sq_norm += layer_sq_norm

        head_sq_norm = sum(
            p.grad.detach().pow(2).sum().item()
            for p in self.head.parameters() if p.grad is not None
        )
        grad_norms["head_grad"] = head_sq_norm ** 0.5
        # With tied embeddings the head shares the embedding matrix, so it is already counted.
        if self.head.weight is not self.token_embedding.weight:
            total_sq_norm += head_sq_norm

        grad_norms["model_grad"] = total_sq_norm ** 0.5
        return grad_norms
