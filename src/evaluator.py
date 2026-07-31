"""Pluggable evaluators.

Each returns a plain dict of metrics rather than logging directly, so they are testable without
a wandb run and the trainer owns all logging. They run on rank 0 only: every one of them is small
next to a training step, and sharding them would buy nothing.

The headline metric for this project is Sudanese, not MMLU. ArabicMMLU contains no Sudan content
at all and no dialectal MMLU covers Sudanese, so MMLU is kept as an MSA regression guard while
`FloresPerplexityEvaluator` tracks what the project is actually for.
"""

import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

data_root = Path("~/sudani_lm/data").expanduser()


class Evaluator(ABC):
    name = "evaluator"

    def __init__(self, frequency: int = 500, run_at_0: bool = True) -> None:
        self.frequency = frequency
        self.run_at_0 = run_at_0

    def should_run(self, step: int) -> bool:
        if step == 0:
            return self.run_at_0
        return self.frequency > 0 and step % self.frequency == 0

    @abstractmethod
    def evaluate(self, model, device, tokenizer) -> dict:
        """Return {metric_name: value}. Values may be floats or wandb objects."""


def last_real_token_logits(output, attention_mask):
    """Logits at each row's final non-padding position.

    `output[:, -1, :]` is the last position of the *padded* batch, which is a <pad> token for
    every row shorter than the batch maximum, so it reads the model's state at padding rather
    than at the end of the prompt.
    """
    last_idx = attention_mask.sum(dim=-1) - 1
    return output[torch.arange(output.shape[0], device=output.device), last_idx]


# --------------------------------------------------------------------------- MMLU -----------

class MMLULetterEvaluator(Evaluator):
    """Score the option *letter* the model would emit after the answer cue.

    Secondary metric. Letter-symbol binding is a capability that emerges well above 110M
    parameters, so this is expected to read near chance for a long time — it is here to catch
    regressions, not to show progress.
    """

    name = "mmlu_letter"

    def __init__(self, dataloader, frequency=500, run_at_0=True) -> None:
        super().__init__(frequency, run_at_0)
        self.dataloader = dataloader

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer) -> dict:
        options_ids = torch.tensor(self.dataloader.dataset.options_ids, device=device)
        correct = total = 0
        chance = 0.0
        predictions = Counter()

        for X, (Y, n_options) in self.dataloader:
            X = {k: (v.to(device) if v is not None else None) for k, v in X.items()}
            Y = Y.to(device)
            n_options = n_options.to(device)

            logits = last_real_token_logits(model(**X), X["attention_mask"])
            scored = logits[:, options_ids]
            slots = torch.arange(scored.shape[1], device=device)
            scored = scored.masked_fill(slots[None, :] >= n_options[:, None], float("-inf"))

            pred = scored.argmax(dim=-1)
            correct += (pred == Y).sum().item()
            total += Y.numel()
            chance += (1.0 / n_options).sum().item()
            predictions.update(pred.cpu().tolist())

        metrics = {
            "mmlu/letter_acc": correct / max(total, 1),
            # Chance is not 0.2: option counts vary from 2 to 5 across the benchmark.
            "mmlu/chance": chance / max(total, 1),
        }
        for i in range(5):
            metrics[f"mmlu/letter_pred_frac_{i}"] = predictions.get(i, 0) / max(total, 1)
        return metrics


class MMLULoglikelihoodEvaluator(Evaluator):
    """Score each option's *text* as a continuation, length-normalized.

    Primary MMLU variant. It asks "which answer does the model find most plausible" rather than
    "can the model bind an answer to a letter symbol", which is the only one of the two a model
    this size can be expected to do.
    """

    name = "mmlu_loglikelihood"

    def __init__(self, dataset, frequency=500, run_at_0=True, max_examples=1000,
                 batch_size=32) -> None:
        super().__init__(frequency, run_at_0)
        self.dataset = dataset
        self.max_examples = max_examples
        self.batch_size = batch_size

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer) -> dict:
        raw = self.dataset.dataset  # the underlying HF dataset
        n = min(self.max_examples, len(raw)) if self.max_examples else len(raw)
        pad_id = tokenizer.pad_token_id

        correct = 0
        chance = 0.0
        for start in range(0, n, self.batch_size):
            rows = [raw[i] for i in range(start, min(start + self.batch_size, n))]
            sequences, spans, groups = [], [], []
            for row in rows:
                options = [row[f"Option {i}"] for i in range(1, 6)
                           if row.get(f"Option {i}") not in (None, "")]
                context = row["Question"]
                if row.get("Context"):
                    context = f"{row['Context']}\n{context}"
                prompt_ids = tokenizer.encode(f"[inst] {context} [/inst]\n",
                                              add_special_tokens=False)
                group = []
                for option in options:
                    option_ids = tokenizer.encode(f" {option}", add_special_tokens=False)
                    if not option_ids:
                        option_ids = [tokenizer.unk_token_id]
                    group.append(len(sequences))
                    sequences.append(prompt_ids + option_ids)
                    spans.append((len(prompt_ids), len(option_ids)))
                groups.append((group, ord(row["Answer Key"].strip().lower()) - ord("a")))
                chance += 1.0 / len(options)

            scores = self._score(model, device, sequences, spans, pad_id)
            for group, gold in groups:
                best = max(group, key=lambda idx: scores[idx])
                if group.index(best) == gold:
                    correct += 1

        return {
            "mmlu/loglikelihood_acc": correct / max(n, 1),
            "mmlu/loglikelihood_chance": chance / max(n, 1),
        }

    def _score(self, model, device, sequences, spans, pad_id):
        """Mean log-probability per continuation token, for each sequence."""
        width = max(len(s) for s in sequences)
        input_ids = torch.full((len(sequences), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq)
            mask[i, : len(seq)] = 1
        input_ids, mask = input_ids.to(device), mask.to(device)

        logits = model(input_ids, mask).float()
        log_probs = logits.log_softmax(dim=-1)

        scores = []
        for i, (prompt_len, option_len) in enumerate(spans):
            # token at position p predicts the token at p+1, so the continuation's log-probs
            # live at positions [prompt_len-1, prompt_len+option_len-1)
            total = 0.0
            for offset in range(option_len):
                pos = prompt_len - 1 + offset
                total += log_probs[i, pos, input_ids[i, pos + 1]].item()
            scores.append(total / option_len)   # length-normalized, or long answers always lose
        return scores


# ------------------------------------------------------------------------ Sudanese ----------

class FloresPerplexityEvaluator(Evaluator):
    """Perplexity on Sudanese_Flores, plus the MSA side as a control.

    The project's independent Sudanese signal. Held-out WhatsApp perplexity is self-referential —
    the same 1,169 people, topics and idiolect — so a model can score well on it while having
    learned nothing generalizable. Flores is unrelated text, and having both sides of the same
    sentences means the Sudanese/MSA gap is measured on identical content.
    """

    name = "flores"

    def __init__(self, split="DEV", frequency=500, run_at_0=True, batch_size=16,
                 max_examples=None) -> None:
        super().__init__(frequency, run_at_0)
        self.path = data_root / "raw" / "sudanese_flores" / f"{split}.jsonl"
        self.batch_size = batch_size
        self.max_examples = max_examples
        self._pairs = None

    def _load(self):
        if self._pairs is None:
            with open(self.path, encoding="utf-8") as fh:
                rows = [json.loads(line)["translation"] for line in fh]
            if self.max_examples:
                rows = rows[: self.max_examples]
            self._pairs = rows
        return self._pairs

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer) -> dict:
        metrics = {}
        for side, key in (("sudanese", "Sud"), ("msa", "Arb")):
            texts = [row[key] for row in self._load()]
            metrics[f"flores/{side}_ppl"] = self._perplexity(model, device, tokenizer, texts)
        # The gap is the interesting number: it should shrink as the model adapts to dialect.
        metrics["flores/sud_minus_msa_ppl"] = (
            metrics["flores/sudanese_ppl"] - metrics["flores/msa_ppl"]
        )
        return metrics

    def _perplexity(self, model, device, tokenizer, texts):
        pad_id = tokenizer.pad_token_id
        total_nll = 0.0
        total_tokens = 0
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = [tokenizer.encode(t, add_special_tokens=False)[:512] for t in batch]
            encoded = [e for e in encoded if len(e) > 1]
            if not encoded:
                continue
            width = max(len(e) for e in encoded)
            ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(encoded), width), dtype=torch.long)
            for i, seq in enumerate(encoded):
                ids[i, : len(seq)] = torch.tensor(seq)
                mask[i, : len(seq)] = 1
            ids, mask = ids.to(device), mask.to(device)

            logits = model(ids, mask).float()
            targets = ids[:, 1:].reshape(-1)
            predictions = logits[:, :-1].reshape(-1, logits.shape[-1])
            # only score positions where both the input and the target are real tokens
            valid = (mask[:, 1:].reshape(-1) == 1)
            nll = F.cross_entropy(predictions[valid], targets[valid], reduction="sum")
            total_nll += nll.item()
            total_tokens += int(valid.sum().item())

        return math.exp(min(total_nll / max(total_tokens, 1), 20))


# ---------------------------------------------------------------------- generation ----------

class GenerationEvaluator(Evaluator):
    """Sample continuations from fixed prompts, for eyeballing.

    Uses the model's KV cache, so this is linear in the number of generated tokens rather than
    quadratic and cheap enough to run often.
    """

    name = "generation"

    def __init__(self, prompts, temperatures=(0.0, 0.7), max_new_tokens=64, top_k=50,
                 frequency=1000, run_at_0=False) -> None:
        super().__init__(frequency, run_at_0)
        self.prompts = list(prompts)
        self.temperatures = list(temperatures)
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k

    @torch.no_grad()
    def evaluate(self, model, device, tokenizer) -> dict:
        samples = []
        for prompt in self.prompts:
            ids = torch.tensor(
                [tokenizer.encode(prompt, add_special_tokens=False) or [tokenizer.bos_token_id]],
                device=device,
            )
            for temperature in self.temperatures:
                out = model.generate(
                    ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=temperature,
                    top_k=self.top_k if temperature > 0 else None,
                    eos_token_id=tokenizer.eos_token_id,
                )
                text = tokenizer.decode(out[0].tolist(), skip_special_tokens=False)
                samples.append({"prompt": prompt, "temperature": temperature, "text": text})
        return {"generation/samples": samples}
