"""Regression tests for the four MMLU bugs that pinned accuracy at chance.

Each test here maps to one of them; test_gold_model_scores_perfect is the end-to-end gate —
it fails if any of the four is reintroduced.
"""

import random

import pytest
import torch

from src.dataset.mmlu import (
    ANSWER_CUE,
    MAX_OPTIONS,
    OPTION_LETTERS,
    MMLU,
    ArabicMMLUDatasetModule,
    get_answer_token_ids,
)
from src.evaluator import last_real_token_logits
from src.tokenizer.utils import get_tokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer()


def make_row(n_options, answer_key="A", question="سؤال"):
    row = {
        "Question": question,
        "Context": "",
        "Answer Key": answer_key,
    }
    for i in range(1, MAX_OPTIONS + 1):
        row[f"Option {i}"] = f"خيار{i}" if i <= n_options else None
    return row


# --- bug 1: the scored token ids were the bare letters, not what the model emits ---------

def test_answer_ids_are_the_in_context_tokens(tokenizer):
    ids = get_answer_token_ids(tokenizer)
    assert len(ids) == MAX_OPTIONS
    assert len(set(ids)) == MAX_OPTIONS, "option letters must map to distinct token ids"

    base = tokenizer.encode(ANSWER_CUE, add_special_tokens=False)
    for letter, token_id in zip(OPTION_LETTERS, ids):
        full = tokenizer.encode(ANSWER_CUE + " " + letter, add_special_tokens=False)
        assert full[len(base)] == token_id
        assert token_id != tokenizer.unk_token_id


def test_answer_ids_differ_from_the_naive_bare_letter_ids(tokenizer):
    """The original bug in one assertion: the naive lookup disagrees with reality."""
    naive = tokenizer.convert_tokens_to_ids(OPTION_LETTERS)
    correct = get_answer_token_ids(tokenizer)
    assert naive != correct


# --- bug 2: logits were read at a padding position ---------------------------------------

def test_last_real_token_logits_skips_padding():
    # row 0 is 2 tokens then padded, row 1 fills all 4
    attention_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    output = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    got = last_real_token_logits(output, attention_mask)

    assert torch.equal(got[0], output[0, 1])   # last *real* token, not output[0, -1]
    assert torch.equal(got[1], output[1, 3])
    assert not torch.equal(got[0], output[0, -1]), "must not read the pad position"


# --- bug 3: the option count was hardcoded to 5 -------------------------------------------

@pytest.mark.parametrize("n_options", [2, 3, 4, 5])
def test_only_existing_options_are_rendered(tokenizer, n_options):
    ds = MMLU([make_row(n_options)], tokenizer)
    text, answer_idx, got_n = ds[0]

    assert got_n == n_options
    assert "None" not in text, "a missing option leaked into the prompt"
    for letter in OPTION_LETTERS[:n_options]:
        assert f"\n{letter} " in text or text.startswith(f"{letter} ")
    for letter in OPTION_LETTERS[n_options:]:
        assert f"\n{letter} " not in text
    assert 0 <= answer_idx < n_options


def test_prompt_ends_with_the_answer_cue(tokenizer):
    ds = MMLU([make_row(4)], tokenizer)
    text, _, _ = ds[0]
    assert text.endswith(ANSWER_CUE), "without a cue the letter is never the natural next token"


# --- bug 4: the option shuffle was unseeded ------------------------------------------------

def test_shuffle_is_deterministic_per_index(tokenizer):
    rows = [make_row(4, answer_key="B") for _ in range(8)]
    a = MMLU(rows, tokenizer)
    b = MMLU(rows, tokenizer)
    random.seed(1234)  # global RNG must not influence the result
    first = [a[i] for i in range(len(rows))]
    random.seed(999)
    second = [b[i] for i in range(len(rows))]
    assert first == second


def test_shuffle_actually_moves_the_answer(tokenizer):
    """Guards against a 'deterministic' fix that just stopped shuffling."""
    rows = [make_row(4, answer_key="A") for _ in range(40)]
    ds = MMLU(rows, tokenizer)
    positions = {ds[i][1] for i in range(len(rows))}
    assert len(positions) > 1, "answer never moves; label position is not being shuffled"


def test_answer_index_tracks_the_shuffled_position(tokenizer):
    for key_i, key in enumerate(["A", "B", "C", "D"]):
        ds = MMLU([make_row(4, answer_key=key)], tokenizer)
        text, answer_idx, _ = ds[0]
        gold_letter = OPTION_LETTERS[answer_idx]
        # the option text originally at position key_i must now sit behind gold_letter
        assert f"{gold_letter} خيار{key_i + 1}" in text


# --- end-to-end: a model that emits the gold letter must score 1.0 -------------------------

class GoldModel(torch.nn.Module):
    """Puts all its mass on the correct option letter, at the last real token."""

    def __init__(self, option_ids, gold_positions, vocab_size):
        super().__init__()
        self.option_ids = option_ids
        self.gold_positions = gold_positions
        self.vocab_size = vocab_size
        self._cursor = 0

    def forward(self, input_ids, attention_mask):
        b, s = input_ids.shape
        out = torch.zeros(b, s, self.vocab_size)
        last = attention_mask.sum(-1) - 1
        for row in range(b):
            gold = self.gold_positions[self._cursor + row]
            out[row, last[row], self.option_ids[gold]] = 10.0
        self._cursor += b
        return out


def test_gold_model_scores_perfect(tokenizer):
    """The gate: with all four bugs fixed, a perfect model reads as perfect."""
    rows = [make_row(n, answer_key=k) for n in (2, 3, 4, 5) for k in ("A", "B")]
    ds = MMLU(rows, tokenizer)
    module = ArabicMMLUDatasetModule.__new__(ArabicMMLUDatasetModule)
    module.tokenizer = tokenizer

    batch = [ds[i] for i in range(len(rows))]
    X, (Y, n_options) = module.colllate_fn(batch)

    model = GoldModel(ds.options_ids, Y.tolist(), len(tokenizer))
    output = model(**X)

    logits = last_real_token_logits(output, X["attention_mask"])
    scored = logits[:, torch.tensor(ds.options_ids)]
    slots = torch.arange(scored.shape[1])
    scored = scored.masked_fill(slots.unsqueeze(0) >= n_options.unsqueeze(1), float("-inf"))
    pred = scored.argmax(dim=-1)

    assert torch.equal(pred, Y), "a model answering correctly did not score 1.0"


def test_predictions_never_exceed_the_option_count(tokenizer):
    """A 2-option question must never be answered 'د'."""
    rows = [make_row(2) for _ in range(6)]
    ds = MMLU(rows, tokenizer)
    module = ArabicMMLUDatasetModule.__new__(ArabicMMLUDatasetModule)
    module.tokenizer = tokenizer
    X, (Y, n_options) = module.colllate_fn([ds[i] for i in range(len(rows))])

    # a model biased entirely toward the last letter
    logits = torch.zeros(len(rows), MAX_OPTIONS)
    logits[:, -1] = 99.0
    slots = torch.arange(MAX_OPTIONS)
    logits = logits.masked_fill(slots.unsqueeze(0) >= n_options.unsqueeze(1), float("-inf"))

    assert (logits.argmax(dim=-1) < n_options).all()
