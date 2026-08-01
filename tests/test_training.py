"""Gate for the trainer rewrite (Part I, Step 4).

Covers the two things that silently corrupt long runs: an LR schedule that never completes its
decay, and a resume that replays data instead of continuing.
"""

import numpy as np
import pytest
import torch

from src.training.data import PackedBatchLoader, SequentialPackedLoader
from src.training.schedule import lr_at_step


class FakePacked:
    """Stands in for PackedDataset: the loaders only need .tokens and .block_size."""

    def __init__(self, n_tokens=100_000, block_size=64):
        self.tokens = np.arange(n_tokens, dtype=np.uint16)
        self.block_size = block_size
        self.n_blocks = (n_tokens - 1) // block_size

    def __len__(self):
        return self.n_blocks


# --- learning-rate schedule -------------------------------------------------------------------

def test_warmup_rises_then_decays():
    peak, steps, warmup = 1e-3, 1000, 100
    lrs = [lr_at_step(s, steps, peak, warmup) for s in range(steps)]
    assert lrs[0] < lrs[warmup - 1]
    assert abs(lrs[warmup - 1] - peak) < 1e-9
    assert lrs[-1] < lrs[warmup]


def test_decay_reaches_the_floor_not_zero():
    """The old SequentialLR annealed to 0 — and, sized to an epoch nothing finished, never got
    there at all. The floor keeps a resumed or extended run learning."""
    peak, steps, warmup = 6e-4, 1000, 20
    final = lr_at_step(steps - 1, steps, peak, warmup, min_lr_ratio=0.1)
    assert final == pytest.approx(peak * 0.1, rel=0.02)
    assert final > 0


def test_schedule_completes_within_max_steps():
    peak, steps, warmup = 6e-4, 500, 10
    assert lr_at_step(steps, steps, peak, warmup) == pytest.approx(peak * 0.1, rel=1e-6)
    # and stays at the floor if the run is extended past max_steps
    assert lr_at_step(steps * 2, steps, peak, warmup) == pytest.approx(peak * 0.1, rel=1e-6)


def test_schedule_is_a_pure_function_of_step():
    """No internal state means resume needs only the step number."""
    a = [lr_at_step(s, 100, 1e-3, 10) for s in range(100)]
    b = [lr_at_step(s, 100, 1e-3, 10) for s in reversed(range(100))][::-1]
    assert a == b


# --- resumable data cursor ---------------------------------------------------------------------

def test_resume_continues_instead_of_replaying():
    """Without the data cursor a resumed run re-reads blocks it already trained on, and the loss
    curve dips in a way that looks like progress."""
    dataset = FakePacked()
    loader = PackedBatchLoader(dataset, batch_size=4, seed=1)
    first = [next(loader)[0] for _ in range(5)]
    state = loader.state_dict()
    expected_next = [next(loader)[0] for _ in range(3)]

    resumed = PackedBatchLoader(dataset, batch_size=4, seed=1)
    resumed.load_state_dict(state)
    got_next = [next(resumed)[0] for _ in range(3)]

    for a, b in zip(expected_next, got_next):
        assert torch.equal(a, b)
    # and it did not restart from the beginning
    assert not torch.equal(got_next[0], first[0])


def test_position_advances_by_global_batch():
    loader = PackedBatchLoader(FakePacked(), batch_size=4, world_size=2)
    next(loader)
    assert loader.position == 8, "position counts samples across all ranks"


def test_ranks_see_disjoint_blocks():
    dataset = FakePacked()
    a = PackedBatchLoader(dataset, batch_size=4, rank=0, world_size=2, seed=3)
    b = PackedBatchLoader(dataset, batch_size=4, rank=1, world_size=2, seed=3)
    xa, _ = next(a)
    xb, _ = next(b)
    starts_a = {int(row[0]) for row in xa}
    starts_b = {int(row[0]) for row in xb}
    assert starts_a.isdisjoint(starts_b), "ranks must not train on the same blocks"


def test_batches_have_constant_shape_across_the_epoch_boundary():
    """A short final batch would make torch.compile recompile and step time lurch."""
    dataset = FakePacked(n_tokens=64 * 20 + 1, block_size=64)   # 20 blocks
    loader = PackedBatchLoader(dataset, batch_size=6, seed=0)
    shapes = {tuple(next(loader)[0].shape) for _ in range(12)}
    assert shapes == {(6, 64)}


def test_labels_are_inputs_shifted_by_one():
    loader = PackedBatchLoader(FakePacked(), batch_size=3, seed=0)
    x, y = next(loader)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_shuffle_off_is_sequential():
    loader = PackedBatchLoader(FakePacked(), batch_size=2, shuffle=False)
    x, _ = next(loader)
    assert int(x[0][0]) == 0 and int(x[1][0]) == 64


# --- validation loader --------------------------------------------------------------------------

def test_sequential_loader_shards_across_ranks():
    dataset = FakePacked()
    a = list(SequentialPackedLoader(dataset, batch_size=4, rank=0, world_size=2, max_batches=3))
    b = list(SequentialPackedLoader(dataset, batch_size=4, rank=1, world_size=2, max_batches=3))
    starts_a = {int(r[0]) for x, _ in a for r in x}
    starts_b = {int(r[0]) for x, _ in b for r in x}
    assert starts_a.isdisjoint(starts_b)


def test_sequential_loader_is_deterministic():
    dataset = FakePacked()
    first = [x for x, _ in SequentialPackedLoader(dataset, batch_size=4, max_batches=3)]
    second = [x for x, _ in SequentialPackedLoader(dataset, batch_size=4, max_batches=3)]
    for a, b in zip(first, second):
        assert torch.equal(a, b)


# --- optimiser param groups ----------------------------------------------------------------------

def test_weight_decay_applies_to_matrices_only():
    """Decaying 1-D parameters (norms, biases) shrinks weights with no scale redundancy."""
    from src.factory import Factory
    from src.models.decoder import DecoderLMHeadModel

    model = DecoderLMHeadModel({
        "vocab_size": 128, "d_model": 32, "num_layers": 2, "num_heads": 4, "max_seq_len": 16,
    })
    factory = Factory({"train": {"optimiser": {"name": "adamw", "config": {
        "lr": 1e-3, "betas": [0.9, 0.95], "weight_decay": 0.1,
    }}}})
    opt = factory.get_optimiser(model)

    decay, no_decay = opt.param_groups[0], opt.param_groups[1]
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    assert len(no_decay["params"]) > 0, "norms should be in the no-decay group"
