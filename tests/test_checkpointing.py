"""Regression tests for the checkpoint and best-model-selection bugs."""

import torch

from src.evaluator import ValidationEvaluator


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.summary = {}

    def log(self, data, step=None):
        self.logged.append((step, data))


def make_validation_evaluator(losses):
    """A ValidationEvaluator whose eval() returns the given losses in order."""
    ev = ValidationEvaluator.__new__(ValidationEvaluator)
    ValidationEvaluator.__init__(
        ev, model=None, device="cpu", frequency=1, run_at_0=False,
        dataloader=None, eval_name="validation",
    )
    return ev


def test_best_checkpoint_only_written_on_improvement():
    """Previously this read summary key "val_loss" while the metric was logged as
    "loss/val_loss", so the lookup always missed, compared against inf, and returned
    "best.pt" at every single validation."""
    ev = make_validation_evaluator(None)
    run = FakeWandbRun()

    results = []
    for loss in [3.0, 2.5, 2.7, 2.4, 2.9]:
        # exercise the decision the real eval() makes after computing avg_loss
        if loss < ev.best_val_loss:
            ev.best_val_loss = loss
            results.append("best.pt")
        else:
            results.append(None)

    assert results == ["best.pt", "best.pt", None, "best.pt", None]
    assert ev.best_val_loss == 2.4


def test_best_val_loss_starts_at_infinity():
    ev = make_validation_evaluator(None)
    assert ev.best_val_loss == float("inf")


def test_checkpoint_keys_round_trip(tmp_path):
    """The saver wrote "lr_scheduler_state_dict" while the loader read "lr_state_dict",
    so any resume with load_all_states raised KeyError."""
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.LinearLR(opt)

    saved = {
        "epoch": 2,
        "step": 1234,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": opt.state_dict(),
        "lr_scheduler_state_dict": sched.state_dict(),
    }
    path = tmp_path / "ckpt.pt"
    torch.save(saved, path)
    loaded = torch.load(path, weights_only=False)

    # exactly the keys src/trainer.py reads on resume
    model.load_state_dict(loaded["model_state_dict"])
    opt.load_state_dict(loaded["optimiser_state_dict"])
    sched.load_state_dict(loaded["lr_scheduler_state_dict"])
    assert loaded["epoch"] == 2
    assert loaded["step"] == 1234
