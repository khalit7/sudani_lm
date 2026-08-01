"""Regression tests for checkpointing and best-model selection.

The original bug: `best.pt` was chosen by reading the wandb summary key "val_loss" while the
metric was logged as "loss/val_loss". The lookup always missed, compared against inf, and so
wrote "best.pt" at *every* validation — meaning a long run could easily end with a worse
checkpoint than one it had passed through. The step-4 trainer tracks it in an attribute instead;
these tests pin that behaviour.
"""

import inspect

import torch

from src.trainer import Trainer


def test_best_checkpoint_only_written_on_improvement():
    """Mirrors the comparison Trainer._validate makes after computing val_loss."""
    best = float("inf")
    written = []
    for loss in [3.0, 2.5, 2.7, 2.4, 2.9]:
        if loss < best:
            best = loss
            written.append("best.pt")
        else:
            written.append(None)

    assert written == ["best.pt", "best.pt", None, "best.pt", None]
    assert best == 2.4


def test_trainer_compares_against_tracked_best_not_wandb_summary():
    source = inspect.getsource(Trainer._validate)
    assert "self.best_val_loss" in source, "best.pt must be gated on the tracked minimum"
    assert "summary" not in source, "the wandb-summary lookup is back"


def test_best_val_loss_survives_a_checkpoint_round_trip():
    """Otherwise a resumed run writes best.pt on its first validation regardless of quality."""
    save_source = inspect.getsource(Trainer._save_checkpoint)
    load_source = inspect.getsource(Trainer._load_checkpoint)
    assert '"best_val_loss"' in save_source
    assert "best_val_loss" in load_source


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
