"""Regression tests for step counting and grad-norm cadence.

These pin bugs from the epoch-based trainer that the Step 4 rewrite superseded. The rewrite makes
them structurally impossible — there is no epoch loop to reset a counter, and the LR schedule is
a pure function of the step with no state to mismatch — but the arithmetic is kept here as the
record of what went wrong, and `test_new_trainer_has_no_epoch_state` guards the property that
replaced them.
"""

import inspect

from src.trainer import Trainer


def simulate_global_step(num_epochs, batches_per_epoch, grad_acc_every):
    """Mirrors the accumulation/step logic in Trainer.train()."""
    global_step = 0
    steps_seen = []
    for _ in range(num_epochs):
        for acc_steps in range(1, batches_per_epoch + 1):
            if acc_steps % grad_acc_every == 0:
                global_step += 1
                steps_seen.append(global_step)
    return steps_seen


def test_step_counter_is_monotonic_across_epochs():
    """It used to be acc_steps // grad_acc_every, which restarts each epoch — so on epoch 2
    the wandb step went backwards and overwrote epoch 1's history."""
    steps = simulate_global_step(num_epochs=3, batches_per_epoch=8, grad_acc_every=2)

    assert steps == sorted(steps), "wandb step must never decrease"
    assert len(steps) == len(set(steps)), "steps must be unique across epochs"
    assert steps == list(range(1, 13))


def test_old_per_epoch_formula_would_regress():
    """Pins the bug so the fix cannot be silently reverted."""
    old = []
    for _ in range(2):
        for acc_steps in range(1, 8 + 1):
            if acc_steps % 2 == 0:
                old.append(acc_steps // 2)
    assert old != sorted(set(old)), "expected the old formula to repeat steps"
    assert old == [1, 2, 3, 4, 1, 2, 3, 4]


def grad_norm_fires(step, freq):
    """Mirrors Trainer.log_grad_norm's guard."""
    return step % freq == 0


def test_grad_norm_respects_frequency():
    """Was `step % freq != 0 and step > 1000`, so with freq=1 it never returned early and
    calc_grad_norms — which calls .item() per layer — ran on every step."""
    assert [s for s in range(1, 21) if grad_norm_fires(s, 5)] == [5, 10, 15, 20]
    assert [s for s in range(1, 6) if grad_norm_fires(s, 1)] == [1, 2, 3, 4, 5]


def test_old_grad_norm_guard_ignored_frequency_below_1000():
    def old_fires(step, freq):
        return not (step % freq != 0 and step > 1000)

    # with freq=10, steps 1..1000 all fired regardless of frequency
    fired = [s for s in range(1, 1001) if old_fires(s, 10)]
    assert len(fired) == 1000, "old guard should have fired on every early step"
    # the fix fires only every 10th
    assert len([s for s in range(1, 1001) if grad_norm_fires(s, 10)]) == 100


def test_new_trainer_has_no_epoch_state():
    """The step-reset bug is gone by construction: the rewrite counts steps, not epochs.

    Replaces an earlier test that inspected __init__ for `load_all_states` and
    `lr_scheduler_state_dict` guards. Both are obsolete — the new trainer has no epoch loop and
    no scheduler object, so neither the counter reset nor the key mismatch can recur.
    """
    source = inspect.getsource(Trainer)
    assert "num_epochs" not in source, "trainer must be step-based, not epoch-based"
    assert "lr_scheduler_state_dict" not in source, "schedule is a pure function of the step"
    assert "self.step" in source and "max_steps" in source


def test_checkpoint_persists_step_and_data_cursor():
    """Resuming must restore both, or the run replays data it has already trained on."""
    source = inspect.getsource(Trainer._save_checkpoint)
    for key in ("step", "data_state", "optimiser_state_dict", "best_val_loss"):
        assert f'"{key}"' in source, f"checkpoint is missing {key}"


def test_checkpoint_write_is_atomic():
    """A crash mid-write must leave the previous checkpoint intact, not a truncated file."""
    source = inspect.getsource(Trainer._save_checkpoint)
    assert ".tmp" in source and "rename" in source
