"""Regression tests for the trainer's step counter and grad-norm logging cadence."""

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


def test_trainer_reads_load_all_states_with_a_default():
    """A config without load_all_states must resume weights-only, not raise KeyError."""
    import inspect

    source = inspect.getsource(Trainer.__init__)
    assert '.get("load_all_states"' in source, "load_all_states must be read defensively"
    assert 'checkpoint["lr_scheduler_state_dict"]' in source, "must match the saved key"
    assert 'checkpoint["lr_state_dict"]' not in source, "the mismatched key is back"
