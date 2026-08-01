"""Learning-rate schedule as a pure function of the step.

A plain function rather than a torch LRScheduler object: resuming then needs only the step
number, with no scheduler state to save, restore, or accidentally mismatch. The previous setup
stored `lr_scheduler_state_dict` and read `lr_state_dict`, which made every resume raise.
"""

import math


def lr_at_step(
    step: int,
    max_steps: int,
    peak_lr: float,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """Linear warmup, then cosine decay to `min_lr_ratio * peak_lr`.

    Decaying to a floor rather than to zero: the last stretch of training at a near-zero LR
    contributes almost nothing, and a floor keeps the model adapting if the run is extended.
    """
    min_lr = peak_lr * min_lr_ratio

    if warmup_steps > 0 and step < warmup_steps:
        # step+1 so the very first step is not exactly zero
        return peak_lr * (step + 1) / warmup_steps

    decay_steps = max(max_steps - warmup_steps, 1)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
