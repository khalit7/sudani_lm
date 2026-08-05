"""Training entry point.

    python train.py --config configs/stage_a.yaml
    torchrun --nproc_per_node=2 train.py --config configs/stage_a.yaml
    python train.py --config configs/stage_a.yaml --resume checkpoints/<project>/<run>/last.pt
"""

import argparse
from pathlib import Path

import yaml

from src.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path, default=None,
                        help="checkpoint to continue from; restores step, optimiser and data cursor")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="start a new stage from these weights only; step, optimiser and "
                             "data cursor all begin fresh")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="override train.max_steps, for smoke runs")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    if args.max_steps is not None:
        config["train"]["max_steps"] = args.max_steps
    if args.no_wandb:
        config["train"]["wandb"] = False

    Trainer(
        config,
        resume_from=str(args.resume) if args.resume else None,
        init_from=str(args.init_from) if args.init_from else None,
    ).train()


if __name__ == "__main__":
    main()
