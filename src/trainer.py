"""Step-based trainer with DDP, bf16 autocast, torch.compile and exact resume.

Rewritten from an epoch-based loop whose schedule was sized to an epoch of ~148k steps that no
run ever came close to finishing, so the cosine decay never happened — and the end-of-schedule
anneal is worth a large share of the final loss. Everything here is denominated in *steps* and
*tokens*, which are also what make runs at different batch and model sizes comparable.

Launch:
    python train.py --config configs/pretraining_packed.yaml               # single GPU
    torchrun --nproc_per_node=2 train.py --config configs/...yaml          # both 5090s
"""

import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import wandb
from src.factory import Factory
from src.tokenizer.utils import get_tokenizer
from src.training.data import PackedBatchLoader, SequentialPackedLoader
from src.training.schedule import lr_at_step

CHECKPOINT_ROOT = Path("~/sudani_lm/checkpoints").expanduser()


def setup_distributed():
    """Read the topology torchrun puts in the environment. Single-process run if absent."""
    if "RANK" not in os.environ:
        return 0, 0, 1, False
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size, True


class Trainer:
    def __init__(self, config, resume_from: str | None = None) -> None:
        self.config = config
        self.rank, self.local_rank, self.world_size, self.distributed = setup_distributed()
        self.is_main = self.rank == 0

        train_cfg = config["train"]
        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

        self.tokenizer = get_tokenizer()
        self.vocab_size = len(self.tokenizer)
        config["model"]["config"]["vocab_size"] = self.vocab_size

        factory = Factory(config)
        self.factory = factory

        # ---- batch arithmetic, in tokens ----------------------------------------------------
        self.micro_batch_size = train_cfg["micro_batch_size"]
        self.block_size = train_cfg.get("block_size", 1024)
        self.grad_accum = train_cfg["grad_accum"]
        self.max_steps = train_cfg["max_steps"]
        self.tokens_per_step = (
            self.micro_batch_size * self.block_size * self.grad_accum * self.world_size
        )

        # ---- data ---------------------------------------------------------------------------
        stage = train_cfg["dataloader"]["stage"]
        module = factory.get_packed_module(stage, self.block_size)
        self.train_loader = PackedBatchLoader(
            module.build_dataset("train"),
            batch_size=self.micro_batch_size,
            rank=self.rank,
            world_size=self.world_size,
            seed=train_cfg.get("seed", 67),
            shuffle=True,
        )
        self.val_loader = SequentialPackedLoader(
            module.build_dataset("val"),
            batch_size=train_cfg.get("val_batch_size", self.micro_batch_size),
            rank=self.rank,
            world_size=self.world_size,
            max_batches=train_cfg.get("val_max_batches"),
        )

        # ---- model --------------------------------------------------------------------------
        self.model = factory.get_model().to(self.device)
        self.raw_model = self.model  # unwrapped handle for state_dict and grad norms
        self.model_stats = self.raw_model.get_model_stats(verbose=self.is_main)

        self.compiled = False
        if train_cfg.get("compile", False):
            # Shapes are constant on the packed path, so this compiles once and never recompiles.
            # Compilation is lazy — it only fails on the first real forward — so trigger it here
            # and fall back to eager rather than let a toolchain problem kill a 16-hour run.
            try:
                # torch.compile() itself is inside the try, not just the trial forward: most
                # toolchain failures surface lazily on the first forward, but not all of them,
                # and either way must degrade to eager rather than kill the run.
                candidate = torch.compile(self.model)
                self._trial_forward(candidate)
                self.model = candidate
                self.compiled = True
            except Exception as exc:  # noqa: BLE001 - any compile/toolchain failure
                if self.is_main:
                    print(
                        f"torch.compile unavailable, continuing in eager mode: "
                        f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}",
                        flush=True,
                    )

        self.optimiser = factory.get_optimiser(self.raw_model)
        self.peak_lr = train_cfg["optimiser"]["config"]["lr"]
        self.warmup_steps = max(1, int(self.max_steps * train_cfg["scheduler"]["warmup_percentage"]))
        self.min_lr_ratio = train_cfg["scheduler"].get("min_lr_ratio", 0.1)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)

        self.step = 0
        self.best_val_loss = float("inf")
        if resume_from:
            self._load_checkpoint(resume_from)

        if self.distributed:
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # bf16 needs no GradScaler: it has the same exponent range as fp32, so there is nothing
        # to rescale. That is the whole reason to prefer it over fp16 here.
        self.autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if train_cfg.get("bf16", True) and torch.cuda.is_available()
            else nullcontext()
        )

        # Pluggable evaluators (MMLU, Flores, generation). Built on rank 0 only — each is small
        # next to a training step, and sharding them would buy nothing.
        self.evaluators = factory.build_evaluators() if self.is_main else []

        self.grad_norm_freq = config.get("monitor", {}).get("grad_norm", {}).get("freq", 50)
        self.eval_every = train_cfg.get("eval_every", 500)
        self.checkpoint_every = train_cfg.get("checkpoint_every", 500)
        # float() because YAML 1.1 parses an unsigned exponent ("105.0e12") as a string
        self.peak_flops = float(train_cfg.get("peak_flops_per_gpu", 105e12))

        self.wandb_run = None
        if self.is_main and train_cfg.get("wandb", True):
            config["model"]["stats"] = self.model_stats
            config["train"]["tokens_per_step"] = self.tokens_per_step
            self.wandb_run = wandb.init(
                project=config["run"]["project_name"],
                name=config["run"]["run_name"],
                config=config,
            )
        self.checkpoint_dir = (
            CHECKPOINT_ROOT / config["run"]["project_name"] / config["run"]["run_name"]
        )
        if self.is_main:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _trial_forward(self, model) -> None:
        """One throwaway forward+backward, purely to force lazy compilation to happen now."""
        ids = torch.zeros((1, self.block_size), dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            model(ids).float().mean().backward()
        for param in self.raw_model.parameters():
            param.grad = None

    # ------------------------------------------------------------------ FLOPs accounting ----

    def flops_per_token(self) -> float:
        """6ND for the parameters, plus the attention score/context matmuls.

        The attention term matters at seq 1024: omitting it understates the work and so
        overstates MFU.
        """
        n = self.model_stats["num_params_non_embedding"] + (
            self.raw_model.d_model * self.vocab_size
        )
        layers = self.raw_model.num_layers
        d_model = self.raw_model.d_model
        return 6 * n + 12 * layers * d_model * self.block_size

    # ------------------------------------------------------------------------- training ----

    def train(self) -> None:
        self.model.train()
        if self.is_main:
            print(
                f"steps {self.max_steps} | tokens/step {self.tokens_per_step:,} | "
                f"total {self.max_steps*self.tokens_per_step/1e9:.2f}B | world {self.world_size}",
                flush=True,
            )

        # Baseline before any optimizer step. Without this the first eval point is step
        # eval_every, and there is nothing to compare it against — a random-init model should
        # sit at chance on MMLU and at ln(vocab) loss, which is the cheapest sanity check there
        # is that the eval plumbing itself is correct.
        if self.step == 0:
            self._run_evaluators(force_step_zero=True)
            self.model.train()

        window_start = time.time()
        window_loss = 0.0

        while self.step < self.max_steps:
            lr = lr_at_step(
                self.step, self.max_steps, self.peak_lr, self.warmup_steps, self.min_lr_ratio
            )
            for group in self.optimiser.param_groups:
                group["lr"] = lr

            accum_loss = 0.0
            for micro in range(self.grad_accum):
                inputs, labels = next(self.train_loader)
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Skip the gradient all-reduce on every micro-step except the last; otherwise
                # DDP synchronises grad_accum times per optimizer step for no benefit.
                sync = (not self.distributed) or (micro == self.grad_accum - 1)
                ctx = nullcontext() if sync else self.model.no_sync()
                with ctx, self.autocast:
                    logits = self.model(inputs)
                    # No explicit .float() here: autocast already runs cross_entropy in fp32
                    # internally, and materializing an fp32 copy of a
                    # (micro_batch * block, 32000) tensor costs 8+ GB on its own.
                    loss = F.cross_entropy(
                        logits.view(-1, self.vocab_size), labels.reshape(-1)
                    )
                (loss / self.grad_accum).backward()
                accum_loss += loss.detach() / self.grad_accum

            # clip_grad_norm_ returns the pre-clip total norm, so grad-norm logging is free —
            # the old per-layer .item() loop forced a GPU sync on every step.
            grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), self.grad_clip)
            self.optimiser.step()
            self.optimiser.zero_grad(set_to_none=True)
            self.step += 1
            window_loss += accum_loss.item()

            if self.step % self.grad_norm_freq == 0:
                self._log({"grad_norm": grad_norm.item()})

            if self.step % self.config["train"].get("log_every", 10) == 0:
                elapsed = time.time() - window_start
                steps = self.config["train"].get("log_every", 10)
                tokens_per_sec = steps * self.tokens_per_step / max(elapsed, 1e-9)
                mfu = (
                    self.flops_per_token() * tokens_per_sec
                    / (self.peak_flops * self.world_size)
                )
                self._log({
                    "loss/train_loss": window_loss / steps,
                    "learning_rate": lr,
                    "throughput/tokens_per_sec": tokens_per_sec,
                    "throughput/mfu": mfu,
                    "progress/tokens_seen": self.step * self.tokens_per_step,
                })
                if self.is_main:
                    print(
                        f"step {self.step:>6}/{self.max_steps}  loss {window_loss/steps:.4f}  "
                        f"lr {lr:.2e}  {tokens_per_sec/1e3:.0f}k tok/s  MFU {mfu*100:.1f}%",
                        flush=True,
                    )
                window_loss = 0.0
                window_start = time.time()

            if self.step % self.eval_every == 0:
                self._validate()
                self._run_evaluators()
                self.model.train()
                window_start = time.time()

            if self.step % self.checkpoint_every == 0:
                self._save_checkpoint("last.pt")

        self._validate()
        self._save_checkpoint("final.pt")
        if self.distributed:
            dist.barrier()
            dist.destroy_process_group()

    # ----------------------------------------------------------------------- validation ----

    @torch.no_grad()
    def _validate(self) -> None:
        self.model.eval()
        total_loss = torch.zeros((), device=self.device)
        total_tokens = torch.zeros((), device=self.device)
        for inputs, labels in self.val_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with self.autocast:
                logits = self.model(inputs)
                # summed, then divided by the true token count — a mean of per-batch means would
                # weight batches equally regardless of how many tokens they carry
                batch_loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size), labels.reshape(-1), reduction="sum"
                )
            total_loss += batch_loss.float()
            total_tokens += labels.numel()

        if self.distributed:
            dist.all_reduce(total_loss)
            dist.all_reduce(total_tokens)

        val_loss = (total_loss / total_tokens.clamp(min=1)).item()
        self._log({"loss/val_loss": val_loss, "loss/val_ppl": math.exp(min(val_loss, 20))})
        if self.is_main:
            print(f"  val loss {val_loss:.4f}  ppl {math.exp(min(val_loss,20)):.1f}", flush=True)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best.pt")

    @torch.no_grad()
    def _run_evaluators(self, force_step_zero: bool = False) -> None:
        """Run the configured evaluators and log whatever they return.

        Each is wrapped: a broken eval must not take down a run that is otherwise healthy, and
        an eval that depends on missing data should degrade to a warning rather than a crash
        several hours in.
        """
        if not self.evaluators:
            return
        self.model.eval()
        for evaluator in self.evaluators:
            if not evaluator.should_run(0 if force_step_zero else self.step):
                continue
            try:
                metrics = evaluator.evaluate(self.raw_model, self.device, self.tokenizer)
            except Exception as exc:  # noqa: BLE001
                print(f"  eval {evaluator.name} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            samples = metrics.pop("generation/samples", None)
            if samples is not None:
                self._log_samples(samples)
            self._log(metrics)
            if metrics:
                summary = "  ".join(f"{k.split('/')[-1]} {v:.4f}" for k, v in metrics.items())
                print(f"  {evaluator.name}: {summary}", flush=True)

    def _log_samples(self, samples) -> None:
        for sample in samples:
            print(
                f"    [T={sample['temperature']}] {sample['text'][:200]!r}",
                flush=True,
            )
        if self.wandb_run is not None:
            table = wandb.Table(columns=["step", "prompt", "temperature", "text"])
            for sample in samples:
                table.add_data(self.step, sample["prompt"], sample["temperature"], sample["text"])
            self.wandb_run.log({"generation": table}, step=self.step)

    # ------------------------------------------------------------------------ bookkeeping ----

    def _log(self, values: dict) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(values, step=self.step)

    def _save_checkpoint(self, name: str) -> None:
        if not self.is_main:
            return
        payload = {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "model_state_dict": self.raw_model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "data_state": self.train_loader.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.config,
        }
        # Write then rename: a crash mid-write leaves the previous checkpoint intact rather
        # than a truncated file that fails to load.
        tmp = self.checkpoint_dir / f".{name}.tmp"
        torch.save(payload, tmp)
        tmp.rename(self.checkpoint_dir / name)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.step = checkpoint["step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        # Restoring the data cursor is what makes a resumed run continue rather than replay:
        # without it the model would see the same blocks again and the loss curve would dip.
        self.train_loader.load_state_dict(checkpoint["data_state"])
        if checkpoint.get("rng_state") is not None:
            torch.set_rng_state(checkpoint["rng_state"].cpu().to(torch.uint8))
        if self.is_main:
            print(
                f"resumed from {path} at step {self.step} "
                f"(data position {self.train_loader.position:,})",
                flush=True,
            )
