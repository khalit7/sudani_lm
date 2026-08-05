"""Batch loader for packed token streams.

Deliberately not a torch DataLoader. Reading a packed block is a memmap slice, so worker
processes and IPC would cost more than they save, and a DataLoader cannot be resumed
mid-epoch — which the trainer needs, since a 16-hour run must survive a crash without
replaying data it has already seen.

Instead: a deterministic permutation of block indices, sharded across ranks, addressed by an
absolute position. Resuming is restoring one integer.
"""

import numpy as np
import torch


class PackedBatchLoader:
    """Yields (inputs, labels) batches of fixed shape from a PackedDataset.

    position counts *samples consumed globally* across all ranks, so it is independent of world
    size and stays meaningful if a run is resumed on a different number of GPUs.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 67,
        shuffle: bool = True,
        position: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.position = position
        self.n_blocks = len(dataset)
        self.block_size = dataset.block_size
        self._epoch = -1
        self._order = None

    def _order_for_epoch(self, epoch: int) -> np.ndarray:
        if self._epoch != epoch:
            if self.shuffle:
                rng = np.random.default_rng(self.seed + epoch)
                self._order = rng.permutation(self.n_blocks)
            else:
                self._order = np.arange(self.n_blocks)
            self._epoch = epoch
        return self._order

    def _fetch(self, block_indices: np.ndarray):
        """One vectorized gather for the whole batch."""
        starts = block_indices.astype(np.int64) * self.block_size
        rows = starts[:, None] + np.arange(self.block_size + 1, dtype=np.int64)
        # uint16 on disk -> int64 for the embedding lookup, cast only for this window
        window = torch.from_numpy(self.dataset.tokens[rows].astype(np.int64))
        inputs, labels = window[:, :-1], window[:, 1:].clone()
        mask = getattr(self.dataset, "mask", None)
        if mask is not None:
            # labels are tokens shifted by one, so label j is governed by mask[j+1]
            keep = torch.from_numpy(np.asarray(mask[rows[:, 1:]], dtype=np.uint8)).bool()
            labels = labels.masked_fill(~keep, -100)
        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        # Each rank takes a disjoint stripe of the global batch, so one optimizer step across
        # world_size ranks covers batch_size * world_size distinct blocks.
        global_batch = self.batch_size * self.world_size
        epoch = self.position // self.n_blocks
        offset = self.position % self.n_blocks

        if offset + global_batch > self.n_blocks:
            # Skip the ragged tail rather than emit a short batch: shapes must stay constant or
            # torch.compile recompiles and the step time becomes uneven.
            self.position = (epoch + 1) * self.n_blocks
            epoch, offset = self.position // self.n_blocks, 0

        order = self._order_for_epoch(epoch)
        chunk = order[offset : offset + global_batch]
        mine = chunk[self.rank :: self.world_size][: self.batch_size]
        self.position += global_batch
        return self._fetch(mine)

    def state_dict(self) -> dict:
        return {"position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]
        self.seed = state.get("seed", self.seed)


class SequentialPackedLoader:
    """Deterministic full pass over a split, for validation.

    Sharded across ranks so each evaluates a disjoint slice; the caller reduces the totals.
    """

    def __init__(self, dataset, batch_size: int, rank: int = 0, world_size: int = 1,
                 max_batches: int | None = None) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.max_batches = max_batches
        self.block_size = dataset.block_size

    def __iter__(self):
        indices = np.arange(len(self.dataset))[self.rank :: self.world_size]
        n_batches = len(indices) // self.batch_size
        if self.max_batches is not None:
            n_batches = min(n_batches, self.max_batches)
        for b in range(n_batches):
            chunk = indices[b * self.batch_size : (b + 1) * self.batch_size]
            starts = chunk.astype(np.int64) * self.block_size
            rows = starts[:, None] + np.arange(self.block_size + 1, dtype=np.int64)
            window = torch.from_numpy(self.dataset.tokens[rows].astype(np.int64))
            inputs, labels = window[:, :-1], window[:, 1:].clone()
            mask = getattr(self.dataset, "mask", None)
            if mask is not None:
                keep = torch.from_numpy(np.asarray(mask[rows[:, 1:]], dtype=np.uint8)).bool()
                labels = labels.masked_fill(~keep, -100)
            yield inputs, labels

    def __len__(self) -> int:
        n = len(np.arange(len(self.dataset))[self.rank :: self.world_size]) // self.batch_size
        return min(n, self.max_batches) if self.max_batches is not None else n
