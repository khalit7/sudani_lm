"""Dataset over a packed uint16 token stream.

Every position is a real token: no padding, no truncation, no attention mask, and every batch
has exactly the same shape — which is what lets torch.compile settle on one graph instead of
recompiling per batch. Tokenization already happened offline, so __getitem__ is a memmap slice
and a dtype cast, nothing more.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset.base import BaseDatasetModule
from src.preprocessing.pack import tokenizer_fingerprint

data_root = Path("~/sudani_lm/data").expanduser()


class PackedDataset(Dataset):
    """Fixed-length blocks cut from a flat token stream.

    Block i is tokens [i*B, i*B+B+1): inputs are the first B, labels the same window shifted by
    one. Blocks do not overlap, so one epoch sees every token exactly once.
    """

    def __init__(self, path: Path, block_size: int) -> None:
        self.path = Path(path)
        self.block_size = block_size
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        # -1 because the last block still needs one token past its end for the label shift.
        self.n_blocks = (len(self.tokens) - 1) // block_size
        if self.n_blocks < 1:
            raise ValueError(f"{self.path} holds {len(self.tokens)} tokens, too few for one block")

    def __len__(self) -> int:
        return self.n_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        window = self.tokens[start : start + self.block_size + 1]
        # uint16 keeps the file small but torch embeddings index with int64, so cast the slice
        # — never the whole file.
        window = torch.from_numpy(window.astype(np.int64))
        return window[:-1], window[1:]


class PackedDatasetModule(BaseDatasetModule):
    def __init__(self, stage: str = "pretrain", block_size: int = 1024) -> None:
        self.stage = stage
        self.block_size = block_size
        self.meta = self._load_meta()

    def _stage_dir(self) -> Path:
        return data_root / "packed" / self.stage

    def _load_meta(self) -> dict:
        meta_path = self._stage_dir() / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — run `python -m src.preprocessing.pack {self.stage}`"
            )
        meta = json.loads(meta_path.read_text())
        # A pack is only meaningful alongside the vocabulary that produced it: the ids are
        # otherwise arbitrary. Fail loudly rather than train on a stale pack.
        expected = meta.get("tokenizer_fingerprint")
        actual = tokenizer_fingerprint()
        if expected != actual:
            raise ValueError(
                f"packed data in {self._stage_dir()} was built with tokenizer {expected} but the "
                f"current tokenizer is {actual}. Repack before training."
            )
        return meta

    def build_dataset(self, split) -> torch.utils.data.Dataset:
        return PackedDataset(self._stage_dir() / f"{split}.bin", self.block_size)

    def colllate_fn(self, batch):
        inputs = torch.stack([x for x, _ in batch])
        labels = torch.stack([y for _, y in batch])
        # attention_mask is all ones by construction; kept so the model signature is unchanged
        # between packed pretraining and the padded evaluation paths.
        attention_mask = torch.ones_like(inputs)
        return {"input_ids": inputs, "attention_mask": attention_mask}, labels
