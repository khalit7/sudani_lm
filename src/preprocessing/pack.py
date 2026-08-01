"""Tokenize once, offline, into flat uint16 token streams.

This replaces tokenizing inside the dataloader collate function, which was doing it *twice per
example* on 12 CPU workers and was almost certainly the real throughput ceiling. It also removes
two silent data losses in the old path: whole documents truncated at 1024 tokens (31.3% of all
tokens discarded) and batches padded to their longest member (only 52% of positions real).
Together those meant the pipeline did useful work on ~36% of what it touched.

Output layout, one directory per training stage:

    data/packed/<stage>/<split>.bin    flat uint16, no padding, no document boundaries
    data/packed/<stage>/meta.json      token count, block size, and the tokenizer fingerprint

uint16 because token ids only need to reach 31,999 and 2 bytes/token keeps the 17B-token stream
at 34 GB — small enough to stay in page cache, where an int64 version at 136 GB would not.
The fingerprint is asserted at load time so a pack made with a different vocabulary can never be
silently trained on.

Usage:  python -m src.preprocessing.pack pretrain
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from datasets import Dataset
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PACKED = REPO_ROOT / "data" / "packed"
TOKENIZER_PATH = REPO_ROOT / "tokenizers" / "v2_32k" / "tokenizer.json"

SEED = 67
DTYPE = np.uint16

# Matches src/preprocessing/arabic.py: half the corpus is dropped, and documents under 20
# whitespace tokens are filtered out. Kept as constants so a future run can lift the fraction
# without touching the logic.
KEEP_FRACTION = 0.5
MIN_WORDS = 20
# ArabicWeb24 ships a per-document language confidence; ~1% of documents score below 0.9.
# Filtering on it here is free — no extra pass over the data.
MIN_LANGUAGE_SCORE = 0.9

# Held-out documents. ~10k docs ≈ 9M tokens: a stable loss estimate that is cheap enough to run
# every few hundred steps.
VAL_DOCS = 10_000

ENCODE_BATCH = 2_000
WRITE_BUFFER_TOKENS = 8_000_000


def tokenizer_fingerprint(path: Path = TOKENIZER_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


class ShardWriter:
    """Buffered append to a flat .bin, counting tokens as it goes."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = open(path, "wb")
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.total = 0

    def add(self, ids) -> None:
        arr = np.asarray(ids, dtype=DTYPE)
        self.buffer.append(arr)
        self.buffered += arr.size
        self.total += arr.size
        if self.buffered >= WRITE_BUFFER_TOKENS:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            np.concatenate(self.buffer).tofile(self.handle)
            self.buffer.clear()
            self.buffered = 0

    def close(self) -> None:
        self.flush()
        self.handle.close()


def arabicweb24_shards() -> list[str]:
    shards = sorted(
        str(p) for p in (DATA_RAW / "arabicweb24").rglob("arabic_web24-train-*-of-*.arrow")
    )
    if not shards:
        raise FileNotFoundError(f"no ArabicWeb24 shards under {DATA_RAW/'arabicweb24'}")
    return shards


def keep_document(text, metadata) -> bool:
    if not isinstance(text, str):
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    score = (metadata or {}).get("labels", {}).get("language_score")
    if score is not None and score < MIN_LANGUAGE_SCORE:
        return False
    return True


def pack_pretrain(limit_shards=None) -> dict:
    """Pack ArabicWeb24 into data/packed/pretrain/{train,val}.bin."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("</s>")
    if eos_id is None:
        raise ValueError("tokenizer has no </s> token")

    out_dir = DATA_PACKED / "pretrain"
    train = ShardWriter(out_dir / "train.bin")
    val = ShardWriter(out_dir / "val.bin")

    shards = arabicweb24_shards()[:limit_shards]
    # The old split came from an unseeded train_test_split, so it was never reproducible. This
    # one is derived per shard from a fixed seed, which also means packing can be resumed or
    # parallelised per shard without changing the result.
    #
    # Val is spread evenly over every shard rather than filled greedily from the front. The
    # corpus is ordered by crawl, so a greedy fill would draw the entire validation set from the
    # first ~11 shards and measure a different domain mix than training.
    val_per_shard = max(1, VAL_DOCS // max(len(shards), 1))
    kept = dropped = 0
    start = time.time()

    for shard_i, shard_path in enumerate(shards):
        ds = Dataset.from_file(shard_path)
        # ds["text"] is a lazy Column. Slicing it returns a real list cheaply, but indexing it
        # element-by-element converts one value at a time and dominates the runtime — that path
        # ran at half the tokenizer's throughput. Always slice.
        text_col = ds["text"]
        meta_col = ds["metadata"]
        rng = random.Random(SEED + shard_i)

        # Reservoir of val slots for this shard, chosen up front so val is spread evenly.
        val_slots = val_per_shard

        for offset in range(0, len(ds), ENCODE_BATCH):
            batch_text = text_col[offset : offset + ENCODE_BATCH]
            batch_meta = meta_col[offset : offset + ENCODE_BATCH]

            texts, targets = [], []
            for text, metadata in zip(batch_text, batch_meta):
                if rng.random() >= KEEP_FRACTION or not keep_document(text, metadata):
                    dropped += 1
                    continue
                to_val = val_slots > 0 and rng.random() < (val_per_shard / max(len(ds) * KEEP_FRACTION, 1))
                if to_val:
                    val_slots -= 1
                texts.append(text)
                targets.append(val if to_val else train)
                kept += 1
            if texts:
                _encode_into(tokenizer, eos_id, texts, targets)

        if (shard_i + 1) % 25 == 0 or shard_i + 1 == len(shards):
            elapsed = time.time() - start
            rate = train.total / max(elapsed, 1e-9)
            print(
                f"  shard {shard_i+1}/{len(shards)}  "
                f"train {train.total/1e9:.2f}B  val {val.total/1e6:.1f}M  "
                f"{rate/1e6:.1f}M tok/s  {elapsed/60:.1f} min",
                flush=True,
            )

    train.close()
    val.close()

    meta = {
        "stage": "pretrain",
        "source": "ArabicWeb24",
        "splits": {"train": train.total, "val": val.total},
        "dtype": np.dtype(DTYPE).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "eos_id": eos_id,
        "documents_kept": kept,
        "documents_dropped": dropped,
        "keep_fraction": KEEP_FRACTION,
        "min_words": MIN_WORDS,
        "min_language_score": MIN_LANGUAGE_SCORE,
        "seed": SEED,
        "built_seconds": round(time.time() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _encode_into(tokenizer, eos_id, texts, targets) -> None:
    for encoding, writer in zip(tokenizer.encode_batch(texts), targets):
        # Documents are concatenated with </s> between them. There are no padding tokens and no
        # truncation: a long document simply spans several training blocks.
        writer.add(encoding.ids)
        writer.add([eos_id])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain"])
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="pack only the first N shards (for a quick smoke run)",
    )
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        print(f"tokenizer not found at {TOKENIZER_PATH}", file=sys.stderr)
        return 1

    print(f"packing stage={args.stage} with tokenizer {tokenizer_fingerprint()}")
    meta = pack_pretrain(limit_shards=args.limit_shards)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
