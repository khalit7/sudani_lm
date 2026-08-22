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
import glob
import hashlib
import json
import random
import re
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
    """Buffered append to a flat .bin, counting tokens as it goes.

    When `with_mask` is set it also writes a parallel `<split>.mask` of uint8 flags, one per
    token, marking which positions contribute to the loss. Needed for SFT, where only the
    owner's own replies should be scored.
    """

    def __init__(self, path: Path, with_mask: bool = False):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = open(path, "wb")
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.total = 0
        self.mask_handle = open(path.with_suffix(".mask"), "wb") if with_mask else None
        self.mask_buffer: list[np.ndarray] = []
        self.scored = 0

    def add(self, ids, mask=None) -> None:
        arr = np.asarray(ids, dtype=DTYPE)
        self.buffer.append(arr)
        self.buffered += arr.size
        self.total += arr.size
        if self.mask_handle is not None:
            # default to "scored" so replay text mixed into an SFT pack still trains normally
            m = np.ones(arr.size, dtype=np.uint8) if mask is None else np.asarray(mask, dtype=np.uint8)
            if m.size != arr.size:
                raise ValueError(f"mask length {m.size} != token length {arr.size}")
            self.mask_buffer.append(m)
            self.scored += int(m.sum())
        if self.buffered >= WRITE_BUFFER_TOKENS:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            np.concatenate(self.buffer).tofile(self.handle)
            self.buffer.clear()
            self.buffered = 0
        if self.mask_handle is not None and self.mask_buffer:
            np.concatenate(self.mask_buffer).tofile(self.mask_handle)
            self.mask_buffer.clear()

    def close(self) -> None:
        self.flush()
        self.handle.close()
        if self.mask_handle is not None:
            self.mask_handle.close()


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


DATA_INTERIM = REPO_ROOT / "data" / "interim"

# Replay fractions, as a share of the stage's own token count. Continued pretraining keeps a
# quarter of its budget on Arabic so dialect adaptation does not erase the base; chat SFT keeps
# a smaller share of instruction data so it retains turn-taking beyond the owner's own style.
CPT_ARABIC_REPLAY = 0.25
SFT_REPLAY = 0.15


def _read_jsonl_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the matching preprocessing script first")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _read_jsonl(path, field="text"):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the matching preprocessing script first")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line)[field] for line in fh]


def _sample_arabic_tokens(target_tokens, seed=SEED):
    """Draw replay text from the packed pretraining stream.

    Reusing the pack rather than re-reading ArabicWeb24 keeps replay drawn from exactly the
    distribution the base model saw, and costs one memmap read.
    """
    source = DATA_PACKED / "pretrain" / "train.bin"
    if not source.exists():
        raise FileNotFoundError(f"{source} not found — pack the pretrain stage first")
    stream = np.memmap(source, dtype=DTYPE, mode="r")
    rng = random.Random(seed)
    start = rng.randrange(0, max(len(stream) - target_tokens - 1, 1))
    return np.asarray(stream[start : start + int(target_tokens)])


def pack_mixture(stage, primary_texts, replay_fraction, replay_texts=None,
                 arabic_replay=False, val_texts=None) -> dict:
    """Pack one stage: primary documents plus a replay mixture."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("</s>")
    out_dir = DATA_PACKED / stage
    train = ShardWriter(out_dir / "train.bin")
    start = time.time()

    for i in range(0, len(primary_texts), ENCODE_BATCH):
        batch = primary_texts[i : i + ENCODE_BATCH]
        _encode_into(tokenizer, eos_id, batch, [train] * len(batch))
    primary_tokens = train.total

    replay_target = int(primary_tokens * replay_fraction / max(1 - replay_fraction, 1e-9))
    if arabic_replay:
        train.add(_sample_arabic_tokens(replay_target))
    elif replay_texts:
        rng = random.Random(SEED)
        rng.shuffle(replay_texts)
        # Small batches on purpose. At ENCODE_BATCH=2000 a single batch of SmolKalam
        # conversations (some average 1,100+ tokens) overshoots the replay target by half again,
        # which silently changes the mixture the stage trains on.
        replay_batch = 64
        added = 0
        for i in range(0, len(replay_texts), replay_batch):
            if added >= replay_target:
                break
            batch = replay_texts[i : i + replay_batch]
            before = train.total
            _encode_into(tokenizer, eos_id, batch, [train] * len(batch))
            added += train.total - before
    replay_tokens = train.total - primary_tokens
    train.close()

    val_total = 0
    if val_texts:
        val = ShardWriter(out_dir / "val.bin")
        for i in range(0, len(val_texts), ENCODE_BATCH):
            batch = val_texts[i : i + ENCODE_BATCH]
            _encode_into(tokenizer, eos_id, batch, [val] * len(batch))
        val_total = val.total
        val.close()

    meta = {
        "stage": stage,
        "splits": {"train": train.total, **({"val": val_total} if val_texts else {})},
        "primary_tokens": primary_tokens,
        "replay_tokens": replay_tokens,
        "replay_fraction_actual": round(replay_tokens / max(train.total, 1), 4),
        "dtype": np.dtype(DTYPE).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "eos_id": eos_id,
        "seed": SEED,
        "built_seconds": round(time.time() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta



# Chats whose owner-token share falls below this go to Stage D's complement split. Measured on
# the corpus the two pools are near-disjoint in character: the low-share side is dominated by
# group chats at ~11% owner tokens, the high side by 1:1 threads at ~44%.
ME_SHARE_THRESHOLD = 0.20

# Stage C's own default. The nine-model ablation (plan.md, "how should C and D divide the chat
# corpus?") settled that Stage C must see the *whole* chat corpus — masking, not data splitting,
# is what justifies Stage D. 1.01 puts every chat below the threshold. Stage D keeps 0.20.
# This constant exists because the shipped stage_c pack was built with an ad-hoc
# `--me-threshold 1.01` override, and a rebuild without it silently reverted to the mixture the
# ablation rejected.
STAGE_C_ME_THRESHOLD = 1.01

ME_TURN_RE = re.compile(r"<\|turn\|>ME:(.*?)(<\|end\|>)", re.S)


def me_loss_mask(tokenizer, text):
    """(token ids, uint8 mask) where the mask marks the owner's own reply tokens.

    Spans are derived from the *whole-string* encoding via offsets. Encoding the segments
    separately and concatenating does NOT reproduce the same ids — the Metaspace pre-tokenizer
    prepends its marker to whatever it sees first, so each segment gains one — which would
    misalign the mask against the tokens actually trained on.

    The closing <|end|> is scored along with the content: the model has to learn to stop. The
    "ME:" label itself is not, since it is part of the prompt.
    """
    encoding = tokenizer.encode(text, add_special_tokens=False)
    spans = [(m.start(1), m.end(2)) for m in ME_TURN_RE.finditer(text)]
    mask = np.zeros(len(encoding.ids), dtype=np.uint8)
    for i, (a, b) in enumerate(encoding.offsets):
        if a == b:
            continue
        for start, end in spans:
            if a >= start and b <= end:
                mask[i] = 1
                break
    return encoding.ids, mask


def chat_me_share(tokenizer, rows):
    """chat -> share of its tokens that are the owner's replies."""
    totals = {}
    for row in rows:
        ids, mask = me_loss_mask(tokenizer, row["text"])
        me, tot = totals.setdefault(row["chat"], [0, 0])
        totals[row["chat"]] = [me + int(mask.sum()), tot + len(ids)]
    return {chat: me / max(tot, 1) for chat, (me, tot) in totals.items()}


def split_chats_by_me_share(tokenizer, rows, threshold=ME_SHARE_THRESHOLD):
    shares = chat_me_share(tokenizer, rows)
    low = {c for c, s in shares.items() if s < threshold}
    return ([r for r in rows if r["chat"] in low],
            [r for r in rows if r["chat"] not in low], shares)


def split_chat(texts, fraction, seed=SEED):
    """Deterministic conversation-level split of the chat training set.

    Used to reserve a slice of chat for a later stage. Split by *conversation*, not by chat:
    a later stage should see new exchanges from the same people, not new people.
    """
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    cut = int(len(order) * fraction)
    first = [texts[i] for i in sorted(order[:cut])]
    second = [texts[i] for i in sorted(order[cut:])]
    return first, second


def _warn_if_threshold_changed(stage, threshold) -> None:
    """Refuse to let a repack silently change the C/D chat split.

    The mixture a stage trains on is part of the experiment; meta.json records it. A different
    threshold on a rebuild is occasionally intentional (an ablation arm) and often a forgotten
    flag, so it is a loud warning rather than an error.
    """
    meta_path = DATA_PACKED / stage / "meta.json"
    if not meta_path.exists():
        return
    previous = json.loads(meta_path.read_text()).get("me_share_threshold")
    if previous is not None and abs(previous - threshold) > 1e-9:
        print(f"  WARNING: repacking {stage} with me_share_threshold={threshold} but the "
              f"existing pack was built with {previous} — the chat mixture will change",
              flush=True)


def pack_stage_c(stage="stage_c", arabic_replay=CPT_ARABIC_REPLAY, sudani_upsample=3,
                 threshold=STAGE_C_ME_THRESHOLD) -> dict:
    """Stage C — dialect exposure (continued pretraining, loss on every token).

    Sees the *whole* chat corpus by default (threshold 1.01 → every chat qualifies). The earlier
    design restricted C to low-owner-share chats to save the owner's writing for Stage D, but the
    ablation showed that starves C: masking in D, not data splitting, is what makes the stages
    different. A lower threshold survives as an explicit override for ablation arms only.
    """
    _warn_if_threshold_changed(stage, threshold)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    rows = _read_jsonl_rows(DATA_INTERIM / "whatsapp" / "train.jsonl")
    low, high, shares = split_chats_by_me_share(tokenizer, rows, threshold)
    sudani = _read_jsonl(DATA_INTERIM / "sudani" / "all.jsonl")
    val = _read_jsonl(DATA_INTERIM / "whatsapp" / "val.jsonl")
    print(f"  chat {len(low):,} conversations from {len({r['chat'] for r in low})} low-owner-share"
          f" chats (<{threshold:.0%}) + sudani {len(sudani):,} x{sudani_upsample}"
          f" + {arabic_replay:.0%} Arabic replay")
    print(f"  (reserved for Stage D: {len(high):,} conversations from"
          f" {len({r['chat'] for r in high})} chats)")
    meta = pack_mixture(stage, [r["text"] for r in low] + sudani * sudani_upsample,
                        arabic_replay, arabic_replay=True, val_texts=val)
    meta["me_share_threshold"] = threshold
    meta["chats_used"] = sorted({r["chat"] for r in low})
    (DATA_PACKED / stage / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def pack_stage_d(stage="stage_d", threshold=ME_SHARE_THRESHOLD) -> dict:
    """Stage D — style (SFT, loss on the owner's replies only).

    Uses the complement of Stage C's pool: the chats where the owner actually writes. Every
    token is still *read* as context; only the owner's replies are *scored*, which is what makes
    this a different gradient rather than more continued-pretraining epochs.
    """
    _warn_if_threshold_changed(stage, threshold)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("</s>")
    rows = _read_jsonl_rows(DATA_INTERIM / "whatsapp" / "train.jsonl")
    _, high, _ = split_chats_by_me_share(tokenizer, rows, threshold)
    val_rows = _read_jsonl_rows(DATA_INTERIM / "whatsapp" / "val.jsonl")

    out_dir = DATA_PACKED / stage
    train = ShardWriter(out_dir / "train.bin", with_mask=True)
    val = ShardWriter(out_dir / "val.bin", with_mask=True)
    start = time.time()
    for writer, source in ((train, high), (val, val_rows)):
        for row in source:
            ids, mask = me_loss_mask(tokenizer, row["text"])
            writer.add(ids, mask)
            writer.add([eos_id], [0])       # the separator is context, not a reply
    scored_train, scored_val = train.scored, val.scored
    train.close(); val.close()

    print(f"  chat {len(high):,} conversations from {len({r['chat'] for r in high})} chats"
          f" (owner-share >= {threshold:.0%})")
    meta = {
        "stage": stage,
        "splits": {"train": train.total, "val": val.total},
        "scored_tokens": {"train": scored_train, "val": scored_val},
        "scored_fraction": round(scored_train / max(train.total, 1), 4),
        "loss_on": "owner replies only",
        "me_share_threshold": threshold,
        "dtype": np.dtype(DTYPE).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "eos_id": eos_id,
        "seed": SEED,
        "built_seconds": round(time.time() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def pack_manifest(manifest_path) -> dict:
    """Pack a stage from a YAML mixture manifest (data expansion v2, plan.md Part IV).

    `pack_stage_c` hardcodes exactly three ingredients; the v2 corpus has many sources with
    per-source upsampling, and the ablation arms need to swap mixtures without touching code.
    A manifest makes each mixture a diffable file:

        stage: stage_c_v2
        sources:                    # packed in order; the train loader shuffles blocks anyway
          - {path: data/interim/whatsapp/train.jsonl, repeat: 2}
          - {path: data/interim/sudani/all.jsonl,     repeat: 2, name: sudani}
        arabic_replay: 0.25         # share of the final stream drawn from the pretrain pack
        val_sources:                # concatenated into val.bin, boundaries recorded in meta
          - {path: data/interim/whatsapp/val.jsonl, name: whatsapp}

    `repeat` is plain repetition of the source's token stream — blocks are shuffled at train
    time, so upsampling needs nothing cleverer. Per-source token counts and val boundaries land
    in meta.json: the counts are the record of what the mixture actually was, and the boundaries
    are what the per-source validation evaluator slices on.

    Keep val_sources ordered with the primary corpus first: the trainer's in-loop val loss reads
    only the first `val_max_batches` blocks, so the head of val.bin is what it measures.

    Usage:  python -m src.preprocessing.pack manifest configs/mixtures/stage_c_v2.yaml
    """
    import yaml

    manifest_path = Path(manifest_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    stage = manifest["stage"]

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    eos_id = tokenizer.token_to_id("</s>")
    out_dir = DATA_PACKED / stage
    train = ShardWriter(out_dir / "train.bin")
    start = time.time()

    def _pack_source(writer, source) -> dict:
        path = REPO_ROOT / source["path"]
        texts = _read_jsonl(path, source.get("field", "text"))
        repeat = int(source.get("repeat", 1))
        before = writer.total
        for _ in range(repeat):
            for i in range(0, len(texts), ENCODE_BATCH):
                batch = texts[i : i + ENCODE_BATCH]
                _encode_into(tokenizer, eos_id, batch, [writer] * len(batch))
        return {
            "name": source.get("name") or path.parent.name,
            "path": source["path"],
            "documents": len(texts),
            "repeat": repeat,
            "tokens": writer.total - before,
        }

    sources_meta = []
    for source in manifest["sources"]:
        entry = _pack_source(train, source)
        sources_meta.append(entry)
        print(f"  {entry['name']:<20} {entry['documents']:>8,} docs x{entry['repeat']}"
              f" -> {entry['tokens']/1e6:8.2f}M tokens", flush=True)

    primary_tokens = train.total
    replay_fraction = float(manifest.get("arabic_replay", 0.0))
    if replay_fraction > 0:
        replay_target = int(primary_tokens * replay_fraction / max(1 - replay_fraction, 1e-9))
        train.add(_sample_arabic_tokens(replay_target))
        print(f"  {'arabic_replay':<20} {'':>8} {'':>4} -> {replay_target/1e6:8.2f}M tokens",
              flush=True)
    replay_tokens = train.total - primary_tokens
    train.close()

    val_sources_meta = []
    val_total = 0
    if manifest.get("val_sources"):
        val = ShardWriter(out_dir / "val.bin")
        for source in manifest["val_sources"]:
            offset = val.total
            entry = _pack_source(val, source)
            entry["start"] = offset            # token offset into val.bin, for per-source ppl
            val_sources_meta.append(entry)
        val_total = val.total
        val.close()

    meta = {
        "stage": stage,
        "manifest": str(manifest_path.relative_to(REPO_ROOT)
                        if manifest_path.is_relative_to(REPO_ROOT) else manifest_path),
        "splits": {"train": train.total, **({"val": val_total} if val_sources_meta else {})},
        "primary_tokens": primary_tokens,
        "replay_tokens": replay_tokens,
        "replay_fraction_actual": round(replay_tokens / max(train.total, 1), 4),
        "sources": sources_meta,
        "val_sources": val_sources_meta,
        "dtype": np.dtype(DTYPE).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "eos_id": eos_id,
        "seed": SEED,
        "built_seconds": round(time.time() - start, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _load_instruction_replay():
    """Multi-turn conversations from SmolKalam, rendered in the project's chat template."""
    import pyarrow.parquet as pq

    from src.tokenizer.special_tokens import ASSISTANT_SPEAKER, USER_SPEAKER, render_conversation

    conversational = [
        "smoltalk_smollm3_everyday_conversations_no_think",
        "smoltalk_smollm3_systemchats_30k_no_think",
        "aya_dataset_Qwen3_32B_think",
    ]
    rendered = []
    for config in conversational:
        for path in sorted(glob.glob(str(DATA_RAW / "smolkalam" / config / "*.parquet"))):
            table = pq.ParquetFile(path).read_row_group(0, columns=["messages"])
            for conversation in table.column("messages").to_pylist():
                turns = []
                for message in conversation:
                    role = message.get("role")
                    if role not in ("user", "assistant"):
                        continue
                    content = (message.get("content") or "").strip()
                    # Some configs keep <think> traces; they teach a behaviour this model has no
                    # capacity for and would only pollute the turn format.
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
                    if content:
                        speaker = USER_SPEAKER if role == "user" else ASSISTANT_SPEAKER
                        turns.append((speaker, content.replace("\n", " ")))
                if len(turns) >= 2:
                    rendered.append(render_conversation(turns))
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain", "stage_c", "stage_d", "manifest"])
    parser.add_argument("manifest_path", nargs="?", default=None,
                        help="manifest only: path to a configs/mixtures/*.yaml mixture file")
    parser.add_argument(
        "--limit-shards", type=int, default=None,
        help="pretrain only: pack the first N shards (for a quick smoke run)",
    )
    parser.add_argument("--out-stage", default=None,
                        help="write to data/packed/<name> instead of the stage's default name")
    parser.add_argument("--arabic-replay", type=float, default=CPT_ARABIC_REPLAY,
                        help="cpt only: share of the stage's tokens drawn from the Arabic pack")
    parser.add_argument("--sudani-upsample", type=int, default=1,
                        help="cpt only: repeat the public Sudanese corpora N times")
    # Default None so each stage resolves its own: stage_c sees all chats (1.01, the ablation
    # winner), stage_d keeps the 0.20 split. Passing a value explicitly is for ablation arms.
    parser.add_argument("--me-threshold", type=float, default=None,
                        help="owner-token share cutoff for the chat split"
                             f" (stage_c default {STAGE_C_ME_THRESHOLD},"
                             f" stage_d default {ME_SHARE_THRESHOLD})")
    args = parser.parse_args()

    if not TOKENIZER_PATH.exists():
        print(f"tokenizer not found at {TOKENIZER_PATH}", file=sys.stderr)
        return 1

    print(f"packing stage={args.stage} with tokenizer {tokenizer_fingerprint()}")
    if args.stage == "pretrain":
        meta = pack_pretrain(limit_shards=args.limit_shards)
    elif args.stage == "manifest":
        if not args.manifest_path:
            print("manifest stage needs a manifest path", file=sys.stderr)
            return 1
        meta = pack_manifest(args.manifest_path)
    elif args.stage == "stage_c":
        meta = pack_stage_c(stage=args.out_stage or "stage_c",
                            arabic_replay=args.arabic_replay,
                            sudani_upsample=args.sudani_upsample,
                            threshold=args.me_threshold if args.me_threshold is not None
                            else STAGE_C_ME_THRESHOLD)
    else:
        meta = pack_stage_d(stage=args.out_stage or "stage_d",
                            threshold=args.me_threshold if args.me_threshold is not None
                            else ME_SHARE_THRESHOLD)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
