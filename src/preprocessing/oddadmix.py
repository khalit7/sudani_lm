"""Sudanese podcast/YouTube transcripts from the three oddadmix collections, as documents.

~200 hours of transcribed spoken Sudanese (Sudan Podcast, Nuuar, Ahmed Gobara) — the closest
public register to WhatsApp chat, and roughly ten times all other public Sudanese text put
together. Provenance caveat recorded in DATASHEET.md: the audio was scraped from YouTube and the
transcripts are AI-generated with QC, so this stays private training data, never redistributed.

Three transcript artifacts have to go before packing:

  - full diacritization: the transcriber emits vowelled text (هُنَا اختَلَفَ الوَضْعُ); nothing
    else in the target distribution is vowelled, so harakat are stripped,
  - production tags: [موسيقى]-style brackets and <laugh>/<pause> markers,
  - stutter runs: ASR repeats a word many times where speech stalls; runs longer than 2 are
    collapsed, because the model already has a repetition failure mode to unlearn.

Episodes are rebuilt from their chunks (ordered by the chunk index), then re-cut into ~1,500-char
documents at sentence-ish boundaries. Split is by episode, never by chunk — same principle as
whatsapp.py's split-by-chat.

Output: data/interim/oddadmix/{train,val}.jsonl

Usage:  python -m src.preprocessing.oddadmix
"""

import argparse
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "oddadmix"
OUT_DIR = REPO_ROOT / "data" / "interim" / "oddadmix"

SEED = 67
VAL_FRACTION = 0.02
TARGET_DOC_CHARS = 1500
MIN_DOC_CHARS = 200

# Arabic diacritics (harakat, Quranic annotation marks) and tatweel
DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-ۨ"
                           r"۪-ۭـ]")
TAG_RE = re.compile(r"\[[^\]\n]{1,30}\]|<[^>\n]{1,20}>|\([^)\n]{0,3}موسيقى[^)\n]{0,3}\)")
CHUNK_INDEX_RE = re.compile(r"_chunk_(\d+)$")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.؟!،])\s+")


def clean(text: str) -> str:
    text = DIACRITICS_RE.sub("", text or "")
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # collapse stutter runs: same word more than twice in a row -> twice
    words, out = text.split(" "), []
    for word in words:
        if len(out) >= 2 and out[-1] == word and out[-2] == word:
            continue
        out.append(word)
    return " ".join(out)


def split_documents(episode_text: str):
    """Cut an episode into ~TARGET_DOC_CHARS documents at sentence-ish boundaries."""
    parts, current = [], ""
    for sentence in SENTENCE_SPLIT_RE.split(episode_text):
        if current and len(current) + len(sentence) > TARGET_DOC_CHARS:
            parts.append(current.strip())
            current = ""
        current += sentence + " "
    if len(current.strip()) >= MIN_DOC_CHARS:
        parts.append(current.strip())
    return [p for p in parts if len(p) >= MIN_DOC_CHARS]


def load_episodes():
    """(collection, video_id) -> episode text, chunks rejoined in order."""
    episodes = {}
    for path in sorted(glob.glob(str(RAW_DIR / "*.jsonl"))):
        collection = Path(path).stem
        chunks = defaultdict(list)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                index = CHUNK_INDEX_RE.search(row.get("chunk_id") or "")
                chunks[row.get("original_video_id") or "unknown"].append(
                    (int(index.group(1)) if index else 10**9, row.get("transcript_text") or ""))
        for video_id, parts in chunks.items():
            text = clean(" ".join(t for _, t in sorted(parts, key=lambda pair: pair[0])))
            if text:
                episodes[(collection, video_id)] = text
    return episodes


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    episodes = load_episodes()
    if not episodes:
        raise FileNotFoundError(f"no transcripts under {RAW_DIR} — "
                                "run scripts/download_sudanese_sources.py first")

    keys = sorted(episodes)
    random.Random(SEED).shuffle(keys)
    val_keys = set(keys[: max(1, int(len(keys) * VAL_FRACTION))])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    chars = {"train": 0, "val": 0}
    handles = {split: open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8")
               for split in counts}
    per_collection = defaultdict(int)
    for key in sorted(episodes):
        collection, video_id = key
        split = "val" if key in val_keys else "train"
        for text in split_documents(episodes[key]):
            handles[split].write(json.dumps(
                {"source": collection, "episode": video_id, "text": text},
                ensure_ascii=False) + "\n")
            counts[split] += 1
            chars[split] += len(text)
            per_collection[collection] += 1
    for handle in handles.values():
        handle.close()

    for collection, n in sorted(per_collection.items()):
        print(f"  {collection:<16} {n:>6,} documents")
    print(f"{len(episodes):,} episodes -> train {counts['train']:,} docs"
          f" ({chars['train']/1e6:.2f}M chars) / val {counts['val']:,} docs"
          f" ({chars['val']/1e6:.2f}M chars) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
