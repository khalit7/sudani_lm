"""Collect the public Sudanese corpora into one document stream.

Small — under 1M tokens against the WhatsApp export's ~7M — but it is the only Sudanese text in
the project that is not the owner's own chat, so it is worth having in the continued-pretraining
mix even at that size.

Five sources (run scripts/download_sudanese_sources.py for the last two):
  - the three original sentiment corpora (labels stripped; two need cleaning, see the loaders),
  - Lisan-Sudanese sentences via the TTS mirrors — the cleanest verified-dialect text that
    publicly exists (CC BY 4.0),
  - the "organic Sudanese" app sample (CC BY 4.0), 300 rows of genuinely spontaneous chat.

A slice of Lisan is held out as its own perplexity eval: unlike the WhatsApp holdout it shares
no people with training, and unlike Flores it is native dialect rather than translationese — so
it is the one number that can catch "memorised the contacts" and "learned translator Sudanese"
at the same time.

Output: data/interim/sudani/all.jsonl, data/interim/sudani/lisan_holdout.jsonl

Usage:  python -m src.preprocessing.sudani
"""

import argparse
import csv
import glob
import json
import random
import re
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "interim" / "sudani"

MIN_CHARS = 20

# The Telegram corpus was collected with a normalisation that rewrote every ي as ى. Left alone it
# would teach a spelling that does not occur in natural Sudanese writing, so it is reversed —
# but only for that corpus, since ى is a legitimate letter elsewhere.
TELE_YA_FIX = str.maketrans({"ى": "ي"})

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")


def clean(text, fix_ya=False):
    # ﻿: SudSenti's files carry a byte-order mark that would otherwise become a token
    text = str(text or "").replace("﻿", "").strip()
    if not text:
        return ""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    # Collection artefacts: trailing ellipsis marking truncation, and raw \r\n
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"…+$", "", text)
    if fix_ya:
        text = text.translate(TELE_YA_FIX)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def load_arrow(folder, column, fix_ya=False):
    files = glob.glob(str(DATA_RAW / folder / "**" / "*.arrow"), recursive=True)
    if not files:
        print(f"  WARNING: {folder} not found, skipping")
        return []
    dataset = Dataset.from_file(files[0])
    return [clean(t, fix_ya) for t in dataset[column]]


def load_sudsenti():
    """SudSenti ships as `text<TAB>label` lines.

    A substantial share is Sudan-*related* MSA news and even Quranic quotation rather than
    dialect, so this is the least dialectal of the three. Kept anyway: it is still Sudanese-topic
    Arabic, which is closer to the target than ArabicWeb24.
    """
    texts = []
    for path in sorted(glob.glob(str(DATA_RAW / "sudsenti" / "*-Tweets.txt"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                texts.append(clean(line.split("\t")[0]))
    return texts


def load_lisan():
    """Lisan-Sudanese sentences, mirrored inside two TTS datasets with identical text.

    Facebook/YouTube comments transcribed and verified by native speakers (Jarrar et al.,
    CC BY 4.0). The two mirrors duplicate each other, which the global dedup absorbs.
    """
    texts = []
    for path in sorted(glob.glob(str(DATA_RAW / "lisan" / "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                texts.append(clean(json.loads(line)["text"]))
    if not texts:
        print("  WARNING: lisan not found, skipping")
    return texts


ARABIC_RE = re.compile(r"[؀-ۿ]")


def load_organic():
    """The 'organic Sudanese dialect' app sample: 300 spontaneous messages, CC BY 4.0.

    Collected from a live translation app, so a handful of rows are foreign one-word probes
    ("paard") or other dialects. Rows must be majority-Arabic; MIN_CHARS handles the probes.
    """
    path = DATA_RAW / "organic_sudanese" / "sudanese_dialect_dataset.csv"
    if not path.exists():
        print("  WARNING: organic_sudanese not found, skipping")
        return []
    texts = []
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            text = clean(row.get("source_text"))
            if len(ARABIC_RE.findall(text)) > len(text) * 0.5:
                texts.append(text)
    return texts


# Held-out share of Lisan, split before training data is written so no pipeline change can leak
# it. ~280 sentences ≈ 9K tokens: small, but perplexity over it is stable enough to compare
# checkpoints, which is all it is for.
LISAN_HOLDOUT_FRACTION = 0.15
SEED = 67


def collect():
    sources = {
        "sudanese_tweets": load_arrow("sudanese_tweets", "Tweet"),
        "sudanese_tweets_tele": load_arrow("sudanese_tweets_tele", "Tweet_Text", fix_ya=True),
        "sudsenti": load_sudsenti(),
        "lisan": load_lisan(),
        "organic_sudanese": load_organic(),
    }
    documents, seen = [], set()
    for name, texts in sources.items():
        kept = 0
        for text in texts:
            if len(text) < MIN_CHARS:
                continue
            # The corpora overlap: SudSenti2 and SudSenti3 share tweets, and the arbml sets
            # were scraped from the same platforms.
            if text in seen:
                continue
            seen.add(text)
            documents.append({"source": name, "text": text})
            kept += 1
        print(f"  {name:<24} {len(texts):>6} raw -> {kept:>6} kept")
    return documents


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    documents = collect()

    lisan = [d for d in documents if d["source"] == "lisan"]
    order = list(range(len(lisan)))
    random.Random(SEED).shuffle(order)
    holdout = {id(lisan[i]) for i in order[: int(len(lisan) * LISAN_HOLDOUT_FRACTION)]}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all.jsonl"
    held_path = OUT_DIR / "lisan_holdout.jsonl"
    kept = held = 0
    with open(out, "w", encoding="utf-8") as fh, open(held_path, "w", encoding="utf-8") as hh:
        for doc in documents:
            if id(doc) in holdout:
                hh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                held += 1
            else:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
                kept += 1
    chars = sum(len(d["text"]) for d in documents)
    print(f"\n{kept:,} documents ({chars/1e6:.2f}M chars) -> {out}")
    print(f"{held:,} Lisan sentences held out -> {held_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
