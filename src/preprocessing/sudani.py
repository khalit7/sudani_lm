"""Collect the three public Sudanese corpora into one document stream.

Small — about 0.64M tokens against the WhatsApp export's ~7M — but it is the only Sudanese text
in the project that is not the owner's own chat, so it is worth having in the continued-pretraining
mix even at that size.

All three were released as *sentiment* corpora, so the labels are stripped and only the text is
kept. Two of them need cleaning first; see the notes on each loader.

Output: data/interim/sudani/all.jsonl

Usage:  python -m src.preprocessing.sudani
"""

import argparse
import glob
import json
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


def collect():
    sources = {
        "sudanese_tweets": load_arrow("sudanese_tweets", "Tweet"),
        "sudanese_tweets_tele": load_arrow("sudanese_tweets_tele", "Tweet_Text", fix_ya=True),
        "sudsenti": load_sudsenti(),
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "all.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    chars = sum(len(d["text"]) for d in documents)
    print(f"\n{len(documents):,} documents, {chars/1e6:.2f}M chars -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
