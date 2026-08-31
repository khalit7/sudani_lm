"""Flores DEVTEST leakage screen for training corpora (plan.md Part IV, step 1.9).

DEVTEST is the one-shot final-comparison eval and must stay out of everything. The
synthesis QC already screens generated data; this applies the same 8-gram rule to the
web-scraped corpora, because web text can quote the same news sentences Flores
translated. Any document sharing a word 8-gram with either side (Sud or Arb) of any
DEVTEST sentence is dropped, and the file is rewritten atomically.

Usage:  python -m src.preprocessing.leakage_screen data/interim/<name>/train.jsonl [...]
"""

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTEST = REPO_ROOT / "data" / "raw" / "sudanese_flores" / "DEVTEST.jsonl"
NGRAM = 8

TOKEN_RE = re.compile(r"[\w؀-ۿ]+")


def ngrams(text: str):
    words = TOKEN_RE.findall(text)
    for i in range(len(words) - NGRAM + 1):
        yield " ".join(words[i:i + NGRAM])


def devtest_ngrams() -> set:
    grams = set()
    for line in DEVTEST.read_text().splitlines():
        if not line.strip():
            continue
        translation = json.loads(line)["translation"]
        for side in ("Sud", "Arb"):
            grams.update(ngrams(translation.get(side, "")))
    return grams


def screen_file(path: Path, grams: set) -> tuple[int, int]:
    tmp_path = path.with_suffix(".jsonl.screening")
    kept = dropped = 0
    with open(tmp_path, "w", encoding="utf-8") as out_fh, \
            open(path, encoding="utf-8") as in_fh:
        for line in in_fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if any(gram in grams for gram in ngrams(row["text"])):
                dropped += 1
                continue
            out_fh.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    tmp_path.replace(path)
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    grams = devtest_ngrams()
    print(f"{len(grams):,} DEVTEST 8-grams loaded", flush=True)
    total_dropped = 0
    for path in args.paths:
        kept, dropped = screen_file(Path(path), grams)
        total_dropped += dropped
        print(f"{path}: kept {kept:,}, dropped {dropped}", flush=True)
    return 0 if total_dropped >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
