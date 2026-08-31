"""BFS-mirrored small sites -> one document per page (plan.md Part IV, step 1.9).

For the sites mirrored by scripts/scrape_small_sites.py (currently aghaniwamthal.com —
proverbs + Haqiba lyrics, the densest dialect found per token). Mirrored pages repeat
their navigation chrome on every page, so boilerplate is removed by document frequency:
a line that appears on more than BOILERPLATE_DF of pages is site furniture, not content.
(Cross-page exact-line dedup would be wrong here — a proverb legitimately appears on an
index page and its own page; frequency separates the two cleanly.)

Output: data/interim/small_sites/<site>/{train,val}.jsonl

Usage:  python -m src.preprocessing.small_sites --site aghaniwamthal
"""

import argparse
import gzip
import html as html_lib
import json
import random
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "small_sites"
OUT_ROOT = REPO_ROOT / "data" / "interim" / "small_sites"

SEED = 67
VAL_FRACTION = 0.02
MIN_DOC_CHARS = 100
BOILERPLATE_DF = 0.05          # line on >5% of pages = chrome

ARABIC_RE = re.compile(r"[؀-ۿ]")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
BLOCK_RE = re.compile(r"</(?:p|div|li|h[1-6]|blockquote|tr)>|<br\s*/?>", re.I)
TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)


def page_lines(raw: bytes) -> tuple[str, list[str]]:
    try:
        page = gzip.decompress(raw).decode("utf-8", errors="replace")
    except (OSError, EOFError):
        return "", []
    title_match = TITLE_RE.search(page)
    title = ""
    if title_match:
        title = html_lib.unescape(TAG_RE.sub(" ", title_match.group(1))).strip()
    text = BLOCK_RE.sub("\n", page)
    text = TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return title, [line for line in lines if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    site_dir = RAW_ROOT / args.site
    paths = sorted(site_dir.glob("*.html.gz"))
    if not paths:
        raise FileNotFoundError(f"no mirrored pages under {site_dir}")

    pages, line_df = [], Counter()
    for path in paths:
        title, lines = page_lines(path.read_bytes())
        if lines:
            pages.append((path.stem, title, lines))
            for key in {re.sub(r"\s+", "", line) for line in lines}:
                line_df[key] += 1
    chrome = {key for key, n in line_df.items() if n / len(pages) > BOILERPLATE_DF}

    documents, seen = [], set()
    for stem, title, lines in pages:
        body = "\n".join(line for line in lines
                         if re.sub(r"\s+", "", line) not in chrome)
        text = ((title + "\n\n") if title else "") + body
        if len(text) < MIN_DOC_CHARS or not ARABIC_RE.search(body):
            continue
        key = re.sub(r"\s+", "", text)[:2000]
        if key in seen:
            continue
        seen.add(key)
        documents.append({"source": f"small_sites/{args.site}", "page": stem,
                          "text": text.strip()})

    order = list(range(len(documents)))
    random.Random(SEED).shuffle(order)
    val_idx = set(order[: max(1, int(len(order) * VAL_FRACTION))])
    out_dir = OUT_ROOT / args.site
    out_dir.mkdir(parents=True, exist_ok=True)
    counts, chars = {"train": 0, "val": 0}, {"train": 0, "val": 0}
    handles = {split: open(out_dir / f"{split}.jsonl", "w", encoding="utf-8")
               for split in counts}
    for i, doc in enumerate(documents):
        split = "val" if i in val_idx else "train"
        handles[split].write(json.dumps(doc, ensure_ascii=False) + "\n")
        counts[split] += 1
        chars[split] += len(doc["text"])
    for handle in handles.values():
        handle.close()
    print(f"{args.site}: {len(pages):,} pages -> {len(documents):,} docs, "
          f"{len(chrome):,} chrome lines removed "
          f"(train {counts['train']:,} / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
