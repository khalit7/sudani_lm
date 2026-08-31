"""Blogger Atom feeds -> one document per post (plan.md Part IV, step 1.9).

The 12-blog cluster (data/raw/blogger/, scripts/scrape_blogger.py) is small but carries
the densest niche registers found: Haqiba lyrics, short stories, novels, personal and
cooking blogs. Feeds are structured Atom with full post bodies in the <content> element
(HTML, entity-escaped), so extraction is a real XML parse, not scraping.

Output: data/interim/blogger/{train,val}.jsonl   (then: dialect_score score <file>)

Usage:  python -m src.preprocessing.blogger
"""

import argparse
import html as html_lib
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "blogger"
OUT_DIR = REPO_ROOT / "data" / "interim" / "blogger"

SEED = 67
VAL_FRACTION = 0.02
MIN_DOC_CHARS = 120
ATOM = "{http://www.w3.org/2005/Atom}"

ARABIC_RE = re.compile(r"[؀-ۿ]")
BLOCK_RE = re.compile(r"</(?:p|div|li|h[1-6]|blockquote)>|<br\s*/?>", re.I)
TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)


def strip_html(rendered: str) -> str:
    text = BLOCK_RE.sub("\n", rendered)
    text = TAG_RE.sub(" ", text)
    text = html_lib.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    documents, seen = [], set()
    for blog_dir in sorted(d for d in RAW_DIR.iterdir() if d.is_dir()):
        n_before = len(documents)
        for feed_path in sorted(blog_dir.glob("feed_*.xml")):
            try:
                root = ET.fromstring(feed_path.read_text())
            except (OSError, ET.ParseError):
                continue
            for entry in root.iter(f"{ATOM}entry"):
                title = (entry.findtext(f"{ATOM}title") or "").strip()
                content = entry.findtext(f"{ATOM}content") or ""
                body = strip_html(content)
                text = (title + "\n\n" + body).strip()
                if len(text) < MIN_DOC_CHARS or not ARABIC_RE.search(text):
                    continue
                key = re.sub(r"\s+", "", text)[:2000]
                if key in seen:
                    continue
                seen.add(key)
                documents.append({"source": "blogger", "blog": blog_dir.name,
                                  "text": text})
        print(f"  {blog_dir.name}: {len(documents) - n_before:,} posts", flush=True)

    order = list(range(len(documents)))
    random.Random(SEED).shuffle(order)
    val_idx = set(order[: max(1, int(len(order) * VAL_FRACTION))])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, chars = {"train": 0, "val": 0}, {"train": 0, "val": 0}
    handles = {split: open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8")
               for split in counts}
    for i, doc in enumerate(documents):
        split = "val" if i in val_idx else "train"
        handles[split].write(json.dumps(doc, ensure_ascii=False) + "\n")
        counts[split] += 1
        chars[split] += len(doc["text"])
    for handle in handles.values():
        handle.close()
    print(f"train {counts['train']:,} docs / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,} -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
