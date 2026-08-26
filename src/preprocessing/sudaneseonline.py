"""sudaneseonline.com crawled threads -> one document per thread (plan.md Part IV, step 1.4).

The forum archives are 25 years of native Sudanese discussion — the discourse register
(argument, storytelling, commentary) the model currently degenerates in. The crawler
(scripts/scrape_sudaneseonline.py) stores one gzipped page per thread; this module extracts the
posts and writes plain documents, split by thread.

Three site-specific quirks handled here:

  - mixed encoding: pages are nominally UTF-8 but legacy segments are cp1256; each page is
    decoded both ways and the decoding with more Arabic wins,
  - post bodies sit in <ul> blocks after each post's header link — the page has no CSS classes
    worth trusting (1999-era HTML),
  - repetition: replies quote parents wholesale and signatures repeat per author, so any line
    already seen in the thread is dropped (a line-level dedup that removes quote pyramids and
    signatures in one stroke — same reasoning as the lyric chorus collapse).

Threads stay whole documents: the packer handles documents longer than a block, and thread-level
coherence is exactly the long-range signal the chat corpus lacks.

Output: data/interim/sudaneseonline/{train,val}.jsonl

Usage:  python -m src.preprocessing.sudaneseonline [--limit N]
"""

import argparse
import gzip
import html as html_lib
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_DIR = REPO_ROOT / "data" / "raw" / "sudaneseonline" / "html"
OUT_DIR = REPO_ROOT / "data" / "interim" / "sudaneseonline"

SEED = 67
VAL_FRACTION = 0.02
MIN_POST_CHARS = 25
MIN_THREAD_CHARS = 300

ARABIC_RE = re.compile(r"[؀-ۿ]")
UL_RE = re.compile(r"<ul>(.*?)</ul>", re.S | re.I)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)


# UTF-8-as-cp1256 mojibake is dominated by ط/ظ (the cp1256 readings of UTF-8 lead bytes).
# Real Arabic never is: in clean text those two letters are a few percent of characters.
MOJIBAKE_SIGNATURE = re.compile(r"[طظ]")


def _mojibake_score(text: str) -> float:
    arabic = ARABIC_RE.findall(text)
    return len(MOJIBAKE_SIGNATURE.findall(text)) / max(len(arabic), 1)


def fix_mojibake(text: str) -> str:
    """Reverse a UTF-8-read-as-cp1256 round trip, line by line.

    The site migrated encodings around 2015 and old posts are double-encoded *within*
    otherwise-cp1256 pages, so no single per-page decoding is right — each line is repaired
    independently, and only when the repair actually looks like Arabic.
    """
    fixed = []
    for line in text.splitlines():
        if _mojibake_score(line) > 0.25:
            try:
                candidate = line.encode("cp1256").decode("utf-8")
                if _mojibake_score(candidate) < 0.25 and ARABIC_RE.search(candidate):
                    line = candidate
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        fixed.append(line)
    return "\n".join(fixed)


def decode_page(raw: bytes) -> str:
    """cp1256 is the site's serving charset; mojibake repair handles the double-encoded parts."""
    try:
        page = raw.decode("utf-8")
    except UnicodeDecodeError:
        page = raw.decode("cp1256", errors="replace")
    return page


# Per-post site furniture: timestamp headers, the site's name banner, "my library" links.
# Left in, these lines recur in thousands of documents and become the strongest n-grams in the
# corpus — the model would learn to emit forum chrome. The audit also caught them dragging
# mojibake into the top dialect band.
BOILERPLATE_RE = re.compile(
    r"^\d{1,2}:\d{2} [AP]M \w{3}, \d{1,2} \d{4}$"
    r"|^سودانيز [اأ]ون لاين$"
    r"|مكتبت[يى] ف[يى] سودانيز\s*اونلاين"
    r"|^SudaneseOnline Images$"
    r"|^رابط مختصر$"
    r"|^مكتبت[يى]$"
)


def clean_fragment(fragment: str) -> str:
    text = TAG_RE.sub("\n", fragment)
    text = html_lib.unescape(text)
    text = fix_mojibake(text)
    text = text.replace("‏", "").replace("�", "")
    lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or BOILERPLATE_RE.search(line):
            continue
        # a line the round-trip repair could not fix stays mojibake forever — drop it rather
        # than let ط/ظ soup masquerade as maximally-dialectal text downstream
        if _mojibake_score(line) > 0.25 and len(ARABIC_RE.findall(line)) > 10:
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_thread(page: str):
    """(title, [post texts]) with quote/signature lines deduped across the thread."""
    title_match = TITLE_RE.search(page)
    title = clean_fragment(title_match.group(1)) if title_match else ""
    posts, seen_lines = [], set()
    for match in UL_RE.finditer(page):
        body = clean_fragment(match.group(1))
        fresh = []
        for line in body.splitlines():
            key = re.sub(r"\s+", "", line)
            if len(key) < 12:               # short lines (اها، فوق) repeat legitimately
                fresh.append(line)
                continue
            if key in seen_lines:
                continue
            seen_lines.add(key)
            fresh.append(line)
        body = "\n".join(fresh).strip()
        if len(body) >= MIN_POST_CHARS and ARABIC_RE.search(body):
            posts.append(body)
    return title, posts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N pages (for the audit pass)")
    args = parser.parse_args()

    pages = sorted(HTML_DIR.glob("*.html.gz"))[: args.limit]
    if not pages:
        raise FileNotFoundError(f"no crawled pages under {HTML_DIR}")

    documents, seen_threads = [], set()
    for path in pages:
        try:
            page = decode_page(gzip.decompress(path.read_bytes()))
        except (OSError, EOFError):
            continue
        title, posts = extract_thread(page)
        if not posts:
            continue
        text = (title + "\n\n" if title else "") + "\n\n".join(posts)
        if len(text) < MIN_THREAD_CHARS:
            continue
        key = re.sub(r"\s+", "", text)[:2000]
        if key in seen_threads:
            continue
        seen_threads.add(key)
        documents.append({"source": "sudaneseonline", "thread": path.stem, "text": text})

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

    print(f"{len(pages):,} pages -> {len(documents):,} threads "
          f"(train {counts['train']:,} / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,}) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
