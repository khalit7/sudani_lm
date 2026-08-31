"""anasudani.net phpBB crawl -> one document per topic (plan.md Part IV, step 1.9).

1.16M posts of Sudanese forum discussion, frozen ~2017, crawled complete (49,559 topics /
68,214 pages). Modern phpBB 3 prosilver markup, clean UTF-8 — none of sudaneseonline's
encoding archaeology. Post bodies live in `<div class="content">`; blockquotes (quoted
parents) nest inside them, so bodies are extracted with a depth-tracking HTML parser
rather than a regex, and the thread-level seen-line dedup then removes the quote pyramids
and repeated signatures in one stroke (same reasoning as sudaneseonline.py).

Multi-page topics are stitched: pages are saved as t<id>_s<offset>.html.gz and grouped by
topic id, ascending offset.

Output: data/interim/anasudani/{train,val}.jsonl   (then: dialect_score score <file>)

Usage:  python -m src.preprocessing.anasudani [--limit N]
"""

import argparse
import gzip
import html as html_lib
import json
import random
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HTML_DIR = REPO_ROOT / "data" / "raw" / "anasudani" / "html"
OUT_DIR = REPO_ROOT / "data" / "interim" / "anasudani"

SEED = 67
VAL_FRACTION = 0.02
MIN_POST_CHARS = 25
MIN_TOPIC_CHARS = 300

ARABIC_RE = re.compile(r"[؀-ۿ]")
PAGE_RE = re.compile(r"t(\d+)_s(\d+)\.html\.gz$")
TITLE_RE = re.compile(r'<h2 class="topic-title">.*?<a[^>]*>(.*?)</a>', re.S)


class PostExtractor(HTMLParser):
    """Collects the text of every `div.content` (post body), tracking div nesting."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.posts, self._chunks, self._depth = [], [], 0

    def handle_starttag(self, tag, attrs):
        if self._depth:
            if tag == "div":
                self._depth += 1
            elif tag in ("br", "p", "blockquote", "li"):
                self._chunks.append("\n")
        elif tag == "div" and ("class", "content") in attrs:
            self._depth = 1

    def handle_endtag(self, tag):
        if self._depth and tag == "div":
            self._depth -= 1
            if not self._depth:
                self.posts.append("".join(self._chunks))
                self._chunks = []
        elif self._depth and tag in ("p", "blockquote", "li"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._depth:
            self._chunks.append(data)


BBCODE_RE = re.compile(r"\[/?[a-zA-Z][^\]\n]{0,40}\]")   # unrendered [glow]/[align]/… leftovers


def clean_post(text: str) -> str:
    # convert_charrefs already unescaped once; &amp;quot; in the source needs one more
    text = html_lib.unescape(text)
    text = BBCODE_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_topic(pages) -> tuple[str, list[str]]:
    """(title, deduped post bodies) across the topic's pages, ascending offset."""
    title, posts, seen_lines = "", [], set()
    for _, path in sorted(pages):
        try:
            page = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        except (OSError, EOFError):
            continue
        if not title:
            match = TITLE_RE.search(page)
            if match:
                title = clean_post(re.sub(r"<[^>]+>", "", match.group(1)))
        parser = PostExtractor()
        parser.feed(page)
        for body in parser.posts:
            fresh = []
            for line in clean_post(body).splitlines():
                key = re.sub(r"\s+", "", line)
                if len(key) >= 12:
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
                        help="process only the first N topics (for the audit pass)")
    args = parser.parse_args()

    topics = defaultdict(list)
    for path in HTML_DIR.glob("t*_s*.html.gz"):
        match = PAGE_RE.search(path.name)
        if match:
            topics[int(match.group(1))].append((int(match.group(2)), path))
    if not topics:
        raise FileNotFoundError(f"no crawled pages under {HTML_DIR}")
    topic_ids = sorted(topics)[: args.limit]

    documents = []
    for topic_id in topic_ids:
        title, posts = extract_topic(topics[topic_id])
        if not posts:
            continue
        text = (title + "\n\n" if title else "") + "\n\n".join(posts)
        if len(text) < MIN_TOPIC_CHARS:
            continue
        documents.append({"source": "anasudani", "topic": topic_id, "text": text})

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

    print(f"{len(topic_ids):,} topics -> {len(documents):,} documents "
          f"(train {counts['train']:,} / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,} / {chars['val']/1e6:.1f}M chars) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
