"""Wayback + Common Crawl dumps of dead Sudanese forums -> documents (plan step 1.9).

ONE extractor for both miners (scripts/scrape_wayback.py, scripts/scrape_commoncrawl.py)
because they capture the same pages: vBulletin archive pages (`div.posttext` — clean),
full-skin showthread pages (`div#post_message_N`), and IPB pages (`div.postcolor`,
sudanesesongs.net). Listing/index pages carry no post container and are skipped.

The same thread routinely exists as multiple captures (different snapshot years, archive
AND showthread renderings, Wayback AND CC) — so post lines >= 12 normalized chars are
deduped globally per domain, which collapses re-captures, quote pyramids and signatures
in one mechanism. Documents are per page; dedup makes the union across sources safe.

Pages declare windows-1256; decoding honours the charset meta and falls back to cp1256,
with sudaneseonline's per-line mojibake repair for double-encoded segments.

Output: data/interim/vbarchive/<domain>/{train,val}.jsonl

Usage:  python -m src.preprocessing.vbarchive [--domains d1,d2]
"""

import argparse
import gzip
import hashlib
import html as html_lib
import json
import random
import re
from html.parser import HTMLParser
from pathlib import Path

from src.preprocessing.sudaneseonline import fix_mojibake

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"wayback": REPO_ROOT / "data" / "raw" / "wayback",
           "cc": REPO_ROOT / "data" / "raw" / "commoncrawl"}
OUT_ROOT = REPO_ROOT / "data" / "interim" / "vbarchive"

SEED = 67
VAL_FRACTION = 0.02
MIN_POST_CHARS = 25
MIN_DOC_CHARS = 300

ARABIC_RE = re.compile(r"[؀-ۿ]")
CHARSET_RE = re.compile(rb'charset=["\']?([\w-]+)', re.I)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)
TAG_RE = re.compile(r"<[^>]+>")


class ForumPostExtractor(HTMLParser):
    """Text of every vB-archive / vB-showthread / IPB post container on a page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.posts, self._chunks, self._depth = [], [], 0

    # vB archive / IPB 2.x lofi / IPB 3.x full-skin post containers
    POST_CLASSES = {"posttext", "postcolor", "entry-content"}

    @classmethod
    def _is_post(cls, tag, attrs):
        if tag != "div":
            return False
        attrs = dict(attrs)
        if set((attrs.get("class") or "").split()) & cls.POST_CLASSES:
            return True
        return (attrs.get("id") or "").startswith("post_message_")

    def handle_starttag(self, tag, attrs):
        if self._depth:
            if tag == "div":
                self._depth += 1
            elif tag in ("br", "p", "blockquote", "li", "tr"):
                self._chunks.append("\n")
        elif self._is_post(tag, attrs):
            self._depth = 1

    def handle_endtag(self, tag):
        if self._depth and tag == "div":
            self._depth -= 1
            if not self._depth:
                self.posts.append("".join(self._chunks))
                self._chunks = []
        elif self._depth and tag in ("p", "blockquote", "li", "tr"):
            self._chunks.append("\n")

    def handle_data(self, data):
        if self._depth:
            self._chunks.append(data)


def decode(raw: bytes) -> str:
    match = CHARSET_RE.search(raw[:2048])
    charset = (match.group(1).decode("ascii", "replace").lower() if match else "cp1256")
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("cp1256", errors="replace")
    return fix_mojibake(text)


# Word-pasted posts leave mangled Office XML tags (<o:p>, half-eaten "<O") as literal text
OFFICE_DEBRIS_RE = re.compile(r"</?[Oo](:[A-Za-z]+)?>?(?=\s|$)", re.M)


def extract_page(raw: bytes):
    page = decode(raw)
    title_match = TITLE_RE.search(page)
    title = ""
    if title_match:
        title = html_lib.unescape(TAG_RE.sub(" ", title_match.group(1)))
        title = re.sub(r"\s+", " ", title).strip()
    parser = ForumPostExtractor()
    parser.feed(page)
    posts = []
    for body in parser.posts:
        body = OFFICE_DEBRIS_RE.sub("", body)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in body.splitlines()]
        body = "\n".join(line for line in lines if line)
        if len(body) >= MIN_POST_CHARS and ARABIC_RE.search(body):
            posts.append(body)
    return title, posts


def process_domain(domain: str, page_dirs):
    documents, seen_lines, n_pages = [], set(), 0
    for provenance, pages_dir in page_dirs:
        for path in sorted(pages_dir.glob("*.html.gz")):
            try:
                raw = gzip.decompress(path.read_bytes())
            except (OSError, EOFError):
                continue
            n_pages += 1
            title, posts = extract_page(raw)
            fresh_posts = []
            for post in posts:
                fresh = []
                for line in post.splitlines():
                    key = re.sub(r"\s+", "", line)
                    if len(key) >= 12:
                        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
                        if digest in seen_lines:
                            continue
                        seen_lines.add(digest)
                    fresh.append(line)
                post = "\n".join(fresh).strip()
                if len(post) >= MIN_POST_CHARS and ARABIC_RE.search(post):
                    fresh_posts.append(post)
            if not fresh_posts:
                continue
            text = (title + "\n\n" if title else "") + "\n\n".join(fresh_posts)
            if len(text) < MIN_DOC_CHARS:
                continue
            documents.append({"source": f"vbarchive/{domain}", "page": path.stem,
                              "provenance": provenance, "text": text})
    return n_pages, documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default=None,
                        help="comma list; default: every domain present in either miner")
    args = parser.parse_args()

    domains = {}
    for provenance, root in SOURCES.items():
        if not root.is_dir():
            continue
        for domain_dir in sorted(root.iterdir()):
            pages_dir = domain_dir / "pages"
            if pages_dir.is_dir():
                domains.setdefault(domain_dir.name, []).append((provenance, pages_dir))
    if args.domains:
        wanted = {d.strip() for d in args.domains.split(",")}
        domains = {d: v for d, v in domains.items() if d in wanted}

    for domain, page_dirs in domains.items():
        n_pages, documents = process_domain(domain, page_dirs)
        order = list(range(len(documents)))
        random.Random(SEED).shuffle(order)
        val_idx = set(order[: max(1, int(len(order) * VAL_FRACTION))]) if documents else set()
        out_dir = OUT_ROOT / domain
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
        print(f"{domain}: {n_pages:,} pages -> {len(documents):,} docs "
              f"(train {counts['train']:,} / {chars['train']/1e6:.1f}M chars, "
              f"val {counts['val']:,}) -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
