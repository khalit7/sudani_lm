"""WordPress wp-json crawls -> documents (plan.md Part IV, step 1.9).

One module for all four WP corpora (alnilin, sudanile, koorasudan, cover_sd) — the crawler
(scripts/scrape_wpjson.py) saved identical JSON for all of them. Two output streams per
site, because they play different mixture roles:

  posts     -> silver: one document per article (title + body), MSA news ranked down by
               the dialect scorer at pack time;
  comments  -> gold path: reader comments are where dialect lives, but most are too short
               to stand alone, so they are grouped under their article (title as header,
               comments in date order). One document per commented article.

Output: data/interim/<name>_posts/{train,val}.jsonl
        data/interim/<name>_comments/{train,val}.jsonl      (when the site has comments)

Usage:  python -m src.preprocessing.wpjson_clean --name alnilin [--what posts,comments]
"""

import argparse
import html as html_lib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "data" / "interim"

SEED = 67
VAL_FRACTION = 0.02
MIN_POST_CHARS = 200
MIN_COMMENT_CHARS = 15
MIN_COMMENT_DOC_CHARS = 60

ARABIC_RE = re.compile(r"[؀-ۿ]")
BLOCK_RE = re.compile(r"</(?:p|div|li|h[1-6]|blockquote)>|<br\s*/?>", re.I)
TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)
BBCODE_RE = re.compile(r"\[/?[a-zA-Z][^\]\n]{0,40}\]")   # raw [B]/[SIZE=4] in comment bodies


def strip_html(rendered: str) -> str:
    text = BLOCK_RE.sub("\n", rendered)
    text = TAG_RE.sub(" ", text)
    text = html_lib.unescape(html_lib.unescape(text))    # &amp;quot; needs two passes
    text = BBCODE_RE.sub("", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_pages(directory: Path):
    for path in sorted(directory.glob("page_*.json")):
        try:
            yield from json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue


def write_split(documents, out_dir: Path, label: str):
    order = list(range(len(documents)))
    random.Random(SEED).shuffle(order)
    val_idx = set(order[: max(1, int(len(order) * VAL_FRACTION))]) if documents else set()
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
    print(f"{label}: {len(documents):,} docs "
          f"(train {counts['train']:,} / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,}) -> {out_dir}")


def post_titles(raw_dir: Path) -> dict:
    titles = {}
    for row in load_pages(raw_dir / "posts"):
        titles[row["id"]] = strip_html(row.get("title", {}).get("rendered", ""))
    return titles


def build_posts(name: str, raw_dir: Path):
    documents, seen = [], set()
    for row in load_pages(raw_dir / "posts"):
        title = strip_html(row.get("title", {}).get("rendered", ""))
        body = strip_html(row.get("content", {}).get("rendered", ""))
        text = (title + "\n\n" + body).strip()
        if len(text) < MIN_POST_CHARS or not ARABIC_RE.search(text):
            continue
        key = re.sub(r"\s+", "", text)[:2000]
        if key in seen:
            continue
        seen.add(key)
        documents.append({"source": f"{name}_posts", "post": row["id"], "text": text})
    write_split(documents, OUT_ROOT / f"{name}_posts", f"{name}/posts")


def build_comments(name: str, raw_dir: Path):
    by_post = defaultdict(list)
    for row in load_pages(raw_dir / "comments"):
        body = strip_html(row.get("content", {}).get("rendered", ""))
        if len(body) >= MIN_COMMENT_CHARS and ARABIC_RE.search(body):
            by_post[row.get("post", 0)].append((row.get("date", ""), body))
    titles = post_titles(raw_dir)
    documents = []
    for post_id in sorted(by_post):
        comments = [body for _, body in sorted(by_post[post_id])]
        title = titles.get(post_id, "")
        text = ((title + "\n\n") if title else "") + "\n\n".join(comments)
        if len(text) < MIN_COMMENT_DOC_CHARS:
            continue
        documents.append({"source": f"{name}_comments", "post": post_id,
                          "n_comments": len(comments), "text": text})
    write_split(documents, OUT_ROOT / f"{name}_comments", f"{name}/comments")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--what", default="posts,comments")
    args = parser.parse_args()
    raw_dir = REPO_ROOT / "data" / "raw" / args.name
    for what in args.what.split(","):
        what = what.strip()
        if not (raw_dir / what).is_dir():
            print(f"{args.name}/{what}: no raw directory, skipped")
            continue
        (build_posts if what == "posts" else build_comments)(args.name, raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
