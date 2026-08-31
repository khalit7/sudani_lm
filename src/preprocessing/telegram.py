"""Crawled Telegram channels -> episode-length documents (plan.md Part IV, step 1.9).

The 16 public channels (data/raw/telegram/, scripts/scrape_telegram.py) are the purest
dialect source in the project — serial fiction, poetry, أمثال, ونسة — but they arrive as
~20-message preview pages. Serial-fiction chapters are posted as *runs* of consecutive
messages, so this module reassembles messages (in id order) into documents, which is
exactly the discourse-length signal the model degenerates without. Quote/proverb channels
come out as quote-collection documents, which the packer treats fine.

Cleaning decisions:
  - promo/ad messages (the fiction channels cross-promote heavily): a message that
    carried a URL and keeps little Arabic after URL removal is dropped;
  - recurring channel furniture (join-us footers, watermark lines) is removed by
    per-channel line dedup for lines >= 12 normalized chars — the same single stroke
    that removed forum signatures in sudaneseonline.py. Fiction text never legitimately
    repeats a >=12-char line inside one channel;
  - documents close at a size target or at a large message-id gap (mass deletions mark
    era boundaries; stitching across them would splice unrelated serials).

Val split: the LAST 2% of documents per channel (contiguous, by id) — adjacent chapters
share a storyline, so a random split would leak plot across the boundary.

Output: data/interim/telegram/{train,val}.jsonl   (then: dialect_score score <file>)

Usage:  python -m src.preprocessing.telegram [--channels a,b] [--limit-pages N]
"""

import argparse
import html as html_lib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "telegram"
OUT_DIR = REPO_ROOT / "data" / "interim" / "telegram"

VAL_FRACTION = 0.02
TARGET_DOC_CHARS = 6_000
MAX_ID_GAP = 200
MIN_DOC_CHARS = 200
MIN_MSG_ARABIC = 10          # keep proverb-length messages
MIN_ARABIC_AFTER_URL = 40    # a linky message must still carry real text to survive

ARABIC_RE = re.compile(r"[؀-ۿ]")
BR_RE = re.compile(r"<br\s*/?>", re.I)
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(?:https?://|t\.me/|www\.)\S+", re.I)
MENTION_RE = re.compile(r"@\w{4,}")


def clean_message(raw_html: str) -> str:
    """Message HTML -> plain text, or '' when the message is promo/furniture."""
    had_url = bool(URL_RE.search(raw_html))   # promo links ride in href attrs, so look
    text = BR_RE.sub("\n", raw_html)          # BEFORE tag-stripping removes them
    text = TAG_RE.sub("", text)
    text = html_lib.unescape(text)
    text = BR_RE.sub("\n", text)   # literal &lt;br&gt; entities surface only after unescape
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    arabic = len(ARABIC_RE.findall(text))
    if arabic < MIN_MSG_ARABIC:
        return ""
    if had_url and arabic < MIN_ARABIC_AFTER_URL:
        return ""
    return text


def load_channel(channel_dir: Path, limit_pages=None):
    """All messages of one channel, ascending by id, page-boundary duplicates removed."""
    messages = {}
    for page_path in sorted(channel_dir.glob("page_*.json"))[:limit_pages]:
        try:
            page = json.loads(page_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for message in page.get("messages", []):
            messages[int(message["id"])] = message["html"]
    return sorted(messages.items())


def build_documents(channel: str, messages):
    """Merge consecutive cleaned messages into documents; per-channel line dedup."""
    seen_lines = set()
    documents, parts, chars, start_id, prev_id = [], [], 0, None, None

    def flush():
        nonlocal parts, chars, start_id
        text = "\n\n".join(parts).strip()
        if len(text) >= MIN_DOC_CHARS:
            documents.append({"source": "telegram", "channel": channel,
                              "start_id": start_id, "text": text})
        parts, chars, start_id = [], 0, None

    for message_id, raw_html in messages:
        text = clean_message(raw_html)
        if text:
            fresh = []
            for line in text.splitlines():
                key = re.sub(r"\s+", "", line)
                if len(key) >= 12:
                    if key in seen_lines:
                        continue
                    seen_lines.add(key)
                fresh.append(line)
            text = "\n".join(fresh).strip()
        if not text:
            continue
        if parts and prev_id is not None and message_id - prev_id > MAX_ID_GAP:
            flush()
        if not parts:
            start_id = message_id
        parts.append(text)
        chars += len(text)
        prev_id = message_id
        if chars >= TARGET_DOC_CHARS:
            flush()
    flush()
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default=None,
                        help="comma list; default: every directory under data/raw/telegram")
    parser.add_argument("--limit-pages", type=int, default=None,
                        help="per-channel page cap (for the audit pass)")
    args = parser.parse_args()

    channel_dirs = ([RAW_DIR / c.strip() for c in args.channels.split(",")]
                    if args.channels else
                    sorted(d for d in RAW_DIR.iterdir() if d.is_dir()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts, chars = {"train": 0, "val": 0}, {"train": 0, "val": 0}
    handles = {split: open(OUT_DIR / f"{split}.jsonl", "w", encoding="utf-8")
               for split in counts}
    for channel_dir in channel_dirs:
        messages = load_channel(channel_dir, args.limit_pages)
        documents = build_documents(channel_dir.name, messages)
        n_val = max(1, int(len(documents) * VAL_FRACTION)) if len(documents) >= 10 else 0
        for i, doc in enumerate(documents):
            split = "val" if n_val and i >= len(documents) - n_val else "train"
            handles[split].write(json.dumps(doc, ensure_ascii=False) + "\n")
            counts[split] += 1
            chars[split] += len(doc["text"])
        print(f"  {channel_dir.name}: {len(messages):,} msgs -> {len(documents):,} docs",
              flush=True)
    for handle in handles.values():
        handle.close()
    print(f"train {counts['train']:,} docs / {chars['train']/1e6:.1f}M chars, "
          f"val {counts['val']:,} docs / {chars['val']/1e6:.1f}M chars -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
