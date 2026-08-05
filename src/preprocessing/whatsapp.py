"""Turn the WhatsApp export into conversations.

A flat CSV of 1M messages is not dialogue. What makes it dialogue is segmenting it into
sessions, merging bursts from one sender, and labelling who is speaking — that structure is what
the model needs in order to learn to *reply* rather than to continue text.

Output: data/interim/whatsapp/{train,val}.jsonl, one rendered conversation per line, using the
chat template from src/tokenizer/special_tokens.py. Both splits feed the packer.

Holdout is by **whole chat**, never by random message. Phrases, names and running jokes recur so
heavily inside one conversation that a random split would leak them across the boundary and
report a flatteringly low perplexity — which would then be the number steering stages C and D.

Usage:  python -m src.preprocessing.whatsapp
"""

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from src.tokenizer.special_tokens import SELF_SPEAKER, render_conversation

csv.field_size_limit(10**9)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_CSV = REPO_ROOT / "data" / "raw" / "whatsapp" / "whatsapp.csv"
OUT_DIR = REPO_ROOT / "data" / "interim" / "whatsapp"

SEED = 67
# A pause longer than this starts a new conversation. Six hours separates "later that evening"
# from "the next day", which is where the thread of a chat actually breaks.
SESSION_GAP_HOURS = 6
# Conversations are capped so that a packed 1024-token block usually holds whole exchanges
# rather than a slice from the middle of a six-month thread.
MAX_TURNS_PER_CONVERSATION = 30
MAX_CHARS_PER_CONVERSATION = 2_000
MIN_TURNS_PER_CONVERSATION = 2
# Fraction of *chats* held out, not messages.
VAL_CHAT_FRACTION = 0.03

PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")


def parse_timestamp(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None


def speaker_for(row, chat_name):
    """Who is talking.

    Outgoing rows carry an empty Sender Name in this export — all 412,830 of them — so the
    owner's own messages have to be labelled from Type, not from the name column.
    """
    if row["Type"] == "Outgoing":
        return SELF_SPEAKER
    name = (row.get("Sender Name") or "").strip()
    return name or chat_name or "UNKNOWN"


def clean_text(text, pseudonymize=False):
    text = (text or "").strip()
    if not text:
        return ""
    # Collapse newlines: a turn is one utterance, and the template already delimits turns.
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    if pseudonymize:
        text = PHONE_RE.sub("<phone>", text)
    return text.strip()


def load_messages(path=CHAT_CSV, pseudonymize=False):
    """Text-bearing messages only, grouped by chat and ordered in time."""
    by_chat = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            # Notifications are system chatter ("X created group", "you were added"), and
            # attachment-only rows are stickers and images with nothing to model. Captions on
            # attachments are kept — the text is real.
            if row["Type"] not in ("Incoming", "Outgoing"):
                continue
            text = clean_text(row.get("Text"), pseudonymize)
            if not text:
                continue
            timestamp = parse_timestamp(row.get("Message Date"))
            if timestamp is None:
                continue
            chat = (row.get("Chat Session") or "").strip()
            by_chat[chat].append((timestamp, speaker_for(row, chat), text))

    for messages in by_chat.values():
        messages.sort(key=lambda m: m[0])
    return by_chat


def segment_sessions(messages, gap_hours=SESSION_GAP_HOURS):
    """Split one chat's history wherever it goes quiet for longer than `gap_hours`."""
    sessions, current = [], []
    previous = None
    for timestamp, speaker, text in messages:
        if previous is not None and (timestamp - previous).total_seconds() > gap_hours * 3600:
            if current:
                sessions.append(current)
            current = []
        current.append((speaker, text))
        previous = timestamp
    if current:
        sessions.append(current)
    return sessions


def merge_consecutive(turns):
    """Collapse a burst from one sender into a single turn.

    People send four messages in a row where one paragraph was meant. Left unmerged the model
    learns to emit a turn marker every few words.
    """
    merged = []
    for speaker, text in turns:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {text}")
        else:
            merged.append((speaker, text))
    return merged


def chunk_conversation(turns):
    """Break a long session into conversation-sized pieces."""
    chunks, current, chars = [], [], 0
    for speaker, text in turns:
        too_long = len(current) >= MAX_TURNS_PER_CONVERSATION or chars + len(text) > MAX_CHARS_PER_CONVERSATION
        if current and too_long:
            chunks.append(current)
            current, chars = [], 0
        current.append((speaker, text))
        chars += len(text)
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= MIN_TURNS_PER_CONVERSATION]


def build_conversations(by_chat):
    """chat -> list of rendered conversation strings."""
    out = {}
    for chat, messages in by_chat.items():
        rendered = []
        for session in segment_sessions(messages):
            for chunk in chunk_conversation(merge_consecutive(session)):
                rendered.append(render_conversation(chunk))
        if rendered:
            out[chat] = rendered
    return out


def split_by_chat(conversations, val_fraction=VAL_CHAT_FRACTION, seed=SEED):
    """Hold out whole chats, chosen to hit the target share of *conversations*.

    Chats are wildly uneven — the largest holds 184k messages — so picking a fixed count of
    chats would give a validation set of unpredictable size.
    """
    chats = sorted(conversations)
    rng = random.Random(seed)
    rng.shuffle(chats)

    total = sum(len(conversations[c]) for c in chats)
    target = total * val_fraction
    val_chats, running = [], 0
    for chat in chats:
        if running >= target:
            break
        # skip any single chat large enough to swallow the whole budget
        if len(conversations[chat]) > target * 1.5 and val_chats:
            continue
        val_chats.append(chat)
        running += len(conversations[chat])

    val = set(val_chats)
    return (
        {c: v for c, v in conversations.items() if c not in val},
        {c: v for c, v in conversations.items() if c in val},
    )


def write_split(conversations, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for chat, rendered in sorted(conversations.items()):
            for text in rendered:
                fh.write(json.dumps({"chat": chat, "text": text}, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudonymize", action="store_true",
                        help="replace phone numbers with a placeholder")
    args = parser.parse_args()

    by_chat = load_messages(pseudonymize=args.pseudonymize)
    print(f"chats with text: {len(by_chat):,}  messages: {sum(len(v) for v in by_chat.values()):,}")

    conversations = build_conversations(by_chat)
    total = sum(len(v) for v in conversations.values())
    print(f"conversations: {total:,} across {len(conversations):,} chats")

    train, val = split_by_chat(conversations)
    n_train = write_split(train, OUT_DIR / "train.jsonl")
    n_val = write_split(val, OUT_DIR / "val.jsonl")

    print(f"train: {n_train:,} conversations from {len(train):,} chats")
    print(f"val:   {n_val:,} conversations from {len(val):,} chats "
          f"({100*n_val/max(total,1):.1f}% of conversations)")
    print(f"held-out chats: {sorted(val)[:5]}{' ...' if len(val) > 5 else ''}")
    (OUT_DIR / "val_chats.json").write_text(
        json.dumps(sorted(val), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
