"""Sudanese song lyrics from the Tarab corpus (CC-BY), one document per song.

~89K verses across ~5K songs, ~0.7M tokens. Lyrics carry vocabulary and imagery the chat corpus
lacks (Haqiba-era poetic register, rural lexicon), which is why they are worth having at all —
but sung verse is a repetitive register and the model already has a repetition failure mode, so
two guards apply here and a third at the mixture level:

  - consecutive duplicate lines are collapsed (choruses),
  - whole-song dedup by normalized text,
  - the mixture manifest caps lyrics at ~2% with repeat: 1 (enforced there, not here).

The Tarab release already contains the older Habibi lyric corpus (`corpus_version` column), so
this single file covers both.

Split is by song, never by verse — same principle as whatsapp.py's split-by-chat.

Output: data/interim/lyrics/{train,val}.jsonl

Usage:  python -m src.preprocessing.lyrics
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "data" / "raw" / "tarab" / "tarab_Sudanese.csv"
OUT_DIR = REPO_ROOT / "data" / "interim" / "lyrics"

SEED = 67
VAL_FRACTION = 0.02
MIN_SONG_CHARS = 80

# Verses arrive with inline ellipses as caesura marks ("انت ما قتلى لي... كلمتنى عيونى").
# Kept: they are part of how lyrics are written online. Only whitespace is normalized.
SPACES_RE = re.compile(r"[ \t]{2,}")


def _norm_line(line: str) -> str:
    return SPACES_RE.sub(" ", line.strip())


def load_songs():
    """csv rows -> one text per song, verses in order, chorus repeats collapsed."""
    csv.field_size_limit(10**8)
    songs = {}
    with open(CSV_PATH, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            key = row["art_id"]
            verse = _norm_line(row["verse_lyrics"] or "")
            if not verse:
                continue
            try:
                order = float(row["verse_order"])
            except (TypeError, ValueError):
                order = 1e9
            songs.setdefault(key, {"title": _norm_line(row["art_title"] or ""), "verses": []})
            songs[key]["verses"].append((order, verse))

    documents = []
    for song in songs.values():
        lines = []
        for _, verse in sorted(song["verses"], key=lambda pair: pair[0]):
            # collapse consecutive duplicates: a chorus sung four times is one training signal,
            # four copies is a repetition lesson
            if lines and verse == lines[-1]:
                continue
            lines.append(verse)
        text = "\n".join(lines)
        if len(text) >= MIN_SONG_CHARS:
            documents.append({"source": "tarab_sudanese", "title": song["title"], "text": text})
    return documents


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    documents = load_songs()

    # whole-song dedup: the corpus merges several collections and popular songs recur
    seen, unique = set(), []
    for doc in documents:
        key = re.sub(r"\s+", "", doc["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)

    order = list(range(len(unique)))
    random.Random(SEED).shuffle(order)
    cut = max(1, int(len(order) * VAL_FRACTION))
    val_idx = set(order[:cut])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, indices in (("train", [i for i in range(len(unique)) if i not in val_idx]),
                           ("val", sorted(val_idx))):
        path = OUT_DIR / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for i in indices:
                fh.write(json.dumps(unique[i], ensure_ascii=False) + "\n")
        counts[split] = len(indices)

    chars = sum(len(d["text"]) for d in unique)
    print(f"{len(documents):,} songs -> {len(unique):,} after dedup, {chars/1e6:.2f}M chars"
          f" (train {counts['train']:,} / val {counts['val']:,}) -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
