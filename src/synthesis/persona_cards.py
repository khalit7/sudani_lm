"""Compile each PKB profile into a card *input*: the distillation request one Claude call turns
into a ~600–900-token persona card (plan.md Part IV, step 2.1).

The profiles run 15K–1.3M characters — far too big to ship per generation request — but they
were written for exactly this: §How we talk is literally a per-relationship style guide. This
module extracts the voice-bearing sections, staples on the machine-readable scaffolding
(vocatives with frequencies, topic distribution, gendered-address counts), pseudonymizes the
lot, and writes one self-contained distillation prompt per person. The generation driver
(generate.py) sends them; the finished cards land in data/interim/synthetic/cards/.

The owner card is compiled from the self-model's "How he talks" sections and is included in
every chat-synthesis request later, so its quality matters most.

Everything written here is already pseudonymized — the card inputs are exactly what will leave
the machine, so the owner can review these files to review the privacy boundary.

Output: data/interim/synthetic/card_inputs/<slug>.md   (+ owner.md)

Usage:  python -m src.synthesis.persona_cards
"""

import argparse
import json
import re
from pathlib import Path

from src.synthesis import blocklist
from src.synthesis.pseudonyms import Pseudonymizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PKB = Path.home() / "personal_knowledge_base"
OUT_DIR = REPO_ROOT / "data" / "interim" / "synthetic" / "card_inputs"

# (heading, char cap) — caps keep a 1.3MB profile's card input under ~25K chars. The sections
# not listed (History, Gifts/money, Watch out for) carry facts, not voice; cards are for voice.
SECTIONS = [
    ("Who", 1500),
    ("What they're like", 2500),
    ("How we talk", 9000),
    ("Running jokes and bits", 4000),
    ("What we talk about", 3000),
    ("Notable quotes", 4000),
]

CARD_SCHEMA = """[PERSONA {name}]
Relationship to K: ...
Gender: ...   Script mix: e.g. Arabic script 80% / Arabizi 15% / English 5%
Register: 2-4 sentences on how they actually write (length, punctuation, emoji, code-switching)
Vocatives they use for K / K uses for them: ...
Topics: comma list, most frequent first
Running jokes: up to 5, one line each
Voice samples: 6-10 short verbatim quotes (keep them EXACTLY as written, typos and all)
K mirrors them by: 1-2 sentences on how K adapts his own register to this person"""

INSTRUCTIONS = """You are distilling a relationship profile into a compact persona card for a
dialogue-synthesis pipeline. The card must let a generator reproduce this person's WhatsApp
voice in Sudanese Arabic. Preserve dialect features exactly: vowel-lengthening (شدييييد),
Arabizi spellings (7abibi, 8oob), emoji habits, taglines. Do not translate, do not clean up,
do not invent anything not in the material. Output ONLY the completed card, nothing else.

Card format:

{schema}

Material follows.
"""


def profile_sections(text: str):
    """h1-section extraction: profiles use a fixed '# Heading' schema."""
    sections = {}
    current, buf = None, []
    for line in text.splitlines():
        match = re.match(r"^# (.+?)\s*$", line)
        if match:
            if current:
                sections[current] = "\n".join(buf).strip()
            current, buf = match.group(1), []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()
    return sections


def markdown_section(text: str, heading_re: str):
    """One '## Heading' section from a self-model file (they use deeper heading levels)."""
    match = re.search(rf"^(#+) {heading_re}\s*$", text, re.M | re.I)
    if not match:
        return ""
    level = len(match.group(1))
    rest = text[match.end():]
    stop = re.search(rf"^#{{1,{level}}} ", rest, re.M)
    return rest[: stop.start()].strip() if stop else rest.strip()


def slug_to_chats():
    """profile slug -> (person key, chat titles), from corpus_index.json."""
    index = json.loads((PKB / "corpus_index.json").read_text())
    mapping = {}
    for person, info in index.items():
        slug = Path(info["file"]).stem
        mapping[slug] = (person, info.get("chats", []))
    return mapping


def scaffolding(chats, vocatives, evidence):
    """The machine-readable half of the card input, already aggregated per chat."""
    lines = []
    for chat in chats:
        if chat in vocatives:
            terms = ", ".join(f"{term}({count})" for term, count
                              in sorted(vocatives[chat].items(), key=lambda kv: -kv[1])[:10])
            lines.append(f"K's address terms for them: {terms}")
        entry = evidence.get(chat)
        if entry:
            topics = ", ".join(f"{topic}({count})" for topic, count
                               in sorted(entry.get("context", {}).items(),
                                         key=lambda kv: -kv[1])[:10])
            if topics:
                lines.append(f"Topic counts: {topics}")
            addressed = entry.get("addressed_as", {})
            if addressed:
                lines.append(f"Gendered address counts: {addressed}")
    return "\n".join(lines)


def compile_person(slug, person, chats, vocatives, evidence, pseudo) -> str | None:
    profile_path = PKB / "profiles" / f"{slug}.md"
    if not profile_path.exists():
        return None
    sections = profile_sections(profile_path.read_text())
    parts = [INSTRUCTIONS.format(schema=CARD_SCHEMA.format(
        name=pseudo.apply(person)))]
    for heading, cap in SECTIONS:
        body = sections.get(heading, "")
        if body:
            parts.append(f"## {heading}\n{body[:cap]}")
    parts.append("## Scaffolding\n" + scaffolding(chats, vocatives, evidence))
    return pseudo.apply("\n\n".join(parts))


def compile_owner(pseudo) -> str:
    """The owner's style card input: §How he talks from both self-model files."""
    material = []
    for name in ("khalid.md", "khalid-character.md"):
        text = (PKB / "self" / "profile" / name).read_text()
        section = markdown_section(text, "How he talks")
        if section:
            material.append(f"## From {name}\n{section[:14000]}")
    instructions = INSTRUCTIONS.format(schema=CARD_SCHEMA.format(name="K (the owner)")) \
        .replace("this person's", "the OWNER'S") \
        .replace("K mirrors them by", "How he mirrors each interlocutor")
    return pseudo.apply("\n\n".join([instructions] + material))


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    pseudo = Pseudonymizer()
    vocatives = json.loads((PKB / "vocatives.json").read_text())
    evidence = json.loads((PKB / "evidence.json").read_text())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for slug, (person, chats) in sorted(slug_to_chats().items()):
        if any(blocklist.is_blocked(i) for i in (slug, person, *chats)):
            print(f"  {slug}: blocklisted, excluded")
            continue
        card_input = compile_person(slug, person, chats, vocatives, evidence, pseudo)
        if card_input is None:
            print(f"  {slug}: no profile, skipped")
            continue
        leftovers = pseudo.scan(card_input)
        if leftovers:
            raise RuntimeError(f"{slug}: real names survived pseudonymization: {leftovers}")
        (OUT_DIR / f"{slug}.md").write_text(card_input)
        written += 1

    owner = compile_owner(pseudo)
    if pseudo.scan(owner):
        raise RuntimeError(f"owner card input: real names survived: {pseudo.scan(owner)}")
    (OUT_DIR / "owner.md").write_text(owner)

    # final gate: nothing blocklisted may exist in the output directory, however it got there
    blocklist.assert_clean((p.stem for p in OUT_DIR.iterdir()), "card_inputs directory")
    print(f"{written} person card inputs + owner -> {OUT_DIR}")
    print("these files are exactly what will be sent to Claude — review them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
