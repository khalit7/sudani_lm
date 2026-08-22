"""Stable pseudonym map for everything that leaves the machine (plan.md Part IV, privacy rule 3).

Persona cards and seed excerpts are sent to Claude during synthesis. Before anything is sent,
every real identity is rewritten to a stable fake Sudanese name — stable so that one person is
one consistent character across every request, fake so that no real name rides along. Phones,
emails and handles are masked outright.

What stays: dialect-bearing address terms (يا زول، حبيبة، عسل...) — they are Sudanese signal,
not identity. What goes: the names attached to them.

The map is built from the PKB's own resolution tables (people_map.csv, corpus_index.json), which
already did the hard work of listing every alias per person. It is written to
data/interim/synthetic/pseudonym_map.json — local only, gitignored, never uploaded — and is
meant to be reviewed and hand-extended: automated name extraction will miss rare nicknames, and
the QC scan (qc.py) plus the owner's card review are the catch for what slips through.

Usage:  python -m src.synthesis.pseudonyms          # build/refresh the map
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

from src.synthesis import blocklist

REPO_ROOT = Path(__file__).resolve().parents[2]
PKB = Path.home() / "personal_knowledge_base"
MAP_PATH = REPO_ROOT / "data" / "interim" / "synthetic" / "pseudonym_map.json"

SEED = 67

# Fake name pool: common Sudanese given names, (arabic, latin) pairs. Deliberately ordinary —
# a pseudonym should read as a plausible contact, not a codename. Assignment skips any name
# that collides with a real variant in the map.
FAKE_FEMALE = [
    ("آمنة", "Amna"), ("تسنيم", "Tasneem"), ("رحاب", "Rehab"), ("سلمى", "Salma"),
    ("نجلاء", "Naglaa"), ("إسراء", "Esraa"), ("وئام", "Weam"), ("رقية", "Rugia"),
    ("مودة", "Mawada"), ("سارة", "Sara"), ("هدى", "Huda"), ("لينا", "Lina"),
    ("ريان", "Rayan"), ("منال", "Manal"), ("تقوى", "Tagwa"), ("أروى", "Arwa"),
    ("دانية", "Dania"), ("شهد", "Shahad"), ("علياء", "Alya"), ("ميساء", "Maysa"),
    ("نورهان", "Nourhan"), ("رنا", "Rana"), ("سماح", "Samah"), ("وفاء", "Wafaa"),
]
FAKE_MALE = [
    ("المعتصم", "Mutasim"), ("صديق", "Siddig"), ("عوض", "Awadalla"), ("الطيب", "Altayeb"),
    ("بكري", "Bakri"), ("مأمون", "Mamoun"), ("حاتم", "Hatim"), ("ياسر", "Yasir"),
    ("عصام", "Isam"), ("نادر", "Nader"), ("همام", "Humam"), ("وليد", "Waleed"),
    ("مجاهد", "Mugahid"), ("أنس", "Anas"), ("قصي", "Gusai"), ("شريف", "Shareef"),
    ("عمار", "Ammar"), ("زين", "Zain"), ("طارق", "Tarig"), ("فارس", "Faris"),
    ("إبراهيم", "Ibrahim"), ("عثمان", "Othman"), ("سيف", "Saif"), ("منتصر", "Muntasir"),
]

FAKE_SURNAMES = [
    ("عبدالرحيم", "Abdelraheem"), ("الأمين", "Alamin"), ("حمدنالله", "Hamdnalla"),
    ("الجزولي", "Aljazouli"), ("أبوزيد", "Abuzaid"), ("الفاضل", "Alfadil"),
    ("عبدالماجد", "Abdelmagid"), ("النور", "Alnour"), ("ميرغني", "Mirghani"),
    ("الصافي", "Alsafi"), ("عبدالقادر", "Abdelgadir"), ("البدوي", "Albadawi"),
]

PHONE_RE = re.compile(r"\+?\d[\d\s\-()]{7,}\d")          # matches whatsapp.py's pattern
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")       # dates are context, not identity
LATIN_STOPWORDS = {
    "the", "new", "and", "not", "this", "that", "male", "female", "friend", "best",
    "likely", "unknown", "none", "old", "chat", "group", "row", "found", "still",
    "active", "her", "his", "him", "she", "with", "from", "for",
}
HANDLE_RE = re.compile(r"(?<![\w@])@\w{3,}")
ARABIC_RE = re.compile(r"[؀-ۿ]")

# Address terms that carry dialect, not identity — never treated as name variants even when
# they show up in you_call_them with high counts.
VOCATIVE_STOPLIST = {
    "بنت", "ولد", "زول", "زوله", "زولة", "عسل", "حبي", "حبيبي", "حبيبتي", "حبيبة",
    "حيوانه", "حيوانة", "يا", "استاذ", "أستاذ", "دكتور", "دكتورة", "باش", "مهندس",
    "اخوي", "أخوي", "اختي", "أختي", "صديق", "صديقي", "جميل", "جميلة", "امي", "أمي",
    "ابوي", "أبوي", "خالة", "خالتو", "عمو", "حبوبة", "مزة", "شباب", "جماعة", "ناس",
}


def _name_tokens(raw: str):
    """(full phrases, all variants): variants are what gets replaced, phrases what may merge.

    Splitting matters for replacement — "هاجر" alone must still be rewritten — but individual
    tokens must never *merge* identities: half of Sudan shares "Mohamed", and token-level
    merging fused unrelated people on first attempt.
    """
    if not raw:
        return [], []
    # "(new)"-style annotations in notion_name are bookkeeping, not part of the name
    phrase = re.sub(r"\s*\([^)]*\)\s*$", "", raw.strip()).strip()
    is_junk_phrase = (
        len(phrase) < 4
        # multi-word all-lowercase ascii phrases are prose fragments from free-text fields
        # ("remove entirly"), not names — single lowercase words ("samar", "7lmi") are real
        # nicknames and stay
        or (" " in phrase and phrase.isascii() and phrase == phrase.lower()
            and not any(c.isdigit() for c in phrase))
    )
    phrases = [] if is_junk_phrase else [phrase]
    tokens = list(phrases)
    for token in re.split(r"[\s|,&/]+", phrase):
        token = token.strip("()♥️😂🎉'’\"").strip()
        if len(token) < 3 or token.lower() in LATIN_STOPWORDS:
            continue
        # Latin single tokens must look like names (capitalized): the CSV fields carry prose
        # fragments ("not on Notion", "best friend"), and lowercase words like "not" and
        # "friend" ended up as name variants on first attempt
        if token.isascii() and not token[0].isupper():
            continue
        tokens.append(token)
    return phrases, tokens


def _arabic_names_from_calls(raw: str):
    """you_call_them looks like 'هاجر(24) | حبيبتي(3)' — keep the name-shaped Arabic tokens."""
    names = []
    for match in re.finditer(r"([؀-ۿ]{3,})\s*\(\d+\)", raw or ""):
        token = match.group(1)
        if token not in VOCATIVE_STOPLIST:
            names.append(token)
    return names


def _guess_gender(addressed_as: str) -> str:
    text = (addressed_as or "").lower()
    if "female" in text or "feminine" in text:
        return "female"
    if "male" in text or "masculine" in text:
        return "male"
    return "unknown"


def build_map() -> dict:
    """One entry per identity: real variants (both scripts) -> one stable fake name."""
    people = {}

    def _add(key, fields, arabic_calls="", gender="unknown"):
        entry = people.setdefault(key, {"variants": set(), "phrases": set(),
                                        "gender": "unknown"})
        for field in fields:
            phrases, tokens = _name_tokens(field)
            entry["phrases"].update(phrases)
            entry["variants"].update(tokens)
        entry["variants"].update(_arabic_names_from_calls(arabic_calls))
        if gender != "unknown":
            entry["gender"] = gender

    with open(PKB / "people_map.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = row["profile_name"] or row["chat_title"]
            if blocklist.is_blocked(key) or blocklist.is_blocked(row["chat_title"]):
                continue
            _add(key,
                 [row.get(f, "") for f in ("chat_title", "profile_name", "all_names",
                                           "notion_name")],
                 arabic_calls=row.get("you_call_them", ""),
                 gender=_guess_gender(row.get("addressed_as", "")))

    index = json.loads((PKB / "corpus_index.json").read_text())
    for key, info in index.items():
        if blocklist.is_blocked(key):
            continue
        _add(key, [key] + list(info.get("chats", [])))

    # the owner is an identity too: his name appears constantly inside message text
    people["Khalid (owner)"] = {
        "variants": {"Khalid", "khalid", "خالد", "Khaled", "Khalid Salman", "خالد سلمان"},
        "phrases": {"Khalid Salman", "خالد سلمان"},
        "gender": "male",
    }

    people = _merge_same_person(people)

    # hand-maintained variants (alias spellings that only occur inside profile prose) —
    # keyed by any known variant of the identity, merged after the automated pass
    extra_path = MAP_PATH.parent / "extra_variants.json"
    if extra_path.exists():
        extra = {k: v for k, v in json.loads(extra_path.read_text()).items()
                 if not k.startswith("_")}
        for alias, variants in extra.items():
            target = next((entry for entry in people.values()
                           if alias.lower() in {v.lower() for v in entry["variants"]}), None)
            if target is None:
                raise RuntimeError(f"extra_variants.json: no identity has variant {alias!r}")
            target["variants"].update(variants)

    rng = random.Random(SEED)
    pools = {"female": FAKE_FEMALE[:], "male": FAKE_MALE[:]}
    rng.shuffle(pools["female"])
    rng.shuffle(pools["male"])
    all_real = {v.lower() for p in people.values() for v in p["variants"]}

    result = {}
    for key in sorted(people):
        entry = people[key]
        pool = pools[entry["gender"] if entry["gender"] != "unknown" else "male"]
        fake = next(((ar, en) for ar, en in pool
                     if en.lower() not in all_real and ar not in all_real), None)
        if fake is not None:
            pool.remove(fake)
        else:
            # pool dry: compound a fresh name from a first name + a surname. The first name
            # must still avoid real variants — a fake containing a real name would be
            # re-substituted by apply() and flagged by scan().
            firsts = [f for f in (FAKE_FEMALE if entry["gender"] == "female" else FAKE_MALE)
                      if f[1].lower() not in all_real and f[0] not in all_real]
            first = rng.choice(firsts)
            surname = rng.choice(FAKE_SURNAMES)
            fake = (f"{first[0]} {surname[0]}", f"{first[1]} {surname[1]}")
        result[key] = {
            "fake_ar": fake[0],
            "fake_en": fake[1],
            "gender": entry["gender"],
            "real_variants": sorted(entry["variants"], key=len, reverse=True),
        }
    return result


def _merge_same_person(people: dict) -> dict:
    """Union entries that share a *full phrase*: people_map keys by profile_name while
    corpus_index keys by person, so the same human often arrives twice under the same chat
    title. Single tokens never merge — half of Sudan shares "Mohamed"."""
    keys = list(people)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_phrase = {}
    for key in keys:
        for phrase in people[key].get("phrases", ()):
            token = phrase.lower()
            if token in by_phrase:
                parent[find(key)] = find(by_phrase[token])
            else:
                by_phrase[token] = key

    merged = {}
    for key in keys:
        root = find(key)
        entry = merged.setdefault(root, {"variants": set(), "gender": "unknown"})
        entry["variants"].update(people[key]["variants"])
        if people[key]["gender"] != "unknown":
            entry["gender"] = people[key]["gender"]
    return merged


def load_map() -> dict:
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"{MAP_PATH} missing — run python -m src.synthesis.pseudonyms")
    return json.loads(MAP_PATH.read_text())


def _compile_patterns(mapping):
    """[(compiled regex, replacement)], longest variants first so full names win."""
    patterns = []
    variants = []
    for entry in mapping.values():
        for variant in entry["real_variants"]:
            variants.append((variant, entry))
    variants.sort(key=lambda pair: len(pair[0]), reverse=True)
    for variant, entry in variants:
        is_arabic = bool(ARABIC_RE.search(variant))
        fake = entry["fake_ar"] if is_arabic else entry["fake_en"]
        if is_arabic:
            # Arabic conjunctions attach to the word (وخالد، لأبرار), so a plain
            # word-boundary lookbehind misses the most common way a name appears mid-sentence.
            # The prefix is captured and kept; only the name is swapped.
            pattern = re.compile(rf"(?<![؀-ۿ])((?:[وفبلك]|ال)?){re.escape(variant)}(?![؀-ۿ])")
            replacement = rf"\g<1>{fake}"
        else:
            pattern = re.compile(rf"(?<![A-Za-z]){re.escape(variant)}(?![A-Za-z])",
                                 re.IGNORECASE)
            replacement = fake
        patterns.append((pattern, replacement))
    return patterns


class Pseudonymizer:
    def __init__(self, mapping=None):
        self.mapping = mapping or load_map()
        self.patterns = _compile_patterns(self.mapping)

    def apply(self, text: str) -> str:
        # ISO dates match the phone pattern; shelter them, mask phones, put them back —
        # dates are context the cards need, phone numbers are identity they must lose
        dates = []

        def _shelter(match):
            dates.append(match.group(0))
            return f"\x00{len(dates)-1}\x00"

        text = ISO_DATE_RE.sub(_shelter, text or "")
        text = PHONE_RE.sub("<phone>", text)
        text = EMAIL_RE.sub("<email>", text)
        text = HANDLE_RE.sub("<handle>", text)
        text = re.sub(r"\x00(\d+)\x00", lambda m: dates[int(m.group(1))], text)
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text

    def scan(self, text: str):
        """Real variants still present — the QC check for anything that slipped through."""
        found = []
        for pattern, _ in self.patterns:
            match = pattern.search(text or "")
            if match:
                found.append(match.group(0))
        return found


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    mapping = build_map()
    blocklist.assert_clean(mapping.keys(), "pseudonym map")
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    n_variants = sum(len(e["real_variants"]) for e in mapping.values())
    print(f"{len(mapping)} identities, {n_variants} name variants -> {MAP_PATH}")
    print("review it: automated extraction misses rare nicknames — add them by hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
