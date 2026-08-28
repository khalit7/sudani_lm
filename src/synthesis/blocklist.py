"""The off-record blocklist, enforced in code rather than by memory.

personal_knowledge_base/DECISIONS.md rules one strand (Elaf Osman) off the record. Every module
that touches PKB material imports these helpers; anything blocklisted must never appear in a
card, a seed, a pair, or even a negative example. `assert_clean` raises rather than warns —
a privacy rule that only prints is not a rule.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKLIST_PATH = REPO_ROOT / "data" / "interim" / "synthetic" / "blocklist.json"


def load() -> dict:
    if not BLOCKLIST_PATH.exists():
        raise FileNotFoundError(
            f"{BLOCKLIST_PATH} missing — the synthesis pipeline must not run without it")
    return json.loads(BLOCKLIST_PATH.read_text())


def blocked_keys() -> set:
    """Every identifier under which blocklisted material could be addressed."""
    blocklist = load()
    return {key.lower()
            for group in ("people", "slugs", "chats")
            for key in blocklist.get(group, [])}


def is_blocked(identifier: str) -> bool:
    return (identifier or "").lower() in blocked_keys()


def assert_clean(identifiers, where: str) -> None:
    """Raise if any identifier in the collection is blocklisted."""
    hits = [i for i in identifiers if is_blocked(i)]
    if hits:
        raise RuntimeError(f"blocklisted identity {hits} reached {where} — refusing to continue")
