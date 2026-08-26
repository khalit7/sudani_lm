"""Sample real conversation excerpts as seeds for chat synthesis (plan.md Part IV, step 2.2).

Frontier models author Sudanese poorly cold but transform/extend real seeds well (AL-QASIDA),
so every synthesis request carries: a real 6–10-turn excerpt from the actual pair being
synthesized, a topic drawn from that pair's measured topic distribution, and the pair's persona
cards. This module supplies the first two.

Sources are the PKB corpora (43 profiled people, K/X speaker-tagged). Two hard exclusions,
enforced here because this is the leakage front door:

  - blocklisted people (DECISIONS.md off-record rule) never yield a seed,
  - chats listed in data/interim/whatsapp/val_chats.json never yield a seed — the WhatsApp
    holdout must stay untouched by synthesis or the chat_holdout metric dies.

Excerpts are pseudonymized before leaving this module: callers only ever see clean text.

Usage:  python -m src.synthesis.seed_sampler --person megat --n 3   # eyeball some seeds
"""

import argparse
import json
import random
import re
from pathlib import Path

from src.synthesis import blocklist
from src.synthesis.pseudonyms import get_pseudonymizer

REPO_ROOT = Path(__file__).resolve().parents[2]
PKB = Path.home() / "personal_knowledge_base"
VAL_CHATS_PATH = REPO_ROOT / "data" / "interim" / "whatsapp" / "val_chats.json"

SEED = 67
MIN_TURNS, MAX_TURNS = 6, 10

BLOCK_RE = re.compile(r"^--- (\d{4}-\d{2}-\d{2})(?: · in group: (.+?))? ---$")
ATTACHMENT_RE = re.compile(r"\[(image|video|audio|sticker|document|gif|contact)[^\]]*\]")


def parse_corpus(path: Path):
    """[{date, group, turns: [(speaker, text)]}] — speaker is 'K', 'X', or a group member."""
    segments = []
    current = None
    for line in path.read_text().splitlines():
        match = BLOCK_RE.match(line)
        if match:
            if current and current["turns"]:
                segments.append(current)
            current = {"date": match.group(1), "group": match.group(2), "turns": []}
            continue
        if current is None or not line.strip() or line.startswith("#"):
            continue
        if line.startswith("K "):
            speaker, text = "K", line[2:]
        elif line.startswith("X "):
            speaker, text = "X", line[2:]
        else:
            # group lines carry the member's full name; first token run before the message.
            # Names are 1-3 capitalized-ish tokens; fall back to treating the line as a
            # continuation of the previous turn when nothing name-shaped leads it.
            match = re.match(r"^([\w'’.-]+(?: [\w'’.-]+){0,2}?) (.+)$", line)
            if match and current["turns"] and not line[0].islower():
                speaker, text = match.group(1), match.group(2)
            elif current["turns"]:
                speaker, text = current["turns"][-1][0], line
            else:
                continue
        text = ATTACHMENT_RE.sub("", text).replace("(re) ", "").strip()
        if text:
            current["turns"].append((speaker, text))
    if current and current["turns"]:
        segments.append(current)
    return segments


def _person_index():
    index = json.loads((PKB / "corpus_index.json").read_text())
    val_chats = set(json.loads(VAL_CHATS_PATH.read_text()))
    people = {}
    for person, info in index.items():
        slug = Path(info["file"]).stem
        if blocklist.is_blocked(slug) or blocklist.is_blocked(person):
            continue
        if any(chat in val_chats for chat in info.get("chats", [])):
            continue        # this person's 1:1 thread is (partly) the eval holdout
        people[slug] = dict(info, person=person)
    return people


class SeedSampler:
    def __init__(self, seed: int = SEED):
        self.people = _person_index()
        self.pseudo = get_pseudonymizer()
        self.evidence = json.loads((PKB / "evidence.json").read_text())
        self.rng = random.Random(seed)
        self._segments = {}

    def slugs(self):
        return sorted(self.people)

    def _load(self, slug):
        if slug not in self._segments:
            segments = parse_corpus(PKB / "corpora" / self.people[slug]["file"])
            # 1:1 segments only, long enough to window, both sides actually speaking
            self._segments[slug] = [
                s for s in segments
                if s["group"] is None and len(s["turns"]) >= MIN_TURNS
                and {sp for sp, _ in s["turns"]} >= {"K", "X"}
            ]
        return self._segments[slug]

    def topics(self, slug, n=8):
        """The pair's measured topic distribution, most frequent first."""
        counts = {}
        for chat in self.people[slug].get("chats", []):
            for topic, count in self.evidence.get(chat, {}).get("context", {}).items():
                counts[topic] = counts.get(topic, 0) + count
        return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]

    def sample(self, slug):
        """One seed: {date, turns: [(speaker, text)], topic} or None."""
        segments = self._load(slug)
        if not segments:
            return None
        segment = self.rng.choice(segments)
        turns = segment["turns"]
        length = self.rng.randint(MIN_TURNS, min(MAX_TURNS, len(turns)))
        start = self.rng.randint(0, len(turns) - length)
        window = turns[start : start + length]
        blocklist.assert_clean([slug], "seed sampling")
        topics = self.topics(slug)
        return {
            "person": slug,
            "date": segment["date"],
            "topic": self.rng.choice(topics) if topics else None,
            "turns": [(speaker, self.pseudo.apply(text)) for speaker, text in window],
        }

    def display_name(self, slug):
        """The person's curated name from corpus_index — chat titles can be junk
        ("Ammmmmaaaaaar", a shared couple chat's title), person keys are clean."""
        return self.people[slug].get("person") or slug

    # ---- non-K chats: mention-graph pairs and real group segments -------------------------

    def pair_candidates(self):
        """[(slug_a, slug_b, weight)]: pairs weighted by how often A's chat mentions B.

        evidence.json's mention keys are short display names; they are matched to people by
        first-token overlap with the person key or chat titles. Fuzzy, but a wrong pairing is
        harmless — worst case two acquaintances get a synthetic conversation.
        """
        by_token = {}
        for slug, info in self.people.items():
            for name in [Path(info["file"]).stem, *info.get("chats", [])]:
                token = name.split()[0].lower().split("-")[0]
                if len(token) >= 3:
                    by_token.setdefault(token, slug)
        weights = {}
        for slug, info in self.people.items():
            for chat in info.get("chats", []):
                for mention, count in self.evidence.get(chat, {}).get("mentions", {}).items():
                    other = by_token.get(mention.split()[0].lower())
                    if other and other != slug:
                        pair = tuple(sorted((slug, other)))
                        weights[pair] = weights.get(pair, 0) + count
        return [(a, b, w) for (a, b), w in sorted(weights.items(), key=lambda kv: -kv[1])]

    def sample_pair_excerpts(self, slug_a, slug_b):
        """Style anchors for a non-K pair: each person's own real excerpt, plus merged topics.

        No real A↔B transcript exists (the corpora only hold K's chats), so each side anchors
        on their own authentic voice and the generator supplies the interaction.
        """
        excerpts = {}
        for slug in (slug_a, slug_b):
            seed = self.sample(slug)
            if seed is None:
                return None
            name = self.display_name(slug)
            excerpts[slug] = "\n".join(
                f"{'K' if sp == 'K' else name}: {text}" for sp, text in seed["turns"])
        topics = list(dict.fromkeys(self.topics(slug_a) + self.topics(slug_b)))
        return {
            "excerpt_a": excerpts[slug_a],
            "excerpt_b": excerpts[slug_b],
            "topic": self.rng.choice(topics) if topics else None,
        }

    def _load_group_segments(self):
        """Real group-chat segments with ≥3 speakers, ≥2 of whom have persona cards."""
        if hasattr(self, "_groups"):
            return self._groups
        display_to_slug = {self.display_name(s): s for s in self.people}
        display_to_slug.update({Path(info["file"]).stem: slug
                                for slug, info in self.people.items()})
        groups = []
        for slug, info in self.people.items():
            host_name = self.display_name(slug)
            for segment in parse_corpus(PKB / "corpora" / info["file"]):
                if segment["group"] is None or len(segment["turns"]) < MIN_TURNS:
                    continue
                # in group segments the corpus writes the host person as literal 'X' —
                # restore their name or the generator learns a speaker called "X"
                segment["turns"] = [(host_name if sp == "X" else sp, text)
                                    for sp, text in segment["turns"]]
                speakers = {sp for sp, _ in segment["turns"]}
                carded = {("K" if sp == "K" else sp) for sp in speakers
                          if sp == "K" or sp in display_to_slug}
                if len(speakers) >= 3 and len(carded - {"K"}) >= 2:
                    groups.append({"host": slug, "segment": segment,
                                   "carded": sorted(carded),
                                   "slugs": sorted({display_to_slug[sp]
                                                    for sp in carded if sp != "K"})})
        # a group thread appears once per member corpus; dedup by group name + date
        seen, unique = set(), []
        for group in groups:
            key = (group["segment"]["group"], group["segment"]["date"],
                   len(group["segment"]["turns"]))
            if key not in seen:
                seen.add(key)
                unique.append(group)
        self._groups = unique
        return unique

    def sample_group(self):
        """One real group excerpt: {group, date, turns, speakers, slugs} or None."""
        groups = self._load_group_segments()
        if not groups:
            return None
        group = self.rng.choice(groups)
        turns = group["segment"]["turns"]
        length = self.rng.randint(MIN_TURNS, min(MAX_TURNS + 4, len(turns)))
        start = self.rng.randint(0, len(turns) - length)
        blocklist.assert_clean(group["slugs"], "group seed sampling")
        return {
            "group": group["segment"]["group"],
            "date": group["segment"]["date"],
            "speakers": group["carded"],
            "slugs": group["slugs"],
            "turns": [(sp, self.pseudo.apply(text))
                      for sp, text in turns[start : start + length]],
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", default=None)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    sampler = SeedSampler()
    slugs = [args.person] if args.person else sampler.slugs()[:3]
    for slug in slugs:
        for _ in range(args.n):
            seed = sampler.sample(slug)
            if seed is None:
                print(f"{slug}: no usable 1:1 segments")
                break
            print(f"--- {slug} · {seed['date']} · topic={seed['topic']}")
            for speaker, text in seed["turns"]:
                print(f"  {speaker}: {text[:90]}")
    print(f"\n{len(sampler.slugs())} people available for seeding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
