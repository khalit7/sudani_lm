"""The one-command synthetic data generator (owner-approved design, 2026-08-27).

    python -m src.synthesis.synth_data gemma3 1000 gemma4 1000 sonnet 1000

Each `model N` pair samples N fresh random seeds for that model's task (the registry below
maps model → task) and generates. Sampling is stateless by design: every invocation draws a
fresh entropy seed, seed configs are practically never repeated, so there is no done-tracking
— the run stamp in every id keeps invocations distinct.

Seed design (plan.md Part IV, seed redesign):
  chats      party count drawn from the EMPIRICAL distribution of the real corpus;
             participants from the mention graph / real group rosters; topic 70% from the
             pair's measured distribution, 30% from the situation bank; always a rotated real
             excerpt as style anchor, chosen independently of the topic (register without
             content bleed); ≥3-party chats carry a participation-imbalance instruction
             sampled from the real group's turn shares.
  monologues writer-persona (50%, incl. the owner) or plain-anchor voice; anchor from
             podcast/forum high-dialect pool; situation from the bank; genre × audience grid.

Every raw output records the full seed config and generator. When generation ends the QC
chain runs automatically: filter → Haiku judge → render, then the diversity report
(compression ratio / self-similarity / embedding NN distance per generator).

Add a model = one MODEL_REGISTRY entry.
"""

import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.synthesis import prompts
from src.synthesis.bakeoff import strip_preamble
from src.synthesis.seed_sampler import SeedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN_DIR = REPO_ROOT / "data" / "interim" / "synthetic"
RAW_DIR = SYN_DIR / "raw"
REQUESTS_DIR = SYN_DIR / "requests"
CARDS_DIR = SYN_DIR / "cards"
BANK_PATH = SYN_DIR / "situations.jsonl"
PARTY_DIST_PATH = SYN_DIR / "party_dist.json"

OLLAMA_OPTIONS = {"temperature": 0.8, "top_p": 0.95, "num_ctx": 8192, "num_predict": 2048}

MODEL_REGISTRY = {
    "sonnet": {"task": "chat", "backend": "claude", "model": "sonnet", "concurrency": 8},
    "opus": {"task": "chat", "backend": "claude", "model": "opus", "concurrency": 8},
    "haiku": {"task": "chat", "backend": "claude", "model": "haiku", "concurrency": 8},
    "gemma3": {"task": "monologue", "backend": "ollama", "model": "gemma3:27b",
               "concurrency": 2},
    "gemma4": {"task": "monologue", "backend": "ollama", "model": "gemma4:12b",
               "concurrency": 2},
    "jais2-8b": {"task": "chat", "backend": "ollama", "model": "jais2-8b", "concurrency": 2},
}


# --------------------------------------------------------------- empirical party counts ----

def party_count_distribution():
    """P(party count) measured from the real rendered conversations, capped at 5."""
    if PARTY_DIST_PATH.exists():
        return {int(k): v for k, v in json.loads(PARTY_DIST_PATH.read_text()).items()}
    speaker_re = re.compile(r"<\|turn\|>([^:]{1,40}):")
    counts = {}
    # iterate the handle, not splitlines(): chat text contains U+2028-style separators that
    # splitlines() breaks on mid-JSON-string
    with open(REPO_ROOT / "data/interim/whatsapp/train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            speakers = set(speaker_re.findall(json.loads(line)["text"]))
            n = min(max(len(speakers), 2), 5)
            counts[n] = counts.get(n, 0) + 1
    total = sum(counts.values())
    dist = {n: c / total for n, c in sorted(counts.items())}
    PARTY_DIST_PATH.write_text(json.dumps(dist))
    return dist


# --------------------------------------------------------------------- seed samplers -------

class ChatSeeds:
    def __init__(self, rng):
        self.rng = rng
        self.sampler = SeedSampler(seed=rng.randrange(2**31))
        self.cards = {p.stem: p.read_text() for p in CARDS_DIR.glob("*.md")}
        self.slugs = [s for s in self.sampler.slugs() if s in self.cards]
        self.pairs = [(a, b) for a, b, _ in self.sampler.pair_candidates()
                      if a in self.cards and b in self.cards]
        self.bank = [json.loads(l) for l in BANK_PATH.read_text().splitlines() if l.strip()]
        self.dist = party_count_distribution()
        self.groups = [g for g in self.sampler._load_group_segments()
                       if all(s in self.cards for s in g["slugs"])]

    def _topic(self, slugs):
        """70% measured pair topics / 30% situation bank; always returned as a situation-ish
        text plus provenance for the seed record."""
        if self.rng.random() < 0.7:
            topics = []
            for slug in slugs:
                topics.extend(self.sampler.topics(slug))
            if topics:
                return {"source": "measured", "text": self.rng.choice(topics)}
        entry = self.rng.choice(self.bank)
        return {"source": "situation_bank", "id": entry["id"], "text": entry["situation"],
                "attrs": entry.get("attrs")}

    def sample(self):
        n_parties = self.rng.choices(list(self.dist), weights=list(self.dist.values()))[0]
        if n_parties == 2:
            if self.rng.random() < 0.6 or not self.pairs:
                slug = self.rng.choice(self.slugs)
                slugs, names = [slug], ["K", self.sampler.display_name(slug)]
            else:
                a, b = self.rng.choice(self.pairs)
                slugs = [a, b]
                names = [self.sampler.display_name(a), self.sampler.display_name(b)]
            imbalance = ""
            excerpt_seed = self.sampler.sample(slugs[0])
            if excerpt_seed is None:
                return None
            anchor_name = self.sampler.display_name(slugs[0])
            excerpt = "\n".join(f"{'K' if sp == 'K' else anchor_name}: {t}"
                                for sp, t in excerpt_seed["turns"])
        else:
            candidates = [g for g in self.groups if len(g["carded"]) >= n_parties]
            if not candidates:
                return None
            group = self.rng.choice(candidates)
            names = list(self.rng.sample(group["carded"], n_parties))
            slugs = [s for s in group["slugs"]
                     if self.sampler.display_name(s) in names]
            segment = group["segment"]
            turn_counts = {}
            for sp, _ in segment["turns"]:
                turn_counts[sp] = turn_counts.get(sp, 0) + 1
            ranked = sorted((sp for sp in names if sp in turn_counts),
                            key=lambda sp: -turn_counts.get(sp, 0))
            imbalance = ("Group texture: mirror real participation imbalance — "
                         + (f"{ranked[0]} writes the most, {ranked[-1]} barely replies."
                            if len(ranked) >= 2 else "not everyone participates equally."))
            window = segment["turns"][:14]
            excerpt = "\n".join(f"{sp}: {self.sampler.pseudo.apply(t)}" for sp, t in window)

        topic = self._topic(slugs)
        cards_block = []
        if "K" in names:
            cards_block.append(f"=== OWNER CARD (K) ===\n{self.cards['owner']}")
        for slug in slugs:
            cards_block.append(f"=== PERSONA: {self.sampler.display_name(slug)} ===\n"
                               f"{self.cards[slug]}")
        return {
            "prompt": prompts.chat_v3_prompt(
                cards_block="\n\n".join(cards_block), roster=" و ".join(names),
                excerpt=excerpt, situation=topic["text"],
                n_turns=self.rng.randint(30, 50), imbalance=imbalance),
            "meta": {"kind": "chat", "speakers": names, "slug": "+".join(slugs)},
            "seed": {"party_count": n_parties, "participants": names, "topic": topic,
                     "prompt_version": prompts.PROMPT_VERSION},
        }


class MonoSeeds:
    def __init__(self, rng):
        self.rng = rng
        self.cards = {p.stem: p.read_text() for p in CARDS_DIR.glob("*.md")}
        self.bank = [json.loads(l) for l in BANK_PATH.read_text().splitlines() if l.strip()]
        self.anchors = []
        for line in (REPO_ROOT / "data/interim/oddadmix/train.jsonl").read_text().splitlines():
            if line.strip():
                self.anchors.append(("podcast", json.loads(line)["text"][:2000]))
        for line in (REPO_ROOT / "data/interim/sudaneseonline/train.jsonl") \
                .read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("dialect", 0) >= 0.8:
                    self.anchors.append(("forum", row["text"][:2000]))

    def sample(self):
        situation = self.rng.choice(self.bank)
        genre = self.rng.choice(prompts.MONOLOGUE_GENRES)
        audience = self.rng.choice(prompts.MONO_AUDIENCES)
        source, anchor = self.rng.choice(self.anchors)
        writer = None
        writer_block = ""
        if self.rng.random() < 0.5:
            writer = self.rng.choice(sorted(self.cards))
            title = "WRITER (this is the OWNER's own voice)" if writer == "owner" \
                else "WRITER"
            writer_block = f"=== {title} ===\n{self.cards[writer]}"
        return {
            "prompt": prompts.mono_v3_prompt(
                anchor=anchor, situation=situation["situation"], genre=genre,
                audience=audience, n_words=self.rng.randint(700, 1200),
                writer_block=writer_block),
            "meta": {"kind": "monologue", "genre": genre, "audience": audience},
            "seed": {"writer": writer, "situation_id": situation["id"],
                     "situation_root": situation["root"], "anchor_source": source,
                     "genre": genre, "audience": audience,
                     "prompt_version": prompts.PROMPT_VERSION},
        }


# ------------------------------------------------------------------------ backends ---------

def _claude_generate(model, prompt):
    result = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json"],
                            input=prompt, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").lower()
        if "limit" in stderr or "overloaded" in stderr:
            raise RuntimeError("usage_limit")
        raise RuntimeError((result.stderr or "claude error")[:200])
    payload = json.loads(result.stdout)
    usage = payload.get("usage", {})
    return payload.get("result", ""), {"cost_usd": payload.get("total_cost_usd"),
                                       "output_tokens": usage.get("output_tokens")}


def _ollama_generate(model, prompt):
    # bakeoff's variant sends think:false with a fallback — without it, thinking-capable
    # models (gemma4, qwen3.x) burn the whole budget inside <think> and return nothing
    from src.synthesis.bakeoff import _ollama_generate as bake_generate
    text, _, _ = bake_generate(model, prompt, timeout=900, options=OLLAMA_OPTIONS)
    return text, {}


USAGE_LIMIT_SLEEP = 900


def run_model(model_key, n_samples, run_stamp):
    spec = MODEL_REGISTRY[model_key]
    rng = random.Random()                       # entropy-seeded: stateless by design
    seeds = ChatSeeds(rng) if spec["task"] == "chat" else MonoSeeds(rng)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)

    pause_until = [0.0]
    pause_lock = threading.Lock()
    counters = {"ok": 0, "failed": 0}
    counter_lock = threading.Lock()
    start = time.time()

    def one(i):
        sample = None
        for _ in range(5):
            sample = seeds.sample()
            if sample is not None:
                break
        if sample is None:
            return "failed"
        request_id = f"sd_{run_stamp}_{model_key}_{i:05d}"
        (REQUESTS_DIR / f"{request_id}.md").write_text(sample["prompt"])
        while True:
            wait = pause_until[0] - time.time()
            if wait > 0:
                time.sleep(min(wait, 30))
                continue
            try:
                if spec["backend"] == "claude":
                    text, extra = _claude_generate(spec["model"], sample["prompt"])
                else:
                    text, extra = _ollama_generate(spec["model"], sample["prompt"])
            except RuntimeError as error:
                if str(error) == "usage_limit":
                    with pause_lock:
                        if time.time() >= pause_until[0]:
                            pause_until[0] = time.time() + USAGE_LIMIT_SLEEP
                            print(f"[{model_key}] usage limit — pausing 15 min", flush=True)
                    continue
                return "failed"
            except Exception:                                        # noqa: BLE001
                return "failed"
            break
        text = strip_preamble(text)
        if not text:
            return "failed"
        (RAW_DIR / f"{request_id}.json").write_text(json.dumps(
            {"id": request_id, "kind": spec["task"], "model": model_key,
             "meta": sample["meta"], "seed": sample["seed"], "text": text, **extra},
            ensure_ascii=False))
        return "ok"

    with ThreadPoolExecutor(max_workers=spec["concurrency"]) as pool:
        futures = [pool.submit(one, i) for i in range(n_samples)]
        for future in as_completed(futures):
            with counter_lock:
                counters[future.result() if future.result() in counters else "failed"] += 1
                done = counters["ok"] + counters["failed"]
                if done % 50 == 0:
                    rate = counters["ok"] / max(time.time() - start, 1) * 3600
                    print(f"[{model_key}] {counters['ok']} ok / {counters['failed']} failed"
                          f" ({rate:.0f}/h, ~{(n_samples-done)/max(rate,1):.1f}h left)",
                          flush=True)
    print(f"[{model_key}] finished: {counters} in {(time.time()-start)/60:.1f} min",
          flush=True)
    return counters


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="+",
                        help="pairs: MODEL N [MODEL N ...], e.g. gemma3 1000 sonnet 1000")
    parser.add_argument("--skip-qc", action="store_true",
                        help="generate only; skip the filter/judge/render/diversity chain")
    args = parser.parse_args()

    if len(args.spec) % 2 != 0:
        parser.error("spec must be MODEL N pairs")
    workload = []
    for model_key, n in zip(args.spec[::2], args.spec[1::2]):
        if model_key not in MODEL_REGISTRY:
            parser.error(f"unknown model {model_key!r} — registry: {list(MODEL_REGISTRY)}")
        workload.append((model_key, int(n)))

    run_stamp = time.strftime("%Y%m%d%H%M")
    print(f"run {run_stamp}: " + ", ".join(
        f"{m}→{MODEL_REGISTRY[m]['task']}×{n}" for m, n in workload))

    # all models run concurrently — claude and ollama backends don't contend
    with ThreadPoolExecutor(max_workers=len(workload)) as pool:
        futures = [pool.submit(run_model, m, n, run_stamp) for m, n in workload]
        for future in as_completed(futures):
            future.result()

    if args.skip_qc:
        return 0
    print("\n=== QC chain: filter → judge → render ===", flush=True)
    from src.synthesis import qc
    qc.filter_cmd()
    qc.judge_cmd(model="haiku")
    qc.render_cmd(min_score=4)
    print("\n=== diversity report ===", flush=True)
    from src.synthesis import diversity
    diversity.report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
