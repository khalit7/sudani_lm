"""Subscription-based generation driver: value-ranked queue, resumable, runs unattended.

Generation runs on the owner's Claude subscription via headless `claude -p`, not the paid API.
Usage-limit windows are the only throttle, which forces the design rule: at any stopping point
the requests completed so far must be the most valuable possible set. Hence a *planned queue*,
ordered before anything runs — persona cards first (everything depends on them), then chat
requests round-robin over all people (breadth before depth), with monologue and MSA→Sudanese
transforms interleaved at their target shares.

Layout under data/interim/synthetic/:
    queue.jsonl        one request per line: {id, kind, model, prompt_path, meta}
    requests/<id>.md   the exact prompt (pseudonymized upstream — reviewable)
    raw/<id>.json      claude's output; existence marks the request done (resume = re-run)
    cards/<slug>.md    finished persona cards (written by `collect` from card requests)

Subcommands:
    plan-cards                       queue the 42+1 card distillations
    plan-pilot --requests N          queue the pilot mix (needs cards/ populated)
    run [--concurrency 3] [--limit N]   process pending requests via `claude -p`
    collect                          parse raw card outputs into cards/
    status                           queue/raw counts

The pilot splits arms across models (sonnet vs opus) per plan.md — QC pass-rate per model
decides what the 50M scale-up uses.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.synthesis import blocklist, prompts
from src.synthesis.pseudonyms import Pseudonymizer
from src.synthesis.seed_sampler import SeedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN_DIR = REPO_ROOT / "data" / "interim" / "synthetic"
QUEUE_PATH = SYN_DIR / "queue.jsonl"
REQUESTS_DIR = SYN_DIR / "requests"
RAW_DIR = SYN_DIR / "raw"
CARDS_DIR = SYN_DIR / "cards"
CARD_INPUTS_DIR = SYN_DIR / "card_inputs"

SEED = 67
# pilot mix by request count: chats dominate because multi-turn dialogue is the register that
# publicly does not exist; transforms are capped as the weakest register signal
KIND_SHARES = {"chat": 0.7, "monologue": 0.2, "transform": 0.1}
CLAUDE_TIMEOUT = 600
USAGE_LIMIT_SLEEP = 900          # 15 min: wait out a usage-limit window, don't hammer it


def _read_queue():
    if not QUEUE_PATH.exists():
        return []
    return [json.loads(line) for line in QUEUE_PATH.read_text().splitlines() if line.strip()]


def _append_queue(entries):
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _write_request(request_id, prompt):
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REQUESTS_DIR / f"{request_id}.md"
    path.write_text(prompt)
    return str(path.relative_to(REPO_ROOT))


def plan_cards() -> int:
    """Card distillations run first: every chat request depends on the cards existing."""
    existing = {entry["id"] for entry in _read_queue()}
    entries = []
    for path in sorted(CARD_INPUTS_DIR.glob("*.md")):
        request_id = f"card_{path.stem}"
        if request_id in existing:
            continue
        blocklist.assert_clean([path.stem], "card queue")
        entries.append({
            "id": request_id, "kind": "card", "model": "opus",   # cards are one-off: best model
            "prompt_path": _write_request(request_id, path.read_text()),
            "meta": {"slug": path.stem},
        })
    _append_queue(entries)
    print(f"queued {len(entries)} card requests")
    return 0


def _load_cards():
    cards = {p.stem: p.read_text() for p in CARDS_DIR.glob("*.md")}
    if "owner" not in cards:
        raise FileNotFoundError("cards/owner.md missing — run plan-cards + run + collect first")
    return cards


def _flores_exemplars(rng, n=8):
    rows = [json.loads(line)["translation"]
            for line in (REPO_ROOT / "data/raw/sudanese_flores/DEV.jsonl").read_text()
            .splitlines() if line.strip()]
    picked = rng.sample(rows, n)
    return "\n".join(f"MSA: {r['Arb']}\nSudanese: {r['Sud']}\n" for r in picked)


def _monologue_seed(rng):
    """A real discourse-register snippet: oddadmix transcript docs are the default pool."""
    path = REPO_ROOT / "data/interim/oddadmix/train.jsonl"
    lines = path.read_text().splitlines()
    return json.loads(rng.choice(lines))["text"][:1200]


def _msa_sources(rng, n):
    """Short, prose-like ArabicWeb24 documents to transform."""
    from datasets import Dataset

    from src.preprocessing.pack import arabicweb24_shards, keep_document
    ds = Dataset.from_file(arabicweb24_shards()[rng.randrange(50)])
    picked = []
    for i in rng.sample(range(len(ds)), min(4000, len(ds))):
        text, metadata = ds[i]["text"], ds[i]["metadata"]
        if keep_document(text, metadata) and 60 <= len(text.split()) <= 160:
            picked.append(text)
            if len(picked) >= n:
                break
    return picked


def plan_pilot(n_requests, models=("sonnet", "opus")) -> int:
    """Queue the pilot mix. Chats round-robin every person so truncation still covers all."""
    rng = random.Random(SEED)
    cards = _load_cards()
    sampler = SeedSampler()
    pseudo = Pseudonymizer()
    existing = {entry["id"] for entry in _read_queue()}

    slugs = [s for s in sampler.slugs() if s in cards]
    print(f"{len(slugs)} people with cards ready")
    counts = {kind: int(n_requests * share) for kind, share in KIND_SHARES.items()}
    entries = []

    msa_pool = _msa_sources(rng, counts["transform"] + 8)
    fake_owner = pseudo.mapping["Khalid (owner)"]["fake_en"]

    for i in range(counts["chat"]):
        slug = slugs[i % len(slugs)]
        request_id = f"chat_{slug}_{i:05d}"
        if request_id in existing:
            continue
        seed = sampler.sample(slug)
        if seed is None:
            continue
        partner = pseudo.mapping.get(slug, {}).get("fake_en") or slug
        seed_text = "\n".join(
            f"{'K' if sp == 'K' else partner}: {text}" for sp, text in seed["turns"])
        prompt = prompts.chat_prompt(
            owner_card=cards["owner"], partner_name=partner, partner_card=cards[slug],
            seed_text=seed_text, date=seed["date"], topic=seed["topic"],
            n_turns=rng.randint(15, 25))
        entries.append({
            "id": request_id, "kind": "chat", "model": models[i % len(models)],
            "prompt_path": _write_request(request_id, prompt),
            "meta": {"slug": slug, "partner": partner, "owner_alias": fake_owner,
                     "topic": seed["topic"], "prompt_version": prompts.PROMPT_VERSION},
        })

    for i in range(counts["monologue"]):
        request_id = f"mono_{i:05d}"
        if request_id in existing:
            continue
        genre = rng.choice(prompts.MONOLOGUE_GENRES)
        topic = rng.choice(["الغربة", "الكهرباء والمويه", "امتحانات الجامعة", "العرس",
                            "رمضان في السودان", "الاسعار", "الاهل", "الشغل", "الكورة",
                            "المطر والخريف", "الجيران", "المواصلات"])
        prompt = prompts.monologue_prompt(_monologue_seed(rng), genre, topic,
                                          n_words=rng.randint(300, 600))
        entries.append({
            "id": request_id, "kind": "monologue", "model": models[i % len(models)],
            "prompt_path": _write_request(request_id, prompt),
            "meta": {"genre": genre, "topic": topic,
                     "prompt_version": prompts.PROMPT_VERSION},
        })

    exemplars = _flores_exemplars(rng)
    for i, source in enumerate(msa_pool[: counts["transform"]]):
        request_id = f"trans_{i:05d}"
        if request_id in existing:
            continue
        prompt = prompts.transform_prompt(exemplars, source)
        entries.append({
            "id": request_id, "kind": "transform", "model": models[i % len(models)],
            "prompt_path": _write_request(request_id, prompt),
            "meta": {"prompt_version": prompts.PROMPT_VERSION},
        })

    # interleave kinds at their target shares: the queue is consumed in order and may be cut
    # short by usage windows, so any prefix must already be a balanced mix
    by_kind = {}
    for entry in entries:
        by_kind.setdefault(entry["kind"], []).append(entry)
    interleaved, cursors = [], {kind: 0 for kind in by_kind}
    while any(cursors[kind] < len(items) for kind, items in by_kind.items()):
        for kind, items in by_kind.items():
            target = KIND_SHARES.get(kind, 0.1)
            quota = max(1, round(target * 10))
            for _ in range(quota):
                if cursors[kind] < len(items):
                    interleaved.append(items[cursors[kind]])
                    cursors[kind] += 1
    _append_queue(interleaved)
    print(f"queued {len(interleaved)}: { {k: len(v) for k, v in by_kind.items()} }")
    return 0


def _run_one(entry):
    if (RAW_DIR / f"{entry['id']}.json").exists():
        return entry, "ok", None            # finished by an earlier pass — never re-spend
    prompt = (REPO_ROOT / entry["prompt_path"]).read_text()
    result = subprocess.run(
        ["claude", "-p", "--model", entry["model"], "--output-format", "json"],
        input=prompt, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").lower()
        kind = "usage_limit" if ("limit" in stderr or "overloaded" in stderr) else "error"
        return entry, kind, (result.stderr or result.stdout)[:500]
    try:
        payload = json.loads(result.stdout)
        text = payload.get("result", "")
    except json.JSONDecodeError:
        return entry, "error", "unparseable claude output"
    if not text.strip():
        return entry, "error", "empty result"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{entry['id']}.json").write_text(json.dumps(
        {"id": entry["id"], "kind": entry["kind"], "model": entry["model"],
         "meta": entry.get("meta", {}), "text": text}, ensure_ascii=False))
    return entry, "ok", None


def run(concurrency=3, limit=None) -> int:
    pending = [entry for entry in _read_queue()
               if not (RAW_DIR / f"{entry['id']}.json").exists()]
    # cards jump the queue: chats can't even be planned without them
    pending.sort(key=lambda entry: entry["kind"] != "card")
    if limit:
        pending = pending[:limit]
    print(f"{len(pending)} pending requests")
    done = failed = 0
    start = time.time()
    index = 0
    while index < len(pending):
        batch = pending[index : index + concurrency]
        hit_limit = False
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for future in as_completed([pool.submit(_run_one, entry) for entry in batch]):
                entry, status, detail = future.result()
                if status == "ok":
                    done += 1
                elif status == "usage_limit":
                    hit_limit = True
                else:
                    failed += 1
                    print(f"  {entry['id']}: {detail}", flush=True)
        if hit_limit:
            print(f"usage limit hit — sleeping {USAGE_LIMIT_SLEEP//60} min "
                  f"({done} done so far)", flush=True)
            time.sleep(USAGE_LIMIT_SLEEP)
            continue                         # retry the same slice (_run_one skips done ones)
        # never mutate `pending` mid-loop: an earlier version refiltered it here while the
        # index kept advancing, which silently skipped half the queue
        index += concurrency
        batch_done = done + failed
        if batch_done and batch_done % 30 < concurrency:
            rate = done / max(time.time() - start, 1) * 3600
            print(f"  {done} done, {failed} failed, {rate:.0f}/h", flush=True)
    print(f"run finished: {done} done, {failed} failed in {(time.time()-start)/60:.1f} min")
    return 0


def collect_cards() -> int:
    """Card outputs -> cards/<slug>.md, with a real-name scan before anything is written."""
    pseudo = Pseudonymizer()
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for path in sorted(RAW_DIR.glob("card_*.json")):
        payload = json.loads(path.read_text())
        slug = payload["meta"]["slug"]
        text = payload["text"].strip()
        leftovers = pseudo.scan(text)
        if leftovers:
            print(f"  {slug}: REJECTED — real names in card: {leftovers}")
            continue
        (CARDS_DIR / f"{slug}.md").write_text(text)
        written += 1
    print(f"{written} cards -> {CARDS_DIR}")
    return 0


def status() -> int:
    queue = _read_queue()
    raw = {p.stem for p in RAW_DIR.glob("*.json")} if RAW_DIR.exists() else set()
    by_kind = {}
    for entry in queue:
        state = "done" if entry["id"] in raw else "pending"
        by_kind.setdefault(entry["kind"], {"done": 0, "pending": 0})[state] += 1
    for kind, counts in sorted(by_kind.items()):
        print(f"  {kind:<10} done {counts['done']:>6}  pending {counts['pending']:>6}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan-cards")
    pilot = sub.add_parser("plan-pilot")
    pilot.add_argument("--requests", type=int, default=200)
    runner = sub.add_parser("run")
    runner.add_argument("--concurrency", type=int, default=3)
    runner.add_argument("--limit", type=int, default=None)
    sub.add_parser("collect")
    sub.add_parser("status")
    args = parser.parse_args()

    if args.cmd == "plan-cards":
        return plan_cards()
    if args.cmd == "plan-pilot":
        return plan_pilot(args.requests)
    if args.cmd == "run":
        return run(args.concurrency, args.limit)
    if args.cmd == "collect":
        return collect_cards()
    return status()


if __name__ == "__main__":
    sys.exit(main())
