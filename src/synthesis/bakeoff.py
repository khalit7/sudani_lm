"""Local-model bake-off: can any open model on the 2×5090 generate usable Sudanese?

Design per ollama_for_synth.md. Every model answers the SAME fixed prompt set (paired
comparison, seeded), built from the same real seed excerpts the Claude pipeline uses. Scoring
runs on two independent axes, because AMIYA showed systems trade dialect vocabulary against
coherence and a blended score hides that:

  (i)  dialect:   the project's char-ngram Sudanese-vs-MSA classifier (free, local), plus
                  structural QC filters (format, degeneration, regurgitation);
  (ii) register:  the same Haiku judge rubric used on the pilot, on structural survivors only.

The Claude pilot's Sonnet outputs are the reference ceiling — same seeds' distribution, same
judges — so "good enough" has a number attached.

Layout: data/interim/bakeoff/{prompts/<id>.md, raw/<model_key>/<id>.json, report.md}

Usage:  python -m src.synthesis.bakeoff plan [--requests 200]
        python -m src.synthesis.bakeoff run --model gemma3      # one model at a time (VRAM)
        python -m src.synthesis.bakeoff run-all
        python -m src.synthesis.bakeoff score                   # classifier + structural axes
        python -m src.synthesis.bakeoff judge [--limit N]       # Haiku axis, survivors only
        python -m src.synthesis.bakeoff report
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.synthesis import prompts
from src.synthesis.seed_sampler import SeedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
BAKE_DIR = REPO_ROOT / "data" / "interim" / "bakeoff"
PROMPTS_DIR = BAKE_DIR / "prompts"
RAW_DIR = BAKE_DIR / "raw"
CARDS_DIR = REPO_ROOT / "data" / "interim" / "synthetic" / "cards"

SEED = 67
OLLAMA_URL = "http://localhost:11434/api/chat"

# key -> (ollama tag, hypothesis being tested)
MODELS = {
    "gemma3": ("gemma3:27b", "control: strongest Arabic-per-parameter general family"),
    # the community GGUF ships a chat template llama.cpp cannot parse ("Unknown statement:
    # raw"); fanar2-fixed is the same weights with the template rewritten in the GGUF metadata
    "fanar2": ("fanar2-fixed", "does the best current Arabic CPT help over its own"
               " Gemma-3 base?"),
    # Jais-2-70B verdict: UNTESTABLE — the official Q4_K_M GGUF (35 downloads) degenerates
    # into token salad on any prompt beyond ~100 tokens, under both the auto-derived and the
    # correct ChatML template; broken conversion, not a model result. The 8B sibling GGUF
    # (585 downloads) stands in so the Arabic-native family still gets a measured data point.
    "jais2": ("jais2-8b", "Arabic-native family stand-in (70B GGUF broken); its 8B scored"
              " 0.208 dialect steering in ArabCulture — does seed-anchoring lift it?"),
    # the iKhalid/ALLaM community upload vanished; Q8_0 GGUF pulled from HF instead
    "allam": ("hf.co/Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q8_0-GGUF",
              "the only open model with measured dialect-steering edge"),
    "llama33": ("llama3.3:70b", "already on disk; best at mirroring dialect it is shown"),
    "qwen38": ("qwen3.8:latest", "owner's pick, already on disk; strong generalist —"
               " tests instruction-following + in-context imitation over Arabic priors"),
    "gemma4": ("gemma4:12b", "owner's pick, already on disk; newest Gemma generation —"
               " does generational lift beat the older 27B's extra capacity?"),
}

GEN_OPTIONS = {"temperature": 0.8, "top_p": 0.95, "num_ctx": 8192, "num_predict": 1600}


def plan(n_requests=200) -> int:
    """One fixed prompt set for every model: 50% k_chat, 50% monologue (the production mix).

    Deliberately no pair/group chats: those lean hardest on instruction-following, and the
    question here is dialect, not orchestration. A model that fails simple chats fails.
    """
    rng = random.Random(SEED + 7)       # fresh stream: don't collide with pilot seed draws
    sampler = SeedSampler(seed=SEED + 7)
    cards = {p.stem: p.read_text() for p in CARDS_DIR.glob("*.md")}
    slugs = [s for s in sampler.slugs() if s in cards]

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i in range(n_requests):
        if i % 2 == 0:
            slug = slugs[(i // 2) % len(slugs)]
            seed = sampler.sample(slug)
            if seed is None:
                continue
            partner = sampler.display_name(slug)
            seed_text = "\n".join(
                f"{'K' if sp == 'K' else partner}: {text}" for sp, text in seed["turns"])
            prompt = prompts.chat_prompt(
                owner_card=cards["owner"], partner_name=partner, partner_card=cards[slug],
                seed_text=seed_text, date=seed["date"], topic=seed["topic"],
                n_turns=rng.randint(30, 50))
            meta = {"kind": "chat", "slug": slug, "partner": partner,
                    "speakers": ["K", partner], "topic": seed["topic"]}
        else:
            genre = rng.choice(prompts.MONOLOGUE_GENRES)
            topic = rng.choice(["الغربة", "الكهرباء والمويه", "امتحانات الجامعة", "العرس",
                                "رمضان في السودان", "الاسعار", "الاهل", "الشغل", "الكورة",
                                "المطر والخريف", "الجيران", "المواصلات"])
            from src.synthesis.generate import _monologue_seed
            prompt = prompts.monologue_prompt(_monologue_seed(rng), genre, topic,
                                              n_words=rng.randint(700, 1200))
            meta = {"kind": "monologue", "genre": genre, "topic": topic}

        request_id = f"bake_{i:04d}"
        (PROMPTS_DIR / f"{request_id}.md").write_text(prompt)
        manifest.append({"id": request_id, "meta": meta})
    (BAKE_DIR / "manifest.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest))
    kinds = [m["meta"]["kind"] for m in manifest]
    print(f"{len(manifest)} prompts ({kinds.count('chat')} chat / "
          f"{kinds.count('monologue')} monologue) -> {PROMPTS_DIR}")
    return 0


def _ollama_generate(tag, prompt, timeout=600, options=None):
    # think: False — reasoning models otherwise spend the whole token budget inside a
    # <think> block and deliver nothing; models without the capability reject the flag,
    # so fall back to a plain request for those.
    for think_flag in (False, None):
        body = {"model": tag, "stream": False, "options": options or GEN_OPTIONS,
                "messages": [{"role": "user", "content": prompt}]}
        if think_flag is not None:
            body["think"] = think_flag
        request = urllib.request.Request(
            OLLAMA_URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            return (payload.get("message", {}).get("content") or "",
                    payload.get("eval_count"), payload.get("eval_duration"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:200]
            if "think" in detail and think_flag is False:
                continue                          # model predates the think flag; retry plain
            raise
    raise RuntimeError("unreachable")


def run(model_key, limit=None) -> int:
    tag, hypothesis = MODELS[model_key]
    manifest = [json.loads(l) for l in (BAKE_DIR / "manifest.jsonl").read_text().splitlines()
                if l.strip()]
    if limit:
        manifest = manifest[:limit]
    out_dir = RAW_DIR / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{model_key} ({tag}): {hypothesis}")
    done = failed = 0
    start = time.time()
    for entry in manifest:
        out = out_dir / f"{entry['id']}.json"
        if out.exists():
            continue
        prompt = (PROMPTS_DIR / f"{entry['id']}.md").read_text()
        try:
            text, eval_count, eval_ns = _ollama_generate(tag, prompt)
        except Exception as error:                                    # noqa: BLE001
            failed += 1
            print(f"  {entry['id']}: {type(error).__name__} {error}", flush=True)
            if failed > 10 and done == 0:
                print("aborting: model not producing"); return 1
            continue
        # strip <think> reasoning blocks some models emit before the answer
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        out.write_text(json.dumps(
            {"id": entry["id"], "model": model_key, "meta": entry["meta"], "text": text,
             "tok_s": round(eval_count / (eval_ns / 1e9), 1) if eval_count and eval_ns else None},
            ensure_ascii=False))
        done += 1
        if done % 20 == 0:
            rate = done / max(time.time() - start, 1) * 3600
            print(f"  {done} done ({rate:.0f}/h)", flush=True)
    print(f"{model_key}: {done} done, {failed} failed, {(time.time()-start)/60:.1f} min")
    return 0


def run_all() -> int:
    for key in MODELS:
        if run(key) != 0:
            print(f"{key} aborted, continuing with next model")
    return 0


def _local_tags():
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=30) as response:
        names = {m["name"] for m in json.loads(response.read()).get("models", [])}
    return names | {name.removesuffix(":latest") for name in names}


def run_available() -> int:
    """Run every roster model that is already local and not yet complete.

    Exit 0 when the whole roster is finished, 2 while some models are still downloading —
    the chain script loops on that, so ready models generate while the rest download.
    """
    manifest_n = len([l for l in (BAKE_DIR / "manifest.jsonl").read_text().splitlines()
                      if l.strip()])
    tags = _local_tags()
    waiting = []
    for key, (tag, _) in MODELS.items():
        out_dir = RAW_DIR / key
        done = len(list(out_dir.glob("bake_*.json"))) if out_dir.exists() else 0
        if done >= manifest_n:
            continue
        if tag in tags:
            run(key)
        else:
            waiting.append(key)
    if waiting:
        print(f"still downloading: {waiting}")
        return 2
    return 0


PREAMBLE_RE = re.compile(
    r"^(here'?s?|sure|certainly|below is|بالتأكيد|إليك|هذه|هاك|دي)\b.*$|^.*(whatsapp|"
    r"conversation|محادثة|مونولوج|monologue).*[:：]\s*$",
    re.IGNORECASE)


def strip_preamble(text: str) -> str:
    """Drop up to 3 leading task-acknowledgment lines ("Here's a new WhatsApp chat: ...").

    Applied to EVERY model identically: the bake-off measures dialect ability, and a chatty
    first line is prompt-compliance noise that is mechanical to normalize in production.
    Conservative: only obvious acknowledgment shapes, only at the top.
    """
    lines = text.strip().splitlines()
    dropped = 0
    while lines and dropped < 3 and PREAMBLE_RE.match(lines[0].strip()):
        lines.pop(0)
        dropped += 1
    return "\n".join(lines).strip()


def score() -> int:
    """Axis 1: structural QC + dialect-classifier probability, per model."""
    import joblib

    from src.synthesis.qc import degenerate, parse_chat, _ngrams
    clf = joblib.load(REPO_ROOT / "data" / "interim" / "dialect_clf.joblib")

    results = {}
    for model_dir in sorted(RAW_DIR.iterdir()):
        rows = [json.loads(p.read_text()) for p in sorted(model_dir.glob("*.json"))
                if not p.name.startswith("_")]
        scored = []
        for row in rows:
            text = strip_preamble(row["text"])
            reason = None
            if row["meta"]["kind"] == "chat":
                payload = {"text": text, "meta": row["meta"]}
                if parse_chat(payload) is None:
                    reason = "format"
            if reason is None:
                reason = degenerate(text)
            if reason is None:
                seed = (PROMPTS_DIR / f"{row['id']}.md").read_text()
                if _ngrams(text) & _ngrams(seed):
                    reason = "seed_regurgitation"
            scored.append({"id": row["id"], "kill": reason,
                           "dialect": float(clf.predict_proba([text[:3000]])[0][1])
                           if not reason else None})
        (model_dir / "_scores.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in scored))
        survivors = [s for s in scored if s["kill"] is None]
        results[model_dir.name] = (len(rows), len(survivors),
                                   sum(s["dialect"] for s in survivors) / max(len(survivors), 1))
    for model, (n, ok, dialect) in results.items():
        print(f"  {model:<10} {ok}/{n} structural pass ({ok/max(n,1):.0%}), "
              f"mean dialect prob {dialect:.3f}")
    return 0


def judge(limit=None, concurrency=8) -> int:
    """Axis 2: the pilot's Haiku rubric over structural survivors. Parallel per model."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from src.synthesis.qc import _judge_one

    for model_dir in sorted(RAW_DIR.iterdir()):
        scores_path = model_dir / "_scores.jsonl"
        if not scores_path.exists():
            continue
        kills = {json.loads(l)["id"]: json.loads(l)["kill"]
                 for l in scores_path.read_text().splitlines() if l.strip()}
        judged_path = model_dir / "_judged.jsonl"
        done = {json.loads(l)["id"] for l in judged_path.read_text().splitlines()
                if l.strip()} if judged_path.exists() else set()
        rows = [json.loads(p.read_text()) for p in sorted(model_dir.glob("*.json"))
                if not p.name.startswith("_")]
        pending = [r for r in rows if kills.get(r["id"]) is None and r["id"] not in done]
        if limit:
            pending = pending[:limit]
        print(f"{model_dir.name}: judging {len(pending)}", flush=True)
        with open(judged_path, "a", encoding="utf-8") as fh, \
                ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_judge_one,
                                   {"id": row["id"],
                                    "qc_text": strip_preamble(row["text"])}, "haiku")
                       for row in pending]
            for future in as_completed(futures):
                fh.write(json.dumps(future.result()) + "\n")
                fh.flush()
    return 0


JAIS_ARM_DIR = BAKE_DIR / "jais_arm"


def jais_arm(n_requests=50) -> int:
    """Mini bake-off arm: jais2-8b on chats with the strict-format prompt and 15-25 turns.

    Owner-requested follow-up: 95/98 of its chats died on format alone (content mostly fine
    on manual read), so this measures whether a hard output contract + shorter chats recover
    it. Prints structural pass, dialect prob, and judge results end-to-end.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import joblib

    from src.synthesis.qc import _judge_one, degenerate, parse_chat
    from src.synthesis.seed_sampler import SeedSampler

    rng = random.Random(SEED + 33)
    sampler = SeedSampler(seed=SEED + 33)
    cards = {p.stem: p.read_text() for p in CARDS_DIR.glob("*.md")}
    slugs = [s for s in sampler.slugs() if s in cards]
    JAIS_ARM_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    clf = joblib.load(REPO_ROOT / "data" / "interim" / "dialect_clf.joblib")
    for i in range(n_requests):
        out = JAIS_ARM_DIR / f"arm_{i:03d}.json"
        if out.exists():
            results.append(json.loads(out.read_text()))
            continue
        slug = slugs[i % len(slugs)]
        seed = sampler.sample(slug)
        if seed is None:
            continue
        partner = sampler.display_name(slug)
        seed_text = "\n".join(
            f"{'K' if sp == 'K' else partner}: {text}" for sp, text in seed["turns"])
        prompt = prompts.jais_chat_prompt(
            owner_card=cards["owner"], partner_name=partner, partner_card=cards[slug],
            seed_text=seed_text, date=seed["date"], topic=seed["topic"],
            n_turns=rng.randint(15, 25))
        text, _, _ = _ollama_generate("jais2-8b", prompt)
        text = strip_preamble(text)
        payload = {"id": f"arm_{i:03d}", "kind": "chat", "model": "jais2-8b",
                   "meta": {"slug": slug, "partner": partner,
                            "speakers": ["K", partner], "topic": seed["topic"]},
                   "text": text}
        out.write_text(json.dumps(payload, ensure_ascii=False))
        results.append(payload)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_requests} generated", flush=True)

    survivors = []
    kills = {}
    for payload in results:
        reason = "format" if parse_chat(payload) is None else degenerate(payload["text"])
        if reason is None:
            survivors.append(payload)
        else:
            kills[reason] = kills.get(reason, 0) + 1
    print(f"\nstructural: {len(survivors)}/{len(results)} pass "
          f"({len(survivors)/max(len(results),1):.0%}); kills {kills}")
    if survivors:
        probs = clf.predict_proba([p["text"][:3000] for p in survivors])[:, 1]
        print(f"mean dialect prob: {probs.mean():.3f}")
        scores = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(_judge_one, {"id": p["id"], "qc_text": p["text"]}, "haiku")
                       for p in survivors]
            for future in as_completed(futures):
                scores.append(future.result()["score"])
        ok = sum(1 for s in scores if s and s >= 4)
        print(f"judge >=4: {ok}/{len(scores)} ({ok/max(len(scores),1):.0%} of survivors)"
              f" -> end-to-end {ok}/{len(results)} ({ok/max(len(results),1):.0%})")
        print("reference: v1-prompt run was 0/98 chats end-to-end;"
              " gemma3 41%, Claude ~82%")
    return 0


def report() -> int:
    lines = ["# Bake-off results\n",
             "| model | structural pass | mean dialect prob | judge ≥4 | median tok/s |",
             "|---|---|---|---|---|"]
    for model_dir in sorted(RAW_DIR.iterdir()):
        scores = [json.loads(l) for l in (model_dir / "_scores.jsonl").read_text().splitlines()
                  if l.strip()] if (model_dir / "_scores.jsonl").exists() else []
        judged = {json.loads(l)["id"]: json.loads(l)["score"]
                  for l in (model_dir / "_judged.jsonl").read_text().splitlines()
                  if l.strip()} if (model_dir / "_judged.jsonl").exists() else {}
        rows = [json.loads(p.read_text()) for p in sorted(model_dir.glob("*.json"))
                if not p.name.startswith("_")]
        survivors = [s for s in scores if s["kill"] is None]
        dialect = sum(s["dialect"] for s in survivors) / max(len(survivors), 1)
        judged_scores = [judged[s["id"]] for s in survivors
                         if judged.get(s["id"]) is not None]
        ok4 = sum(1 for s in judged_scores if s >= 4)
        speeds = sorted(r["tok_s"] for r in rows if r.get("tok_s"))
        lines.append(f"| {model_dir.name} | {len(survivors)}/{len(scores)}"
                     f" ({len(survivors)/max(len(scores),1):.0%})"
                     f" | {dialect:.3f} | {ok4}/{len(judged_scores)}"
                     f" ({ok4/max(len(judged_scores),1):.0%})"
                     f" | {speeds[len(speeds)//2] if speeds else '—'} |")
    lines.append("\nReference ceiling (Claude pilot, same harness): structural 85%, "
                 "judge ≥4 on 97.5%, chats 99% / monologues 100%.")
    report_path = BAKE_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    planner = sub.add_parser("plan")
    planner.add_argument("--requests", type=int, default=200)
    runner = sub.add_parser("run")
    runner.add_argument("--model", required=True, choices=list(MODELS))
    runner.add_argument("--limit", type=int, default=None)
    sub.add_parser("run-all")
    sub.add_parser("run-available")
    sub.add_parser("score")
    judger = sub.add_parser("judge")
    judger.add_argument("--limit", type=int, default=None)
    arm = sub.add_parser("jais-arm")
    arm.add_argument("--requests", type=int, default=50)
    sub.add_parser("report")
    args = parser.parse_args()

    if args.cmd == "plan":
        return plan(args.requests)
    if args.cmd == "run":
        return run(args.model, args.limit)
    if args.cmd == "run-all":
        return run_all()
    if args.cmd == "run-available":
        return run_available()
    if args.cmd == "score":
        return score()
    if args.cmd == "judge":
        return judge(args.limit)
    if args.cmd == "jais-arm":
        return jais_arm(args.requests)
    return report()


if __name__ == "__main__":
    sys.exit(main())
