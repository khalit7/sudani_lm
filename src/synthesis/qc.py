"""QC gate between raw generations and training data (plan.md Part IV, step 2.5).

Unfiltered synthesis teaches the model the generator's failure modes: MSA-tinted Arabic,
repetition, mode-collapsed phrasing, and — worst — regurgitated real text or real names.
Every filter logs its kill count; a filter that never fires is as informative as one that
fires constantly.

Order matters and is cheapest-first:
  1. format validity (chat parses into turns, expected speakers, sane turn count)
  2. degeneration (repeated n-grams, line repeats, compression ratio)
  3. real-name scan (pseudonym map — nothing real may survive into training data)
  4. leakage (8-gram overlap vs WhatsApp val chats and Flores DEVTEST → hard reject;
     8-gram overlap vs the request's own seed → regurgitation reject)
  5. near-duplicate dedup across the kept pool (shingle Jaccard)
  6. dialect judge (separate subcommand: claude -p with a rubric, keep score ≥ 4)

Outputs (only after filters):
  chats_me.jsonl       chat template rendered with the owner's turns labelled ME
  chats_pseudo.jsonl   same conversations, owner under his pseudonym (the ablation's other arm)
  monologue.jsonl      {"source": "syn_monologue", "text": ...}
  transformed.jsonl    {"source": "syn_msa2sud", "text": ...}

Usage:  python -m src.synthesis.qc filter
        python -m src.synthesis.qc judge [--model haiku] [--limit N]
        python -m src.synthesis.qc render [--min-score 4]
"""

import argparse
import json
import re
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

from src.synthesis.pseudonyms import get_pseudonymizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN_DIR = REPO_ROOT / "data" / "interim" / "synthetic"
RAW_DIR = SYN_DIR / "raw"
FILTERED_PATH = SYN_DIR / "filtered.jsonl"
JUDGED_PATH = SYN_DIR / "judged.jsonl"

VAL_CHATS = REPO_ROOT / "data" / "interim" / "whatsapp" / "val.jsonl"
FLORES_DEVTEST = REPO_ROOT / "data" / "raw" / "sudanese_flores" / "DEVTEST.jsonl"

# v3 asks for 30-50 turns, but bursty texture means one "turn" is often several lines and
# good conversations run to ~120 lines; true runaway loops are caught by degenerate(), so
# the ceiling only guards absurd outputs.
MIN_CHAT_TURNS, MAX_CHAT_TURNS = 8, 120
ARABIC_RE = re.compile(r"[؀-ۿ]")

JUDGE_PROMPT = """أنت خبير في اللهجة السودانية. قيّم النص التالي من 1 إلى 5:
5 = لهجة سودانية أصيلة تماما، زي ما يكتبها سوداني في الواتساب
4 = سودانية واضحة مع هفوات بسيطة
3 = عربية عامية لكن مش سودانية بالتحديد (مصرية/شامية/خليجية أو خليط)
2 = فصحى متنكرة بكلمات عامية
1 = فصحى أو نص ركيك

رد برقم واحد فقط، بدون أي كلام تاني.

النص:
{text}"""


def _ngrams(text, n=8):
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _load_leakage_ngrams():
    """8-gram sets of everything that must never be echoed by synthetic data."""
    grams = set()
    for line in VAL_CHATS.read_text().splitlines():
        if line.strip():
            grams |= _ngrams(json.loads(line)["text"])
    if FLORES_DEVTEST.exists():
        for line in FLORES_DEVTEST.read_text().splitlines():
            if line.strip():
                pair = json.loads(line)["translation"]
                grams |= _ngrams(pair.get("Sud", ""))
                grams |= _ngrams(pair.get("Arb", ""))
    return grams


def parse_chat(payload):
    """NAME: text lines -> [(speaker, text)] or None if malformed."""
    turns = []
    for line in payload["text"].strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^([^:：]{1,40})[:：]\s*(.+)$", line)
        if not match:
            return None
        turns.append((match.group(1).strip(), match.group(2).strip()))
    if not (MIN_CHAT_TURNS <= len(turns) <= MAX_CHAT_TURNS):
        return None
    # k_chats declare ["K", partner]; pair/group chats declare their own speaker rosters.
    # Matching is lenient where it is unambiguous — models naturally shorten "Safaa Ismaeel"
    # to "Safaa" and sometimes change case ("NAJLA") — and every accepted variant is
    # canonicalized back to the declared label so training data stays consistent.
    declared = payload["meta"].get("speakers") or ["K", payload["meta"].get("partner", "")]
    canon = {name.lower(): name for name in declared}
    firsts = {}
    for name in declared:
        first = name.split()[0].lower()
        firsts[first] = None if first in firsts else name      # None = ambiguous prefix
    normalized = []
    for speaker, text in turns:
        key = speaker.lower()
        match = canon.get(key) or (firsts.get(key) if firsts.get(key) else None)
        if match is None:
            return None
        normalized.append((match, text))
    if len({speaker for speaker, _ in normalized}) < 2:
        return None
    return normalized


def degenerate(text):
    words = text.split()
    if len(words) < 20:
        return "too_short"
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if trigrams:
        counts = Counter(trigrams)
        if sum(c - 1 for c in counts.values() if c > 1) / len(trigrams) > 0.3:
            return "repeated_ngrams"
    if max(Counter(words).values()) > max(6, len(words) * 0.15):
        return "repeated_word"
    if len(zlib.compress(text.encode())) / max(len(text.encode()), 1) < 0.22:
        return "low_entropy"
    arabic = len(ARABIC_RE.findall(text))
    if arabic < len(text) * 0.25:
        return "not_arabic_enough"
    return None


KILLS_PATH = SYN_DIR / "kills.jsonl"


def filter_cmd() -> int:
    """Incremental: documents already filtered (kept or killed) are never re-processed.

    filtered.jsonl holds the kept payloads, kills.jsonl the rejected ids with reasons —
    together they are the seen-set, so the on-demand QC suite only pays for new raw outputs.
    """
    pseudo = get_pseudonymizer()
    leak_grams = _load_leakage_ngrams()
    kills = Counter()

    kept = []
    if FILTERED_PATH.exists():
        kept = [json.loads(l) for l in FILTERED_PATH.read_text().splitlines() if l.strip()]
    seen = {p["id"] for p in kept}
    if KILLS_PATH.exists():
        seen |= {json.loads(l)["id"] for l in KILLS_PATH.read_text().splitlines()
                 if l.strip()}
    # the near-duplicate screen compares new docs against the whole kept pool
    kept_shingles = [_ngrams(p["qc_text"], n=4) for p in kept]
    n_before = len(kept)

    new_kills = []
    for path in sorted(RAW_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload["kind"] == "card" or payload["id"] in seen:
            continue
        reason = None
        shingles = None
        text = payload["text"].strip()
        if payload["kind"] == "chat":
            turns = parse_chat(payload)
            if turns is None:
                reason = "format"
            else:
                text = "\n".join(f"{speaker}: {t}" for speaker, t in turns)
                payload["turns"] = turns
        if reason is None:
            reason = degenerate(text)
        if reason is None and pseudo.scan(text):
            reason = "real_name"
        if reason is None:
            grams = _ngrams(text)
            if grams & leak_grams:
                reason = "leakage"
            else:
                seed_path = SYN_DIR / "requests" / f"{payload['id']}.md"
                if seed_path.exists() and grams & _ngrams(seed_path.read_text()):
                    reason = "seed_regurgitation"
        if reason is None:
            shingles = _ngrams(text, n=4)
            for other in kept_shingles:
                union = len(shingles | other)
                if union and len(shingles & other) / union > 0.6:
                    reason = "near_duplicate"
                    break
        if reason is not None or shingles is None:
            kills[reason or "format"] += 1
            new_kills.append({"id": payload["id"], "kill": reason or "format"})
            continue
        assert shingles is not None
        kept_shingles.append(shingles)
        payload["qc_text"] = text
        kept.append(payload)

    # append-only persistence: this is the seen-set for future incremental runs
    with open(FILTERED_PATH, "a", encoding="utf-8") as fh:
        for payload in kept[n_before:]:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    with open(KILLS_PATH, "a", encoding="utf-8") as fh:
        for row in new_kills:
            fh.write(json.dumps(row) + "\n")
    new_total = (len(kept) - n_before) + len(new_kills)
    print(f"new this run: kept {len(kept)-n_before}/{new_total}"
          f" ({(len(kept)-n_before)/max(new_total,1):.0%}); pool total {len(kept)} kept")
    for reason, count in kills.most_common():
        print(f"  killed {count:>5}  {reason}")
    return 0


def _judge_one(row, model):
    result = subprocess.run(
        ["claude", "-p", "--model", model, "--output-format", "json"],
        input=JUDGE_PROMPT.format(text=row["qc_text"][:6000]),
        capture_output=True, text=True, timeout=240)
    score = None
    if result.returncode == 0:
        try:
            match = re.search(r"[1-5]", json.loads(result.stdout).get("result", ""))
            score = int(match.group(0)) if match else None
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"id": row["id"], "score": score}


def judge_cmd(model="haiku", limit=None, concurrency=8) -> int:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    rows = [json.loads(line) for line in FILTERED_PATH.read_text().splitlines() if line.strip()]
    scored_ids = set()
    if JUDGED_PATH.exists():
        scored_ids = {json.loads(line)["id"] for line in JUDGED_PATH.read_text().splitlines()
                      if line.strip()}
    pending = [row for row in rows if row["id"] not in scored_ids]
    if limit:
        pending = pending[:limit]
    print(f"{len(pending)} to judge")
    done = 0
    with open(JUDGED_PATH, "a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_judge_one, row, model) for row in pending]
        for future in as_completed(futures):
            fh.write(json.dumps(future.result()) + "\n")
            fh.flush()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(pending)}", flush=True)
    return 0


def render_cmd(min_score=4) -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from src.tokenizer.special_tokens import render_conversation

    scores = {}
    if JUDGED_PATH.exists():
        for line in JUDGED_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                scores[row["id"]] = row["score"]

    outputs = {name: open(SYN_DIR / name, "w", encoding="utf-8")
               for name in ("chats_me.jsonl", "chats_pseudo.jsonl", "monologue.jsonl",
                            "transformed.jsonl")}
    counts = Counter()
    for line in FILTERED_PATH.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        score = scores.get(payload["id"])
        if score is None or score < min_score:
            counts["below_score" if score is not None else "unjudged"] += 1
            continue
        # provenance travels into the training files: generator + full seed config
        provenance = {"generator": payload.get("model"), "seed": payload.get("seed")}
        if payload["kind"] == "chat":
            alias = payload["meta"].get("owner_alias", "Khalid")
            # only the owner's label differs between the arms; pair/group chats without K
            # come out identical in both, which is correct — the arms differ on the ME
            # identity, not on third-party dialogue
            me = [("ME" if speaker == "K" else speaker, text)
                  for speaker, text in payload["turns"]]
            pseudo = [(alias if speaker == "K" else speaker, text)
                      for speaker, text in payload["turns"]]
            outputs["chats_me.jsonl"].write(json.dumps(
                {"chat": f"syn:{payload['meta']['slug']}",
                 "text": render_conversation(me), **provenance},
                ensure_ascii=False) + "\n")
            outputs["chats_pseudo.jsonl"].write(json.dumps(
                {"chat": f"syn:{payload['meta']['slug']}",
                 "text": render_conversation(pseudo), **provenance},
                ensure_ascii=False) + "\n")
            counts["chat"] += 1
        elif payload["kind"] == "monologue":
            outputs["monologue.jsonl"].write(json.dumps(
                {"source": "syn_monologue", "text": payload["qc_text"], **provenance},
                ensure_ascii=False) + "\n")
            counts["monologue"] += 1
        else:
            outputs["transformed.jsonl"].write(json.dumps(
                {"source": "syn_msa2sud", "text": payload["qc_text"], **provenance},
                ensure_ascii=False) + "\n")
            counts["transform"] += 1
    for handle in outputs.values():
        handle.close()
    print(dict(counts))
    return 0


def suite(model="haiku", min_score=4) -> int:
    """The on-demand judge suite: incremental filter → judge (cached by id) → full render."""
    filter_cmd()
    judge_cmd(model=model)
    render_cmd(min_score=min_score)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("all", help="filter -> judge -> render, incrementally (the judge suite)")
    sub.add_parser("filter")
    judge = sub.add_parser("judge")
    judge.add_argument("--model", default="haiku")
    judge.add_argument("--limit", type=int, default=None)
    render = sub.add_parser("render")
    render.add_argument("--min-score", type=int, default=4)
    args = parser.parse_args()

    if args.cmd == "all":
        return suite()
    if args.cmd == "filter":
        return filter_cmd()
    if args.cmd == "judge":
        return judge_cmd(args.model, args.limit)
    return render_cmd(args.min_score)


if __name__ == "__main__":
    sys.exit(main())
