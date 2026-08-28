"""Build the situation bank: two-sentence Sudanese situations with a causal hook.

The seed redesign (plan.md Part IV) replaces flat topic nouns with SODA-style *situations*:
"الكورة" underdetermines a conversation, "فريقو خسر امبارح وأخوهو ما بطل يرسل ستيكرات" determines
one. Roots come from the personas' MEASURED topic distributions (evidence.json) plus the
monologue topic list; an orthogonal attribute grid (TinyStories-style cross products) forces
spread; generation uses Verbalized Sampling — ask for k candidates WITH probabilities in one
call — which restores most of the diversity that direct prompting collapses
(arXiv 2510.01171: ~24% → ~67% of base-model diversity).

Built offline, once; generation-time sampling then only ever draws from the bank, so the
generator never free-chooses a topic (the mode-collapse regime).

Output: data/interim/synthetic/situations.jsonl  {id, situation, root, attrs}

Usage:  python -m src.synthesis.situations build [--target 2000] [--concurrency 8]
        python -m src.synthesis.situations stats
"""

import argparse
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PKB = Path.home() / "personal_knowledge_base"
BANK_PATH = REPO_ROOT / "data" / "interim" / "synthetic" / "situations.jsonl"

SEED = 67
K_PER_CALL = 8

ATTRS = {
    "time": ["الصباح بدري", "بعد الضهر", "بالليل", "متأخر بالليل"],
    "valence": ["فرح", "ضيق", "قلق", "حنين", "عادي يومي", "قهر ساخر"],
    "arc": ["الموضوع بنحل في الآخر", "الموضوع بفضل معلق", "الموضوع بتحول لحاجة تانية"],
    "media": ["فويس نوت", "صورة", "لينك", "فيديو", "بدون ميديا"],
}

VS_PROMPT = """أنت سوداني بتعرف تفاصيل الحياة اليومية في السودان وبره للمغتربين.

الموضوع الجذر: {root}
الجو العام: الزمن {time} · الإحساس {valence} · {arc} · فيها {media}

اقترح {k} مواقف مختلفة تماماً عن بعض، كل موقف جملتين بالعامية السودانية:
الجملة الأولى تحدد الوضع بتفاصيل ملموسة، والجملة التانية فيها "سبب" يخلي الناس تتونس أو
تتجادل حوله (خبر، مشكلة، مفارقة، طلب). لازم المواقف تغطي زوايا مختلفة من الموضوع الجذر —
ما تكرر نفس الفكرة بصياغة تانية.

أخرج JSON فقط، سطر لكل موقف:
{{"p": <احتمال أن موقف زي دا يحصل فعلاً، رقم بين 0 و 1>, "s": "<الموقف في جملتين>"}}"""


def _roots():
    """Measured topics across all personas (weighted by count) + the monologue topic list."""
    from src.synthesis.prompts import MONO_TOPICS
    evidence = json.loads((PKB / "evidence.json").read_text())
    counts = {}
    for entry in evidence.values():
        for topic, count in entry.get("context", {}).items():
            counts[topic] = counts.get(topic, 0) + count
    # evidence.json context keys include tokenizer noise ("cohere", "aveni") — keep only
    # Arabic-script roots plus a small allowlist of genuine Latin topics
    allowed_latin = {"work", "phd", "wedding", "gym", "football"}
    measured = [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:120]
                if re.search(r"[؀-ۿ]", t) or t.lower() in allowed_latin]
    return list(dict.fromkeys(measured + MONO_TOPICS))


def _one_call(root, attrs):
    prompt = VS_PROMPT.format(root=root, k=K_PER_CALL, **attrs)
    result = subprocess.run(["claude", "-p", "--model", "sonnet", "--output-format", "json"],
                            input=prompt, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return []
    try:
        text = json.loads(result.stdout).get("result", "")
    except json.JSONDecodeError:
        return []
    situations = []
    for line in text.splitlines():
        match = re.search(r'\{.*"s"\s*:\s*"(.+?)"\s*\}', line)
        try:
            row = json.loads(line[line.index("{"):])
            if isinstance(row.get("s"), str) and len(row["s"]) > 30:
                situations.append(row["s"].strip())
        except (ValueError, json.JSONDecodeError):
            if match:
                situations.append(match.group(1).strip())
    return situations


def _ngrams(text, n=4):
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def build(target=2000, concurrency=8) -> int:
    rng = random.Random(SEED)
    roots = _roots()
    print(f"{len(roots)} root topics")

    calls = []
    call_index = 0
    while len(calls) * K_PER_CALL < target * 1.6:      # headroom for dedup + parse losses
        root = roots[call_index % len(roots)]
        calls.append((root, {key: rng.choice(values) for key, values in ATTRS.items()}))
        call_index += 1

    bank, shingle_sets = [], []
    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BANK_PATH, "w", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_one_call, root, attrs): (root, attrs)
                   for root, attrs in calls}
        for future in as_completed(futures):
            root, attrs = futures[future]
            for situation in future.result():
                grams = _ngrams(situation)
                if any(len(grams & other) / max(len(grams | other), 1) > 0.5
                       for other in shingle_sets):
                    continue
                shingle_sets.append(grams)
                entry = {"id": f"sit_{len(bank):05d}", "situation": situation,
                         "root": root, "attrs": attrs}
                bank.append(entry)
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fh.flush()
            if len(bank) % 200 < K_PER_CALL:
                print(f"  {len(bank)} situations banked", flush=True)
            if len(bank) >= target:
                break
    print(f"{len(bank)} situations -> {BANK_PATH}")
    return 0


def stats() -> int:
    rows = [json.loads(l) for l in BANK_PATH.read_text().splitlines() if l.strip()]
    from collections import Counter
    roots = Counter(r["root"] for r in rows)
    print(f"{len(rows)} situations across {len(roots)} roots; top:",
          roots.most_common(5))
    for row in random.Random(0).sample(rows, min(5, len(rows))):
        print(f"  [{row['root']}] {row['situation'][:110]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--target", type=int, default=2000)
    builder.add_argument("--concurrency", type=int, default=8)
    sub.add_parser("stats")
    args = parser.parse_args()
    return build(args.target, args.concurrency) if args.cmd == "build" else stats()


if __name__ == "__main__":
    sys.exit(main())
