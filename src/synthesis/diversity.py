"""Per-generator diversity instrumentation — the mode-collapse alarm.

The measurement literature's non-redundant minimal set (Shaib et al., arXiv 2403.00553),
implemented with what's already in the environment:

  - compression ratio (gzip): cheap, captures nearly the same signal as slow n-gram
    homogenization scores; higher = more diverse
  - self-similarity: mean max pairwise 4-gram Jaccard over a sample (Self-BLEU proxy);
    lower = more diverse
  - embedding nearest-neighbour distance: char-ngram hashing vectors + cosine (Magpie's
    operational dedup metric); higher = more diverse

Each generator's numbers are compared against a REAL-corpus reference row (WhatsApp
conversations for chat, podcast+forum for monologue) — synthetic text is expected to sit
below real text on diversity; the alarm is a generator drifting further below over time or
far below its peers.

Usage:  python -m src.synthesis.diversity report [--sample 300]
"""

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYN_DIR = REPO_ROOT / "data" / "interim" / "synthetic"
RAW_DIR = SYN_DIR / "raw"
REPORT_PATH = SYN_DIR / "diversity_report.json"

SEED = 67


def _ngrams(text, n=4):
    words = text.split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def compression_ratio(texts):
    blob = "\n".join(texts).encode()
    return len(gzip.compress(blob)) / max(len(blob), 1)


def self_similarity(texts, rng, sample=150):
    picked = rng.sample(texts, min(sample, len(texts)))
    grams = [_ngrams(t) for t in picked]
    scores = []
    for i, g in enumerate(grams):
        best = 0.0
        for j, other in enumerate(grams):
            if i == j or not g or not other:
                continue
            best = max(best, len(g & other) / len(g | other))
        scores.append(best)
    return sum(scores) / max(len(scores), 1)


def nn_distance(texts, rng, sample=300):
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    picked = rng.sample(texts, min(sample, len(texts)))
    vectors = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=2**18,
                                alternate_sign=False).transform(picked)
    sims = cosine_similarity(vectors)
    import numpy as np
    np.fill_diagonal(sims, -1)
    return float((1 - sims.max(axis=1)).mean())


def _metrics(texts, rng):
    if len(texts) < 5:
        return None
    return {"n": len(texts),
            "compression_ratio": round(compression_ratio(texts), 4),
            "self_similarity": round(self_similarity(texts, rng), 4),
            "nn_distance": round(nn_distance(texts, rng), 4)}


def _references(rng):
    refs = {}
    chat_texts = []
    # handle iteration, not splitlines(): chat text contains U+2028-style separators
    with open(REPO_ROOT / "data/interim/whatsapp/train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chat_texts.append(json.loads(line)["text"])
    refs["REAL_chat"] = _metrics(rng.sample(chat_texts, min(1000, len(chat_texts))), rng)
    mono_texts = []
    for path in ("data/interim/oddadmix/train.jsonl", "data/interim/sudaneseonline/train.jsonl"):
        for line in (REPO_ROOT / path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("dialect", 1) >= 0.8:
                    mono_texts.append(row["text"][:4000])
    refs["REAL_discourse"] = _metrics(rng.sample(mono_texts, min(1000, len(mono_texts))), rng)
    return refs


def report(sample=300) -> int:
    rng = random.Random(SEED)
    by_generator = defaultdict(list)
    for path in RAW_DIR.glob("*.json"):
        row = json.loads(path.read_text())
        if row.get("kind") == "card":
            continue
        by_generator[(row.get("model", "?"), row.get("kind", "?"))].append(row["text"])

    results = {"references": _references(rng)}
    print(f"{'generator':<22} {'kind':<10} {'n':>6} {'comp.ratio↑':>12} "
          f"{'self-sim↓':>10} {'nn-dist↑':>9}")
    for name, metrics in results["references"].items():
        if metrics:
            print(f"{name:<22} {'—':<10} {metrics['n']:>6} "
                  f"{metrics['compression_ratio']:>12} {metrics['self_similarity']:>10} "
                  f"{metrics['nn_distance']:>9}")
    for (model, kind), texts in sorted(by_generator.items()):
        metrics = _metrics(texts, rng)
        if metrics is None:
            continue
        results[f"{model}/{kind}"] = metrics
        print(f"{model:<22} {kind:<10} {metrics['n']:>6} "
              f"{metrics['compression_ratio']:>12} {metrics['self_similarity']:>10} "
              f"{metrics['nn_distance']:>9}")
    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print(f"-> {REPORT_PATH}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    reporter = sub.add_parser("report")
    reporter.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()
    return report(args.sample)


if __name__ == "__main__":
    sys.exit(main())
