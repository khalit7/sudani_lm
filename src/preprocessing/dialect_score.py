"""Sudanese-dialect scorer: char-ngram logistic classifier, Sudanese vs MSA (plan.md Part IV).

Built for the forum corpus problem: sudaneseonline.com yields far more text than stage C can
absorb, and its register runs from pure dialect storytelling to pasted MSA news columns. Rather
than a keep/drop threshold, the mixture takes the top-N tokens by this score — ranking needs
calibration only at the margin, which is much less demanding than the ≥0.9-precision gate the
FineWeb2 mining step will require (that gate still applies before any mined text enters a
manifest; this ranking use is deliberately weaker).

Training data is what the project already trusts: the public Sudanese stream + podcast
transcripts as positives, ArabicWeb24 + the Flores MSA side as negatives. Char 2-5-grams
capture orthographic dialect signal (زول، شنو، دايرة، بتاع، هسة) without tokenizer coupling.

Usage:  python -m src.preprocessing.dialect_score train
        python -m src.preprocessing.dialect_score score data/interim/sudaneseonline/train.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_INTERIM = REPO_ROOT / "data" / "interim"
MODEL_PATH = DATA_INTERIM / "dialect_clf.joblib"

SEED = 67
N_NEGATIVES = 25_000


def _positives():
    texts = []
    for path in (DATA_INTERIM / "sudani" / "all.jsonl",
                 DATA_INTERIM / "oddadmix" / "train.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                texts.append(json.loads(line)["text"])
    flores = REPO_ROOT / "data" / "raw" / "sudanese_flores" / "DEV.jsonl"
    for line in flores.read_text().splitlines():
        if line.strip():
            texts.append(json.loads(line)["translation"]["Sud"])
    return texts


def _negatives(rng):
    from datasets import Dataset

    from src.preprocessing.pack import arabicweb24_shards, keep_document
    texts = []
    flores = REPO_ROOT / "data" / "raw" / "sudanese_flores" / "DEV.jsonl"
    for line in flores.read_text().splitlines():
        if line.strip():
            texts.append(json.loads(line)["translation"]["Arb"])
    ds = Dataset.from_file(arabicweb24_shards()[rng.randrange(100)])
    for i in rng.sample(range(len(ds)), min(60_000, len(ds))):
        text = ds[i]["text"]
        if keep_document(text, ds[i]["metadata"]):
            texts.append(text[:1500])
            if len(texts) >= N_NEGATIVES:
                break
    return texts


def _pipeline():
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    return make_pipeline(
        HashingVectorizer(analyzer="char_wb", ngram_range=(2, 5), n_features=2**20,
                          alternate_sign=False),
        LogisticRegression(max_iter=1000, C=1.0),
    )


def train() -> int:
    import joblib
    from sklearn.model_selection import train_test_split

    rng = random.Random(SEED)
    positives, negatives = _positives(), _negatives(rng)
    texts = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.1, random_state=SEED, stratify=labels)

    model = _pipeline()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    joblib.dump(model, MODEL_PATH)
    print(f"{len(positives):,} pos / {len(negatives):,} neg -> holdout acc {accuracy:.3f}"
          f" -> {MODEL_PATH}")
    return 0


def score_file(path, chunk_rows: int = 20_000) -> int:
    """Adds a `dialect` probability to every row, rewriting the file atomically.

    Streams in chunks: the wave-3 corpora (alnilin_posts 1.7GB) OOM-killed the previous
    load-everything version, and an in-place rewrite killed mid-write truncates the file —
    hence chunked scoring into a temp file swapped in at the end.
    """
    import joblib
    import numpy as np

    model = joblib.load(MODEL_PATH)
    path = Path(path)
    tmp_path = path.with_suffix(".jsonl.scoring")
    all_probs, n_rows, chunk = [], 0, []

    def score_chunk(fh):
        nonlocal chunk
        # score on a bounded prefix: a 100KB thread's tail adds nothing to its register call
        probs = model.predict_proba([row["text"][:3000] for row in chunk])[:, 1]
        for row, prob in zip(chunk, probs):
            row["dialect"] = round(float(prob), 4)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_probs.extend(probs.tolist())
        chunk = []

    with open(tmp_path, "w", encoding="utf-8") as out_fh, open(path, encoding="utf-8") as in_fh:
        for line in in_fh:
            if not line.strip():
                continue
            chunk.append(json.loads(line))
            n_rows += 1
            if len(chunk) >= chunk_rows:
                score_chunk(out_fh)
        if chunk:
            score_chunk(out_fh)
    tmp_path.replace(path)
    print(f"{path}: {n_rows:,} rows scored, dialect quartiles "
          f"{np.percentile(all_probs, [25, 50, 75]).round(3).tolist()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train")
    scorer = sub.add_parser("score")
    scorer.add_argument("path")
    args = parser.parse_args()
    return train() if args.cmd == "train" else score_file(args.path)


if __name__ == "__main__":
    sys.exit(main())
