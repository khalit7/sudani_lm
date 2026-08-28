"""Fetch the public Sudanese sources for data expansion v2 (plan.md Part IV, step 1.1/1.2/1.3).

Everything lands under data/raw/, one folder per source, ready for its preprocessing module.
Each fetch is idempotent: a source whose output already exists is skipped, so the script can be
re-run after a network failure without re-downloading what landed.

The oddadmix collections are audio datasets (~14 GB of parquet), but the transcripts are a
plain string column — reading just that column over the hf:// filesystem costs a few MB of
HTTP range requests instead of the full download, so the audio bytes never touch the disk.

Usage:  python scripts/download_sudanese_sources.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = REPO_ROOT / "data" / "raw"

ODDADMIX_REPOS = {
    "sudan_podcast": "oddadmix/arabic-audio-collection-sudanese-sudan-podcast",
    "nuuar": "oddadmix/arabic-audio-collection-sudanese-nuuar",
    "ahmed_gobara": "oddadmix/arabic-audio-collection-sudanese-ahmed-gobara",
}
ODDADMIX_COLUMNS = ["chunk_id", "transcript_text", "duration", "original_video_id"]


def fetch_oddadmix() -> None:
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    out_dir = DATA_RAW / "oddadmix"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("hf")

    for name, repo in ODDADMIX_REPOS.items():
        out = out_dir / f"{name}.jsonl"
        if out.exists():
            print(f"  oddadmix/{name}: exists, skipping")
            continue
        shards = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                        if f.endswith(".parquet"))
        rows = 0
        with open(out, "w", encoding="utf-8") as fh:
            for i, shard in enumerate(shards):
                with fs.open(f"datasets/{repo}/{shard}") as f:
                    table = pq.read_table(f, columns=ODDADMIX_COLUMNS)
                for row in table.to_pylist():
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
                print(f"  oddadmix/{name}: shard {i+1}/{len(shards)}  {rows:,} rows", flush=True)
        print(f"  oddadmix/{name}: {rows:,} rows -> {out}")


LISAN_TTS_REPOS = {
    "lisan_tts": "AymanMansour/Lisan-Sudanese-TTS-Dataset",
    "lisan_tts_new": "AymanMansour/New-Lisan-Sudanese-TTS-Dataset",
}


def fetch_lisan_tts() -> None:
    """The Lisan-Sudanese *text* rides inside these TTS datasets as a plain string column.

    The full Lisan corpus (52K tokens, CC BY 4.0) is behind a Google form at
    sina.birzeit.edu/currasat — worth requesting manually; this gets the freely-mirrored
    sentences now, same column-projection trick as oddadmix so the audio stays remote.
    """
    import fsspec
    import pyarrow.parquet as pq
    from huggingface_hub import list_repo_files

    out_dir = DATA_RAW / "lisan"
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("hf")

    for name, repo in LISAN_TTS_REPOS.items():
        out = out_dir / f"{name}.jsonl"
        if out.exists():
            print(f"  lisan/{name}: exists, skipping")
            continue
        shards = sorted(f for f in list_repo_files(repo, repo_type="dataset")
                        if f.endswith(".parquet"))
        rows = 0
        with open(out, "w", encoding="utf-8") as fh:
            for shard in shards:
                with fs.open(f"datasets/{repo}/{shard}") as f:
                    table = pq.read_table(f, columns=["text"])
                for row in table.to_pylist():
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
        print(f"  lisan/{name}: {rows:,} rows -> {out}")


def fetch_file(repo, filename, out_path, repo_type="dataset") -> None:
    from huggingface_hub import hf_hub_download

    if out_path.exists():
        print(f"  {out_path.relative_to(DATA_RAW)}: exists, skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo, filename, repo_type=repo_type)
    out_path.write_bytes(Path(cached).read_bytes())
    print(f"  {out_path.relative_to(DATA_RAW)}: {out_path.stat().st_size/1e6:.2f} MB")


def main() -> int:
    print("Tarab (Sudanese lyric slice, CC-BY):")
    fetch_file("drelhaj/Tarab", "tarab_by_dialect/tarab_Sudanese.csv",
               DATA_RAW / "tarab" / "tarab_Sudanese.csv")

    print("Alexandria SD subset (eval-grade conversations):")
    from huggingface_hub import list_repo_files

    # not every dialect ships every split — list what actually exists rather than assuming
    for f in list_repo_files("UBC-NLP/alexandria", repo_type="dataset"):
        if f.startswith("SD/") and f.endswith(".parquet"):
            fetch_file("UBC-NLP/alexandria", f,
                       DATA_RAW / "alexandria_sd" / f.split("/")[-1])

    print("Organic Sudanese dialect sample (CC-BY):")
    fetch_file("ebubekr53/organic-sudanese-arabic-dialect-dataset",
               "sudanese_dialect_dataset.csv",
               DATA_RAW / "organic_sudanese" / "sudanese_dialect_dataset.csv")

    print("Lisan-Sudanese sentences (via the TTS mirrors, text column only):")
    fetch_lisan_tts()

    print("oddadmix podcast transcripts (transcript column only, no audio):")
    fetch_oddadmix()
    return 0


if __name__ == "__main__":
    sys.exit(main())
