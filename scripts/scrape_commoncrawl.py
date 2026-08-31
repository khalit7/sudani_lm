"""Common Crawl bolt-on miner for the dead Sudanese forums (SCRAPESHEET queue).

Complements scrape_wayback.py: CC's historical crawls (2017-2023) hold ~5-8k unique
status-200 thread pages across the four dead forums, retrievable by byte-range GETs
against data.commoncrawl.org — fast and unthrottled, unlike Wayback. Heavy overlap
with the Wayback miner is expected; preprocessing dedups by content.

Caveats measured in the 2026-08-31 sweep: many CC records are status 406 (the forums
refused CC's UA in some years) — only status 200 records are fetched; payloads are
windows-1256 and are stored raw for the existing cp1256 repair in preprocessing.

Two phases per domain, both resumable:
  index -> data/raw/commoncrawl/<domain>/index.jsonl   (one record per unique urlkey)
  fetch -> data/raw/commoncrawl/<domain>/pages/<hash>.html.gz

Usage:  python scripts/scrape_commoncrawl.py [--domains d1,d2] [--delay 0.4]
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "commoncrawl"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"

DOMAINS = ["sudanyat.org", "mugrn.net", "algorer.net", "wadmadani.com"]
COLLINFO = "https://index.commoncrawl.org/collinfo.json"
DATA = "https://data.commoncrawl.org/"
THREAD_KEYS = ("archive/index.php", "showthread", "viewtopic", "/vb/",
               "showtopic", "lofiversion")


def build_index(session, domain, out_dir, delay):
    index_path = out_dir / "index.jsonl"
    done_path = out_dir / "collections_done.txt"
    done = set(done_path.read_text().split()) if done_path.exists() else set()
    collections = [c["id"] for c in session.get(COLLINFO, timeout=60).json()]
    seen_keys = set()
    if index_path.exists():
        for line in open(index_path):
            seen_keys.add(json.loads(line)["urlkey"])
    for coll in collections:
        if coll in done:
            continue
        try:
            response = session.get(
                f"https://index.commoncrawl.org/{coll}-index",
                params={"url": f"{domain}/*", "output": "json",
                        "filter": "=status:200"}, timeout=120)
        except requests.RequestException:
            time.sleep(15)
            continue
        if response.status_code == 200:
            with open(index_path, "a") as fh:
                for line in response.text.strip().splitlines():
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("urlkey") in seen_keys:
                        continue
                    if not any(k in rec.get("url", "") for k in THREAD_KEYS):
                        continue
                    seen_keys.add(rec["urlkey"])
                    fh.write(json.dumps(rec) + "\n")
        # 404 = domain absent from this collection: also done
        if response.status_code in (200, 404):
            done.add(coll)
            with open(done_path, "a") as fh:
                fh.write(coll + "\n")
        time.sleep(delay)
    return len(seen_keys)


def fetch(session, domain, out_dir, delay):
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    fetched = failures = 0
    for line in open(out_dir / "index.jsonl"):
        rec = json.loads(line)
        digest = hashlib.sha1(rec["url"].encode()).hexdigest()[:16]
        out = pages_dir / f"{digest}.html.gz"
        if out.exists():
            continue
        start = int(rec["offset"])
        end = start + int(rec["length"]) - 1
        try:
            response = session.get(DATA + rec["filename"],
                                   headers={"Range": f"bytes={start}-{end}"},
                                   timeout=90)
        except requests.RequestException:
            failures += 1
            time.sleep(min(2 ** min(failures, 6), 120))
            continue
        if response.status_code not in (200, 206):
            failures += 1
            time.sleep(min(2 ** min(failures, 6), 120))
            continue
        failures = 0
        try:
            record = gzip.decompress(response.content)
            # WARC record = warc headers \r\n\r\n http headers \r\n\r\n body
            body = record.split(b"\r\n\r\n", 2)[2]
        except (OSError, IndexError):
            continue
        if len(body) > 1000:
            out.write_bytes(gzip.compress(body))
            fetched += 1
            if fetched % 200 == 0:
                print(f"  {domain}: {fetched} pages", flush=True)
        time.sleep(delay)
    print(f"  {domain}: fetch pass done ({fetched} new)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--delay", type=float, default=0.4)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for domain in args.domains.split(","):
        domain = domain.strip()
        out_dir = RAW_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        n = build_index(session, domain, out_dir, args.delay)
        print(f"=== {domain}: {n} unique thread urls indexed", flush=True)
        fetch(session, domain, out_dir, args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
