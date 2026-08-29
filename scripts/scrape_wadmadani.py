"""Slow-drip crawler for wadmadani.com/vb (SCRAPESHEET queue).

The forum's robots.txt allows everything but sets `crawl-delay: 60` — obeyed as written, so
this runs for weeks by design (one request per minute). It sweeps the light vBulletin archive
pages `/vb/archive/index.php/t-N.html` (thread ids observed up to ≥41,443). The host is on
failing storage (intermittent 507/508); those count as transient, 404s are tombstoned like
anasudani's dead ids.

Usage:  python scripts/scrape_wadmadani.py [--max-id 42000]
"""

import argparse
import gzip
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wadmadani"
BASE = "http://www.wadmadani.com/vb/archive/index.php"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
CRAWL_DELAY = 60          # mandated by the site's robots.txt — do not lower


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-id", type=int, default=42000)
    args = parser.parse_args()

    html_dir = RAW_DIR / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    dead_path = RAW_DIR / "dead.txt"
    dead = set(dead_path.read_text().split()) if dead_path.exists() else set()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    fetched = 0
    for thread_id in range(1, args.max_id + 1):
        tid = str(thread_id)
        out = html_dir / f"t{tid}.html.gz"
        if out.exists() or tid in dead:
            continue
        response = None
        try:
            response = session.get(f"{BASE}/t-{tid}.html", timeout=60)
            status = response.status_code
        except requests.RequestException:
            status = 0
        if status == 200 and response is not None and len(response.content) > 1000:
            out.write_bytes(gzip.compress(response.content))
            fetched += 1
            if fetched % 50 == 0:
                print(f"{fetched} threads (at id {tid})", flush=True)
        elif status == 404:
            dead.add(tid)
            with open(dead_path, "a") as fh:
                fh.write(tid + "\n")
        # 507/508/timeouts: transient (failing host) — just move on after the delay
        time.sleep(CRAWL_DELAY)
    print(f"sweep complete: {fetched} threads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
