"""Polite, resumable crawler for sudaneseonline.com forum threads (plan.md Part IV, step 1.4).

The forum's 25 years of archives are the largest native-Sudanese text reservoir on the open web.
robots.txt disallows only /admin/ and lists ~80 sitemap files whose union enumerates ~110K
thread URLs (through ~2015; later years need board-page walking, a separate increment).

Politeness: one request at a time, a fixed delay between requests, an honest identifying
User-Agent, exponential backoff on errors, and a hard stop after too many consecutive failures
so a site outage never turns into a hammering loop.

Resumability: one gzipped HTML file per thread, named by its stable message id — a thread whose
file exists is never fetched again, so the crawl can be killed and restarted freely.

Input:   data/raw/sudaneseonline/thread_urls.txt  (one "board/N/msg/....html" path per line)
Output:  data/raw/sudaneseonline/html/<board>_<msgid>.html.gz
         data/raw/sudaneseonline/crawl.log

Usage:   python scripts/scrape_sudaneseonline.py [--delay 0.5] [--limit N]
"""

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sudaneseonline"
URL_FILE = RAW_DIR / "thread_urls.txt"
HTML_DIR = RAW_DIR / "html"

BASE = "https://sudaneseonline.com/"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
MSG_ID_RE = re.compile(r"-(\d+)\.html?$")
BOARD_RE = re.compile(r"^board/(\d+)/")

MAX_CONSECUTIVE_FAILURES = 20


def out_path(rel_url: str) -> Path | None:
    msg = MSG_ID_RE.search(rel_url)
    board = BOARD_RE.search(rel_url)
    if not msg or not board:
        return None
    return HTML_DIR / f"{board.group(1)}_{msg.group(1)}.html.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.5,
                        help="seconds between requests (default 0.5 ≈ 2 req/s, single-threaded)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after fetching N new threads (for a pilot)")
    args = parser.parse_args()

    urls = [line.strip() for line in URL_FILE.read_text().splitlines() if line.strip()]
    HTML_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    fetched = skipped = failed = consecutive = 0
    start = time.time()
    for rel_url in urls:
        target = out_path(rel_url)
        if target is None:
            continue
        if target.exists():
            skipped += 1
            continue
        if args.limit is not None and fetched >= args.limit:
            break

        try:
            response = session.get(BASE + rel_url, timeout=30)
            if response.status_code == 200 and len(response.content) > 2000:
                # atomic write so a kill mid-write never leaves a truncated "done" file
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(response.content))
                tmp.rename(target)
                fetched += 1
                consecutive = 0
            else:
                failed += 1
                consecutive += 1
        except requests.RequestException:
            failed += 1
            consecutive += 1
            time.sleep(min(2 ** min(consecutive, 6), 60))   # backoff on top of the base delay

        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            print(f"aborting: {consecutive} consecutive failures — site down or blocking",
                  flush=True)
            return 1

        if fetched and fetched % 200 == 0:
            rate = fetched / max(time.time() - start, 1)
            remaining = len(urls) - fetched - skipped
            print(f"  {fetched:,} fetched  {skipped:,} already  {failed:,} failed  "
                  f"{rate:.1f}/s  ~{remaining/max(rate,0.1)/3600:.1f}h left", flush=True)
        time.sleep(args.delay)

    print(f"done: {fetched:,} fetched, {skipped:,} already present, {failed:,} failed "
          f"in {(time.time()-start)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
