"""Polite scraper for alnilin.com via the WordPress REST API (audited 2026-08-28).

The audit found ~550K articles (sitemap depth) and — the real prize — 561K reader comments,
both exposed wholesale through the standard open wp-json endpoints, which return clean JSON
(no HTML parsing fragility). robots.txt permits general crawling; the REST route is the
lightest-touch way to read the same public content.

Articles are Sudan-topical MSA (silver tier — ranked down by the dialect classifier at
mixture time); comments are where dialect lives (gold tier).

Resumable via page-numbered files; single-threaded; fixed delay; backoff; hard stop.

Output: data/raw/alnilin/{posts,comments}/page_<n>.json   (100 records per page)
Usage:  python scripts/scrape_alnilin.py [--what posts,comments] [--delay 1.0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "alnilin"

BASE = "https://www.alnilin.com/wp-json/wp/v2"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
PER_PAGE = 100
MAX_CONSECUTIVE_FAILURES = 15

FIELDS = {
    "posts": "id,date,title,content,link,categories",
    "comments": "id,post,date,content",
}


def scrape(session, what, delay):
    out_dir = RAW_DIR / what
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem.split("_")[1]) for p in out_dir.glob("page_*.json")}
    page = max(done) + 1 if done else 1
    failures = 0
    while True:
        url = (f"{BASE}/{what}?per_page={PER_PAGE}&page={page}"
               f"&orderby=id&order=asc&_fields={FIELDS[what]}")
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"{what}: aborting after {failures} failures", flush=True)
                return
            time.sleep(min(2 ** failures, 120))
            continue
        if response.status_code == 400:                # past the last page
            print(f"{what}: complete at page {page - 1}", flush=True)
            return
        if response.status_code != 200:
            failures += 1
            time.sleep(min(2 ** failures, 300))
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"{what}: aborting at page {page} (HTTP {response.status_code})",
                      flush=True)
                return
            continue
        failures = 0
        rows = response.json()
        if not rows:
            print(f"{what}: complete at page {page - 1}", flush=True)
            return
        (out_dir / f"page_{page:06d}.json").write_text(
            json.dumps(rows, ensure_ascii=False))
        if page % 100 == 0:
            total = response.headers.get("X-WP-Total", "?")
            print(f"{what}: page {page} ({PER_PAGE}/pg of {total} total)", flush=True)
        page += 1
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", default="comments,posts",
                        help="comments first: they are the dialect-bearing half")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for what in args.what.split(","):
        print(f"=== {what}", flush=True)
        scrape(session, what.strip(), args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
