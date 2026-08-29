"""Generic polite scraper for open WordPress REST APIs (SCRAPESHEET queue).

Same method as scrape_alnilin.py, parameterized by site so one script covers sudanile,
alsudaninews, dabangasudan, and any future WP site with an open /wp-json route. Clean JSON,
no HTML fragility, resumable by page file.

Usage:  python scripts/scrape_wpjson.py --name sudanile --base https://sudanile.com \
            [--what posts,comments] [--delay 1.0]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
PER_PAGE = 100
MAX_CONSECUTIVE_FAILURES = 15
FIELDS = {"posts": "id,date,title,content,link,categories", "comments": "id,post,date,content"}


def scrape(session, name, base, what, delay):
    out_dir = REPO_ROOT / "data" / "raw" / name / what
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {int(p.stem.split("_")[1]) for p in out_dir.glob("page_*.json")}
    page = max(done) + 1 if done else 1
    failures = 0
    while True:
        url = (f"{base}/wp-json/wp/v2/{what}?per_page={PER_PAGE}&page={page}"
               f"&orderby=id&order=asc&_fields={FIELDS[what]}")
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"{name}/{what}: aborting after {failures} failures", flush=True)
                return
            time.sleep(min(2 ** failures, 120))
            continue
        if response.status_code == 400:
            print(f"{name}/{what}: complete at page {page - 1}", flush=True)
            return
        if response.status_code != 200:
            failures += 1
            time.sleep(min(2 ** failures, 300))
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"{name}/{what}: aborting at page {page}"
                      f" (HTTP {response.status_code})", flush=True)
                return
            continue
        failures = 0
        rows = response.json()
        if not rows:
            print(f"{name}/{what}: complete at page {page - 1}", flush=True)
            return
        (out_dir / f"page_{page:06d}.json").write_text(json.dumps(rows, ensure_ascii=False))
        if page % 100 == 0:
            print(f"{name}/{what}: page {page} of ~"
                  f"{response.headers.get('X-WP-Total', '?')}÷{PER_PAGE}", flush=True)
        page += 1
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--what", default="posts,comments")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for what in args.what.split(","):
        scrape(session, args.name, args.base, what.strip(), args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
