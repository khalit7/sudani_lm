"""Polite same-domain BFS mirror for small Sudanese sites (SCRAPESHEET queue).

For sites too small to deserve a bespoke scraper (proverb/lyric collections, essay pages):
breadth-first crawl within the domain, HTML pages only, capped page budget, robots-respected
by the caller's site selection (each site here was audited as permissive or robots-absent).

Usage:  python scripts/scrape_small_sites.py --site aghaniwamthal --base https://aghaniwamthal.com \
            [--max-pages 3000] [--delay 1.0]
"""

import argparse
import gzip
import hashlib
import re
import sys
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
LINK_RE = re.compile(r'href=["\']([^"\'#]+)')
SKIP_EXT = re.compile(r"\.(jpg|jpeg|png|gif|webp|css|js|ico|svg|mp3|mp4|pdf|zip|woff2?)(\?|$)",
                      re.I)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--max-pages", type=int, default=3000)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = REPO_ROOT / "data" / "raw" / "small_sites" / args.site
    out_dir.mkdir(parents=True, exist_ok=True)
    host = urlparse(args.base).netloc
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    seen_index = out_dir / "_seen.txt"
    seen = set(seen_index.read_text().splitlines()) if seen_index.exists() else set()
    queue = deque([args.base])
    fetched = failures = 0

    with open(seen_index, "a") as seen_fh:
        while queue and fetched < args.max_pages:
            url = queue.popleft()
            if url in seen or SKIP_EXT.search(url):
                continue
            seen.add(url)
            seen_fh.write(url + "\n")
            try:
                response = session.get(url, timeout=45)
            except requests.RequestException:
                failures += 1
                if failures > 20:
                    print("aborting: too many failures")
                    return 1
                time.sleep(min(2 ** min(failures, 6), 60))
                continue
            if response.status_code != 200 or "text/html" not in \
                    response.headers.get("content-type", ""):
                continue
            failures = 0
            digest = hashlib.sha1(url.encode()).hexdigest()[:16]
            (out_dir / f"{digest}.html.gz").write_bytes(gzip.compress(response.content))
            fetched += 1
            for link in LINK_RE.findall(response.text):
                absolute = urljoin(url, link)
                if urlparse(absolute).netloc == host and absolute not in seen:
                    queue.append(absolute.split("#")[0])
            if fetched % 100 == 0:
                print(f"{args.site}: {fetched} pages, queue {len(queue)}", flush=True)
            time.sleep(args.delay)
    print(f"{args.site}: done — {fetched} pages fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
