"""Wayback Machine miner for dead Sudanese vBulletin forums (SCRAPESHEET queue).

Five dead forums (sudanyat, mugrn, algorer, sudanelite, alhasahisa) plus hurriyatsudan share
the predictable vBulletin archive URL shape `/vb/archive/index.php/t-N.html` — light pages,
ideal for Wayback retrieval. CDX enumerates what the archive holds; then each snapshot is
fetched once via the `id_` raw endpoint. archive.org tolerates ~1 req/s from polite clients.

Two phases per domain, both resumable:
  cdx     -> data/raw/wayback/<domain>/cdx.txt   (url, timestamp per captured page)
  fetch   -> data/raw/wayback/<domain>/pages/<hash>.html.gz

Usage:  python scripts/scrape_wayback.py [--domains d1,d2] [--delay 1.0]
"""

import argparse
import gzip
import hashlib
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "wayback"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"

DOMAINS = ["sudanyat.org", "mugrn.net", "algorer.net", "sudanelite.com",
           "alhasahisa.org", "hurriyatsudan.com"]
CDX = "http://web.archive.org/cdx/search/cdx"


def cdx_enumerate(session, domain, out_dir, delay):
    cdx_path = out_dir / "cdx.txt"
    if cdx_path.exists() and cdx_path.stat().st_size > 0:
        return sum(1 for _ in open(cdx_path))
    rows = []
    for page in range(0, 200):
        try:
            response = session.get(CDX, params={
                "url": f"{domain}/*", "matchType": "domain",
                "filter": ["statuscode:200", "mimetype:text/html"],
                "collapse": "urlkey", "fl": "original,timestamp",
                "page": page}, timeout=120)
        except requests.RequestException:
            time.sleep(30)
            continue
        if response.status_code != 200 or not response.text.strip():
            break
        rows.extend(response.text.strip().splitlines())
        time.sleep(delay)
    with open(cdx_path, "w") as fh:
        fh.write("\n".join(rows))
    return len(rows)


def fetch(session, domain, out_dir, delay):
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    fetched = failures = 0
    for line in open(out_dir / "cdx.txt"):
        parts = line.split()
        if len(parts) != 2:
            continue
        url, timestamp = parts
        # forum content only: archive/thread/forum pages, skip assets and login/reply forms
        if not any(k in url for k in ("archive/index.php", "showthread", "viewtopic", "/vb/")):
            continue
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        out = pages_dir / f"{digest}.html.gz"
        if out.exists():
            continue
        try:
            response = session.get(
                f"http://web.archive.org/web/{timestamp}id_/{url}", timeout=90)
        except requests.RequestException:
            failures += 1
            time.sleep(min(2 ** min(failures, 6), 120))
            continue
        if response.status_code == 200 and len(response.content) > 1500:
            out.write_bytes(gzip.compress(response.content))
            fetched += 1
            failures = 0
            if fetched % 200 == 0:
                print(f"  {domain}: {fetched} pages", flush=True)
        time.sleep(delay)
    print(f"  {domain}: fetch pass done ({fetched} new)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for domain in args.domains.split(","):
        domain = domain.strip()
        out_dir = RAW_DIR / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        n = cdx_enumerate(session, domain, out_dir, args.delay)
        print(f"=== {domain}: {n} captured urls in CDX", flush=True)
        fetch(session, domain, out_dir, args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
