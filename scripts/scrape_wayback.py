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
           "alhasahisa.org", "hurriyatsudan.com",
           "sudanesesongs.net"]   # IPB lyrics forum, 58 CDX page-blocks (added 2026-08-31)
CDX = "http://web.archive.org/cdx/search/cdx"


def cdx_enumerate(session, domain, out_dir, delay):
    """Enumerate captures page by page. A transient archive.org outage must NOT truncate
    the listing (mugrn.net lost its entire 31k-url /vb/ archive to exactly that on
    2026-09-02): non-200 responses and HTML bodies (the "Temporarily Offline" page) are
    retried with backoff, and a truncated listing raises instead of being persisted —
    only a clean empty-page terminator writes cdx.txt."""
    cdx_path = out_dir / "cdx.txt"
    if cdx_path.exists() and cdx_path.stat().st_size > 0:
        return sum(1 for _ in open(cdx_path))
    seen, rows, failures = set(), [], 0
    params = {"url": f"{domain}/*", "matchType": "domain",
              "filter": ["statuscode:200", "mimetype:text/html"],
              "fl": "original,timestamp", "limit": 25000, "showResumeKey": "true"}
    while True:
        try:
            response = session.get(CDX, params=params, timeout=120)
        except requests.RequestException:
            response = None
        bad = (response is None or response.status_code != 200
               or response.text.lstrip().startswith("<"))
        if bad:
            failures += 1
            if failures > 12:
                raise RuntimeError(
                    f"{domain}: CDX enumeration failing persistently — refusing to "
                    f"write a truncated listing (got {len(rows)} rows)")
            time.sleep(min(30 * failures, 300))
            continue
        failures = 0
        lines = response.text.splitlines()
        # response layout: data lines, then a blank line + resume key when more remain
        resume_key = None
        if "" in lines:
            split = lines.index("")
            resume_key = next((l for l in lines[split + 1:] if l.strip()), None)
            lines = lines[:split]
        for line in lines:
            url = line.split(" ", 1)[0]
            if url and url not in seen:          # client-side collapse (resumeKey is
                seen.add(url)                    # incompatible with collapse=urlkey)
                rows.append(line)
        if not resume_key:
            break                                # clean end of listing
        params["resumeKey"] = resume_key
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
        if not any(k in url for k in ("archive/index.php", "showthread", "viewtopic", "/vb/",
                                      "showtopic", "lofiversion")):   # last two: IPB boards
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
