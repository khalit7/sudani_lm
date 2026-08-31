"""Scraper for the Sudanese Blogger cluster via the Atom feed API (SCRAPESHEET queue).

Blogger exposes full post bodies at /feeds/posts/default?max-results=500&start-index=N —
structured Atom, no HTML fragility, tiny volume but the densest dialect registers found
(Haqiba lyrics, short stories, novels).

Usage:  python scripts/scrape_blogger.py [--delay 1.0]
"""

import argparse
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"

BLOGS = {
    "hageebatalfun": "http://hageebatalfun.blogspot.com",
    "sudaneseshortstories": "http://sudaneseshortstorieswriters.blogspot.com",
    "sudanese_novels": "http://sudanese-novels.blogspot.com",
    "katabsudsnese": "http://katabsudsnese.blogspot.com",
    # 2026-08-31 delta from mtwersd.com/sudanese-blogs/ (personal/women's/cooking registers)
    "unothati": "https://unothati.blogspot.com",
    "olive2020": "https://olive2020.blogspot.com",
    "ajba77": "https://ajba77.blogspot.com",
    "sudanesemollified": "https://sudanesemollified.blogspot.com",
    "salahamza2": "https://salahamza2.blogspot.com",
    "ar_cher": "https://ar-cher.blogspot.com",
    "montser2019": "https://22montser2019.blogspot.com",
    "trendsudani": "https://trendsudani.blogspot.com",
}
BATCH = 150      # blogger caps max-results in practice; 150 is reliably honoured


def scrape(session, name, base, delay):
    out_dir = REPO_ROOT / "data" / "raw" / "blogger" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        out = out_dir / f"feed_{index:06d}.xml"
        if out.exists():
            index += BATCH
            continue
        url = f"{base}/feeds/posts/default?max-results={BATCH}&start-index={index}"
        try:
            response = session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(30)
            continue
        if response.status_code != 200:
            print(f"  {name}: HTTP {response.status_code} at index {index} — stopping",
                  flush=True)
            return
        entries = response.text.count("<entry>")
        if entries == 0:
            print(f"  {name}: complete ({index - 1} posts max)", flush=True)
            return
        out.write_text(response.text)
        print(f"  {name}: index {index} (+{entries} entries)", flush=True)
        index += BATCH
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for name, base in BLOGS.items():
        print(f"=== {name}", flush=True)
        scrape(session, name, base, args.delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
