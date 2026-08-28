"""Polite scraper for PUBLIC Telegram channels via the t.me/s/ preview pages.

The highest dialect-density source found in the 2026-08-28 acquisition survey: Sudanese
serial-fiction and chat channels publish tens of thousands of messages of pure عامي, and
Telegram's public preview (t.me/s/<channel>) serves them as static HTML, ~20 messages per
page, paginated by `?before=<message_id>`, with no robots restrictions (t.me/robots.txt is
a 404).

Resumable: one JSON file per page-fetch keyed by (channel, before-cursor); a crawl can be
killed and rerun freely. Same politeness rules as the sudaneseonline crawler: single-threaded,
fixed delay, honest UA, backoff, hard stop on consecutive failures.

Output: data/raw/telegram/<channel>/page_<cursor>.json   (raw message HTML blocks)
Usage:  python scripts/scrape_telegram.py [--channels a,b,c] [--delay 1.0] [--limit-pages N]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "telegram"

USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
MAX_CONSECUTIVE_FAILURES = 15

# Sudanese channels verified dialect-dense in the acquisition survey; extend freely.
CHANNELS = [
    "novelsforus2", "klam_sudany", "sudanesenovels", "Sd_rewaya3t",
    "sudanes0", "Diwansha3r",
]

MESSAGE_RE = re.compile(
    r'data-post="[^"]+/(\d+)".*?tgme_widget_message_text[^>]*>(.*?)</div>', re.S)
ANY_POST_RE = re.compile(r'data-post="[^"]+/(\d+)"')


def scrape_channel(session, channel, delay, limit_pages):
    out_dir = RAW_DIR / channel
    out_dir.mkdir(parents=True, exist_ok=True)
    # resume: continue below the lowest message id already fetched
    done_cursors = sorted(
        (int(p.stem.split("_")[1]) for p in out_dir.glob("page_*.json")), reverse=True)
    cursor = min(done_cursors) if done_cursors else None

    fetched = failures = 0
    while limit_pages is None or fetched < limit_pages:
        url = f"https://t.me/s/{channel}" + (f"?before={cursor}" if cursor else "")
        try:
            response = session.get(url, timeout=30)
        except requests.RequestException:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {channel}: aborting after {failures} failures", flush=True)
                return fetched
            time.sleep(min(2 ** failures, 120))
            continue
        if response.status_code != 200:
            failures += 1
            time.sleep(min(2 ** failures, 120))
            if failures >= MAX_CONSECUTIVE_FAILURES:
                return fetched
            continue
        failures = 0

        ids = [int(i) for i in ANY_POST_RE.findall(response.text)]
        if not ids:
            print(f"  {channel}: reached channel start", flush=True)
            return fetched
        messages = [{"id": int(mid), "html": html}
                    for mid, html in MESSAGE_RE.findall(response.text)]
        lowest = min(ids)
        (out_dir / f"page_{lowest:08d}.json").write_text(json.dumps(
            {"channel": channel, "before": cursor, "lowest": lowest,
             "messages": messages}, ensure_ascii=False))
        fetched += 1
        if fetched % 100 == 0:
            print(f"  {channel}: {fetched} pages, at id {lowest}", flush=True)
        if lowest <= 1:
            print(f"  {channel}: complete", flush=True)
            return fetched
        cursor = lowest
        time.sleep(delay)
    return fetched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", default=",".join(CHANNELS))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit-pages", type=int, default=None)
    args = parser.parse_args()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    for channel in args.channels.split(","):
        channel = channel.strip()
        print(f"=== {channel}", flush=True)
        pages = scrape_channel(session, channel, args.delay, args.limit_pages)
        print(f"  {channel}: {pages} new pages", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
