"""Polite crawler for the anasudani.net phpBB forum (SCRAPESHEET queue item #3).

1,161,442 posts / 52,973 topics, frozen ~Jan 2017, and the site serves NO robots.txt (404 →
unrestricted). Topic ids are sparse, so discovery walks the viewforum listings rather than
sweeping ids.

Two resumable phases:
  discover  walk every forum's listing pages -> data/raw/anasudani/topics.txt (forum, topic)
  fetch     per topic page (viewtopic, paginated) -> html/t<ID>_s<START>.html.gz

Same politeness contract as every crawler here: single-threaded, fixed delay, honest UA,
backoff, hard stop on consecutive failures.

Usage:  python scripts/scrape_anasudani.py discover [--delay 0.7]
        python scripts/scrape_anasudani.py fetch [--delay 0.7]
"""

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "anasudani"
TOPICS_PATH = RAW_DIR / "topics.txt"
HTML_DIR = RAW_DIR / "html"

BASE = "https://www.anasudani.net/forum/"
USER_AGENT = "Mozilla/5.0 (compatible; sudani-lm-crawler; personal research use)"
MAX_CONSECUTIVE_FAILURES = 15
# 20, not 40: the forum's own pagination links step by 20. The raw count of viewtopic links
# per listing page overstates the step (stickies repeat on every page) — stepping 40 silently
# skipped every other page and halved the first discovery run.
TOPICS_PER_LISTING = 20

TOPIC_RE = re.compile(r"viewtopic\.php\?[^\"']*t=(\d+)")
FORUM_RE = re.compile(r"viewforum\.php\?f=(\d+)")


def _get(session, url, failures):
    """(text, failures, status). 404 is NOT a failure — the forum's listings reference
    deleted topics, and a run of dead ids must not trip the consecutive-failure abort."""
    try:
        response = session.get(url, timeout=45)
        if response.status_code == 200:
            return response.text, 0, 200
        if response.status_code == 404:
            return None, failures, 404
        status = response.status_code
    except requests.RequestException:
        status = 0
    time.sleep(min(2 ** (failures + 1), 120))
    return None, failures + 1, status


def discover(session, delay) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    if TOPICS_PATH.exists():
        seen = {line.split()[1] for line in TOPICS_PATH.read_text().splitlines() if line}
    index, failures, _ = _get(session, BASE, 0)
    if index is None:
        print("index unreachable")
        return 1
    forums = sorted({int(f) for f in FORUM_RE.findall(index)})
    print(f"{len(forums)} forums; {len(seen)} topics already known")

    with open(TOPICS_PATH, "a", encoding="utf-8") as fh:
        for forum in forums:
            start, new_in_forum = 0, 0
            prev_topics = None
            while True:
                page, failures, status = _get(
                    session, f"{BASE}viewforum.php?f={forum}&start={start}", failures)
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    print("aborting: repeated failures")
                    return 1
                if status == 404:
                    break
                if page is None:
                    continue
                topics = set(TOPIC_RE.findall(page))
                fresh = [t for t in topics if t not in seen]
                for topic in fresh:
                    seen.add(topic)
                    fh.write(f"{forum} {topic}\n")
                new_in_forum += len(fresh)
                fh.flush()
                # Termination: past-the-end pages are not empty (a sticky persists) and
                # breaking on no-FRESH-topics broke resume (seen is preloaded, so page 1 of
                # every forum looked stale). The reliable signal is REPETITION — past the
                # last page phpBB serves the identical topic set every time.
                if not topics or topics == prev_topics:
                    break
                prev_topics = topics
                start += TOPICS_PER_LISTING
                time.sleep(delay)
            print(f"  forum {forum}: {new_in_forum} topics", flush=True)
    print(f"total topics known: {len(seen)}")
    return 0


def fetch(session, delay) -> int:
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    rows = [line.split() for line in TOPICS_PATH.read_text().splitlines() if line.strip()]
    done_first_pages = {p.name for p in HTML_DIR.glob("t*_s0.html.gz")}
    dead_path = RAW_DIR / "dead.txt"
    dead = set(dead_path.read_text().split()) if dead_path.exists() else set()
    fetched = failures = 0
    start_time = time.time()
    for forum, topic in rows:
        first = HTML_DIR / f"t{topic}_s0.html.gz"
        if first.name in done_first_pages or topic in dead:
            continue
        page, failures, status = _get(
            session, f"{BASE}viewtopic.php?f={forum}&t={topic}", failures)
        if failures >= MAX_CONSECUTIVE_FAILURES:
            print("aborting: repeated failures")
            return 1
        if status == 404:
            dead.add(topic)
            with open(dead_path, "a") as dh:
                dh.write(topic + "\n")
            time.sleep(delay)
            continue
        if page is None:
            continue
        first.write_bytes(gzip.compress(page.encode()))
        # follow this topic's own pagination — offsets must come from links to THIS topic,
        # not from "similar topics" links elsewhere on the page
        topic_start_re = re.compile(
            rf"viewtopic\.php\?[^\"']*t={topic}(?:&(?:amp;)?[^\"']*)?start=(\d+)")
        starts = sorted({int(s) for s in topic_start_re.findall(page)})
        for offset in [s for s in starts if s > 0]:
            extra = HTML_DIR / f"t{topic}_s{offset}.html.gz"
            if extra.exists():
                continue
            time.sleep(delay)
            page2, failures, _ = _get(
                session, f"{BASE}viewtopic.php?f={forum}&t={topic}&start={offset}", failures)
            if page2 is not None:
                extra.write_bytes(gzip.compress(page2.encode()))
        fetched += 1
        if fetched % 200 == 0:
            rate = fetched / max(time.time() - start_time, 1)
            remaining = len(rows) - fetched
            print(f"  {fetched:,} topics fetched ({rate:.1f}/s,"
                  f" ~{remaining/max(rate,0.01)/3600:.1f}h left)", flush=True)
        time.sleep(delay)
    print(f"fetch complete: {fetched:,} new topics")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["discover", "fetch"])
    parser.add_argument("--delay", type=float, default=0.7)
    args = parser.parse_args()
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return discover(session, args.delay) if args.phase == "discover" \
        else fetch(session, args.delay)


if __name__ == "__main__":
    sys.exit(main())
