#!/bin/bash
# v4: watches the final-sweep fleet — wayback miner(s), telegram delta, koorasudan
# wp-json, common-crawl bolt-on. One line per state change per crawl. Process-gone is
# reported once as DONE-or-DIED (finished crawls and dead crawls look the same here;
# the log tail says which).
cd /home/khalid/sudani_lm

check() {  # name process_pattern current_count
  local name=$1 pattern=$2 count=$3
  local alive
  alive=$(pgrep -fc "$pattern")
  alive=${alive:-0}
  local last_var="last_$name" state_var="state_$name"
  local last=${!last_var:--1} state=${!state_var}
  if [ "$alive" -eq 0 ] 2>/dev/null; then
    [ "$state" != gone ] && echo "DONE-or-DIED $name: process gone at $count items"
    printf -v "$state_var" gone
  elif [ "$count" = "$last" ]; then
    [ "$state" != stalled ] && echo "PROBLEM $name: stalled at $count items for 15+ min"
    printf -v "$state_var" stalled
  else
    printf -v "$state_var" alive
  fi
  printf -v "$last_var" "%s" "$count"
}

while true; do
  sleep 900
  check wayback     'scrape_wayback.p[y]'      "$(find data/raw/wayback -name '*.html.gz' 2>/dev/null | wc -l)"
  check telegram    'scrape_telegram.p[y]'     "$(find data/raw/telegram -name 'page_*.json' 2>/dev/null | wc -l)"
  check commoncrawl 'scrape_commoncrawl.p[y]'  "$(( $(find data/raw/commoncrawl -name '*.html.gz' 2>/dev/null | wc -l) + $(cat data/raw/commoncrawl/*/index.jsonl 2>/dev/null | wc -l) ))"
done
