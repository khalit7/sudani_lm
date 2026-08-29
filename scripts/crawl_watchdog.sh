#!/bin/bash
# Emits one line per STATE CHANGE only: a crawler dying without its completion marker,
# stalling (once, until progress resumes), or completing. Silence = healthy.
cd /home/khalid/sudani_lm
declare -A state prev_count stall_seen
while true; do
  sleep 600
  while IFS='|' read -r name pattern dir donemark; do
    [ -z "$name" ] && continue
    count=$(find "$dir" -type f 2>/dev/null | wc -l)
    alive=$(pgrep -fc "$pattern")            # pgrep -c prints 0 on no match
    alive=${alive:-0}
    log="$(dirname "$dir")/crawl.log"; [ -f "$dir/../crawl.log" ] && log="$dir/../crawl.log"
    if [ "$alive" -eq 0 ] 2>/dev/null; then
      if tail -5 "$log" 2>/dev/null | grep -qiE "$donemark"; then
        [ "${state[$name]}" != done ] && echo "INFO $name: completed ($count files)"
        state[$name]=done
      else
        [ "${state[$name]}" != dead ] && echo "PROBLEM $name: process gone without completion ($count files)"
        state[$name]=dead
      fi
    else
      if [ "$count" = "${prev_count[$name]}" ]; then
        if [ "${stall_seen[$name]}" = pending ]; then
          echo "PROBLEM $name: alive but stalled at $count files for 20+ min"
          stall_seen[$name]=alerted
        elif [ "${stall_seen[$name]}" != alerted ]; then
          stall_seen[$name]=pending
        fi
      else
        stall_seen[$name]=""
      fi
      state[$name]=alive
    fi
    prev_count[$name]=$count
  done <<'LIST'
anasudani|scrape_anasudani.py fetch|data/raw/anasudani/html|FETCH LOOP FINISHED
alnilin|scrape_alnilin|data/raw/alnilin/posts|complete at page
sudanile|scrape_wpjson.*sudanile|data/raw/sudanile/posts|complete at page
aghaniwamthal|scrape_small_sites.*aghaniwamthal|data/raw/small_sites/aghaniwamthal|done —
wayback|scrape_wayback|data/raw/wayback|WADMADANI WAYBACK DONE|
LIST
done
