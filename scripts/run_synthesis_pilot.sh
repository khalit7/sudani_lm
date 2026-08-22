#!/usr/bin/env bash
# Autonomous synthesis pilot chain (plan.md Part IV, sequencing step 6).
#
# Waits for the persona-card generation already running to finish, collects the cards
# (real-name scan included), plans the pilot queue (balanced mix, round-robin people), and
# starts the pilot generation. Every stage is resumable, so this script can be re-run after
# any interruption and continues where things stopped.
#
#   nohup bash scripts/run_synthesis_pilot.sh >> data/interim/synthetic/pilot.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[$(date +%H:%M)] waiting for card generation to finish..."
while true; do
    pending=$(uv run python -m src.synthesis.generate status 2>/dev/null \
              | awk '/card/ {print $5}')
    [ "${pending:-1}" = "0" ] && break
    sleep 120
done

echo "[$(date +%H:%M)] collecting cards"
uv run python -m src.synthesis.generate collect

cards=$(ls data/interim/synthetic/cards/*.md 2>/dev/null | wc -l)
echo "[$(date +%H:%M)] $cards cards ready"
if [ "$cards" -lt 30 ]; then
    echo "too few cards ($cards) — a card generation or scan problem needs a look; stopping"
    exit 1
fi

echo "[$(date +%H:%M)] planning pilot queue"
uv run python -m src.synthesis.generate plan-pilot --requests 10000

echo "[$(date +%H:%M)] starting pilot generation (resumable; usage windows are waited out)"
uv run python -m src.synthesis.generate run --concurrency 3

echo "[$(date +%H:%M)] pilot generation complete"
