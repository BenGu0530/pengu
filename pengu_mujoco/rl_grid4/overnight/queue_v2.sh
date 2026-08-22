#!/usr/bin/env bash
# Sequential generation queue. Each entry runs to completion (train -> select ->
# confirm -> render) before the next starts, so the machine holds 3 concurrent
# trainings and nothing oversubscribes.
set -u
cd "$(dirname "$0")/.."
for spec in "$@"; do
  gen="${spec%%:*}"; cfg="${spec#*:}"
  [ -d "overnight/$gen" ] && [ -f "overnight/$gen.log" ] && \
    grep -q complete "overnight/$gen.log" && { echo "[skip] $gen done"; continue; }
  echo "[queue] $gen <- $cfg  $(date '+%F %T')"
  bash overnight/run_gen_v2.sh "$gen" "$cfg" > "overnight/$gen.log" 2>&1
done
echo "[queue] all done $(date '+%F %T')"
