#!/usr/bin/env bash
# Frozen queue. Do not edit while running -- bash reads a script by byte offset.
set -u
cd "$(dirname "$0")/.."
for g in e2_a e2_b e2_c; do
  echo "[queue] $g <- overnight/cfgs/$g.txt  $(date '+%F %T')" >> overnight/queue_e2.log
  bash overnight/run_gen_v4.sh "$g" "overnight/cfgs/$g.txt" > "overnight/$g.log" 2>&1
done
echo "[queue] all done $(date '+%F %T')" >> overnight/queue_e2.log
