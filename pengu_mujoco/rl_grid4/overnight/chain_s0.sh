#!/usr/bin/env bash
# Waits out the a1 ablation queue and the baseline, then starts S0 (band).
cd "$(dirname "$0")/.."
while true; do
  grep -q "all done" overnight/queue_e2.log 2>/dev/null \
    && grep -q "complete" overnight/e2_base.log 2>/dev/null \
    && [ "$(pgrep -cf 'train_grid4.py --mode e2')" -eq 0 ] && break
  sleep 120
done
echo "[chain] a1 queue + baseline done, starting S0 $(date '+%F %T')" >> overnight/queue_prog2.log
bash overnight/run_gen_v5.sh s0_band overnight/cfgs/s0_band.txt > overnight/s0_band.log 2>&1
echo "[chain] S0 done $(date '+%F %T')" >> overnight/queue_prog2.log
