#!/bin/bash
# GRID-5 c6 realism filter on the Mac: the 117,670 passing c6 cells (1.20-1.70 Hz) at the
# drag-measured friction 0.12 / 0.45, variants act (354 deg/s hard cap) and both (cap +
# 56 ms torso lag), torso PID kappa=2 clamp 45 as flashed. 471k rollouts, ~1.2 s each,
# over N single-core shards; resumable (each shard appends by cell) and mergeable.
#
#   bash grid6/run_realism_local.sh [shards=8]
#   CONFIG=c6 python grid6/realism_check.py --merge      # when every shard is done
#
# Runs at nice 10 so the Mac stays usable. Logs in results/grid6_report/realism_c6.<i>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-8}"
P=/opt/anaconda3/envs/pengu/bin/python
export CONFIG=c6 OMP_NUM_THREADS=1
mkdir -p results/grid6_report
for ((i = 0; i < N; i++)); do
  nohup nice -n 10 "$P" grid6/realism_check.py \
      --cells-file results/grid6_hw/filter_cells_c6.csv \
      --mu 0.12 0.45 --variants act both --shard "$i" --of "$N" \
      > "results/grid6_report/realism_c6.$i.log" 2>&1 &
done
echo "launched $N shards (pids: $(pgrep -f 'realism_check.py --cells-file' | tr '\n' ' '))"
echo "progress: tail -n1 results/grid6_report/realism_c6.*.log"
