#!/bin/bash
# 2x2 probe, SINGLE-STAGE protocol (C7 adopted: no warm-start, no stage A/B).
# Each cell: from scratch, mu ~ U(0.1,0.4), vx-curriculum c2 (cmd 0.12->0.47),
# 3M steps, seed 0, frozen eval after. Cells:
#   a1p1: frozen band + priors      (baseline, re-run under this protocol)
#   a2p1: crank band (0.0 +- 1.9)   [covers the c6 command domain]
#   a1p0: priors removed (swing=0 scrub=0)
#   a2p0: both
# Usage: nohup bash rl_grid4/run_e2x2.sh >> rl_grid4/runs/e2x2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
S="${SEED:-0}"

run_cell () {  # $1=cell, $2=EVALARGS, rest = train args
  local cell="$1"; shift
  local EVALARGS="$1"; shift
  echo "=== e2x2 $cell (single-stage) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --curriculum --name "e2x2/$cell" "$@"
  echo "=== e2x2 $cell eval $(date) ==="
  "$PY" rl_grid4/eval_grid4_policy.py "rl_grid4/runs/e2x2/$cell/ckpts/final.zip" \
    --repeats 5 $EVALARGS || true
}

run_cell a1p1 ""
run_cell a2p1 "--crank-band 0.0 1.9" --crank-band 0.0 1.9
run_cell a1p0 "" --rw swing=0.0 scrub=0.0
run_cell a2p0 "--crank-band 0.0 1.9" --crank-band 0.0 1.9 --rw swing=0.0 scrub=0.0
echo "=== e2x2 complete $(date) ==="
