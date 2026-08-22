#!/bin/bash
# 2x2 probe under REWARD r3 (r2 + hf high-freq residual penalty, w=0.5).
# Same single-stage protocol as run_e2x2.sh; r3 is the code default so no --rw
# needed. Cells identical: a1p1 / a2p1 / a1p0 / a2p0.
#   a1p1: frozen band + priors      (baseline, re-run under this protocol)
#   a2p1: crank band (0.0 +- 1.9)   [covers the c6 command domain]
#   a1p0: priors removed (swing=0 scrub=0)
#   a2p0: both
# Usage: nohup bash rl_grid4/run_e2x2_hf.sh >> rl_grid4/runs/e2x2hf.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
S="${SEED:-0}"

run_cell () {  # $1=cell, $2=EVALARGS, rest = train args
  local cell="$1"; shift
  local EVALARGS="$1"; shift
  echo "=== e2x2hf $cell (single-stage) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --curriculum --name "e2x2hf/$cell" "$@"
  echo "=== e2x2hf $cell eval $(date) ==="
  "$PY" rl_grid4/eval_grid4_policy.py "rl_grid4/runs/e2x2hf/$cell/ckpts/final.zip" \
    --repeats 5 $EVALARGS || true
}

run_cell a1p1 ""
run_cell a2p1 "--crank-band 0.0 1.9" --crank-band 0.0 1.9
run_cell a1p0 "" --rw swing=0.0 scrub=0.0
run_cell a2p0 "--crank-band 0.0 1.9" --crank-band 0.0 1.9 --rw swing=0.0 scrub=0.0
echo "=== e2x2hf complete $(date) ==="
