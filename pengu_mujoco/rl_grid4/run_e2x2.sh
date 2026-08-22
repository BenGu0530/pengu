#!/bin/bash
# 2x2 probe from the reward audit (2026-08-21), local Mac. Cells:
#   baseline a1p1 = existing runs/e2/s0 (frozen recipe, not re-run)
#   a2p1: crank band widened to (0.0 +- 1.9) [covers c6 command domain]
#   a1p0: stepping priors removed (swing=0 scrub=0)
#   a2p0: both
# Full two-stage recipe per cell, seed 0, mu-curriculum stage A at 0.4.
# Usage: nohup bash rl_grid4/run_e2x2.sh >> rl_grid4/runs/e2x2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
S="${SEED:-0}"

run_cell () {  # $1=cell name, $2=EVALARGS (crank band for eval), rest train args
  local cell="$1"; shift
  local EVALARGS="$1"; shift
  echo "=== e2x2 $cell stage A $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --mu-fixed 0.4 --name "e2x2/$cell/stageA" "$@"
  echo "=== e2x2 $cell stage B $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --curriculum --init-from "rl_grid4/runs/e2x2/$cell/stageA/ckpts/final.zip" \
    --name "e2x2/$cell/stageB" "$@"
  echo "=== e2x2 $cell eval $(date) ==="
  "$PY" rl_grid4/eval_grid4_policy.py "rl_grid4/runs/e2x2/$cell/stageB/ckpts/final.zip" \
    --repeats 5 $EVALARGS || true
}

run_cell a2p1 "--crank-band 0.0 1.9" --crank-band 0.0 1.9
run_cell a1p0 "" --rw swing=0.0 scrub=0.0
run_cell a2p0 "--crank-band 0.0 1.9" --crank-band 0.0 1.9 --rw swing=0.0 scrub=0.0
echo "=== e2x2 complete $(date) ==="
