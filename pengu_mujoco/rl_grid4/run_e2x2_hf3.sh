#!/bin/bash
# Round 3 (REWARD r3c): executed-signal HF penalty (act_filt cascade),
# hips+torso only, w=6.0. Full 2x2 re-run (a1 pricing changed vs r3 too).
# Usage: nohup bash rl_grid4/run_e2x2_hf3.sh >> rl_grid4/runs/e2x2hf3.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
S="${SEED:-0}"

run_cell () {
  local cell="$1"; shift
  local EVALARGS="$1"; shift
  echo "=== e2x2hf3 $cell (single-stage, r3c) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --curriculum --name "e2x2hf3/$cell" "$@"
  echo "=== e2x2hf3 $cell eval $(date) ==="
  "$PY" rl_grid4/eval_grid4_policy.py "rl_grid4/runs/e2x2hf3/$cell/ckpts/final.zip" \
    --repeats 5 $EVALARGS || true
}

run_cell a1p1 ""
run_cell a2p1 "--crank-band 0.0 1.9" --crank-band 0.0 1.9
run_cell a1p0 "" --rw swing=0.0 scrub=0.0
run_cell a2p0 "--crank-band 0.0 1.9" --crank-band 0.0 1.9 --rw swing=0.0 scrub=0.0
echo "=== e2x2hf3 complete $(date) ==="
