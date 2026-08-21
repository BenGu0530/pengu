#!/bin/bash
# Ice-arm training pipeline (frozen protocol, post Gate 0 2026-08-21).
# Per seed, two stages (the validated recipe from the gate):
#   A: from scratch, fixed vx_cmd 0.47, 3M steps  -> learns balance + creep
#   B: warm-start from A + curriculum c2, 3M steps -> ramps speed
# mu ~ U(0.1, 0.4) per episode throughout (mode e2). Frozen eval afterwards.
# Usage: nohup bash rl_grid4/run_e2_arm.sh >> rl_grid4/runs/e2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
SEEDS="${SEEDS:-0 1 2 3}"
for s in $SEEDS; do
  echo "=== e2 seed $s stage A (scratch, cmd 0.47) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$s" --n-envs 8
  echo "=== e2 seed $s stage B (warm + curriculum) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$s" --n-envs 8 \
    --curriculum --init-from "rl_grid4/runs/e2_r2a1e1_s$s/ckpts/final.zip"
done
echo "=== e2 arm complete $(date) ==="
