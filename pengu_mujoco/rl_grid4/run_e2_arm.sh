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
# mu-curriculum amendment 2026-08-21: stage A at fixed mu=0.4 (the easy end
# of the arm's range) — under full U(0.1,0.4) stage A dash-locked (1M steps,
# ep_len 16-18, fall 1.00). Stage B runs the full range.
for s in $SEEDS; do
  echo "=== e2 seed $s stage A (scratch, cmd 0.47, mu 0.4) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$s" --n-envs 8 \
    --mu-fixed 0.4 --name "e2/s$s/stageA"
  echo "=== e2 seed $s stage B (warm + curriculum, mu U(0.1,0.4)) $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$s" --n-envs 8 \
    --curriculum --init-from "rl_grid4/runs/e2/s$s/stageA/ckpts/final.zip" \
    --name "e2/s$s/stageB"
done
echo "=== e2 arm complete $(date) ==="
