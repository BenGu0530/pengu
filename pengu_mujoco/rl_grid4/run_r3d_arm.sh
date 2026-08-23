#!/bin/bash
# Definitive independent-seed arm of the KEPT recipe (Ben 2026-08-23):
#   REWARD r3d (hf=0.6, commanded residual, hips+torso), a2 band (0.0 +-1.9),
#   priors off (swing=0 scrub=0), single-stage from scratch, mu~U(0.1,0.4),
#   c2 curriculum, 3M steps. Seed 0 = runs/e2x2hf4b/a2p0 (already trained,
#   best-ckpt 2000k confirmed 16/20). This script adds seeds 1..7.
# Waits for the a2p0_ext probe (pid arg) to release the CPU first.
# Usage: nohup bash rl_grid4/run_r3d_arm.sh <wait_pid> >> rl_grid4/runs/r3d_arm.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=/opt/anaconda3/envs/pengu/bin/python
WAIT_PID="${1:-0}"

if [ "$WAIT_PID" -gt 0 ]; then
  echo "=== waiting for pid $WAIT_PID (a2p0_ext) $(date) ==="
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "=== pid $WAIT_PID done $(date) ==="
fi

for S in 1 2 3 4 5 6 7; do
  echo "=== r3d_arm s$S train $(date) ==="
  nice -n 10 "$PY" rl_grid4/train_grid4.py --mode e2 --seed "$S" --n-envs 8 \
    --curriculum --crank-band 0.0 1.9 --rw swing=0.0 scrub=0.0 \
    --name "r3d_arm/s$S"
  echo "=== r3d_arm s$S eval $(date) ==="
  "$PY" rl_grid4/eval_grid4_policy.py "rl_grid4/runs/r3d_arm/s$S/ckpts/final.zip" \
    --repeats 5 --crank-band 0.0 1.9 || true
done
echo "=== r3d_arm complete $(date) ==="
