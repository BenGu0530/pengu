#!/usr/bin/env bash
# GRID-3 launcher: sharded, resumable, reboot-safe. Parametrized by KAPPA so the same
# launcher drives every torso-control cell (kappa 0 / 0.5 / 1 / 1.5 / 2).
# Idempotent (safe from cron @reboot + */10): flock, pidfile+/proc liveness, .done skips.
#   KAPPA=0 bash physics/run_grid3.sh          # this machine's default shards 0-11
#   GRID3_SHARDS="12 13 14 15" KAPPA=0 ...      # a helper machine's slice
set -u
export KAPPA="${KAPPA:-0}"
KTAG="k$(printf '%g' "$KAPPA" | tr '.' 'p')"
exec 9>"/tmp/pengu_grid3_${KTAG}.lock"
flock -n 9 || exit 0

cd "$(dirname "$0")/.."          # repo root pengu_mujoco/
export PENGU_MODEL=v3
PY="${GRID3_PY:-/home/ben/miniconda3/envs/mujoco/bin/python}"
N=16
SHARDS="${GRID3_SHARDS:-$(seq 0 11)}"
CSV="results/gait_sweep/sweep_v3_grid3_${KTAG}_freq_hip_phi_leg_amp_hip_amp_hip_off.csv"
LOG="results/gait_sweep/grid3_${KTAG}_autoresume.log"

[ -f "$CSV.done" ] && exit 0
$PY physics/grid3_kappa_sweep.py initcsv >> "$LOG" 2>&1

for s in $SHARDS; do
  [ -f "$CSV.shard${s}of${N}.done" ] && continue
  PIDF="results/gait_sweep/.grid3_${KTAG}_shard${s}.pid"
  if [ -f "$PIDF" ] && [ -d "/proc/$(cat "$PIDF" 2>/dev/null)" ]; then
    continue
  fi
  N_SHARDS=$N SHARD_ID=$s nohup $PY physics/grid3_kappa_sweep.py >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "$(date '+%F %T') launched grid3 $KTAG shard $s pid $!" >> "$LOG"
done
