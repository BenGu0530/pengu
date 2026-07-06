#!/usr/bin/env bash
# GRID-2 Stage A launcher: sharded, resumable, reboot-safe (clone of run_grid.sh pattern).
# Safe to call repeatedly (cron @reboot + */10): flock stops double-launch, pidfile
# /proc-liveness skips running shards, per-shard .done skips finished ones.
set -u
exec 9>/tmp/pengu_grid2.lock
flock -n 9 || exit 0

cd "$(dirname "$0")/.."          # repo root pengu_mujoco/
export PENGU_MODEL=v3
PY="${GRID2_PY:-/home/ben/miniconda3/envs/mujoco/bin/python}"
N=16
# which shards THIS machine owns (Mac takes 12-15 by hand, see docs/grid2_mac_memo.md)
SHARDS="${GRID2_SHARDS:-$(seq 0 11)}"
CSV=results/gait_sweep/sweep_v3_grid2_freq_hip_phi_leg_amp_hip_amp_torso_amp_torso_phi_hip_off.csv
LOG=results/grid2_autoresume.log

[ -f "$CSV.done" ] && exit 0

$PY physics/grid2_sweep.py initcsv >> "$LOG" 2>&1

for s in $SHARDS; do
  [ -f "$CSV.shard${s}of${N}.done" ] && continue
  PIDF="results/gait_sweep/.grid2_shard${s}.pid"
  if [ -f "$PIDF" ] && [ -d "/proc/$(cat "$PIDF" 2>/dev/null)" ]; then
    continue                      # this shard is alive
  fi
  N_SHARDS=$N SHARD_ID=$s nohup $PY physics/grid2_sweep.py \
      >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "$(date '+%F %T') launched grid2 shard $s pid $!" >> "$LOG"
done
