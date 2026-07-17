#!/usr/bin/env bash
# GRID-2 Stage B launcher: reboot-safe + resumable (the guard Stage B was missing when
# a reboot killed 12 bare-nohup workers at 68%).
# Idempotent: safe from cron '@reboot' and '*/10 * * * *'.
#   flock  -> no double-launch
#   pidfile + /proc liveness -> skip shards already running
#   per-shard .done sentinel  -> skip finished shards
# Resume is by 7-tuple inside the worker, so a killed shard loses nothing.
set -u
exec 9>/tmp/pengu_stageB.lock
flock -n 9 || exit 0

cd "$(dirname "$0")/.."          # repo root pengu_mujoco/
export PENGU_MODEL=v3
PY="${STAGEB_PY:-/home/ben/miniconda3/envs/mujoco/bin/python}"
N=16
SHARDS="${STAGEB_SHARDS:-$(seq 0 15)}"   # Mac finished 12-15, but its wrapped-angle rows
                                          # were dropped for recompute -> run all 16 here
CSV=results/gait_sweep/grid2_stageB_minmu.csv
LOG=results/gait_sweep/stageB_autoresume.log

[ -f "$CSV.done" ] && exit 0
[ -f results/gait_sweep/grid2_cleanwalkers.csv ] || \
  gunzip -kf results/gait_sweep/grid2_cleanwalkers.csv.gz

$PY physics/stage_b_minmu.py initcsv >> "$LOG" 2>&1

for s in $SHARDS; do
  [ -f "$CSV.shard${s}of${N}.done" ] && continue
  PIDF="results/gait_sweep/.stageB_shard${s}.pid"
  if [ -f "$PIDF" ] && [ -d "/proc/$(cat "$PIDF" 2>/dev/null)" ]; then
    continue
  fi
  N_SHARDS=$N SHARD_ID=$s nohup $PY physics/stage_b_minmu.py >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "$(date '+%F %T') launched stageB shard $s pid $!" >> "$LOG"
done
