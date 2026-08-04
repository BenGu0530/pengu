#!/usr/bin/env bash
# k0 DR re-sweep launcher: sharded, resumable, reboot-safe. GATED on the k2 sweep finishing
# first, so it never oversubscribes CPU against the running kappa ladder. Idempotent under
# cron @reboot + */10 (flock + per-shard pidfile/.done). Runs grid3_dr_sweep.py (K=5 default:
# each cell scored over 5 randomized mu/mass/pose repeats).
#   bash physics/run_k0dr.sh                    # this machine's default shards 0-11 (N=12)
set -u
cd "$(dirname "$0")/.."                          # repo root pengu_mujoco/
K2_DONE="results/gait_sweep/sweep_v3_grid3_k2_freq_hip_phi_leg_amp_hip_amp_hip_off.csv.done"
[ -f "$K2_DONE" ] || exit 0                      # wait for k2 to finish before starting DR

exec 9>/tmp/pengu_k0dr.lock
flock -n 9 || exit 0

export PENGU_MODEL=v3
PY="${GRID3_PY:-/home/ben/miniconda3/envs/mujoco/bin/python}"
N="${K0DR_N:-12}"
SHARDS="${K0DR_SHARDS:-$(seq 0 11)}"
CSV="results/gait_sweep/sweep_v3_grid3_k0dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv"
LOG="results/gait_sweep/k0dr_autoresume.log"

[ -f "$CSV.done" ] && exit 0
$PY physics/grid3_dr_sweep.py initcsv >> "$LOG" 2>&1

for s in $SHARDS; do
  [ -f "$CSV.shard${s}of${N}.done" ] && continue
  PIDF="results/gait_sweep/.k0dr_shard${s}.pid"
  if [ -f "$PIDF" ] && [ -d "/proc/$(cat "$PIDF" 2>/dev/null)" ]; then
    continue
  fi
  N_SHARDS=$N SHARD_ID=$s nohup $PY physics/grid3_dr_sweep.py >> "$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "$(date '+%F %T') launched k0dr shard $s pid $!" >> "$LOG"
done
