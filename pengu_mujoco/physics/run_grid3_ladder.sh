#!/usr/bin/env bash
# GRID-3 kappa LADDER: drive every torso-control cell (kappa 1.5 / 2 / 1 / 0.5) on THIS
# machine, one kappa at a time, reboot-safe and resumable. Self-contained: runs all 16
# shards here (no Mac split), so each kappa's master .done actually fires and the ladder
# can advance. Idempotent under cron @reboot + */10 (also a watchdog if the ladder dies):
#   @reboot            bash <repo>/physics/run_grid3_ladder.sh
#   */10 * * * *       bash <repo>/physics/run_grid3_ladder.sh
# Order front-loads the two Gait-2 lean points (kappa>1); override with GRID3_LADDER.
# k0 (Gait 1) is already done and is skipped by its .done sentinel.
set -u
cd "$(dirname "$0")/.."                       # repo root pengu_mujoco/
exec 9>/tmp/pengu_grid3_ladder.lock
flock -n 9 || exit 0                          # one ladder at a time; a live ladder wins

LADDER="${GRID3_LADDER:-1.5 2 1 0.5}"
# Self-contained here: N=12 shards cover the whole grid (modulo 12), master .done fires when
# all 12 land, and 12 of 24 threads stay free for other work. Override GRID3_N/GRID3_SHARDS
# to split across machines (then that machine must run 0..N-1 between them for master .done).
export GRID3_N="${GRID3_N:-12}"
export GRID3_SHARDS="${GRID3_SHARDS:-$(seq 0 11)}"

for K in $LADDER; do
  KTAG="k$(printf '%g' "$K" | tr '.' 'p')"
  CSV="results/gait_sweep/sweep_v3_grid3_${KTAG}_freq_hip_phi_leg_amp_hip_amp_hip_off.csv"
  [ -f "$CSV.done" ] && continue              # this kappa fully swept already
  # launch (and, each tick, re-arm any shard killed by a reboot) until master .done appears
  until [ -f "$CSV.done" ]; do
    KAPPA="$K" bash physics/run_grid3.sh      # idempotent: flock/pidfile/.done inside
    sleep 60
  done
done
