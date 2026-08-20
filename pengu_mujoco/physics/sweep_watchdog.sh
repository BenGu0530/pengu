#!/usr/bin/env bash
# GRID-4 shard watchdog: revive a dead/partial sweep + rotate a 6-hourly local snapshot.
#
#   bash physics/sweep_watchdog.sh c5 [n_shards]        # one-shot check (safe from cron)
#   bash physics/sweep_watchdog.sh install c5 [n]       # add @reboot + every-10-min crontab
#
# - If fewer than n_shards workers are alive, kills the stragglers and relaunches via
#   run_sweep.sh (resume is by axis-tuple, so this never loses or duplicates work).
# - Refuses to relaunch onto a header-less CSV (the known resume-killer) — fix that first.
# - Every 6 h it rotates a local snapshot  <csv>.snap.gz  (protection against accidental
#   file damage; NOT a substitute for the final ship-back).
# - To stop a sweep ON PURPOSE without the watchdog fighting you:
#       touch results/gait_sweep/WATCHDOG_OFF
#   (remove the file to re-enable).
# WSL note: cron only runs if systemd is enabled (add [boot]\nsystemd=true to /etc/wsl.conf,
# then `wsl --shutdown` once) — otherwise use Windows Task Scheduler (see grid4_xps_memo.md).
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/
REPO="$PWD"

if [ "${1:-}" = "install" ]; then
  shift; CFG="${1:?usage: sweep_watchdog.sh install cN [n_shards]}"; N="${2:-}"
  NPFX=""; [ "${SWEEP_NICE:-0}" != "0" ] && NPFX="SWEEP_NICE=${SWEEP_NICE} "
  L1="@reboot sleep 90 && cd '$REPO' && ${NPFX}bash physics/sweep_watchdog.sh $CFG $N >> results/gait_sweep/watchdog.log 2>&1"
  L2="*/10 * * * * cd '$REPO' && ${NPFX}bash physics/sweep_watchdog.sh $CFG $N >> results/gait_sweep/watchdog.log 2>&1"
  ( crontab -l 2>/dev/null | grep -v sweep_watchdog ; echo "$L1"; echo "$L2" ) | crontab -
  echo "watchdog installed for $CFG:"; crontab -l | grep sweep_watchdog
  exit 0
fi

CFG="${1:?usage: sweep_watchdog.sh cN [n_shards]  |  sweep_watchdog.sh install cN [n]}"
CORES="$( { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; } )"
N="${2:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
CSV="results/gait_sweep/sweep_grid4_${CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"

[ -f results/gait_sweep/WATCHDOG_OFF ] && { echo "$(date '+%F %T') $CFG: WATCHDOG_OFF present, skipping"; exit 0; }
[ -f "$CSV.done" ] && { echo "$(date '+%F %T') $CFG: config complete (.done), skipping"; exit 0; }

LIVE=$(pgrep -f 'grid4_sweep[.]py' | wc -l | tr -d ' ')
if [ "$LIVE" -lt "$N" ]; then
  echo "$(date '+%F %T') $CFG: $LIVE/$N shards alive -> reviving"
  if [ -f "$CSV" ] && ! head -1 "$CSV" | grep -q '^freq,'; then
    echo "$(date '+%F %T') $CFG: CSV HAS NO HEADER — refusing to relaunch (repair per fleet memo pre-flight 4)"
    exit 1
  fi
  pkill -f 'grid4_sweep[.]py' 2>/dev/null; sleep 2
  GRID3_PY="${GRID3_PY:-$REPO/.sweep_venv/bin/python}" nice -n "${SWEEP_NICE:-0}" bash physics/run_sweep.sh "$CFG" "$N"
else
  echo "$(date '+%F %T') $CFG: ok ($LIVE/$N shards)"
fi

# 6-hourly local snapshot rotation
SNAP="$CSV.snap.gz"
if [ -f "$CSV" ] && { [ ! -f "$SNAP" ] || [ -n "$(find "$SNAP" -mmin +360 2>/dev/null)" ]; }; then
  gzip -c "$CSV" > "$SNAP.tmp" && mv "$SNAP.tmp" "$SNAP"
  echo "$(date '+%F %T') $CFG: snapshot rotated ($(du -h "$SNAP" | cut -f1), $(wc -l < "$CSV") lines)"
fi
