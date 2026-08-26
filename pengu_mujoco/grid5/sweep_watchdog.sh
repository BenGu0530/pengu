#!/usr/bin/env bash
# GRID-5 shard watchdog: revive a dead/partial sweep + rotate a 6-hourly local snapshot.
# Duplicated from physics/sweep_watchdog.sh (GRID-4, untouched backup); crontab entries
# reference grid5/sweep_watchdog.sh so grid4 watchdog lines are never disturbed.
#
#   bash grid5/sweep_watchdog.sh c5 [n_shards]        # one-shot check (safe from cron)
#   bash grid5/sweep_watchdog.sh install c5 [n]       # add @reboot + every-10-min crontab
#
# To stop a sweep ON PURPOSE:  touch results/gait_sweep/WATCHDOG_OFF
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/
REPO="$PWD"

if [ "${1:-}" = "install" ]; then
  shift; CFG="${1:?usage: sweep_watchdog.sh install cN [n_shards]}"; N="${2:-}"
  NPFX=""; [ "${SWEEP_NICE:-0}" != "0" ] && NPFX="SWEEP_NICE=${SWEEP_NICE} "
  L1="@reboot sleep 90 && cd '$REPO' && ${NPFX}bash grid5/sweep_watchdog.sh $CFG $N >> results/gait_sweep/watchdog5.log 2>&1"
  L2="*/10 * * * * cd '$REPO' && ${NPFX}bash grid5/sweep_watchdog.sh $CFG $N >> results/gait_sweep/watchdog5.log 2>&1"
  ( crontab -l 2>/dev/null | grep -v 'grid5/sweep_watchdog' ; echo "$L1"; echo "$L2" ) | crontab -
  echo "grid5 watchdog installed for $CFG:"; crontab -l | grep 'grid5/sweep_watchdog'
  exit 0
fi

CFG="${1:?usage: sweep_watchdog.sh cN [n_shards]  |  sweep_watchdog.sh install cN [n]}"
CORES="$( { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; } )"
N="${2:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
CSV="results/gait_sweep/sweep_grid5_${CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"

[ -f results/gait_sweep/WATCHDOG_OFF ] && { echo "$(date '+%F %T') $CFG: WATCHDOG_OFF present, skipping"; exit 0; }
[ -f "$CSV.done" ] && { echo "$(date '+%F %T') $CFG: config complete (.done), skipping"; exit 0; }

LIVE=$(pgrep -f 'grid5_sweep[.]py' | wc -l | tr -d ' ')
if [ "$LIVE" -lt "$N" ]; then
  echo "$(date '+%F %T') $CFG: $LIVE/$N shards alive -> reviving"
  if [ -f "$CSV" ] && ! head -1 "$CSV" | grep -q '^freq,'; then
    echo "$(date '+%F %T') $CFG: CSV HAS NO HEADER — refusing to relaunch (repair per fleet memo pre-flight 4)"
    exit 1
  fi
  pkill -f 'grid5_sweep[.]py' 2>/dev/null; sleep 2
  GRID3_PY="${GRID3_PY:-$REPO/.sweep_venv/bin/python}" nice -n "${SWEEP_NICE:-0}" bash grid5/run_sweep.sh "$CFG" "$N"
else
  echo "$(date '+%F %T') $CFG: ok ($LIVE/$N shards)"
fi

# 6-hourly local snapshot rotation (find -mmin: portable across Linux/Mac)
SNAP="$CSV.snap.gz"
if [ -f "$CSV" ] && { [ ! -f "$SNAP" ] || [ -n "$(find "$SNAP" -mmin +360 2>/dev/null)" ]; }; then
  gzip -c "$CSV" > "$SNAP.tmp" && mv "$SNAP.tmp" "$SNAP"
  echo "$(date '+%F %T') $CFG: snapshot rotated ($(du -h "$SNAP" | cut -f1), $(wc -l < "$CSV") lines)"
fi
