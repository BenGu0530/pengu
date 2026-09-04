#!/usr/bin/env bash
# GRID-5 c2 slice runner with a LIVE shard governor — range-parameterized.
#
#   nohup bash grid5/run_c2_range.sh <N> <from> <to> [ctrlfile] > results/gait_sweep/c2_slices_rml2.log 2>&1 &
#
# Control file:  results/gait_sweep/SHARDS_RML2   (a single integer)
#   echo 14 > .../SHARDS_RML2   -> use all 14 shards (desktop free)
#   echo 8  > .../SHARDS_RML2   -> drop to 8 (someone needs the desktop)
#   echo 0  > .../SHARDS_RML2   -> pause (shards killed, governor keeps waiting)
# The governor reconciles every 20 s: kills + relaunches shards on a change.
# Cheap by design: slice CSVs are <=115,200 rows, so a bitmap-resume restart
# costs <1 s of reload — real-time adjustment is lossless (rows flush per line).
# Also acts as the watchdog: dead shards are relaunched automatically.
set -u
cd "$(dirname "$0")/.."
PY="$PWD/.sweep_venv/bin/python"
OUT=results/gait_sweep
CTRL="$OUT/${4:-SHARDS_RML2}"
export CONFIG=c2
[ -f "$CTRL" ] || echo 8 > "$CTRL"          # default: 60% (8 of 14)

launch() {  # $1 = phi, $2 = n shards
  for s in $(seq 0 $(($2 - 1))); do
    N_SHARDS=$2 SHARD_ID=$s nohup "$PY" -u grid5/slice_phi.py "$1" >> "$OUT/c2_$(printf phi%03d "$1")_run.log" 2>&1 &
  done
}
stopall() { pkill -f 'slice_phi[.]py' 2>/dev/null; sleep 2; }

for P in $(seq "${2:-30}" 10 "${3:-350}"); do
  TAG=$(printf "phi%03d" "$P")
  CSV="$OUT/sweep_grid5_c2_${TAG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
  [ ! -f "$CSV" ] && [ -f "$CSV.gz" ] && gunzip -kf "$CSV.gz"
  ROWS=0; [ -f "$CSV" ] && ROWS=$(($(wc -l < "$CSV") - 1))
  if [ "$ROWS" -ge 115200 ]; then echo "== $TAG complete ($ROWS), skip"; continue; fi
  "$PY" grid5/slice_phi.py "$P" initcsv >/dev/null
  CUR=$(cat "$CTRL" 2>/dev/null || echo 8); CUR=${CUR:-8}
  echo "== $(date '+%F %T') $TAG starting from $ROWS rows at $CUR shards"
  [ "$CUR" -gt 0 ] && launch "$P" "$CUR"
  while :; do
    sleep 20
    ROWS=$(($(wc -l < "$CSV") - 1))
    if [ "$ROWS" -ge 115200 ]; then
      stopall; gzip -kf "$CSV"
      echo "== $(date '+%F %T') $TAG COMPLETE ($ROWS rows)"
      break
    fi
    WANT=$(cat "$CTRL" 2>/dev/null || echo "$CUR"); WANT=${WANT:-$CUR}
    LIVE=$(pgrep -fc 'slice_phi[.]py' || true); LIVE=${LIVE:-0}
    if [ "$WANT" != "$CUR" ]; then
      echo "== $(date '+%F %T') $TAG governor: $CUR -> $WANT shards (rows $ROWS)"
      stopall; CUR=$WANT
      [ "$CUR" -gt 0 ] && launch "$P" "$CUR"
    elif [ "$CUR" -gt 0 ] && [ "$LIVE" -lt "$CUR" ]; then
      echo "== $(date '+%F %T') $TAG reviving ($LIVE/$CUR alive, rows $ROWS)"
      stopall; launch "$P" "$CUR"
    fi
  done
done
echo "== $(date '+%F %T') all c2 slices done on rml2"
