#!/usr/bin/env bash
# rml2 continuation of GRID-5 c2 (NCSA Delta ran out of core-hours at phi=30 partial).
# Runs the remaining hip_phi slices sequentially with a REDUCED shard count so the
# desktop stays usable (Ben, 2026-09-02: 60% of shards -> 8 of 14).
#
#   nohup bash grid5/run_c2_slices.sh 8 > results/gait_sweep/c2_slices_rml2.log 2>&1 &
#
# Per slice: recover any shipped partial gz -> initcsv+manifest -> N shards -> wait
# -> verify 115,200 rows -> gzip. Slices already complete (local or shipped) skip.
# Delta claimed phi 0,10,20 (complete) + 30 (partial, we finish it); we go 30 -> 350.
set -u
cd "$(dirname "$0")/.."
N="${1:-8}"
PY="$PWD/.sweep_venv/bin/python"
OUT=results/gait_sweep
export CONFIG=c2

for P in $(seq 30 10 350); do
  TAG=$(printf "phi%03d" "$P")
  CSV="$OUT/sweep_grid5_c2_${TAG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
  # recover shipped slice gz (Delta partials/completes) if no local CSV yet
  if [ ! -f "$CSV" ] && [ -f "$CSV.gz" ]; then gunzip -kf "$CSV.gz"; fi
  ROWS=0; [ -f "$CSV" ] && ROWS=$(($(wc -l < "$CSV") - 1))
  if [ "$ROWS" -ge 115200 ]; then echo "== $TAG already complete ($ROWS rows), skip"; continue; fi
  echo "== $(date '+%F %T') slice $TAG: starting from $ROWS rows, $N shards"
  "$PY" grid5/slice_phi.py "$P" initcsv >/dev/null
  PIDS=()
  for s in $(seq 0 $((N - 1))); do
    N_SHARDS=$N SHARD_ID=$s nohup "$PY" -u grid5/slice_phi.py "$P" >> "$OUT/c2_${TAG}_run.log" 2>&1 &
    PIDS+=($!)
  done
  wait "${PIDS[@]}"
  ROWS=$(($(wc -l < "$CSV") - 1))
  if [ "$ROWS" -ge 115200 ]; then
    gzip -kf "$CSV"
    echo "== $(date '+%F %T') slice $TAG COMPLETE ($ROWS rows), gz ready"
  else
    echo "== $(date '+%F %T') slice $TAG ENDED SHORT ($ROWS rows) — rerun this script to resume"
  fi
done
echo "== $(date '+%F %T') all c2 slices done on rml2"
