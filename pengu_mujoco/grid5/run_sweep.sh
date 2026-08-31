#!/usr/bin/env bash
# One-line, CPU-only launcher for the GRID-5 co-design sweeps (docs/grid5_design.md).
# Duplicated from physics/run_sweep.sh (GRID-4, untouched backup) and adapted.
#
#   bash grid5/run_sweep.sh c1         # c1..c6 = GRID-4 configs, c7..c10 = COM 1.10/1.40
#   bash grid5/run_sweep.sh c10 8      # config c10, force 8 shards (default = cores-2)
#   GRID3_PY=/path/to/python bash grid5/run_sweep.sh c1   # use a specific python
#
# (1) finds a python that can import mujoco+numpy or builds .sweep_venv; (2) recovers a
# committed .csv.gz snapshot; (3) initcsv + manifest; (4) launches N shards. Resume is
# by axis-tuple; randomization is seeded by (cell,mu,rep) -> machine-independent rows.
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/

JOB="${1:-c1}"
case "$JOB" in
  c[1-9]|c10)  export CONFIG="$JOB" ; TAGN="grid5_$JOB" ; SCRIPT=grid5/grid5_sweep.py
               # grid5-v2: map is deterministic, K fixed at 1 in grid5_sweep.py (no DR_K)
               CSV="results/gait_sweep/sweep_grid5_${JOB}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv" ;;
  *) echo "usage: bash grid5/run_sweep.sh [c1..c10] [n_shards]"; exit 2 ;;
esac

# ---- pick a python that can import mujoco, else build a local venv ----
pick_py() {
  for c in "${GRID3_PY:-}" "$PWD/.sweep_venv/bin/python" python3 python; do
    [ -z "$c" ] && continue
    if "$c" -c "import mujoco, numpy; assert mujoco.__version__.startswith('3.8')" >/dev/null 2>&1; then echo "$c"; return 0; fi
  done
  return 1
}
PY="$(pick_py || true)"
if [ -z "${PY:-}" ]; then
  echo "no python with mujoco found -> building .sweep_venv (one-time)"
  BASE="${GRID3_PY:-python3}"
  "$BASE" -m venv .sweep_venv
  PY="$PWD/.sweep_venv/bin/python"
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q "mujoco>=3.8,<3.9" numpy cma matplotlib
fi
echo "using python: $PY  ($("$PY" -c 'import mujoco;print("mujoco",mujoco.__version__)'))"

# ---- shard count: default = cores - 2 (>=1) ----
CORES="$( { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; } )"
N="${2:-$(( CORES > 3 ? CORES - 2 : 1 ))}"

LOG="results/gait_sweep/${TAGN}_run.log"
mkdir -p results/gait_sweep

# recover partial CSV from committed snapshot if present and not yet unpacked
if [ ! -f "$CSV" ] && [ -f "$CSV.gz" ]; then
  gunzip -kf "$CSV.gz"; echo "recovered $CSV from committed .gz"
fi

"$PY" "$SCRIPT" initcsv >> "$LOG" 2>&1
CNT="$("$PY" "$SCRIPT" count 2>/dev/null)"
done0=0; [ -f "$CSV" ] && done0=$(($(wc -l < "$CSV") - 1))

for s in $(seq 0 $((N - 1))); do
  N_SHARDS=$N SHARD_ID=$s nohup "$PY" -u "$SCRIPT" >> "$LOG" 2>&1 &   # -u: unbuffered, startup line reaches the log
done

echo "launched $TAGN: $N shards, resuming from $done0 done rows"
echo "  $CNT"
echo "watch:  wc -l $CSV"
echo "log:    tail -f $LOG"
echo "when done: awk 'NF' \"$CSV\" > t && mv t \"$CSV\" && gzip -kf \"$CSV\" && split -b 90m -d \"$CSV.gz\" \"$CSV.gz.part\" && git add -f \"$CSV.gz.part\"* \"${CSV%.csv}.manifest.json\" && git commit && git push   # gz >100MB exceeds GitHub; reassemble: cat <csv>.gz.part* > <csv>.gz. Confirm branch with Ben first"
