#!/usr/bin/env bash
# One-line, CPU-only launcher for the GRID-4 co-design sweeps (docs/grid4_guide.md).
# No GPU needed. After `git clone && cd pengu_mujoco`, just run one line:
#
#   bash physics/run_sweep.sh c1         # config c1 (kappa=0, COM 1.05) ... c2..c6
#   bash physics/run_sweep.sh c4 8       # config c4, force 8 shards (default = cores-2)
#   GRID3_PY=/path/to/python bash physics/run_sweep.sh c1   # use a specific python
#
# It (1) finds a python that can import mujoco+numpy, or builds a local .sweep_venv and
# pip-installs mujoco/numpy/cma; (2) recovers any committed .csv.gz snapshot; (3) initcsv;
# (4) launches N shards. Resume is automatic by axis-tuple (already-done rows are skipped),
# and randomization is seeded by (cell,mu,rep), so any machine reproduces a row exactly.
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/

JOB="${1:-c1}"
case "$JOB" in
  # GRID-4 (docs/grid4_guide.md): c1..c6 = {kappa 0,2} x {COM 1.05, 1.20, 1.31}
  c[1-6])      export CONFIG="$JOB" ; TAGN="$JOB" ; SCRIPT=physics/grid4_sweep.py
               export DR_K="${DR_K:-1}"   # staged-K amendment: map at K=1 (override only for topup experiments)
               CSV="results/gait_sweep/sweep_grid4_${JOB}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv" ;;
  # legacy GRID-3 DR jobs (superseded by GRID-4; kept for resuming old data only)
  k0dr|K0DR)   export KAPPA=0 ; TAGN=k0dr ; SCRIPT=physics/grid3_dr_sweep.py
               CSV="results/gait_sweep/sweep_v3_grid3_k0dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv" ;;
  k2dr|K2DR)   export KAPPA=2 ; TAGN=k2dr ; SCRIPT=physics/grid3_dr_sweep.py
               CSV="results/gait_sweep/sweep_v3_grid3_k2dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv" ;;
  *) echo "usage: bash physics/run_sweep.sh [c1..c6|k0dr|k2dr] [n_shards]"; exit 2 ;;
esac

# ---- pick a python that can import mujoco, else build a local venv ----
pick_py() {
  for c in "${GRID3_PY:-}" python3 python; do
    [ -z "$c" ] && continue
    if "$c" -c "import mujoco, numpy" >/dev/null 2>&1; then echo "$c"; return 0; fi
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
echo "when done: awk 'NF' \"$CSV\" > t && mv t \"$CSV\" && gzip -kf \"$CSV\" && git add -f \"$CSV.gz\" && git commit -m \"GRID-4 $TAGN complete\" && git push"
