#!/usr/bin/env bash
# One-line, CPU-only launcher for the GRID-3 DR sweeps (k0dr / k2dr) on ANY fresh clone.
# No GPU needed. After `git clone && cd pengu_mujoco`, just run one line:
#
#   bash physics/run_sweep.sh            # default: k0dr (Gait 1, kappa=0)
#   bash physics/run_sweep.sh k2dr       # Gait 2 re-sweep (kappa=2)
#   bash physics/run_sweep.sh k2dr 8     # force 8 shards (default = cores-2)
#   GRID3_PY=/path/to/python bash physics/run_sweep.sh k2dr   # use a specific python
#
# It (1) finds a python that can import mujoco+numpy, or builds a local .sweep_venv and
# pip-installs mujoco/numpy/cma; (2) recovers any committed .csv.gz snapshot; (3) initcsv;
# (4) launches N shards. Resume is automatic by axis-tuple (already-done cells are skipped),
# and DR randomization is seeded by cell index, so any machine reproduces a cell exactly.
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/
export PENGU_MODEL=v3

JOB="${1:-k0dr}"
case "$JOB" in
  k0dr|K0DR|0) export KAPPA=0 ; TAGN=k0dr ;;
  k2dr|K2DR|2) export KAPPA=2 ; TAGN=k2dr ;;
  *) echo "usage: bash physics/run_sweep.sh [k0dr|k2dr] [n_shards]"; exit 2 ;;
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
  "$PY" -m pip install -q "mujoco>=3.8,<3.9" numpy cma
fi
echo "using python: $PY  ($("$PY" -c 'import mujoco;print("mujoco",mujoco.__version__)'))"

# ---- shard count: default = cores - 2 (>=1) ----
CORES="$( { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; } )"
N="${2:-$(( CORES > 3 ? CORES - 2 : 1 ))}"

CSV="results/gait_sweep/sweep_v3_grid3_${TAGN}_freq_hip_phi_leg_amp_hip_amp_hip_off.csv"
LOG="results/gait_sweep/${TAGN}_run.log"
mkdir -p results/gait_sweep

# recover partial CSV from committed snapshot if present and not yet unpacked
if [ ! -f "$CSV" ] && [ -f "$CSV.gz" ]; then
  gunzip -kf "$CSV.gz"; echo "recovered $CSV from committed .gz"
fi

"$PY" physics/grid3_dr_sweep.py initcsv >> "$LOG" 2>&1
CNT="$("$PY" physics/grid3_dr_sweep.py count 2>/dev/null)"
done0=0; [ -f "$CSV" ] && done0=$(($(wc -l < "$CSV") - 1))

for s in $(seq 0 $((N - 1))); do
  N_SHARDS=$N SHARD_ID=$s nohup "$PY" physics/grid3_dr_sweep.py >> "$LOG" 2>&1 &
done

echo "launched $TAGN (kappa=$KAPPA): $N shards, resuming from $done0 done cells"
echo "  $CNT"
echo "watch:  wc -l $CSV        # target 454500 (+1 header)"
echo "log:    tail -f $LOG"
echo "when done (454500 rows): gzip -kf \"$CSV\" && git add -f \"$CSV.gz\" && git commit -m \"$TAGN DR sweep complete\" && git push"
