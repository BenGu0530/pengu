#!/bin/bash
# Build the Python environment for the C2 sweep on a Delta LOGIN node.
#
#   (in the tmux session named 'delta')
#   cd /projects/beht/bgu/pengu
#   bash delta/delta_setup.sh
#
# Run this on the login node, NOT inside a job: compute nodes have no outbound
# internet, so pip must happen here. Verified 2026-09-01: dt-login03 reaches
# pypi.org (HTTP 200).
#
# The venv lands in /projects/beht (persistent, 500G project quota, 304K used at
# setup time) -- never in /u/bgu, which is a different filesystem with a 100G
# per-user quota, and never in /work/*, whose purge policy is unverified.
#
# Delta has no $PROJECT environment variable (that is a Bridges-2 thing), so the
# location is spelled out below and overridable with DELTA_WORK.
#
# Costs zero allocation. Login nodes are free.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
DELTA_WORK="${DELTA_WORK:-/projects/beht/bgu}"
VENV="${DELTA_WORK}/mjx_venv"

echo "tree : $REPO"
echo "work : $DELTA_WORK"
echo "venv : $VENV"
echo

[ -f "$REPO/DELTA_ISOLATED_TREE" ] || {
    echo "ERROR: $REPO is not the isolated duplicate tree (no DELTA_ISOLATED_TREE)." >&2
    exit 1
}
[ -d "$DELTA_WORK" ] || mkdir -p "$DELTA_WORK"

# ---- pick a base python ---------------------------------------------------
# Delta has no anaconda3 module (that was the Bridges-2 route). Available as of
# 2026-09-01: system /usr/bin/python3 = 3.9.21, and module python/3.13.5-gcc13.3.1.
#
# Either works as long as "mujoco==3.8.1" has a wheel for it. cp39 wheels
# certainly exist (that is what the existing GRID-5 data was produced with);
# cp313 is likely but unverified, so if the pinned install fails on 3.13 this
# script retries once on the system 3.9 rather than leaving a half-built venv.
pick_python() {
    if [ -n "${DELTA_BASE_PY:-}" ]; then echo "$DELTA_BASE_PY"; return; fi
    if module load python/3.13.5-gcc13.3.1 2>/dev/null; then
        command -v python3
    else
        echo "(no python module; using system python3)" >&2
        echo /usr/bin/python3
    fi
}

build_venv() {   # build_venv <base-python>
    local py="$1"
    echo "base python: ${py} -> $(${py} --version 2>&1)"
    rm -rf "${VENV}"
    "${py}" -m venv "${VENV}"
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    python -m pip install -q --upgrade pip wheel

    # mujoco is PINNED to 3.8.1: grid5_sweep.py records mujoco_version in the manifest
    # and check_manifest() (:194-210) REFUSES to append rows under a manifest written
    # by a different version. The existing GRID-5 data is 3.8.1.
    #
    # matplotlib is NOT optional -- gait_sweep.py:30-32 imports it at module level, so
    # without it every shard dies at import with no rows written.
    python -m pip install -q "mujoco==3.8.1" numpy matplotlib
}

BASE_PY="$(pick_python)"
if ! build_venv "$BASE_PY"; then
    echo
    echo "install failed on ${BASE_PY} -- retrying on system python3 (3.9.x)" >&2
    BASE_PY=/usr/bin/python3
    build_venv "$BASE_PY"
fi

echo
echo "--- installed ---"
python - <<'EOF'
import mujoco, numpy, matplotlib
print("mujoco    ", mujoco.__version__, "(must be 3.8.1)")
print("numpy     ", numpy.__version__)
print("matplotlib", matplotlib.__version__)
assert mujoco.__version__ == "3.8.1", "WRONG MUJOCO -- manifest check will reject writes"
EOF

echo
echo "--- smoke: can the driver see the C2 grid? ---"
# Must print config=c2 kappa=0.0 com=1.2 and 115,200 rows. Writes nothing, costs nothing.
python "$HERE/c2_driver.py" 0 count

mkdir -p "$HERE/state/logs"

# ---- record the balance BEFORE anything is charged -------------------------
# The whole campaign rests on one unknown: whether `accounts` reports node-hours
# or core-hours (a factor of 128). It cannot be settled by reasoning, only by
# watching the balance move across a job of known size. That measurement needs a
# baseline taken before the first job -- take it now, while it is still free.
if command -v accounts >/dev/null 2>&1; then
    accounts | tee "$HERE/state/balance_at_setup.txt"
    echo "(baseline written to delta/state/balance_at_setup.txt)"
fi

cat <<EOF

Setup OK. Next:

  1. CALIBRATE FIRST. Every cost estimate depends on it, and it is the only thing
     that can tell you whether "Hours" means node-hours or core-hours:
       cd $REPO && sbatch delta/c2_calib.slurm

  2. Feed the results in, then start the campaign:
       echo <node-hour|core-hour>       > delta/state/unit.txt
       echo <cost-per-slice from above> > delta/state/est_per_slice.txt
       bash delta/c2_ctl.sh start 0

  3. Watch it:
       bash delta/c2_ctl.sh status

  Budget cap is $(cat "$HERE/state/budget.txt" 2>/dev/null || echo "unset") (delta/state/budget.txt),
  expressed in whatever unit delta/state/unit.txt names. It is deliberately tiny
  until calibration replaces the guess -- raise it then.
EOF
