#!/bin/bash
# STEP 0 -- calibration. Run this BEFORE committing any real allocation.
#
#   cd /projects/beht/bgu/pengu && sbatch delta/c2_calib.slurm
#
# Two things stand between here and a budget you can defend:
#
#   1. CONFIRMING THE CHARGE UNIT. `accounts` reports "431 Hours". The NCSA Delta
#      user guide (Job Accounting) says a node-exclusive hour costs 128 SUs
#      (128 cores x 1 hour), i.e. one Hour is one CORE-hour -- so 431 buys under
#      three of C2's thirty-six slices. That is the documented answer, not a
#      measured one, and PSC's most expensive lesson was exactly that documented
#      numbers can be wrong (its throughput doc was off by 6-20x). It could not be
#      cross-checked locally: sacct cannot see the other project members' jobs,
#      bgu has no job history, and TRESBillingWeights (CPU=1000,Mem=512G,
#      MAX_TRES) gives the weighting but not the conversion to "Hours". This job
#      is small enough to be affordable under EITHER reading, so running it is how
#      the documented answer gets confirmed before anything expensive is submitted.
#
#   2. THROUGHPUT ON THIS HARDWARE FOR THIS CONFIG. PSC measured 904 rows/hr/core
#      on 8 EPYC 7742 cores and 754 on 128 -- but that was C1 (com=1.05), the most
#      stable config in the set, whose trials run closest to the full 21,000 steps.
#      C2 (com=1.20) should fall more often and cost less. EPYC 7763 (Milan) is
#      also ~10-20% faster per core than 7742 (Rome). Both effects point the same
#      way and neither has been measured.
#
# PSC's most expensive lesson was estimating from documentation instead of
# measuring: a 2 SU calibration there moved the 34-slice total from "2,900-3,900"
# to 5,270 -- a 40% correction, bought before asking the PI for budget.
#
# COST: 8 cores x <=30 min. Under the node-hour reading that is ~0.03 Hours;
# under the core-hour reading ~4 Hours (under 1% of the 431 balance). Cheap
# either way, which is the point -- this job is affordable precisely BECAUSE it
# is small enough that being wrong about the unit does not matter yet.
#
# It also proves out the plumbing -- venv, model paths, CSV + manifest creation,
# shard coordination -- on 8 cores instead of discovering a typo on 128.

#SBATCH --job-name=c2calib
#SBATCH --account=beht-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=delta/state/logs/c2calib-%j.out
#SBATCH --error=delta/state/logs/c2calib-%j.out

set -uo pipefail

# SLURM copies the batch script to a node-local spool dir and runs it from
# there, so ${BASH_SOURCE[0]} points at /var/spool/slurm/.../jobNNNN/ -- NOT at
# the submitted path. Deriving the tree from it fails with a cryptic
# "can't open file .../c2_driver.py". Use the submit directory instead, and
# verify it really is the isolated tree so a wrong CWD fails loudly.
REPO="${C2_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$REPO/delta"
if [ ! -f "$REPO/DELTA_ISOLATED_TREE" ]; then
    echo "ERROR: '$REPO' is not the isolated tree (no DELTA_ISOLATED_TREE)." >&2
    echo "  sbatch was run from the wrong directory. Do:" >&2
    echo "    cd /projects/beht/bgu/pengu && sbatch delta/c2_calib.slurm" >&2
    echo "  or set C2_REPO=/projects/beht/bgu/pengu" >&2
    exit 1
fi
PHI=0
SAMPLE_SECONDS=900        # measure over 15 minutes of steady state
NSHARDS=8

PY="${DELTA_PY:-${DELTA_WORK:-/projects/beht/bgu}/mjx_venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

echo "=== calibration: C2 throughput on this node, and the charge unit ==="
echo "job    : ${SLURM_JOB_ID}"
echo "node   : $(hostname)"
lscpu 2>/dev/null | grep -E "^Model name|^CPU\(s\)|^Thread" | sed 's/^/  /'
echo "python : $PY"
"$PY" -c "import mujoco; print('  mujoco', mujoco.__version__)"
echo

# Balance at the start of the job. Compute nodes may not have `accounts`; if not,
# the baseline written by delta_setup.sh on the login node still serves.
echo "--- balance before (may be stale; NCSA's accounting can lag) ---"
command -v accounts >/dev/null 2>&1 && accounts || echo "  (accounts not available on compute nodes)"
echo

"$PY" "$HERE/c2_driver.py" "$PHI" initcsv || exit 1
rows() { "$PY" "$HERE/c2_driver.py" "$PHI" --rows; }

R0=$(rows); T0=$(date +%s)
echo "starting $NSHARDS shards from $R0 rows"
for s in $(seq 0 $((NSHARDS - 1))); do
    N_SHARDS=$NSHARDS SHARD_ID=$s nohup "$PY" -u "$HERE/c2_driver.py" "$PHI" \
        >>"$REPO/pengu_mujoco/results/gait_sweep/c2_calib.log" 2>&1 &
done

sleep "$SAMPLE_SECONDS"
R1=$(rows); T1=$(date +%s)
pkill -f "c2_driver.py $PHI" 2>/dev/null || true
sleep 2

DR=$((R1 - R0)); DT=$((T1 - T0))
echo
echo "=== THROUGHPUT ==="
echo "  rows produced : $DR in ${DT}s on $NSHARDS cores"
if [ "$DR" -le 0 ] || [ "$DT" -le 0 ]; then
    echo "  NO ROWS PRODUCED -- something is broken; do not submit the real jobs."
    exit 1
fi
awk -v dr="$DR" -v dt="$DT" -v n="$NSHARDS" 'BEGIN {
    per_core   = dr * 3600.0 / dt / n
    slice_core = 115200.0 / per_core          # core-hours for one 115,200-row slice
    slice_node = slice_core / 128.0           # same slice, expressed in node-hours
    printf "  rows/hr/core  : %.0f\n", per_core
    printf "\n  ONE SLICE (115,200 rows) COSTS:\n"
    printf "    %8.1f core-hours   -> %8.0f for all 36 slices\n", slice_core, slice_core * 36
    printf "    %8.2f node-hours   -> %8.1f for all 36 slices\n", slice_node, slice_node * 36
    printf "    %8.0f min wall clock on a full 128-core node\n", slice_core / 128.0 * 60.0
    printf "\n  REFERENCES\n"
    printf "    PSC C1 on EPYC 7742, 8 cores  : 904 rows/hr/core   (this node is %.2fx)\n", per_core / 904.0
    printf "    PSC C1 on EPYC 7742, 128 cores: 754 rows/hr/core\n"
    printf "    PSC C1 measured slice cost    : 156 core-hours\n"
    printf "    Mac M2 Max reference          : 2000 rows/hr/core  (this node is %.2fx)\n", per_core / 2000.0
    printf "\n    If this slice cost is well under 156 core-hours, the expected\n"
    printf "    \"C2 falls more than C1, so it is cheaper\" effect is real.\n"
}'

echo
echo "=== CHARGE UNIT -- finish this from the login node ==="
echo "This job cannot measure its own charge: SLURM records the charge after the"
echo "job exits, and NCSA's accounting may lag behind that (on PSC the equivalent"
echo "command lagged 1-2 HOURS and nearly caused a 128x misreading)."
echo
echo "Once this job has finished, from the login node:"
echo
echo "    bash delta/c2_ctl.sh unit"
echo
echo "and re-run it every ~30 min until the balance moves or an hour or two has"
echo "passed with no movement. What the two outcomes mean:"
echo
awk -v cores=8 -v dt="$DT" 'BEGIN {
    h = dt / 3600.0
    printf "    balance drops by ~%.1f  -> CORE-hours, as documented. This is the\n", cores * h
    printf "                              expected outcome. 431 buys only 2-3 of 36\n"
    printf "                              slices, so STOP: take the measured cost per\n"
    printf "                              slice above to the PI and ask for a real\n"
    printf "                              allocation before submitting c2_phi.slurm.\n"
    printf "    balance does not move  -> the documentation would be wrong and Hours\n"
    printf "                              would be NODE-hours (this job is %.3f of one,\n", cores * h / 128.0
    printf "                              below the integer resolution of accounts).\n"
    printf "                              Do not conclude that from one reading: wait\n"
    printf "                              out the lag first, then re-run this check.\n"
}'
echo
echo "    Cross-check with SLURM's own record, which does not lag:"
echo "      sacct -X -j ${SLURM_JOB_ID} -o JobID,AllocCPUS,AllocNodes,ElapsedRaw,AllocTRES%%60,State"
echo "      seff ${SLURM_JOB_ID}"
echo "      sshare -A beht-delta-cpu -o Account,User,RawUsage"
echo
echo "NOTE: these rows are real C2 data and are kept -- phi=0 resumes from here."
