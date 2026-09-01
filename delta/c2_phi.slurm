#!/bin/bash
# One hip_phi slice of GRID-5 C2 on a full Delta cpu node.
#
#   cd /projects/beht/bgu/pengu && sbatch delta/c2_phi.slurm 0
#
# Normally you do not run this by hand -- delta/c2_ctl.sh submits it and each job
# asks the controller for the next one, so the campaign advances on its own until
# the budget guard or the STOP flag halts it.
#
# COST: Delta charges 1 SU per core-hour; a node-exclusive hour is 128 SU
# (NCSA Delta user guide, Job Accounting). Charging is on elapsed time, so the
# --time headroom below is free unless it is used. A slice is 115,200 rows; at
# PSC's measured 754 rows/hr/core that is ~71 min => ~152 SU. C2 (com=1.20) falls
# more often than PSC's C1 (com=1.05) and EPYC 7763 is faster than 7742, so the
# real number should be lower -- delta/c2_calib.slurm measures it.
#
# Whole node rather than a shared slice: we want all 128 cores anyway, the price
# per core-hour is identical, and an exclusive node has no neighbour competing for
# memory bandwidth -- which for 128 concurrent MuJoCo processes is the binding
# resource. On PSC, 128 cores ran at 0.83x the per-core rate of 8 cores.

#SBATCH --job-name=c2phi
#SBATCH --account=beht-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=delta/state/logs/c2phi-%j.out
#SBATCH --error=delta/state/logs/c2phi-%j.out

set -uo pipefail

PHI="${1:?usage: sbatch c2_phi.slurm <phi>}"

# SLURM copies the batch script to a node-local spool dir and runs it from
# there, so ${BASH_SOURCE[0]} points at /var/spool/slurm/.../jobNNNN/ -- NOT at
# the submitted path. Use the submit directory, and verify it really is the
# isolated tree so a wrong CWD fails loudly instead of cryptically.
REPO="${C2_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$REPO/delta"
REPO_RESULTS="$REPO/pengu_mujoco/results/gait_sweep"
if [ ! -f "$REPO/DELTA_ISOLATED_TREE" ]; then
    echo "ERROR: '$REPO' is not the isolated tree (no DELTA_ISOLATED_TREE)." >&2
    echo "  Submit from the tree root:  cd /projects/beht/bgu/pengu && sbatch delta/c2_phi.slurm $PHI" >&2
    echo "  or set C2_REPO=/projects/beht/bgu/pengu" >&2
    exit 1
fi

echo "=== job ${SLURM_JOB_ID} : C2 slice phi=${PHI} ==="
echo "node     : $(hostname)"
echo "cores    : ${SLURM_CPUS_ON_NODE}"
echo "started  : $(date)"
echo "restarts : ${SLURM_RESTART_COUNT:-0}"
echo

T_START=$(date +%s)
bash "$HERE/c2_node.sh" "$PHI"
RC=$?
T_END=$(date +%s)

# ---- record what this job cost, NOW, while we still know ------------------
# The budget guard reads sacct, but slurmdbd records are purged eventually and
# the guard would then silently under-count and overspend. A job knows its own
# cost without help: AllocCPUS x elapsed. Written here it outlives sacct.
# (Slightly under-counts -- the job has not exited yet -- so treat it as a floor
# and reconcile against sacct while the record still exists.)
#
# Both units are recorded. Delta's documented charge unit is the core-hour, but
# recording node-hours too costs nothing and means a wrong assumption about the
# unit is recoverable from the log rather than requiring a re-run.
ELAPSED=$(( T_END - T_START ))
CORES="${SLURM_CPUS_ON_NODE:-128}"
NODES="${SLURM_JOB_NUM_NODES:-1}"
CORE_H=$(awk -v c="$CORES" -v e="$ELAPSED" 'BEGIN{printf "%.1f", c*e/3600.0}')
NODE_H=$(awk -v n="$NODES" -v e="$ELAPSED" 'BEGIN{printf "%.3f", n*e/3600.0}')
mkdir -p "$HERE/state"
echo "${SLURM_JOB_ID} ${PHI} ${CORES} ${NODES} ${ELAPSED} ${CORE_H} ${NODE_H} $(date -Iseconds) rc=${RC}" \
    >> "$HERE/state/costs.txt"

echo
echo "finished : $(date)   c2_node.sh rc=$RC"
echo "cost     : ${CORES} cores x ${ELAPSED}s = ${CORE_H} core-hours (= ${NODE_H} node-hours)"
echo "           recorded in delta/state/costs.txt"
echo "--- reconcile while sacct still has it ---"
echo "sacct -X -j ${SLURM_JOB_ID} -o JobID,AllocCPUS,AllocNodes,ElapsedRaw,Elapsed,State"

# ---- package the finished slice for retrieval -----------------------------
# Unlike Bridges-2 (which ships neither scp nor rsync) Delta has both, so the pull
# is a plain rsync -- but it still has to be initiated from the laptop, which sits
# behind CMU NAT and cannot be reached from here. Compressing means the pull moves
# ~5x less; the .csv stays in place so resume still works.
if [ "$RC" -eq 0 ]; then
    OUT="$REPO_RESULTS"
    CSV=$(ls "$OUT"/sweep_grid5_c2_phi$(printf '%03d' "$PHI")*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ]; then
        gzip -kf "$CSV" && echo "packaged : $(basename "$CSV").gz ($(du -h "$CSV.gz" | cut -f1))"
    fi
fi

if [ "$RC" -ne 0 ]; then
    echo
    echo "SLICE DID NOT COMPLETE (rc=$RC) -- NOT chaining to the next slice."
    echo "Submitting onward would bury the failure and spend allocation on a broken setup."
    echo "Investigate this log, then: bash delta/c2_ctl.sh next"
    exit "$RC"
fi

# Hand off. The controller re-checks the STOP flag and the budget before it
# submits anything, so this is a request, not a guarantee.
bash "$HERE/c2_ctl.sh" next
