#!/bin/bash
# Burn the remaining allocation on C2, one job, slices back-to-back.
#
# Do not sbatch this by hand -- the whole point is that --time and --ntasks are
# computed from the LIVE balance, and this file's defaults are deliberately
# small so that a hand-submitted job cannot spend the account by accident:
#
#   bash delta/c2_ctl.sh burn          # show the plan, submit nothing
#   bash delta/c2_ctl.sh burn --go     # size it from the balance and submit
#
# The controller overrides --time and --ntasks-per-node on the sbatch command
# line, which wins over the directives below.
#
# COST MODEL: Delta charges 1 SU per core-hour on elapsed time, so a job of N
# cores for T hours costs exactly N*T -- there is no headroom to leave, and
# --time is therefore not a safety margin but the budget itself. This is the
# opposite of the usual advice to over-request time, and it is why the defaults
# here are 1 hour and 8 cores (8 core-hours) rather than something node-sized.

#SBATCH --job-name=c2burn
#SBATCH --account=beht-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=delta/state/logs/c2burn-%j.out
#SBATCH --error=delta/state/logs/c2burn-%j.out

set -uo pipefail

# SLURM copies the batch script to a node-local spool dir and runs it from there,
# so ${BASH_SOURCE[0]} points at /var/spool/slurm/.../jobNNNN/ -- NOT at the
# submitted path. Use the submit directory, and verify it really is the isolated
# tree so a wrong CWD fails loudly instead of cryptically.
REPO="${C2_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
HERE="$REPO/delta"
if [ ! -f "$REPO/DELTA_ISOLATED_TREE" ]; then
    echo "ERROR: '$REPO' is not the isolated tree (no DELTA_ISOLATED_TREE)." >&2
    echo "  Submit from the tree root:  cd /projects/beht/bgu/pengu && bash delta/c2_ctl.sh burn --go" >&2
    exit 1
fi

CORES="${SLURM_CPUS_ON_NODE:-8}"

echo "=== job ${SLURM_JOB_ID} : C2 BURN ==="
echo "node     : $(hostname)"
echo "cores    : ${CORES}"
echo "time cap : $(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | sed -n 's/^TimeLimit=//p' | head -1)"
echo "started  : $(date)"
echo "restarts : ${SLURM_RESTART_COUNT:-0}"
echo "budget   : this job may spend up to (cores x time limit) core-hours"
echo

bash "$HERE/c2_burn.sh" "$CORES"
RC=$?

echo
echo "finished : $(date)   c2_burn.sh rc=$RC"
echo "--- reconcile while sacct still has it ---"
echo "sacct -X -j ${SLURM_JOB_ID} -o JobID,AllocCPUS,ElapsedRaw,Elapsed,State"
echo "bash delta/c2_ctl.sh status"
echo "bash delta/c2_ctl.sh unit"

# No chaining. A burn job stops because the allocation is gone, and submitting
# another one is a decision about other people's budget, not an automatic step.
exit "$RC"
