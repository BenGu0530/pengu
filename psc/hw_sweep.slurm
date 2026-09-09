#!/bin/bash
# hardware-model sweep on Bridges-2 RM-shared: grid6/hw_sweep.py (ff_sweep protocol, FF
# torso locked to the leg phase, 354 deg/s hard cap on the legs, 56 ms torso lag, and for
# kappa=2 also the PID loop the robot ran) at the drag-measured friction 0.12 / 0.45.
#
#   sbatch --array=0-255 psc/hw_sweep.slurm c1 0.12      # c1 kappa 0  pengu1_05_hw_updated
#   sbatch --array=0-255 psc/hw_sweep.slurm c2 0.12      # c2 kappa 0  pengu1_20_hw_updated
#   sbatch --array=0-255 psc/hw_sweep.slurm c5 0.12      # c5 kappa 2  pengu1_20_hw_updated
#   sbatch --array=0-255 psc/hw_sweep.slurm c6 0.12      # c6 kappa 2  pengu1_31
#   ... and the same four with 0.45
#
# COST per (config, mu) with the margin-1 cell lists (freq 0.02, phi all 36, leg 70-130,
# hip 12-32, off 20-40; GRID-5 black regions pruned): c1 ~200k cells x 4 = ~265 SU,
# c2 ~240k x 4 = ~315 SU, ~1-1.3 h wall over 256 single-core tasks. Ask Ben first.

#SBATCH --job-name=hwsweep
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000M
#SBATCH --time=03:00:00
# NODE_FAIL is not rare here -- 74 of 1,280 tasks hit it in the first
# hardware batch (5.8%). Both scripts resume from their own shard file
# (hw_sweep reads the rows already written into `done`; realism_check
# reports "N already done"), so a requeued task picks up where it left
# off instead of redoing the shard.
#SBATCH --requeue
#SBATCH --output=logs/hwsweep-%A_%a.out
#SBATCH --error=logs/hwsweep-%A_%a.out

# Re-running a subset of failed shards: SLURM_ARRAY_TASK_COUNT becomes the
# size of the RESUBMITTED array, not the original one, so --array=13,23,81
# would make each task compute 1/3 of the grid instead of 1/256 -- silently
# the wrong cells. Pass the original width as the LAST POSITIONAL ARGUMENT.
# (An environment variable does not survive submission on Bridges-2 -- neither
# `OF=256 sbatch` nor `sbatch --export=ALL,OF=256` reaches the job.)
#   sbatch --array=13,23,81 psc/hw_sweep.slurm ...
set -euo pipefail
CFG="${1:?config c1|c2|c5|c6}"
MU="${2:?mu, e.g. 0.12}"

CODE="${PROJECT}/pengu_hw"
source "${PROJECT}/mjx_venv/bin/activate"
cd "${CODE}/pengu_mujoco"
mkdir -p results/grid6_hw

export CONFIG="${CFG}"
export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  mu ${MU}  $(date)"

# cell list from grid6/hw_mask.py (GRID-5 black regions pruned, margin 1); falls back to
# the full grid if the list is absent
CELLS="results/grid6_hw/${CFG}/cells_${CFG}_mu${MU/0./0}_r1.csv"      # 0.12 -> mu012, 0.45 -> mu045
[ -f "${CELLS}" ] && CF=(--cells-file "${CELLS}") || CF=()
echo "cells: ${CELLS} ($([ -f "${CELLS}" ] && wc -l < "${CELLS}" || echo none))"
T_START=$(date +%s)
# -e would abort here on a non-zero exit and skip the record below,
# which is exactly the case worth recording. Capture it instead.
set +e
python grid6/hw_sweep.py --mu "${MU}" "${CF[@]}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${3:-${SLURM_ARRAY_TASK_COUNT}}"

RC=$?
set -e
T_END=$(date +%s)

# Record what this task did, in the tree, now. Two SSH drops during the c1 and
# c5/c6 batches killed the watching session while the jobs ran on regardless;
# squeue and sacct are then the only witnesses, and sacct records are purged
# eventually. A line per task in the tree outlives both, and is what a merge
# should be checked against.
#
# 256 tasks append concurrently. Each line is well under PIPE_BUF, so O_APPEND
# writes do not interleave -- the same property the sweep already relies on for
# its per-row CSV flushes.
mkdir -p "${CODE}/state"
printf '%s %s %s %s %s %s %.1f %s %s\n' \
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "hwsweep" \
    "${CFG}" "${MU}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" \
        'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" \
    >> "${CODE}/state/tasks.txt"

echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"
