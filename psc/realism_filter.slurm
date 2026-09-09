#!/bin/bash
# GRID-5 realism filter on Bridges-2 RM-shared: re-run the passing cells of the c1 and c6
# maps (1.20-1.70 Hz, lists in pengu_mujoco/results/grid6_hw/filter_cells_c*.csv) at the
# drag-measured friction 0.12 / 0.45 with the 354 deg/s + 29 ms leg servo model, and for
# c6 also with the 56 ms torso lag on the kappa = 2 loop.
#
#   sbatch --array=0-255 psc/realism_filter.slurm c6 act,both
#   (c1 dropped 2026-09-08: the slide-tuned c1 is superseded by the hardware-model sweep)
#
# COST: one core per array task, billed on elapsed time. ~1.2 s per rollout, so
#   c1: 192,152 cells x 2 mu x 1 variant = 384k rollouts / 256 tasks ~ 30 min each
#   c6: 117,670 cells x 2 mu x 2 variants = 471k rollouts / 256 tasks ~ 37 min each
# Total ~285 RM SU if every task runs to its estimate. Ask Ben before submitting.

#SBATCH --job-name=realism
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
#SBATCH --output=logs/realism-%A_%a.out
#SBATCH --error=logs/realism-%A_%a.out

# Re-running a subset of failed shards: SLURM_ARRAY_TASK_COUNT becomes the
# size of the RESUBMITTED array, not the original one, so --array=13,23,81
# would make each task compute 1/3 of the grid instead of 1/256 -- silently
# the wrong cells. Pass the original width as the LAST POSITIONAL ARGUMENT.
# (An environment variable does not survive submission on Bridges-2 -- neither
# `OF=256 sbatch` nor `sbatch --export=ALL,OF=256` reaches the job.)
#   sbatch --array=13,23,81 psc/realism_filter.slurm ...
set -euo pipefail
CFG="${1:?config c1|c6}"
VARIANTS="${2:-act}"

CODE="${PROJECT}/pengu_hw"
source "${PROJECT}/mjx_venv/bin/activate"
cd "${CODE}/pengu_mujoco"
mkdir -p results/grid6_report

export CONFIG="${CFG}"
export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cfg ${CFG}  variants ${VARIANTS}  $(date)"

T_START=$(date +%s)
# -e would abort here on a non-zero exit and skip the record below,
# which is exactly the case worth recording. Capture it instead.
set +e
python grid6/realism_check.py \
    --cells-file "results/grid6_hw/filter_cells_${CFG}.csv" \
    --mu 0.12 0.45 \
    --variants ${VARIANTS//,/ } \
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
    "${SLURM_ARRAY_JOB_ID}" "${SLURM_ARRAY_TASK_ID}" "realism" \
    "${CFG}" "${VARIANTS}" \
    "${SLURM_CPUS_ON_NODE:-1}" \
    "$(awk -v c="${SLURM_CPUS_ON_NODE:-1}" -v e="$(( T_END - T_START ))" \
        'BEGIN{print c*e/3600}')" \
    "rc=${RC}" "$(date -Iseconds)" \
    >> "${CODE}/state/tasks.txt"

echo "done $(date)  rc=${RC}  $(( T_END - T_START ))s"
exit "${RC}"
