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
#SBATCH --output=logs/realism-%A_%a.out
#SBATCH --error=logs/realism-%A_%a.out

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

python grid6/realism_check.py \
    --cells-file "results/grid6_hw/filter_cells_${CFG}.csv" \
    --mu 0.12 0.45 \
    --variants ${VARIANTS//,/ } \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${SLURM_ARRAY_TASK_COUNT}"

echo "done $(date)"
