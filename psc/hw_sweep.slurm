#!/bin/bash
# hardware_c1 sweep on Bridges-2 RM-shared: grid6/hw_sweep.py (ff_sweep protocol, FF torso,
# 354 deg/s + 29 ms legs, 56 ms torso lag) at the drag-measured friction.
#
#   sbatch --array=0-255 psc/hw_sweep.slurm 0.12
#   sbatch --array=0-255 psc/hw_sweep.slurm 0.45
#
# COST: 111,540 cells x (1 held + 3 ff) = 446k rollouts per mu, ~1.2 s each, over 256
# single-core tasks ~ 35 min each -> ~150 RM SU per mu, ~300 for both. Ask Ben first.

#SBATCH --job-name=hwsweep
#SBATCH --account=cis250009p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --output=logs/hwsweep-%A_%a.out
#SBATCH --error=logs/hwsweep-%A_%a.out

set -euo pipefail
MU="${1:?mu, e.g. 0.12}"

CODE="${PROJECT}/pengu_hw"
source "${PROJECT}/mjx_venv/bin/activate"
cd "${CODE}/pengu_mujoco"
mkdir -p results/grid6_hw

export OMP_NUM_THREADS=1
echo "node $(hostname)  task ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  mu ${MU}  $(date)"

python grid6/hw_sweep.py --mu "${MU}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${SLURM_ARRAY_TASK_COUNT}"

echo "done $(date)"
