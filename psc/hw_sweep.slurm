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
python grid6/hw_sweep.py --mu "${MU}" "${CF[@]}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --of "${3:-${SLURM_ARRAY_TASK_COUNT}}"

echo "done $(date)"
