#!/usr/bin/env bash
# Resume the k0 DR sweep on ANY machine (e.g. a fresh clone). PORTABLE: no hardcoded paths,
# no k2 gate, no flock. Recovers the CSV from the committed .gz, then launches N shards.
# Resume is automatic by axis-tuple (grid3_dr_sweep.py skips cells already in the CSV), and
# randomization is seeded by cell index, so a resumed cell reproduces the same mu/mass/pose
# EXACTLY on any machine.
#   GRID3_PY=/path/to/python bash physics/resume_k0dr.sh     # python needs mujoco 3.8.x + numpy
#   K0DR_N=8 GRID3_PY=... bash physics/resume_k0dr.sh        # different shard count is fine
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/
export PENGU_MODEL=v3
PY="${GRID3_PY:-python}"
N="${K0DR_N:-12}"
CSV="results/gait_sweep/sweep_v3_grid3_k0dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv"
LOG="results/gait_sweep/k0dr_autoresume.log"
mkdir -p results/gait_sweep

# recover the uncompressed CSV from the committed snapshot if it isn't here yet
if [ ! -f "$CSV" ] && [ -f "$CSV.gz" ]; then
  gunzip -kf "$CSV.gz"
  echo "recovered $CSV from committed .gz"
fi
"$PY" physics/grid3_dr_sweep.py initcsv >> "$LOG" 2>&1

done0=0; [ -f "$CSV" ] && done0=$(($(wc -l < "$CSV") - 1))
for s in $(seq 0 $((N - 1))); do
  N_SHARDS=$N SHARD_ID=$s nohup "$PY" physics/grid3_dr_sweep.py >> "$LOG" 2>&1 &
done
echo "resuming k0dr from $done0 / 454500 done cells; launched $N shards (PY=$PY)"
echo "watch:  wc -l $CSV"
