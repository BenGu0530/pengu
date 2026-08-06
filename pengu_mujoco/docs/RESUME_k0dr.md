# Resume the k0 DR sweep on another machine

The k0 domain-randomization re-sweep (`physics/grid3_dr_sweep.py`) is **resumable and
machine-independent**: resume is by axis-tuple (a cell already in the CSV is skipped), and
the per-trial randomization (mu ~ U(0.45,0.90), torso mass ±15%, initial-pose jitter) is
**seeded by cell index**, so a resumed cell reproduces the exact same mu/mass/pose on any
machine. A clone can pick up where this one left off.

## Steps

1. Clone the repo and checkout the branch:
   ```bash
   git clone <repo> && cd pengu_mujoco
   git checkout fable/friction-experiments
   ```
2. Get a Python with **mujoco 3.8.x + numpy** (the model + all code are committed, incl.
   `penguV3/scene.xml` + meshes). Note its path.
3. From repo root `pengu_mujoco/`:
   ```bash
   GRID3_PY=/path/to/python bash physics/resume_k0dr.sh
   ```
   This recovers the partial CSV from the committed `.gz`, then launches 12 shards; each
   skips cells already present. Watch progress:
   ```bash
   wc -l results/gait_sweep/sweep_v3_grid3_k0dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv
   # target: 454,500 (+1 header)
   ```
4. When it reaches 454,500, gzip + commit + push:
   ```bash
   CSV=results/gait_sweep/sweep_v3_grid3_k0dr_freq_hip_phi_leg_amp_hip_amp_hip_off.csv
   gzip -kf "$CSV" && git add -f "$CSV.gz"
   git commit -m "k0dr DR re-sweep complete (454,500)" && git push
   ```

## Notes
- `K0DR_N=<n>` changes shard count; any sharding works (resume keys on axis-tuple, not shard id).
- The committed `.csv.gz` is a periodic **snapshot**; re-running only fills the missing cells.
- The columns are: `freq,hip_phi,leg_amp,hip_amp,hip_off, pass_rate,surv_rate,net_fwd_mean,
  net_fwd_min,slip_mean,head_mean` (K=5 randomized repeats per cell, `DR_K` to change).
- Don't run two machines against the same output unless you later merge (concat + dedupe on
  the 5 axis columns).
- Deterministic single-mu sweeps k0 / k1.5 / k2 are already complete and committed as
  `results/gait_sweep/sweep_v3_grid3_{k0,k1p5,k2}_*.csv.gz` (454,500 × 29-col each).
