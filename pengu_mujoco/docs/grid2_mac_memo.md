# GRID-2 Stage A — Mac runbook (shards 12–15)

Shard split: **Linux box owns shards 0–11 (already running), Mac owns 12–15.**
Disjoint by construction (`global_index % 16`), so nothing can collide; each machine
grows its own copy of the CSV and we merge at the end.

## Run (from repo root `pengu_mujoco/`, branch `fable/friction-experiments`)
Requirements: python env with `mujoco` + `numpy` only (**no `cma` needed**).
`PENGU_MODEL=v3` is set by the script itself.

```bash
git pull
python physics/grid2_sweep.py count      # sanity: cells=2225232
python physics/grid2_sweep.py initcsv    # writes header once (safe if exists)
for s in 12 13 14 15; do
  N_SHARDS=16 SHARD_ID=$s nohup python physics/grid2_sweep.py \
    > results/grid2_mac_s$s.log 2>&1 &
done
```

Do NOT use `physics/run_grid2.sh` on the Mac — it relies on `flock` and `/proc`
(Linux-only). The four commands above are the whole thing.

## Monitor
```bash
wc -l results/gait_sweep/sweep_v3_grid2_*.csv   # Mac's 4/16 share ≈ 556,000 rows + header
tail -2 results/grid2_mac_s12.log
ls results/gait_sweep/*.shard1[2-5]of16.done    # appears as each shard finishes
```
ETA on an M-series: ~1.5–2 days for the 4 shards (≈139k cells/shard, ~1 s/cell).
Resume is automatic: if the Mac sleeps/reboots, just re-run the same for-loop —
finished cells are skipped (resume by axis-tuple), the shard picks up where it left.

## When the 4 shards are done — ship the data back
The CSV is git-ignored and ~70 MB; commit the compressed copy:
```bash
gzip -k results/gait_sweep/sweep_v3_grid2_freq_hip_phi_leg_amp_hip_amp_torso_amp_torso_phi_hip_off.csv
git add -f results/gait_sweep/sweep_v3_grid2_*.csv.gz
git commit -m "GRID-2 Mac shards 12-15 complete (csv.gz)" && git push
```
Linux side merges (concat + de-dupe by the 7 axis columns) and runs Stage B.
Note: the master `.done` sentinel never appears on either machine alone (each sees
only its own shard sentinels) — completion = all 16 `.shardNof16.done` exist across
both machines; the merge script checks row count = 2,225,232 instead.

## What this sweep is
Stage A of docs/minmu_grid_design.md: walkability of 2.22M gait cells on penguV3 at
μ=0.7 — freq 1.00–2.00@0.01, hip_phi@10°, leg{95,105,115}, hip{16,20,24},
torso_amp{0,10,20} (0 = upright slice; torso_phi collapsed there), torso_phi@45°,
hip_off{10,20,30,40}. Stage B (min_mu ladder on the clean walkers) runs after merge.
