# GRID-2 Stage B — Mac runbook (shards 12–15)

Stage B = min_mu ladder + torso_corr + foot roll/pitch on the 484,157 Stage-A clean
walkers. Linux runs shards 0–11 (started), Mac runs **12–15**. Both partition the SAME
committed clean-walker list by `index % 16`, so shards are disjoint — no collisions.

## Run (repo root `pengu_mujoco/`, branch `fable/friction-experiments`)
Needs `mujoco` + `numpy` only (no `cma`).

```bash
git pull
gunzip -kf results/gait_sweep/grid2_cleanwalkers.csv.gz   # -> grid2_cleanwalkers.csv (the shared input)
python physics/stage_b_minmu.py initcsv                    # header once (safe if exists)
for s in 12 13 14 15; do
  N_SHARDS=16 SHARD_ID=$s nohup python physics/stage_b_minmu.py \
    > results/gait_sweep/stageB_mac_s$s.log 2>&1 &
done
```
Do NOT run `initlist` on the Mac — use the committed `grid2_cleanwalkers.csv.gz` so both
machines index the identical list. (initlist would rebuild it from the merged CSV; same
result if you have grid2_merged.csv, but the committed list removes any doubt.)

## Monitor / ETA
```bash
wc -l results/gait_sweep/grid2_stageB_minmu.csv   # Mac's 4/16 ≈ 121,000 rows when done
tail -2 results/gait_sweep/stageB_mac_s12.log
```
Linux rate ≈ 14k cells/h on 12 shards → ~25 h for its 363k share. Mac's 121k share on 4
shards ≈ **~1–1.5 days**. Resume is automatic (resume-by-7-tuple): re-run the for-loop
after any sleep/reboot; finished cells are skipped.

## Ship back when the 4 shards finish
```bash
gzip -kf results/gait_sweep/grid2_stageB_minmu.csv
git add -f results/gait_sweep/grid2_stageB_minmu.csv.gz
git commit -m "Stage B Mac shards 12-15" && git push
```
Linux merges (concat + de-dupe by the 7 axis cols) → analysis: min_mu vs torso mode
(corr-classified) within speed bins, foot roll-vs-pitch single-peak, the landscape law.

## Output columns
`freq,hip_phi,leg_amp,hip_amp,torso_amp,torso_phi,hip_off, a_netfwd,a_single,a_mureq
(Stage-A refs), min_mu_to_walk, torso_corr, foot_roll_amp, foot_pitch_amp`
