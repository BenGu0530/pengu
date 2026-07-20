# GRID-3 (kappa=0) — Mac runbook (shards 12-15)

One design/control cell of the co-design table:
- FIXED: penguV3, COM ratio 1.108 (current), foot gap current, floor mu=0.7,
  torso control = PID at **kappa=0** (Gait 1, world-upright torso).
- SWEPT: freq (1.00-2.00@0.01), hip_phi (0-350@10), leg_amp {85,95,105,115,125},
  hip_amp {12,16,20,24,28}, hip_off {10,20,30,40,50}  = 454,500 cells.
- Two extra readouts per cell: torso_roll_rms (Gait-1 benchmark, ~0 = held upright)
  and torso_sat_frac (fraction of walk window the +-4.1 N.m torso motor was clamped).

Linux runs shards 0-11 (started). Mac runs **12-15**. Disjoint by index % 16.

## Run (repo root pengu_mujoco/, branch fable/friction-experiments)
Needs mujoco + numpy (no cma). KAPPA and PENGU_MODEL are set below.
```bash
git pull
KAPPA=0 python physics/grid3_kappa_sweep.py count      # expect cells=454500
KAPPA=0 python physics/grid3_kappa_sweep.py initcsv
for s in 12 13 14 15; do
  N_SHARDS=16 SHARD_ID=$s KAPPA=0 PENGU_MODEL=v3 \
    nohup python physics/grid3_kappa_sweep.py \
    > results/gait_sweep/grid3_k0_mac_s$s.log 2>&1 &
done
```
Do NOT use run_grid3.sh on the Mac (flock//proc are Linux-only). Resume is automatic
(by axis-tuple): after any sleep/reboot just re-run the for-loop.

## Monitor
```bash
wc -l results/gait_sweep/sweep_v3_grid3_k0_freq_hip_phi_leg_amp_hip_amp_hip_off.csv
tail -2 results/gait_sweep/grid3_k0_mac_s12.log
```
Linux ~14.7k cells/h on 12 shards -> ~23 h for its 341k share. Mac's 114k on 4 shards
~1-1.5 days.

## Ship back when the 4 shards finish
```bash
CSV=results/gait_sweep/sweep_v3_grid3_k0_freq_hip_phi_leg_amp_hip_amp_hip_off.csv
gzip -kf $CSV && git add -f $CSV.gz
git commit -m "GRID-3 k0 Mac shards 12-15" && git push
```
Linux merges (concat + dedupe by the 5 axis cols, verify 454,500) then analyzes:
best kappa=0 gait in this cell + the torso_roll_rms / net_fwd / min_mu landscape.

## Next cells (later)
Same script, `KAPPA=0.5 / 1 / 1.5 / 2`; then repeat the whole thing per COM variant and
foot gap. Each cell is an independent CSV tagged by kappa.
