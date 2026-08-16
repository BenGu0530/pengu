# GRID-4 fleet memo — who runs what

Full spec: `docs/grid4_guide.md`. Protocol is FROZEN (commit `a22f80b`): base = hardened
`models/pengu1_31` (2.2724 kg), COM rungs {1.05, 1.20, 1.31} built in-memory at load,
μ axis {0.1, 0.3, 0.5, 0.7} ±5% rel., K=5, pose jitter, no mass jitter,
pass = survive ∧ heading>0.5 ∧ net_fwd>0.05 (slip recorded, not gated).
Do NOT change any parameter mid-fleet — one changed value = a different sweep.

## Assignment (one machine = one config = one CSV; no cross-machine merging)

| batch | Mac (Ben's, running) | machine B | machine C |
|---|---|---|---|
| **1 (now)** | **c1** (κ=0, COM 1.05) | **c2** (κ=0, 1.20) | **c3** (κ=0, 1.31) |
| **2 (next)** | c4 (κ=2, 1.05) | c5 (κ=2, 1.20) | c6 (κ=2, 1.31) |

Each config = 1,818,000 rows (+1 header), ~2 weeks/machine at ~10 shards — slower
machines just take longer; resume is automatic, restarts are safe.

## New machine setup (Linux / Mac; Windows → use WSL2; Linux VM fine)

```bash
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
bash physics/run_sweep.sh c2          # machine C: c3   (2nd arg = shard count, default cores-2)
```

No mujoco python found → the launcher builds `.sweep_venv` and pip-installs
mujoco/numpy/cma automatically.

- Windows: install WSL2 (`wsl --install`), work inside `~/` (NOT `/mnt/c` — slow I/O),
  set Windows power to never sleep.
- VM: give it all cores, local virtual disk (no shared folders), host must not sleep.
- Mac: `caffeinate -dimsu &` to prevent sleep. Linux: disable suspend.

## Pre-flight (2 min, before leaving a machine unattended)

1. `git rev-parse --short HEAD` matches the other machines.
2. `CONFIG=cN python physics/grid4_sweep.py count` prints
   `cells=454500 mus=[0.1, 0.3, 0.5, 0.7] rows=1818000 K=5` (and the right config/kappa/com).
3. After launch: `wc -l` on the CSV grows; startup line in
   `results/gait_sweep/cN_run.log` shows `com=<target> ... mass=2.2724kg`.

## Watching

```bash
wc -l results/gait_sweep/sweep_grid4_cN_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
tail -f results/gait_sweep/cN_run.log
```

## Ship-back when a config completes (1,818,000 rows)

```bash
CSV=results/gait_sweep/sweep_grid4_cN_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
awk 'NF' "$CSV" > t && mv t "$CSV"                     # strip stray blank lines
# integrity: rows=1,818,001 incl header; all NF==12; unique 6-tuples = 1,818,000
gzip -kf "$CSV" && git add -f "$CSV.gz"
git commit -m "GRID-4 cN complete" && git push
```
Then start the machine's batch-2 config (`run_sweep.sh c4/c5/c6`).
No AI attribution in commit messages. Branch: `friction-experiments`.

## Open items (Ben)

- pitch (torso forward-lean) axis: definition + full-cross vs staged — decides whether
  more configs are queued after batch 2. Does not block batches 1–2.
- RL (6 policies) and the mass-effect control: deferred until the table is in.
