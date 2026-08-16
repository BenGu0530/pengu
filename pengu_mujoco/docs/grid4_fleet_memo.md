# GRID-4 fleet memo — who runs what

Full spec: `docs/grid4_guide.md`. Protocol is FROZEN (commit `a22f80b`): base = hardened
`models/pengu1_31` (2.2724 kg), COM rungs {1.05, 1.20, 1.31} built in-memory at load,
μ axis {0.1, 0.3, 0.5, 0.7} ±5% rel., K=5, pose jitter, no mass jitter,
pass = survive ∧ heading>0.5 ∧ net_fwd>0.05 (slip recorded, not gated).
Do NOT change any parameter mid-fleet — one changed value = a different sweep.

## Assignment (one machine = one config = one CSV; no cross-machine merging)

| machine | now | after that |
|---|---|---|
| **Mac** (Ben's) | **c1** (κ=0, 1.05) — running | c5 (κ=2, 1.20) |
| **B** (Ryzen/WSL2, `grid4_machineB_memo.md`) | **c2** (κ=0, 1.20) — running, slowest box (~60 d) | — (gets help, see below) |
| **C** (XPS/WSL2, `grid4_xps_memo.md`) | **c3** (κ=0, 1.31) — running | c6 (κ=2, 1.31) |
| **D** (Linux, `grid4_machineD_memo.md`) | **c4** (κ=2, 1.05) | help finish c2 |

When a machine frees up and c2 is still far from done, it joins c2 with a DISJOINT
shard split (e.g. B keeps `N_SHARDS=11 SHARD_ID 0..10` as-is; helper runs
`N_SHARDS=33 SHARD_ID 11..32`-style non-overlapping params into its own copy, then
concat + dedupe on the 6 axis cols). Coordinate in this file before starting.

Each config = 1,818,000 rows (+1 header). Measured rates: Mac ~4.3k rows/h,
B ~1.24k, C ~3.96k — ETA ~17/60/19 days respectively; resume is automatic,
restarts are safe.

## New machine setup (Linux / Mac; Windows → use WSL2; Linux VM fine)

```bash
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
bash physics/run_sweep.sh c2          # machine C: c3   (2nd arg = shard count, default cores-2)
```

No mujoco python found → the launcher builds `.sweep_venv` and pip-installs
mujoco/numpy/cma automatically.

- Windows: install WSL2 (`wsl --install`), work inside `~/` (NOT `/mnt/c` — slow I/O),
  set Windows power to never sleep. **Native Windows does NOT work** (mujoco 3.8.x DLL
  fails to load — measured on machine C; WSL2 is mandatory).
- **WSL2 keepalive is REQUIRED**: with no terminal attached the distro shuts down within
  ~45 s and kills every shard (measured on machine B). Arm an anchor from Windows before
  walking away, and re-arm after every reboot/logoff:
  ```
  powershell -c "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden"
  ```
- VM: give it all cores, local virtual disk (no shared folders), host must not sleep.
- Mac: `caffeinate -dimsu &` to prevent sleep. Linux: disable suspend.

## Pre-flight (2 min, before leaving a machine unattended)

1. `git rev-parse --short HEAD` matches the other machines.
2. `CONFIG=cN python physics/grid4_sweep.py count` prints
   `cells=454500 mus=[0.1, 0.3, 0.5, 0.7] rows=1818000 K=5` (and the right config/kappa/com).
   (numpy ≥2.3 prints `np.float64(0.1)` — cosmetic, values identical.)
3. After launch: `wc -l` on the CSV grows; startup line in
   `results/gait_sweep/cN_run.log` shows `com=<target> ... mass=2.2724kg`.
   To verify COM/mass without waiting for the (buffered) log, run a zero-row probe:
   ```bash
   CONFIG=cN N_SHARDS=999999 SHARD_ID=999998 .sweep_venv/bin/python -u physics/grid4_sweep.py
   ```
   Expected slides: 1.05→−86.05 mm, 1.20→−31.37 mm, 1.31→+8.73 mm, always `mass=2.2724kg`.

Gotcha: pass `GRID3_PY` as an ABSOLUTE path (e.g. `$HOME/...`), not `$PWD/...` — it
expands before the script's `cd` (machine-C finding).

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

## Per-machine routine (compiled from the machine memos)

**B (WSL2, c2)** — details `grid4_machineB_memo.md`:
1. After every Windows reboot/logoff: re-arm the WSL anchor (command above), THEN
   relaunch: `GRID3_PY=$HOME/pengu/pengu_mujoco/.sweep_venv/bin/python bash physics/run_sweep.sh c2`.
2. Check-in: `pgrep -fc grid4_sweep.py` = 11; `wc -l` on the c2 CSV grows.
3. Ship-back at 1,818,000 rows per the block below.

**C (WSL2, c3)** — details `grid4_xps_memo.md`:
1. Same anchor + relaunch rule as B (`... run_sweep.sh c3`; expect 14 shards).
2. Check-in additionally watches for the shard-death incident: if `pgrep -fc` < 14,
   relaunch (safe), and note it in the memo — twice means it needs diagnosis.
3. Keep on AC; Modern-Standby power settings already applied (see memo).
4. After c3 ships: start c6.

**D (Linux, c4)** — details `grid4_machineD_memo.md`: no WSL anchor needed; disable
suspend once; daily `pgrep`/`wc -l`; after c4 ships, help c2 per the split above.

**Mac (c1)** — Claude-managed on Ben's machine (caffeinate + completion monitor armed);
after c1 ships: c5.

## Open items (Ben)

- pitch (torso forward-lean) axis: definition + full-cross vs staged — decides whether
  more configs are queued after batch 2. Does not block batches 1–2.
- RL (6 policies) and the mass-effect control: deferred until the table is in.
