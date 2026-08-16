# GRID-4 machine D handoff — Linux box, config c4

Fleet assignment: `docs/grid4_fleet_memo.md`. Full spec: `docs/grid4_guide.md`.
Protocol FROZEN at `a22f80b` — do not change any sweep parameter.

**Assignment: c4 = kappa 2 (Gait 2), COM ratio 1.05.** After c4 ships, check the fleet
memo for the next task (plan: help finish machine B's c2 via disjoint shards).

## Setup (native Linux — no WSL gymnastics needed)

```bash
sudo apt install -y git python3-venv        # Debian/Ubuntu; skip if already present
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
bash physics/run_sweep.sh c4                # 2nd arg = shard count, default cores-2
```

The launcher auto-detects a python with mujoco, else builds `.sweep_venv` and installs
mujoco/numpy/cma/matplotlib. To use a specific python:
`GRID3_PY=/abs/path/to/python bash physics/run_sweep.sh c4` (ABSOLUTE path — `$PWD/...`
expands before the script's `cd`, machine-C finding).

Ubuntu 24.04 note (machine-C finding): if the venv build fails with an `ensurepip`
error, `sudo apt install python3.12-venv` (or bootstrap pip via get-pip.py, see
`docs/grid4_xps_memo.md` §2) and rerun the launcher.

## Keep-alive

- **Disable suspend** (server install: usually already off; desktop:
  `sudo systemctl mask sleep.target suspend.target hibernate.target` or GUI settings).
- `nohup`ed shards survive closing the terminal on a standard Linux. If your distro
  kills user processes on logout (`loginctl show-user $USER` → `Linger=no` AND
  KillUserProcesses=yes), run `loginctl enable-linger $USER` once.
- Machine must stay on AC power.

## Pre-flight (do all four before walking away)

1. `git rev-parse --short HEAD` — must match the fleet's current HEAD.
2. `CONFIG=c4 .sweep_venv/bin/python physics/grid4_sweep.py count` →
   `config=c4 kappa=2.0 com=1.05  cells=454500  rows=1818000  K=5  trials=9090000`
   (numpy ≥2.3 prints `np.float64(0.1)` in the mus list — cosmetic).
3. Zero-row probe for the startup line (COM variant + mass check, touches no data):
   ```bash
   CONFIG=c4 N_SHARDS=999999 SHARD_ID=999998 .sweep_venv/bin/python -u physics/grid4_sweep.py
   # expect: com=1.0500 slide=-86.05mm mass=2.2724kg
   ```
4. `pgrep -fc grid4_sweep.py` equals the shard count and
   `wc -l results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv`
   grows over a couple of minutes.

## Daily check-in (30 s)

```bash
cd ~/pengu/pengu_mujoco
pgrep -fc grid4_sweep.py            # = shard count; if lower, see Restart
wc -l results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
```
After the first hour, note the rows/h in this file for fleet planning (B measured
~112 rows/h/shard, C ~283 — expect this box in between depending on CPU).

## Restart (after reboot, or if shards died)

```bash
cd ~/pengu/pengu_mujoco
GRID3_PY=$HOME/pengu/pengu_mujoco/.sweep_venv/bin/python bash physics/run_sweep.sh c4
```
Resume is automatic by 6-axis tuple; interrupting at any moment is safe (fleet-verified:
SIGTERM leaves no torn rows).

## Ship-back at 1,818,000 rows

```bash
CSV=results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
awk 'NF' "$CSV" > t && mv t "$CSV"
# verify: wc -l = 1,818,001 (incl header); awk -F, 'NF!=12' finds nothing;
#         cut -d, -f1-6 | sort -u | wc -l = 1,818,000
gzip -kf "$CSV" && git add -f "$CSV.gz"
git commit -m "GRID-4 c4 complete" && git push
```
Branch `friction-experiments`; no AI attribution in commit messages.
Then check `docs/grid4_fleet_memo.md` for this machine's next assignment.
