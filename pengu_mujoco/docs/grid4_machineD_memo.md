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

**As actually deployed (2026-08-16).** The clone on this box is NOT at `~/pengu` — it is:

```
REPO=$HOME/Documents/ben_gu/ben_pengu/pengu/pengu_mujoco
```

16 cores → 14 shards. Python 3.13.5 has no system mujoco, so the launcher built
`.sweep_venv` (mujoco 3.8.1, numpy 2.5.2). Use `$REPO` everywhere `~/pengu/pengu_mujoco`
appears below.

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
3. Zero-row probe for the startup line (COM variant + mass check, writes no data rows —
   but it DOES create an empty CSV, so run it **after** the launcher, never before):
   ```bash
   CONFIG=c4 N_SHARDS=999999 SHARD_ID=999998 .sweep_venv/bin/python -u physics/grid4_sweep.py
   # expect: com=1.0500 slide=-86.05mm mass=2.2724kg
   ```
   Then check the header — `head -1 $CSV` must be the `freq,hip_phi,...` line, not a data
   row. See fleet memo pre-flight step 4 for why and how to repair. Probe-before-launch
   cost this machine its header on the first try (caught at 321 rows, repaired, relaunched).
4. Shard count check — `pgrep -fc grid4_sweep.py` over-counts (it matches the checking
   shell itself); use an exact-argv count instead:
   ```bash
   ps -eo args | grep -c "^$REPO/.sweep_venv/bin/python -u physics/grid4_sweep.py$"   # = 14
   ```
   and
   `wc -l results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv`
   grows over a couple of minutes.

## Daily check-in (30 s)

```bash
cd $HOME/Documents/ben_gu/ben_pengu/pengu/pengu_mujoco
ps -eo args | grep -c '^.*/\.sweep_venv/bin/python -u physics/grid4_sweep\.py$'   # = 14
wc -l results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
```
After the first hour, note the rows/h in this file for fleet planning (B measured
~112 rows/h/shard, C ~283 — expect this box in between depending on CPU).

## Restart (after reboot, or if shards died)

```bash
REPO=$HOME/Documents/ben_gu/ben_pengu/pengu/pengu_mujoco
cd $REPO
CSV=results/gait_sweep/sweep_grid4_c4_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
head -1 "$CSV"      # MUST be the freq,hip_phi,... header — if not, repair it FIRST
                    # (fleet memo pre-flight 4); relaunching headerless duplicates work
GRID3_PY=$REPO/.sweep_venv/bin/python bash physics/run_sweep.sh c4
# confirm: "resuming from <N> done rows", N = wc -l minus 1 (not -1, not 0)
```
To stop shards cleanly, kill by exact argv — a bare `pkill -f grid4_sweep.py` also
matches (and kills) the shell issuing it:
```bash
ps -eo pid,args | awk '$4=="physics/grid4_sweep.py"{print $1}' | xargs -r kill -TERM
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
