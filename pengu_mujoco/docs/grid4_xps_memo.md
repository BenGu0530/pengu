# GRID-4 c3 — XPS 15 (machine C) status memo

Machine-C status report for the GRID-4 fleet. Assignment table: `docs/grid4_fleet_memo.md`.
Full spec: `docs/grid4_guide.md`. Protocol FROZEN at commit `a22f80b` — nothing below
changes a protocol parameter.

## Machine

| | |
|---|---|
| Host | Dell XPS 15 9530 — i7-13700H (6 P + 8 E = 14C/20T), 64 GB DDR5-4800 |
| OS | Windows 11 Home build 26200 + WSL2, Ubuntu 24.04.4 LTS |
| WSL limits | 16 vCPU, 24 GB, swap=0 (`C:\Users\bengu\.wslconfig`) |
| Repo path | `~/pengu` on WSL ext4 (NOT `/mnt/c` — per fleet memo) |
| Branch / HEAD | `friction-experiments` / `ccf088a` |
| Python | 3.12.3, venv at `pengu_mujoco/.sweep_venv` |
| Packages | mujoco **3.8.1**, numpy 2.5.2, cma 4.4.4, matplotlib |

## Assignment and launch

**c3 = kappa 0, COM ratio 1.31.** Launched 2026-08-15 23:21 EDT with 14 shards
(`N_SHARDS=14`, default `cores-2`). Shard 10 relaunched 23:26 (see Incidents).

## Pre-flight — all three items pass

1. **HEAD** — `ccf088a`, matches the fleet.
2. **count**
   ```
   config=c3 kappa=0.0 com=1.31  cells=454500  mus=[0.1, 0.3, 0.5, 0.7]
   rows=1818000  K=5  trials=9090000
   csv=sweep_grid4_c3_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
   ```
3. **Startup line**
   ```
   # GRID4 c3 (kappa=0.0 com=1.3100 slide=+8.73mm mass=2.2724kg)
     cells=454500  K=5  shard=10/14
   ```
   COM ratio and mass were also verified independently of the running shards by
   re-running `apply_com_variant(model, 1.31)` in a separate process:
   `com_ratio=1.3100  slide=+8.73mm  mass=2.2724kg`, xml = `models/pengu1_31/scene.xml`.

## Measured throughput

Two samples taken while all 14 shards were up and the load average was saturated
(~14 on 16 vCPU):

| sample | window | rows | rate |
|---|---|---|---|
| t+2 min | 90 s | +95 | ~3,800 rows/h |
| t+30 min | 120 s | +132 | ~3,960 rows/h |

At ~3,960 rows/h, the remaining 1,816,175 rows are **~458 h ≈ 19 days**. The fleet memo's
planning figure is ~2 weeks/machine at ~10 shards. Both samples are from the first half
hour; a longer sample after a full day would be firmer. Rate is recorded here as measured,
without adjustment.

## Environment deviations (none touch the protocol)

Recorded so machine B and any later Windows machine can skip the same debugging.

1. **Native Windows is not viable — WSL2 is mandatory, not a preference.**
   mujoco 3.8.0 and 3.8.1 both fail to load on Windows with
   `OSError: [WinError 1114] DLL initialization routine failed`, under numpy 1.x and 2.x,
   and with a PATH stripped of Anaconda. `mujoco.dll` is present (4.5 MB), both VC++
   runtimes are present, and the DLL has no extra same-directory dependencies. Only
   mujoco 3.1.4 (Anaconda's) loads — wrong version for the protocol. The same wheel
   version loads fine under WSL2.

2. **`ensurepip` is absent on Ubuntu 24.04** (it ships in the separate `python3.12-venv`
   package), so `run_sweep.sh`'s automatic venv build fails. Installing that package needs
   sudo. Worked around without sudo: `python3 -m venv --without-pip .sweep_venv`, then
   bootstrapped pip inside the venv from the official PyPA `get-pip.py`. Resulting venv is
   equivalent to what the launcher would have built.

3. **`run_sweep.sh` does not install matplotlib, but the sweep needs it.**
   Line 44 installs `mujoco numpy cma` only; `physics/gait_sweep.py:30` does
   `import matplotlib`, pulled in by `physics/grid4_sweep.py:33`. Every shard dies at
   startup with `ModuleNotFoundError: No module named 'matplotlib'` until it is installed
   by hand. Fix belongs in the launcher's pip line (or make the import lazy in
   `gait_sweep.py`).

4. **`GRID3_PY=$PWD/...` on the launcher command line expands too early** — it resolves
   against the shell's starting directory, not the repo root, so `pick_py` fails and the
   script falls into its "build venv" branch. It recovers, because after the script's own
   `cd` the fallback `PY="$PWD/.sweep_venv/bin/python"` points at the right interpreter,
   but the run emits a confusing `line 41: ... No such file or directory` first. Pass an
   absolute path instead.

5. **`.sweep_venv/` is not in `.gitignore`** and shows up as untracked in `git status`.

## Incidents

**Shard 10 died during startup; cause not identified.** 14 shards were launched, 13 were
alive one minute later (SHARD_IDs 0-9, 11-13 present; 10 absent). No traceback in
`c3_run.log`, no OOM record in `dmesg`, and memory use was 3.8 GB of 24 GB, so memory
pressure is ruled out but the actual signal is unknown. Relaunched at 23:26 as a single
shard; all 14 have been up since. Resume is by axis-tuple, so no rows were lost.
**Worth a process count on each check-in — if it recurs it needs a real diagnosis.**

**The log stays nearly empty; this is expected, not a stall.** `grid4_sweep.py` prints
progress to stdout, which is block-buffered when redirected to a file, and only the CSV
handle is flushed (`grid4_sweep.py:183`). Progress lines therefore appear in bursts of
several KB. Use the CSV row count, not the log, to judge progress. Shard 10 was relaunched
with `python -u`, so its lines appear immediately.

## Power

AC-only settings; the machine must stay plugged in for the duration.

| setting | before | now |
|---|---|---|
| system sleep (AC) | never | never |
| hibernate (AC) | never | never |
| disk timeout (AC) | 30 min | never |
| lid close (AC) | — | do nothing |

The lid-close setting is hidden from `powercfg /query` on this machine; it was set with
`/setacvalueindex` and confirmed via the registry (`ACSettingIndex=0`). This host uses
Modern Standby (S0); S1/S2 are unavailable.

## Monitor

```bash
cd ~/pengu/pengu_mujoco
wc -l results/gait_sweep/sweep_grid4_c3_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
pgrep -fc grid4_sweep.py        # expect 14
```

## Restart after a reboot or a stop

```bash
cd ~/pengu/pengu_mujoco
GRID3_PY=$HOME/pengu/pengu_mujoco/.sweep_venv/bin/python bash physics/run_sweep.sh c3
```

Resume is automatic by axis-tuple; already-completed rows are skipped.

## Ship-back

Unchanged from `docs/grid4_fleet_memo.md` — at 1,818,000 rows, strip blank lines, verify
1,818,001 lines incl. header / all `NF==12` / 1,818,000 unique 6-tuples, gzip,
`git add -f`, commit, push. Then this machine starts its batch-2 config.

## Open items

- Longer throughput sample after 24 h, to replace the ~19 day figure with a firmer one.
- Shard-10 death: recurrence check on every visit; diagnose if it happens again.
- Launcher fixes (matplotlib dependency, `.gitignore` entry) — not applied here, since
  editing the launcher mid-fleet touches a shared file.
