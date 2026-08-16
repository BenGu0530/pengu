# GRID-4 c3 — XPS 15 (machine C) status memo

Machine-C status report for the GRID-4 fleet. Assignment table: `docs/grid4_fleet_memo.md`.
Full spec: `docs/grid4_guide.md`. Protocol FROZEN at commit `a22f80b` — nothing below
changes a protocol parameter, a sweep axis, or a scoring rule.

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
(`N_SHARDS=14`, default `cores-2`).

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

All samples taken with 14 shards up and load average saturated (~14 on 16 vCPU):

| sample | window | rows | rate |
|---|---|---|---|
| t+2 min | 90 s | +95 | ~3,800 rows/h |
| t+30 min | 120 s | +132 | ~3,960 rows/h |
| **t+72 min (cumulative)** | **72 min** | **4,127** | **~3,440 rows/h** |

The two short windows land high; the cumulative hour-scale figure is **~3,440 rows/h**,
which puts the remaining rows at **~527 h ≈ 22 days**. Use the cumulative number. A
24 h sample is still the one to trust and is listed under open items. Rates are recorded
as measured, without adjustment.

## Bug found: `grid4_sweep.py` crashes on resume (fleet-wide)

Found on this machine while testing keepalive recovery. **It affects every machine in the
fleet**, not just this one.

```
NameError: cannot access free variable 'row' where it is not associated with a value
  physics/grid4_sweep.py:186
```

**Mechanism.** `row` is bound inside the inner `for mi, mu0 in enumerate(MUS)` loop
(line 175). The resume guard `if key in done: continue` (line 157) can skip all four mu
values of a cell, leaving `row` unbound. The periodic progress print at line 184
(`if n_mine % 25 == 0:`) then dereferences `row` unconditionally and the shard dies.

**Why it only bites on resume.** On a first launch `done` is empty, nothing is skipped,
so `row` is bound on the very first cell and the print is always safe. On a resume, a
shard skips its already-finished cells; if the first 25 cells it owns are all complete,
`row` is never bound and it crashes within seconds of start.

**Blast radius.** Each shard owns ~32,464 cells and walks them in `i % n_shards` order,
so once a config is more than `25 x n_shards` cells in, *every* shard crashes on *every*
restart. c3 passed that threshold within the first few minutes. Left unfixed this turns
any keepalive into a crash loop: shards restart, die in seconds, CSV stops growing. The
same code runs c1 on the Mac and will run c2 on machine B — neither has restarted yet,
which is the only reason it has not shown up there.

**Fix** (2 lines, logging branch only — no computation path, no protocol value, no effect
on any row already written):

```python
    n_mine = 0
    row = None                                    # initialise

        if n_mine % 25 == 0 and row is not None:  # guard
```

**Status: applied and committed on this machine only, NOT pushed.** The fleet still
carries the bug. Verified here by killing shard 5 and letting the watchdog resume it at
`done=5061`: pre-fix it died in ~3 s, post-fix it ran through the skip path and kept
going. Whole-run log shows exactly one `NameError`, the pre-fix one.

## Keepalive (this machine)

Three layers. Scripts are local to this machine and deliberately not in the repo.

| layer | what | covers |
|---|---|---|
| 1 | `~/bin/run_c3_watchdog.sh` | a shard dying silently |
| 2 | user crontab: `@reboot` + `*/10 * * * *` | WSL restart, shard loss within 10 min |
| 3 | Windows task `PenguGrid4C3Keepalive` (at logon + every 30 min, via a hidden-window VBS) | Windows reboot — starts WSL, then the watchdog |

The watchdog follows the `run_grid2.sh` idiom (flock, per-shard `.done` skip) with one
change: **liveness is read from `/proc/<pid>/environ` (`SHARD_ID` + `CONFIG`) rather than
from pidfiles**, because the shards launched by `run_sweep.sh` write no pidfile. A
pidfile-only check would have seen 14 absent shards and launched 14 duplicates, and a
duplicated `SHARD_ID` writes duplicate rows — which breaks the unique-6-tuple ship-back
check.

Verified: with 14 shards alive the watchdog starts 0; after `kill -9` on shard 5 it
restarts exactly shard 5; the Windows task returns `LastTaskResult: 0` and adds no
duplicates.

**Limitation.** The Windows trigger is *at logon*. After an unattended reboot (e.g. a
forced Windows Update restart) the machine sits at the lock screen and nothing runs until
someone logs in. Making it run without logon requires storing account credentials in the
task, which was deliberately not done.

## Incidents

**Shard 10 died during startup on the first launch; cause still unidentified.** 14 were
launched, 13 were alive a minute later (SHARD_IDs 0-9, 11-13). No traceback in
`c3_run.log`, no OOM in `dmesg`, memory use 3.8 GB of 24 GB. Note this is **not** the
row-guard bug above: `done` was empty at first launch, so nothing was skipped and `row`
was bound normally. Relaunched by hand at 23:26. Still worth a shard count on each
check-in; the watchdog now covers recurrence automatically.

**The run log stays nearly empty; this is expected, not a stall.** `grid4_sweep.py` prints
progress to stdout, block-buffered when redirected to a file, and only the CSV handle is
flushed (line 183). Progress appears in multi-KB bursts. Judge progress by CSV row count.
Shards started by the watchdog use `python -u`, so their lines appear immediately.

## Power

AC-only settings; the machine must stay plugged in for the duration.

| setting | before | now |
|---|---|---|
| system sleep (AC) | never | never |
| hibernate (AC) | never | never |
| disk timeout (AC) | 30 min | never |
| lid close (AC) | — | do nothing |

The lid-close setting is hidden from `powercfg /query` on this host; it was set with
`/setacvalueindex` and confirmed in the registry (`ACSettingIndex=0`). This host uses
Modern Standby (S0); S1/S2 are unavailable.

## Environment deviations (none touch the protocol)

Recorded so machine B and any later Windows machine can skip the same debugging.

1. **Native Windows is not viable — WSL2 is mandatory, not a preference.**
   mujoco 3.8.0 and 3.8.1 both fail to load on Windows with
   `OSError: [WinError 1114] DLL initialization routine failed`, under numpy 1.x and 2.x,
   and with a PATH stripped of Anaconda. `mujoco.dll` is present (4.5 MB), both VC++
   runtimes are present, and the DLL has no extra same-directory dependencies. Only
   mujoco 3.1.4 loads — wrong version for the protocol. The same wheel loads fine on WSL2.

2. **`ensurepip` is absent on Ubuntu 24.04** (it ships in the separate `python3.12-venv`
   package), so `run_sweep.sh`'s automatic venv build fails, and installing that package
   needs sudo. Worked around without sudo:
   `python3 -m venv --without-pip .sweep_venv`, then pip bootstrapped inside the venv from
   the official PyPA `get-pip.py`. The result is equivalent to what the launcher builds.

3. **`run_sweep.sh` does not install matplotlib, but the sweep needs it.** Line 44 installs
   `mujoco numpy cma` only; `physics/gait_sweep.py:30` does `import matplotlib`, pulled in
   by `physics/grid4_sweep.py:33`. Every shard dies at startup with
   `ModuleNotFoundError: No module named 'matplotlib'` until it is installed by hand.

4. **`GRID3_PY=$PWD/...` on the launcher command line expands too early** — it resolves
   against the shell's starting directory, not the repo root, so `pick_py` fails and the
   script falls into its "build venv" branch. It recovers, because after the script's own
   `cd` the fallback `PY="$PWD/.sweep_venv/bin/python"` is correct, but the run emits a
   confusing `line 41: ... No such file or directory` first. Pass an absolute path.

5. **`.sweep_venv/` is not in `.gitignore`** and shows up as untracked in `git status`.

## Monitor

```bash
cd ~/pengu/pengu_mujoco
wc -l results/gait_sweep/sweep_grid4_c3_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
pgrep -fc grid4_sweep.py                       # expect 14
tail -5 results/gait_sweep/c3_watchdog.log     # what the watchdog has restarted
```

## Restart by hand (the watchdog normally does this)

```bash
~/bin/run_c3_watchdog.sh
```

Resume is automatic by axis-tuple; completed rows are skipped.

## Ship-back

Unchanged from `docs/grid4_fleet_memo.md` — at 1,818,000 rows, strip blank lines, verify
1,818,001 lines incl. header / all `NF==12` / 1,818,000 unique 6-tuples, gzip,
`git add -f`, commit, push. Then this machine starts its batch-2 config.

## Open items

- **The row-guard fix is not on the fleet.** Ben to decide when to push it; until then any
  restart of c1 or c2 hits the crash.
- 24 h throughput sample to replace the ~3,440 rows/h figure.
- Shard-10 first-launch death: unexplained. The watchdog now auto-recovers it, but the
  cause is still open.
- Launcher gaps (matplotlib dependency, `.gitignore` entry) — not applied, since editing
  a shared launcher mid-fleet was out of scope here.
