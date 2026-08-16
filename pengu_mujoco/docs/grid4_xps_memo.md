# GRID-4 c3 — XPS 15 (machine C) status memo

Machine-C status report for the GRID-4 fleet. Assignment table: `docs/grid4_fleet_memo.md`.
Full spec: `docs/grid4_guide.md`. Protocol FROZEN at commit `a22f80b`; this machine follows
the STAGED-K amendment (K=1 map, top up later with `topup_k.py`). Nothing recorded here
changes a sweep axis, a μ level, a jitter, or a scoring rule.

## Machine

| | |
|---|---|
| Host | Dell XPS 15 9530 — i7-13700H (6 P + 8 E = 14C/20T), 64 GB DDR5-4800 |
| OS | Windows 11 Home build 26200 + WSL2, Ubuntu 24.04.4 LTS |
| WSL limits | **20 vCPU** (all logical CPUs), 24 GB, swap=0 (`C:\Users\bengu\.wslconfig`) |
| Repo path | `~/pengu` on WSL ext4 (NOT `/mnt/c` — per fleet memo) |
| Python | 3.12.3, venv at `pengu_mujoco/.sweep_venv` |
| Packages | mujoco **3.8.1**, numpy 2.5.2, cma 4.4.4, matplotlib |

## Current run

**c3 = kappa 0, COM ratio 1.31, K=1, 20 shards.** Relaunched 2026-08-16 19:26 EDT.

Originally launched 2026-08-15 23:21 at K=5 / 14 shards / 16 vCPU. Switched per the
staged-K amendment; the K=5 partial (**35,005 rows**, header intact) is archived beside the
live CSV as `…_mu.csv.k5partial` and was never mixed into the K=1 file.

## Pre-flight

1. **HEAD** matches the fleet.
2. **count** — `config=c3 kappa=0.0 com=1.31 cells=454500 mus=[0.1,0.3,0.5,0.7] rows=1818000`.
3. **Startup line** — `com=1.3100 slide=+8.73mm mass=2.2724kg … K=1 shard=n/20`.
   COM/mass also verified in a separate process via `apply_com_variant(model, 1.31)`:
   `com_ratio=1.3100  slide=+8.73mm  mass=2.2724kg`, xml = `models/pengu1_31/scene.xml`.
   Matches the fleet memo's expected slide for the 1.31 rung.
4. **Header check** — `head -1` on the live CSV is the 12-column header.

## Measured throughput

| configuration | sample | rate | ETA |
|---|---|---|---|
| K=5, 14 shards, 16 vCPU | 4 h 11 min cumulative | ~3,370 rows/h | ~22 d |
| **K=1, 20 shards, 20 vCPU** | **90 s** | **~20,320 rows/h** | **~3.7 d** |

The ~6x jump decomposes as ~5x from K=5→K=1 (one trial per (cell,μ) instead of five) and
~1.2x from 14→20 shards on 16→20 vCPU. The 3.7 d figure agrees with the fleet memo's
independent ~3.8 d estimate for this box. The K=1 sample is short — a longer one after a
day is still worth taking.

Note the sublinear core scaling: this is a hybrid CPU (6 P-cores with SMT + 8 E-cores), so
the four vCPUs added past 16 are E-cores and SMT siblings, worth well under a P-core each.

## Bug found: `grid4_sweep.py` crashes on resume (fleet-wide)

```
NameError: cannot access free variable 'row' where it is not associated with a value
  physics/grid4_sweep.py:186
```

**Mechanism.** `row` is bound inside the inner `for mi, mu0 in enumerate(MUS)` loop. The
resume guard `if key in done: continue` can skip all μ values of a cell, leaving `row`
unbound; the periodic print at `if n_mine % 25 == 0:` then dereferences it and the shard
dies within seconds.

**Only on resume.** A first launch has `done` empty, so `row` is always bound. On a resume,
a shard whose first 25 owned cells are already complete crashes immediately — which means
once a config is past `25 x n_shards` cells, *every* shard dies on *every* restart. Left
unfixed this turns any keepalive into a crash loop that produces no rows.

**Fix** (2 lines, logging branch only — no computation path, no protocol value, no effect
on rows already written):

```python
    n_mine = 0
    row = None                                    # initialise

        if n_mine % 25 == 0 and row is not None:  # guard
```

**Status: committed on this machine, NOT pushed.** The fleet still carries the bug. It is
dormant right now because the staged-K switch gave every machine a fresh CSV, but it will
fire on the first restart after each machine has some rows.

Verified here twice: killing a shard mid-run reproduced the crash pre-fix and resumed
cleanly post-fix; and the post-reboot relaunch resumed all 14 shards at `done=14081` with
no crash. Whole-run log carries exactly one `NameError`, the pre-fix one.

## Keepalive (this machine, local scripts, not in the repo)

| layer | what | covers |
|---|---|---|
| 1 | `~/bin/run_c3_watchdog.sh` (`N=20`, `export DR_K=1`) | a shard dying silently |
| 2 | user crontab: `@reboot` + `*/10 * * * *` | WSL restart, shard loss within 10 min |
| 3 | Windows task `PenguGrid4C3Keepalive` (at logon + every 30 min, hidden-window VBS) | Windows reboot — starts WSL, then the watchdog |

Follows the `run_grid2.sh` idiom (flock, per-shard `.done` skip) with one change:
**liveness is read from `/proc/<pid>/environ` (`SHARD_ID` + `CONFIG`) rather than pidfiles**,
because shards launched by `run_sweep.sh` write no pidfile. A pidfile-only check would have
seen every shard as absent and launched a full duplicate set, and duplicate SHARD_IDs write
duplicate rows, which breaks the unique-6-tuple ship-back check.

The watchdog must also carry `DR_K=1`: a relaunch without it defaults to K=5 and mixes K
values into one CSV.

### Watchdog bug: leaked lock fd (fixed)

The lock was taken with `exec 9>/tmp/pengu_c3.lock` + `flock -n 9`, and `nohup … &` let
every shard **inherit fd 9**. The 20 shards then held the lock for their whole lifetime, so:

- every later watchdog run hit `flock -n 9 || exit 0` and exited silently — **shard-death
  recovery was dead** (reboot recovery still worked, since a fresh boot has no holders);
- any *blocking* `flock` on that file hung forever, deadlocking against the very shards it
  was about to kill.

Fixed by closing the fd in the children:

```bash
CONFIG=$CFG N_SHARDS=$N SHARD_ID=$s nohup "$PY" -u "$SCRIPT" >> "$RUNLOG" 2>&1 9>&- &
```

Verified: with all 20 shards running, 0 processes hold the lock file and `flock -n` reports
it free.

## Incidents

**Reboot 2026-08-16 03:34 cost ~10 h of idle time.** The Windows keepalive task triggers
*at logon*; after the update restart the machine sat at the lock screen and nothing ran
until logon at 13:34. Recovery itself was clean and immediate — the task fired at 13:34:31,
the watchdog relaunched all 14 shards at 13:34:39, and every one resumed at `done=14081`
without crashing. This is the documented limitation of the logon trigger, not a failure of
the chain. Running without logon needs stored account credentials in the task, deliberately
not done.

**Shard 10 died during startup on the very first launch; cause still unidentified.** No
traceback, no OOM in `dmesg`, memory 3.8 GB of 24 GB. **Not** the row-guard bug — `done`
was empty on a first launch, so nothing was skipped. Has not recurred; the watchdog now
recovers it automatically.

## Power

AC-only; the machine must stay plugged in.

| setting | before | now |
|---|---|---|
| system sleep / hibernate (AC) | never | never |
| disk timeout (AC) | 30 min | never |
| lid close (AC) | — | do nothing |
| Windows Update active hours | 08:00–21:00 | **18:00–12:00** (18 h, the maximum span) |

The 03:34 reboot fell outside the old 08:00–21:00 window, which is exactly why it happened
then; the new window covers the whole night. Active hours only defers automatic restarts —
pausing updates outright is the stronger measure and is a manual step in Settings.

Lid close is hidden from `powercfg /query` on this host; set via `/setacvalueindex` and
confirmed in the registry (`ACSettingIndex=0`). Modern Standby (S0); S1/S2 unavailable.

## Environment deviations (none touch the protocol)

1. **Native Windows is not viable — WSL2 is mandatory.** mujoco 3.8.0 and 3.8.1 both fail
   with `OSError: [WinError 1114] DLL initialization routine failed`, under numpy 1.x and
   2.x and with a PATH stripped of Anaconda. The DLL is present, both VC++ runtimes are
   present, no extra same-directory dependencies. Only mujoco 3.1.4 loads — wrong version.
   The same wheel loads fine on WSL2.
2. **`ensurepip` is absent on Ubuntu 24.04** (it lives in `python3.12-venv`), so the
   launcher's venv build fails and installing that package needs sudo. Worked around
   without sudo: `python3 -m venv --without-pip`, then pip bootstrapped inside the venv
   from the official PyPA `get-pip.py`.
3. **PowerShell 5.1 mangles `|` and `&` even inside quotes** when passing arguments to
   `wsl.exe`; a quoted grep pattern got split into three commands, and a quoted `sed`
   expression arrived truncated. On Windows hosts, put anything containing shell
   metacharacters in a script file and run `wsl -d <distro> -- bash /mnt/c/path/script.sh`.
4. Fixed upstream since this memo was first written: matplotlib missing from the launcher's
   pip line, `-u` on shard stdout, `.sweep_venv` in `.gitignore`, and the `GRID3_PY=$PWD`
   early-expansion trap.

## Monitor

```bash
cd ~/pengu/pengu_mujoco
wc -l results/gait_sweep/sweep_grid4_c3_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
pgrep -fc grid4_sweep.py                       # expect 20
tail -5 results/gait_sweep/c3_watchdog.log     # what the watchdog has restarted
```

Restart by hand (the watchdog normally does this): `~/bin/run_c3_watchdog.sh`.

## Ship-back

Unchanged from `docs/grid4_fleet_memo.md`. The `.k5partial` archive is separate data at a
different K and must not be concatenated into the K=1 CSV.

## Open items

- **The row-guard fix is not on the fleet.** Dormant while every config has a fresh CSV;
  fires on the first restart after that. Ben to decide when to push.
- Longer throughput sample at K=1 to firm up the ~3.7 d figure.
- Windows Update pause (5 weeks) — manual step, not yet applied; active hours alone does
  not stop an update from installing.
- Shard-10 first-launch death: still unexplained, has not recurred.
