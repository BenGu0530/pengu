# GRID-4 machine B — c2 status

Spec: `docs/grid4_guide.md`. Fleet assignment: `docs/grid4_fleet_memo.md`.
Machine B = **c2** (kappa=0, COM 1.20). Launched 2026-08-15 at K=5; switched to the
**staged-K amendment (`DR_K=1`) on 2026-08-16** per the fleet memo.
**RETIRED 2026-08-19 — this box is out of the fleet (Ben: CPU too slow).** c2 stopped
at 264949 of 1,818,000 rows (14.57%). See Handoff below.

## Machine

Windows 11 Home laptop, Ryzen 7 5800H (8 physical / 16 logical), 15.4 GB RAM.
Sweep runs in WSL2 / Ubuntu 22.04, repo at `~/pengu` (Linux fs, not `/mnt/c`).

`~/.wslconfig`: `memory=8GB`, `processors=16`, `swap=4GB` -> WSL sees all 16 logical
cores, launcher default gives **14 shards**, nothing reserved for Windows (Ben,
2026-08-16). Earlier settings this box ran at: `processors=12` (10 shards) and
`processors=13` (11 shards) — see throughput, the shard count made no measurable
difference.

Env from the launcher's auto-venv path: mujoco 3.8.1, numpy 2.2.6, cma 4.4.4,
matplotlib 3.10.9. Python 3.10.12.

## Pre-flight

1. HEAD on `friction-experiments`, matching the fleet.
2. `CONFIG=c2 ... count` ->
   `config=c2 kappa=0.0 com=1.2 cells=454500 rows=1818000 K=1 trials=1818000`
   (`K=5 trials=9090000` before the staged-K switch).
3. Startup line, now visible in the log within seconds thanks to the `-u` fix:
   ```
   # GRID4 c2 (kappa=0.0 com=1.2000 slide=-31.37mm mass=2.2724kg)  cells=454500
     done=628/1818000  K=1  shard=4/14
   ```
   `mass=2.2724kg` and `slide=-31.37mm` both match the fleet memo's expected values
   for the 1.20 rung.
4. Header check passes on every relaunch (machine-D trap); the CSV has never been
   headerless on this box.

## Measured throughput

All samples with every shard alive and `wa=0`.

| K | shards | window | rate | implied remaining |
|---|---|---|---|---|
| 5 | 10 | 900 s | 1,288 rows/h | 58.8 d |
| 5 | 11 | 900 s | 1,236 rows/h | 61.3 d |
| 5 | 11 | **13.9 h** | **1,511 rows/h** | 49.5 d |
| 1 | 14 | 900 s | **7,492 rows/h** | **10.1 d** |

The 13.9 h figure supersedes the two 15-minute K=5 samples, which ran ~18% low. Working
back from it, the rate was near-flat at ~1,520 rows/h for the whole run — the early
samples were not a slow region, just short windows.

**Shard count makes no difference on this box.** K=1 divides the per-row trial count by 5,
so the expected speedup from the switch alone is 5.00x. Measured 7,492 / 1,511 = **4.96x**,
with shards raised 11 -> 14 at the same time. The three extra shards therefore contributed
nothing measurable (if anything, slightly negative). Consistent with the earlier
10 -> 11 comparison, which was also flat: 8 physical cores saturate at ~10 processes and
further shards land on SMT siblings. CPU at 14 shards: 87.3% user, 12.0% idle,
loadavg 14.06 against 16 allocated.

Independent cross-check of the K=5 rate from the sweep parameters: `SIM_DURATION=24.0` s
x `K=5` = 120 s simulated per row; 1,818,000 rows = 2,525 simulated days per config; the
measured aggregate ~43x realtime -> ~58.7 days. Matches the short-sample K=5 figures.

### Against the other machines

| | B (this) | C (XPS) | D (Linux) |
|---|---|---|---|
| CPU | Ryzen 7 5800H, 8C/16T | i7-13700H, 6P+8E | — |
| shards | 14 | 14 | 14 |
| rate (K=5) | 1,511 rows/h | ~3,960 rows/h | ~5,005 rows/h |

At K=1 this box measures 10.1 d for c2, against the fleet memo's ~12 d prediction for B.
Recorded as measured, no adjustment.

## Staged-K switch (2026-08-16)

Followed the fleet memo procedure exactly. The K=5 partial was **archived, never mixed**:

- 22,496 data rows + header, all lines `NF==12`, unique 6-tuples = 22,496 (zero duplicates)
- `mv` to `...csv.k5partial`, gzipped to 325 KB, committed `725f34a`
- fresh CSV created by `initcsv` with header intact; relaunched at `DR_K=1`

## RETIRED 2026-08-19 — handoff

Ben retired this box from the fleet: the Ryzen 7 5800H is the slowest machine in the
group by a wide margin (see the cross-machine table below), so c2 is better restarted
elsewhere than finished here.

**State at shutdown**

| | |
|---|---|
| rows | 264,949 of 1,818,000 (14.57%) |
| protocol | staged-K, `DR_K=1` |
| shards | 14, stopped with SIGTERM |
| integrity | every line 12 fields, header present, **0 duplicate 6-tuples** |

**To continue c2 on another machine** — the data is committed in the launcher's own
recovery format, so no manual steps are needed:

```bash
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
DR_K=1 bash physics/run_sweep.sh c2
```

`run_sweep.sh` finds `...csv.gz`, gunzips it, and resumes from row 264,949 by axis-tuple.
The `...INCOMPLETE_264949_of_1818000_rows.zip` next to it is a labelled human-readable copy
of the same data with a README — it is not the resume path.

**Left running on this box: nothing.** Shards stopped, WSL keepalive anchor stopped.

## WSL interop broke after the 2026-08-17 package update (fleet-relevant)

Separate from the outage below, and it silently broke `git push` for hours.

```
/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe: Exec format error
error: git-remote-https died of signal 15
```

`WSLInterop` was **not registered in binfmt_misc**, so WSL could not execute *any*
Windows `.exe` — including `cmd.exe` and the Git Credential Manager this box uses for
push auth. Git therefore had no credential helper and sat waiting instead of failing
cleanly, which reads exactly like a slow network.

This box runs `[boot] systemd=true` in `/etc/wsl.conf`; under systemd the interop
registration is done by a unit that did not come back after the WSL package was updated
to 2.7.12.0. Repair (root, non-destructive, survives until the next distro restart):

```bash
echo ':WSLInterop:M::MZ::/init:PF' > /proc/sys/fs/binfmt_misc/register
```

Check on any WSL box whose `git push` hangs, or that uses GCM for auth:

```bash
cat /proc/sys/fs/binfmt_misc/WSLInterop   # should print: enabled / interpreter /init
/mnt/c/Windows/System32/cmd.exe /c "echo interop_ok"
```

Machine C is also WSL2 and may be exposed to the same failure.

## Outage 2026-08-17 -> 2026-08-19: Microsoft Store updated the WSL package

**40.7 h of compute lost (~330,000 rows at the measured rate).** No data was damaged.

```
2026-08-17 20:55:16  Store installed MicrosoftCorporationII.WindowsSubsystemforLinux
2026-08-17 20:55:39  last CSV write -- 23 s later
2026-08-17 20:57:52  Store installed CanonicalGroupLimited.Ubuntu
2026-08-18 05:07 / 05:10 / 05:37  three Windows Update reboots (later, not the cause)
2026-08-19 13:40     found dead, relaunched from 249,767 rows
```

Ruled out from the event log: no sleep (the last Kernel-Power 42/107 pair on this box is
from 2025 -- it has not slept once in 2026), no power-source change, no OOM, no traceback
in the run log. On AC at 100% throughout.

**Updating the WSL MSIX package force-terminates the WSL VM.** Neither existing
protection covers this, and it applies to every WSL machine in the fleet:

- **The keepalive anchor cannot help** -- it is a process *inside* the VM being torn
  down, so it dies with the shards.
- **`sweep_watchdog.sh` installed via WSL crontab cannot help either.** Its `@reboot`
  entry fires when the *distro* boots, but after a package update the VM is simply down
  and nothing on the Windows side ever starts it. The script's own header says as much
  (use Windows Task Scheduler on WSL).

Mitigations that would actually cover it (not yet applied here, Ben's call):

1. A **Windows Task Scheduler** job, at logon and every ~15 min, running
   `wsl.exe -d Ubuntu -e bash -lc "cd ~/pengu/pengu_mujoco && bash physics/sweep_watchdog.sh c2 14"`.
   `wsl.exe` boots the distro and the watchdog revives shards, so one mechanism covers
   package updates, reboots, and individual shard deaths.
2. Turn off **Microsoft Store -> Settings -> App updates**, and pause Windows Update for
   the duration of the run.

### Shard 12/13 loss on the recovery relaunch (self-inflicted)

Recorded so it is not mistaken for machine C's incident. The recovery launch was run as
`bash physics/run_sweep.sh c2 2>&1 | head -3`. `head` exits after three lines and closes
the pipe; the launcher's trailing `echo`s then take SIGPIPE and the parent dies. The 14
shards are spawned in the loop *before* those echoes, but the last two spawned (12 and
13) had not fully detached and went with the parent -- shards 0-11 survived. The
signature looks identical to machine C's silent shard death (no traceback, no OOM), but
the cause here was the pipe, not the launcher. **Do not pipe `run_sweep.sh` output
through `head`.** Recovered by launching the two missing SHARD_IDs directly; IDs 0-13
verified unique afterwards.


## Environment notes

Items 1-3 were reported from this box and have since been fixed upstream
(`612617f`); kept here as the record of what was measured.

1. **`matplotlib` missing from the launcher deps** — `gait_sweep.py:30` imports it at
   module level via `grid4_sweep.py:33`, but `run_sweep.sh` installed only
   `mujoco numpy cma`. Clean machines died at `initcsv`. **Fixed** (`98c0535`).
2. **Shard stdout was block-buffered**, so the `# GRID4 cN (...)` startup line never
   reached the log while shards ran normally — it only flushed when a process exited.
   **Fixed** by `-u` in `run_sweep.sh:65`; verified working here (startup line appears
   within seconds of launch).
3. **WSL2 terminates the distro when no session is attached, killing every shard.**
   Measured on this box: distro gone within 45 s, `nohup`ed process killed with it.
   **Now a REQUIRED fleet step.** Anchor, re-armed after every reboot/logoff:
   ```
   powershell -c "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden"
   ```
   Held for 13.9 h unattended with 11 shards, zero losses. **Does not protect against
   a WSL package update** -- see Outage above.
4. **No shard deaths observed on this box** across the K=5 run (13.9 h, 11 shards) and
   since the K=1 relaunch (14 shards). Machine C's shard-10 incident has not reproduced
   here. One box is not a diagnosis, but the launcher path itself looks sound.
5. **Push auth**: this box uses Windows Git Credential Manager from WSL —
   `credential.helper = /mnt/c/Program\ Files/Git/mingw64/bin/git-credential-manager.exe`,
   OAuth in the browser, token in Windows Credential Manager. HTTPS remote.

## Power

| setting | AC | battery |
|---|---|---|
| system sleep | never (was already) | 5 min (unchanged) |
| hibernate | never (was already) | 3 h (unchanged) |
| lid close | **do nothing** (changed) | sleep (deliberately unchanged) |

Battery-side settings were left alone on purpose: if the machine is unplugged and the lid
is closed, it should still sleep rather than run 14 saturated threads in a bag.

## Monitor

```bash
cd ~/pengu/pengu_mujoco
wc -l results/gait_sweep/sweep_grid4_c2_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
pgrep -c -f 'grid4_sweep[.]py'        # expect 14
```

## Restart after a reboot or a stop

Interrupting is safe. Verified twice here: SIGTERM to all shards left every line at 12
fields with the header intact, and the relaunch resumed from the exact row count.

```bash
# 1. re-arm the WSL anchor from Windows FIRST (command above)
# 2. then:
cd ~/pengu/pengu_mujoco
DR_K=1 GRID3_PY=$HOME/pengu/pengu_mujoco/.sweep_venv/bin/python bash physics/run_sweep.sh c2
```

`GRID3_PY` absolute on purpose: `$PWD/...` expands before the script's `cd`
(machine-C finding). `DR_K=1` must be present on every relaunch, or the box silently
reverts to K=5 and starts mixing K values into the same CSV.

## Data shipped so far

c2 is **NOT complete**. Two partials are committed, and K values are never mixed:

| file | protocol | rows | commit |
|---|---|---|---|
| `...csv.k5partial.gz` | K=5 | 22,496 | `725f34a` |
| `...csv.gz` (launcher resume path) | K=1 | 264,949 | this commit |
| `...c2_INCOMPLETE_264949_of_1818000_rows.zip` | K=1 | 264,949 | this commit |

The zip carries a `README_INCOMPLETE.txt` saying the same. The snapshot was taken from
the live CSV without stopping the sweep, and verified before packing: every line 12
fields, header present, **0 duplicate 6-tuples** across all the restarts so far.

## Open items

- `docs/grid4_fleet_memo.md` still describes B from the first-hour samples: line 34
  `slowest box (~60 d)` and line 44 `B ~1.24k`. Current measurements are 1,511 rows/h at
  K=5 and 7,492 rows/h at K=1 (10.1 d for c2). The line-36 plan for D to help finish c2
  was sized against the old figure — Ben's call whether it still applies.
- **Watchdog via Windows Task Scheduler + Store auto-update off** (see Outage). Until
  one of these is in place this box will keep dying to Store updates, silently.
- Whether to return `processors` to 13 (11 shards): measurements say throughput is
  unchanged and it gives Windows back 3 logical cores. Currently at 16 per Ben's
  instruction.
- numpy skew across the fleet (B 2.2.6, C 2.5.2, D unrecorded). Values identical; noted
  only because the guide states any machine reproduces a row exactly.
