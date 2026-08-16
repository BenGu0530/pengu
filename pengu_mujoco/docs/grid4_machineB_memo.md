# GRID-4 machine B — c2 status

Spec: `docs/grid4_guide.md`. Fleet assignment: `docs/grid4_fleet_memo.md`.
Machine B = **c2** (kappa=0, COM 1.20). Launched 2026-08-15 at K=5; switched to the
**staged-K amendment (`DR_K=1`) on 2026-08-16** per the fleet memo. Running.

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
   Held for 13.9 h unattended with 11 shards, zero losses.
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

## Open items

- `docs/grid4_fleet_memo.md` still describes B from the first-hour samples: line 34
  `slowest box (~60 d)` and line 44 `B ~1.24k`. Current measurements are 1,511 rows/h at
  K=5 and 7,492 rows/h at K=1 (10.1 d for c2). The line-36 plan for D to help finish c2
  was sized against the old figure — Ben's call whether it still applies.
- Whether to return `processors` to 13 (11 shards): measurements say throughput is
  unchanged and it gives Windows back 3 logical cores. Currently at 16 per Ben's
  instruction.
- numpy skew across the fleet (B 2.2.6, C 2.5.2, D unrecorded). Values identical; noted
  only because the guide states any machine reproduces a row exactly.
