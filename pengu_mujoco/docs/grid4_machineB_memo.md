# GRID-4 machine B — c2 status

Spec: `docs/grid4_guide.md`. Fleet assignment: `docs/grid4_fleet_memo.md`.
Machine B = **c2** (kappa=0, COM 1.20). Launched 2026-08-15, running.

## Machine

Windows 11 Home laptop, Ryzen 7 5800H (8 physical / 16 logical), 15.4 GB RAM.
Sweep runs in WSL2 / Ubuntu 22.04, repo at `~/pengu` (Linux fs, not `/mnt/c`).

`~/.wslconfig`: `memory=8GB`, `processors=13`, `swap=4GB` -> WSL sees 13 cores,
launcher default gives **11 shards**, 3 logical cores left to Windows.

Env built by the launcher's auto-venv path: mujoco 3.8.1, numpy 2.2.6, cma 4.4.4,
matplotlib 3.10.9.

## Pre-flight (all three pass)

1. `HEAD = ccf088a` on `friction-experiments`.
2. `CONFIG=c2 ... count` ->
   `config=c2 kappa=0.0 com=1.2 cells=454500 rows=1818000 K=5 trials=9090000`
3. Startup line (see "stdout buffering" below for how it was obtained):
   ```
   # GRID4 c2 (kappa=0.0 com=1.2000 slide=-31.37mm mass=2.2724kg)  cells=454500
   ```
   `mass=2.2724kg` matches the guide's native total mass; `slide=-31.37mm` against the
   guide's approximate `1.20 -> -31.5 mm`.

## Measured throughput

Two samples, both with all shards alive and no I/O wait (`wa=0`):

| shards | window | rows | rate | implied remaining |
|---|---|---|---|---|
| 10 | 900 s | +322 | 1,288 rows/h | 58.8 d |
| 11 | 900 s | +309 | 1,236 rows/h | 61.3 d |

Going 10 -> 11 shards produced no gain (-4%): 8 physical cores are already saturated at
10 shards, and the extra process lands on an SMT sibling. CPU at 11 shards: 84.7% user,
14.8% idle, `wa=0`, loadavg 11.1 against 13 allocated cores.
Caveat: the two samples cover different regions of the grid, so this is not a controlled
A/B.

Independent cross-check from the sweep parameters, which lands on the same number:
`SIM_DURATION=24.0` s x `K=5` = 120 s of simulated time per row; 1,818,000 rows =
2,525 simulated days per config. Measured aggregate is ~43x realtime -> ~58.7 days.

For comparison with the existing estimates:

| source | stated | rate that would require |
|---|---|---|
| `grid4_guide.md` | ~20-24 h/config @ 10 shards | 75,750 rows/h (59x measured) |
| `grid4_fleet_memo.md` | ~2 weeks/machine @ ~10 shards | 5,410 rows/h (4.2x measured) |

Both samples were taken in a survival-heavy region (`surv_rate=1.0` on early rows).
Failed trials exit before `SIM_DURATION` (`gait_sweep.py:254`), so cells that fall early
are cheaper; throughput may change as the sweep moves through the grid. Numbers above are
first-hours measurements, not a converged estimate.

### Against machine C (`docs/grid4_xps_memo.md`)

| | machine B (this) | machine C |
|---|---|---|
| CPU | Ryzen 7 5800H, 8C/16T | i7-13700H, 6P+8E, 14C/20T |
| RAM (to WSL) | 15.4 GB (8 GB) | 64 GB (24 GB) |
| distro / Python | Ubuntu 22.04 / 3.10.12 | Ubuntu 24.04 / 3.12.3 |
| numpy | 2.2.6 | 2.5.2 |
| shards | 11 | 14 |
| rate | 1,236 rows/h | ~3,960 rows/h |
| per shard | 112 rows/h | 283 rows/h |
| implied remaining | ~61 d | ~19 d |

Aggregate 3.2x, per-shard 2.5x; both on mujoco 3.8.1. Neither machine's measurement is
near the guide's 20-24 h figure. Machine C's ~19 d sits within ~40% of the fleet memo's
~2 weeks. Both sets are first-hours samples. Recorded as measured, no adjustment.

## Issues found on a clean machine

**1. `matplotlib` missing from the launcher's dependency list.**
`physics/gait_sweep.py:30` imports matplotlib at module level and `grid4_sweep.py:33`
imports `gait_sweep`, but `run_sweep.sh` installed only `mujoco numpy cma`. A clean
machine following the fleet memo's one-liner fails at `initcsv` with
`ModuleNotFoundError: No module named 'matplotlib'`. Fixed in `run_sweep.sh`.

**2. Shard stdout is block-buffered, so the startup line never reaches the log.**
`run_sweep.sh` launches shards as `nohup "$PY" "$SCRIPT" >> "$LOG" 2>&1 &`. With stdout
redirected to a file, Python block-buffers (8 KB), and a shard writes nothing else to
stdout during the run, so the `# GRID4 cN (...)` line stays in the buffer until the
process exits. Pre-flight step 3 cannot be satisfied while the shards are running
normally; the line only surfaces when a process exits and flushes — which is how machine
C saw it (`grid4_xps_memo.md`: a shard that died and was relaunched, plus a separate
verification process). `nohup "$PY" -u "$SCRIPT"` would fix it. NOT changed yet.

Workaround used here to verify COM/mass without touching the data: run one probe with a
shard id outside the range, which builds the model, prints the startup line, and writes
zero rows (`i % 999999` never equals `999998` for `i < 454500`):
```bash
CONFIG=c2 N_SHARDS=999999 SHARD_ID=999998 .sweep_venv/bin/python -u physics/grid4_sweep.py
```

**3. WSL2 terminates the distro when no session is attached, killing the shards.**
Measured: with no terminal attached, the distro was gone within 45 s and a `nohup`ed
background process was killed with it. A sweep started per the memo and left unattended
stops almost immediately. Anchor process required on the Windows side:
```
powershell -c "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden"
```
The anchor does not survive reboot or logoff; it must be re-armed before relaunching.

`grid4_xps_memo.md` does not mention a keepalive, so it is unknown whether machine C's
WSL build behaves the same way. Worth checking there before its terminal is closed — the
failure is silent, and the shards are simply gone on the next visit.

**4. `.sweep_venv/` is not in `.gitignore`.** The launcher creates it on every machine;
a `git add -A` on a fleet machine would stage the whole virtualenv. Not changed yet.

## Restart procedure on this machine

Interrupting is safe (resume is by axis-tuple). Verified once: SIGTERM to all shards left
the CSV with every line at 12 fields and no torn row; relaunch resumed from the exact
row count.

```bash
# 1. re-arm the anchor from Windows (after any reboot)
# 2. then:
cd ~/pengu/pengu_mujoco
GRID3_PY=$HOME/pengu/pengu_mujoco/.sweep_venv/bin/python bash physics/run_sweep.sh c2
```
Absolute path deliberately: `GRID3_PY=$PWD/...` on the command line expands against the
shell's starting directory, not the repo root (machine C, `grid4_xps_memo.md` item 4).

## Open items

- Whether to apply the `-u` fix (issue 2) and the `.gitignore` entry (issue 4).
- numpy version skew across the fleet: this machine runs numpy 2.2.6, which prints
  `mus=[np.float64(0.1), ...]` where the fleet memo shows `mus=[0.1, ...]`. Values are
  identical; flagging only because the guide states any machine reproduces a row exactly.
- Machine C (c3) will hit issues 1-3 as well.
