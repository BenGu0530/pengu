# GRID-5 machine C (laptop) — status, slice tooling, and why this box is slow

Machine-C report for the GRID-5 fleet. Deployment card: `fleet_launch.md`.
Operations guide: `RUN_GRID5.md`. Protocol is grid5-v2 (deterministic map, exact
nominal mu, no pose jitter, no RNG, K=1); nothing here changes a protocol value.

## Machine

| | |
|---|---|
| Host | Dell XPS 15 9530 — i7-13700H, **6 P-cores + 8 E-cores = 14 physical / 20 threads** |
| Memory | 64 GB DDR5-4800; WSL2 capped at 48 GB, swap 8 GB |
| OS | Windows 11 + WSL2 Ubuntu 24.04, repo at `~/pengu` on ext4 |
| Engine | mujoco 3.8.1 in `.sweep_venv` |
| Queue | c1 (kappa 0, COM 1.05) |
| Shards | **19** (not the `cores-2` default — see "Shard count" below) |

## Current state (2026-09-01)

**Shipped:** `sweep_grid5_c1_PARTIAL_hipoff0-10_1382400_of_4147200_rows.csv.gz`
with the config manifest. The two outermost blocks `hip_off = 0` and `hip_off = 10`
are complete at 691,200 rows each: every 6-axis tuple unique, no duplicate lines,
all 31 columns present, header identical to the working CSV.

**Running:** a `hip_phi ∈ {0, 10}` slice across the remaining `hip_off` blocks
(see "Slice tooling"). 144,675 of 230,400 slice rows done. The full-config sweep
is stopped and `results/gait_sweep/WATCHDOG_OFF` is in place so the 10-minute
watchdog does not revive it alongside the slice.

Working CSV holds 1,503,289 rows of the 4,147,200-row config.

## Slice tooling (local, `~/bin/`, not in the repo)

`run_slice.sh <config> <axis> <values> [shards]` → `slice_sweep.py`, which imports
`grid5_sweep` and **wraps `cells()` with a filter, nothing else**. All module-level
axis definitions, the seeds, the physics and the manifest are untouched, so:

- rows produced are bit-identical to the same rows from a full sweep (grid5-v2 is
  deterministic);
- `check_manifest` still passes — it validates protocol / config / K /
  mujoco_version / slip, not the axis lists;
- output goes to the config's normal CSV, so resume-by-axis-tuple skips whatever
  the full sweep already did.

```bash
~/bin/run_slice.sh c1 hip_phi 0,10 19
```

Useful when a slow box should contribute a usable sub-map instead of blocking a
whole config. Any of the five cell axes works.

## Why this box is slow — it is the config, not the hardware

The laptop measures ~10,300 rows/h on c1 against rml2's reported ~51,000 rows/h on
c6. Diagnosis on this machine: **no zombies, no strays, no throttling** — 19 shards
at 99.9% CPU each, all cores at 2918 MHz (118% of the 2400 MHz base) after five
days of sustained load, on AC, zero OOM.

The gap decomposes as roughly **1.8x hardware and 4.3x workload**:

- *Hardware.* 14 physical cores is not 14 equal cores. Counting E-cores at ~0.6 of
  a P-core and SMT at ~+25%, the chip is about **12.3 P-core-equivalents**, so 19
  shards each get ~0.65 of a P-core. Measured 6.8 s per row per shard.
- *Workload.* Configs differ enormously in cost. Fraction of rows with
  `net_fwd_mean = 0` (trial ends early) in the shipped GRID-4 maps:

  | config | rows ending early | shipped gz |
  |---|---|---|
  | **c1** | **2.6%** | 24.0 MB |
  | c2 | — | 16.7 MB |
  | c3 | 46.7% | 10.9 MB |
  | c4 | — | 23.2 MB |
  | c5 | — | 16.6 MB |
  | **c6** | **48.6%** | 10.5 MB |

  About half of c6's cells stop early and cost almost nothing; only 2.6% of c1's
  do, so nearly every c1 cell runs the full duration. **c1 is the most expensive
  config of the six and it is assigned to the weakest box.** (Numbers only — what
  the early-termination difference means physically is Ben's call.)

This is cheap to check before assigning Phase B (c7–c10): count `net_fwd_mean = 0`
per config in the existing maps and rank.

## Shard count: pick N coprime with leg_amp × hip_amp

Cells are handed out as `i % N == shard_id`, and the two innermost loops cycle with
period `len(LEG_AMPS) * len(HIP_AMPS)` = **60** under v2.3. If `g = gcd(N, 60) > 1`,
every shard is locked to `60/g` of the 60 (leg_amp, hip_amp) pairs:

| N | gcd(N,60) | pairs per shard |
|---|---|---|
| 18 (`cores-2` here) | 6 | 10 / 60 |
| **19 (in use)** | **1** | **60 / 60** |
| 20 | 20 | 3 / 60 |
| 30 (`cores-2` on a 32-core box) | 30 | **2 / 60** |

Row *counts* stay even either way — striding handles that — but the *mix* does not,
and cell cost varies several-fold with parameters, so a locked shard finishes early
and idles while others grind. Observed directly under GRID-4 (period 36, N=18): the
`hip_amp=28` group reached 93.3% while the other four sat at 25–34%.

**Machines running `cores-2` on 16 or 32 cores (14, 30) both collide with 60.** 19,
29 or 31 cost one core and remove the tail.

## Watchdog caveat: revival can duplicate rows

`sweep_watchdog.sh` compares only the shard *count*, not which SHARD_IDs are alive.
If some shards die it relaunches the full set, so surviving ids run twice and their
rows are written twice. This produced 9,445 duplicate rows during the GRID-4 c6
topup here.

Under grid5-v2 the damage is bounded — determinism means duplicates are always
byte-identical, so `awk '!seen[$0]++'` removes them losslessly — but **check the row
count before shipping**: more rows than the config's target means this happened.

Also note `run_sweep.sh` has no liveness guard at all. Re-running it while shards
are alive doubles them. Revival is safe only through the watchdog, or from a
genuinely dead state.

## Measured rates on this box

| what | rate |
|---|---|
| c1 full sweep, `hip_off = 0` block | ~11,520 rows/h |
| c1 full sweep, later blocks | ~10,300 rows/h |
| `hip_phi ∈ {0,10}` slice | ~11,060 rows/h |

Block-to-block variation is small (~12% between the first block and the rest), so
the earlier worry that later blocks would be much cheaper did not hold. At ~10,300
rows/h the remaining full config would need roughly 11 more days.

## Open items

- Slice finishes in ~8 h; then decide whether to slice more `hip_phi` values or
  remove `WATCHDOG_OFF` and let the full c1 sweep resume.
- The slice has **no watchdog** — a crash or reboot needs the launch line re-run by
  hand. `WATCHDOG_OFF` must stay while the slice runs.
- `grid5/run_machine.sh` carries an uncommitted local edit (`NSH=19` for laptop).
- Consider reassigning c1 to a stronger box, or splitting it with disjoint shard
  ids as `fleet_launch.md` allows.
