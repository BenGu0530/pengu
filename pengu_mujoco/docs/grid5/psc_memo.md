# PSC memo — running GRID-5 c1 on Bridges-2

Written 2026-09-01 during the run, closed out 2026-09-02 when it finished.
Companion to
`docs/grid5/fleet_launch.md` (the laptop fleet) — this is the same sweep on a
supercomputer, and the numbers are very different.

Operational detail (paths, rules, command cheatsheet, every gotcha hit) lives
outside this repo in `../PenguMujoco_psc/PSC_MEMO.md`. This file is the summary
a labmate or the PI should be able to read on its own.

---

## 1. What was run

One SLURM job, **44960443**, on one Bridges-2 RM node (128 cores), working
through **every remaining c1 hip_phi slice serially**: phi = 40, 50, … 350
(32 slices; phi=20 and 30 were done earlier as separate jobs).

**COMPLETE as of 2026-09-02 17:35 EDT: 32/32 slices, 40.75 h, zero failures.**
Together with phi=20 and 30 that is **34 of c1's 36 hip_phi slices,
3,916,800 rows** (0 and 10 were produced elsewhere).

**Why one job instead of one job per slice.** Each `sbatch` re-enters the
allocation's admission check. Running deep into overdraft, that is 32 chances
to be refused halfway through and left with a partial dataset. One job clears
admission once, at submit time, and SLURM posts the charge when it ends.
(Verified: the `cis250009p` association has no `GrpTRESMins`,
`GrpTRESRunMins`, `MaxTRESMins`, `MaxWall` or `MaxJobs` set, so nothing in
SLURM interrupts a job already running. The `rm` QOS caps concurrency at
`MaxTRESPerUser=cpu=25600` = 200 nodes, which is not a constraint here.)

Each slice is gzipped the moment it finishes, so a job that dies at hour 42
still leaves every completed slice packaged and ready to pull.

---

## 2. Measured numbers

Everything below is measured on Bridges-2, not extrapolated from the fleet.

### Billing

| | |
|---|---|
| RM | **1 SU = 1 core-hour**, charged on `AllocCPUS x Elapsed` — the cores you *reserve*, not the ones you use |
| Reconciliation | 372 → 370 (2.04 SU calib) → 217 (152.6 SU slice); agrees with `sacct` to **1.3%** |
| `projects` | **lags 1–2 h.** A 152.6 SU job showed no balance change 40 min after it ended. Do not use it for real-time decisions — use `sacct -X -o AllocCPUS,ElapsedRaw` |

### Throughput

| machine | rows/hr/core |
|---|---|
| M2 Max (12 cores, 11 shards) | **2,000** |
| EPYC 7742, 8 cores | **904** |
| EPYC 7742, **128 cores** | **754** |

An EPYC 7742 core is **0.45x** an M2 Max core on this workload. At 128 cores
the rate drops a further **17%** (Lustre append contention on the shared CSV +
memory bandwidth). That 0.83 ratio was the deciding number for keeping the
shared-CSV layout rather than switching to one file per shard: the switch
could recover at most those 17 points and would permanently freeze the shard
count, because per-shard resume only de-duplicates within a shard.

### Cost per slice (115,200 rows each)

Measured across all 34: **152.6 – 173.2 SU**, mean ~163. The expensive band is
phi 70–110 (~172 SU); the cheap end is 300–350 (~156 SU).

A trial that falls costs ~45% of one that survives (9.6 s of simulated time vs
21.1 s), so per-slice cost varies with that slice's survival rate. The most
expensive slices (phi 70–110, ~172 SU) are also the highest-survival ones
(87–90%); the cheapest (300–350, ~156 SU) are lower-survival.

A cost prediction made mid-run from c3's phase profile put the turn toward
cheaper slices at 130–150; measured, it came at 150→160 (168.1 → 163.8 SU).

### Scale, for context

| | trials | simulated | PSC core-hours |
|---|---|---|---|
| 1 trial | — | 21 s (21,001 steps) | 4.77 core-s |
| 1 hip_phi slice | 115,200 | 28 days | ~165 SU |
| c1, full 36 slices | 4,147,200 | **2.76 years** | ~5,900 SU |
| GRID-5, all 10 configs | 41,472,000 | **27.6 years** | ~59,000 SU |

MuJoCo runs this model at about **4.4x real time** on one EPYC core. That
single number explains the whole cost picture: simulating 27.6 years of
walking at 4.4x real time needs ~6.3 years of single-core compute.

---

## 2b. Readouts across the 34 slices

Reported as measured. `walking` = fraction of rows with `net_fwd_mean > 0.05 m/s`
(the `NET_MIN` gate); `survival` = fraction with `surv_rate > 0`.

| hip_phi | survival | walking | max net_fwd |
|---|---|---|---|
| 20 | 77.4% | 24.5% | 0.983 |
| 30 | 78.4% | 22.3% | 1.014 |
| 40 | 80.6% | 18.5% | 1.004 |
| 50 | 83.1% | 14.3% | 0.903 |
| 60 | 85.7% | 9.7% | 0.943 |
| 70 | 87.4% | 6.5% | 0.881 |
| 80 | 88.8% | 5.0% | 0.793 |
| 90 | 89.5% | 4.3% | 0.926 |
| 100 | 90.0% | 3.8% | 0.772 |
| 110 | 89.5% | 3.4% | 0.826 |
| 120 | 88.7% | 3.3% | 0.776 |
| 130 | 87.5% | 3.5% | 0.814 |
| 140 | 86.7% | 4.1% | 0.917 |
| 150 | 85.5% | 4.5% | 0.892 |
| 160 | 83.4% | 4.4% | 0.819 |
| 170 | 81.0% | 4.0% | 0.871 |
| 180 | 79.1% | 3.7% | 0.811 |
| 190 | 78.4% | 3.4% | 0.960 |
| 200 | 78.8% | 3.5% | 0.916 |
| 210 | 79.3% | 4.3% | 0.838 |
| 220 | 79.7% | 8.1% | 0.903 |
| 230 | 80.4% | 14.1% | 1.017 |
| 240 | 81.5% | 23.4% | 1.072 |
| 250 | 83.9% | 32.4% | 1.173 |
| 260 | 86.3% | 38.5% | 1.012 |
| 270 | 88.1% | 41.0% | 1.099 |
| 280 | 88.3% | 42.7% | 1.069 |
| 290 | 87.1% | 41.3% | **1.210** |
| 300 | 85.6% | 36.9% | 1.079 |
| 310 | 84.3% | 34.5% | 1.122 |
| 320 | 83.2% | 34.0% | 1.172 |
| 330 | 82.4% | 32.7% | 1.068 |
| 340 | 82.1% | 31.1% | 1.099 |
| 350 | 81.0% | 28.8% | 1.020 |

Whole set: 3,916,800 rows, survival 83.8%, walking 17.4%, max net_fwd 1.210 m/s
at `freq` / `leg_amp` / `hip_amp` / `hip_off` / `mu` still to be read off that row.

hip_phi 0 and 10 are not in this dataset (produced elsewhere) — the phase axis
is 34 of 36 points.

## 3. Budget reality

| | SU |
|---|---|
| RM allocation, total | **400** |
| Balance at start of this work | 372 |
| Spent before job 44960443 | 314 (calibration + phi 20 + phi 30) |
| Job 44960443, **actual** | **5,214.9** |
| **Campaign total, actual** | **5,527** |
| Allocation after the increase Ben requested | **5,228** |
| **Final overdraft** | **~410** |

The run went ahead on Ben's decision to get a complete c1 dataset in one piece
rather than a fragment. The allocation increase he requested mid-run landed
(400 -> 5,228 SU), so the final overdraft is **~410 SU** rather than the ~4,400
that was projected when the decision was made.

If more RM SU is requested, the number to ask for is **~5,900 per config**,
measured, not estimated.

---

## 4. How c1 differs from the fleet runs

| | laptop fleet | PSC |
|---|---|---|
| launch | `bash grid5/run_sweep.sh c3` | one `sbatch`, 32 slices inside |
| shards | cores−2 (11 on the Mac) | 128, whole node |
| output | one CSV per config | **one CSV per hip_phi slice** |
| revival | `run_machine.sh`, 5-min poll | in-job restart + `--requeue` |
| stopping | `touch WATCHDOG_OFF` | `scancel`; resume by resubmitting |
| cost | electricity | SU, ~165 per slice |

The per-slice split exists because `gs._load_done()` re-reads the entire CSV in
every shard at startup — with 32 slices in one file that is a ~600 MB read
times 128 shards per job.

**Code reuse.** `grid5_sweep.py` and its four dependencies run **byte-identical**
to this repo; the PSC driver imports the module and patches two globals
(`HIP_PHIS` to the one slice, `TAG` to the per-slice filename) before calling
`main()`. No fork of the sweep logic exists, so there is nothing to keep in
sync. `check_manifest()` validates protocol/config/K/mujoco_version/slip and
not the axis set, so a restricted-axis run is accepted.

---

## 5. Getting the data back, and its format

Bridges-2 has **neither `rsync` nor `scp`** (not a PATH problem — not installed,
and no module provides them). Transfer is `tar` over `ssh`, initiated from the
laptop; PSC cannot push (compute nodes have no outbound internet, and the
laptop is behind CMU NAT). The `tar` pipe does not self-verify, so every
transfer is followed by a sha256 comparison of both ends.

`psc/format_for_analysis.py` merges the per-slice files into the canonical
single-file form and repairs the three manifest fields the slice runs distort:

| field | slice value | repaired to |
|---|---|---|
| `axes.hip_phi` | the 1 value that slice ran | the full 36-value axis |
| `rows` | 115,200 | 4,147,200 — the *expected* total, matching what the local artifact means by `rows` (c3's manifest says 4,147,200 while its CSV holds 92% of that) |
| `repo_commit` | empty (the isolated tree is deliberately not a git repo) | `58a71216…`, the commit the copied sources came from |

Verified against the local c3 artifact: **15 of 17 manifest fields are
byte-identical**; the two that differ are `config` and `com_target`, which
should. The CSV header is byte-identical, and rows are copied verbatim as text,
never round-tripped through `float()`.

PSC provenance (which slices, which machine, per-file sha256, the
cross-platform-FP caveat) goes in a separate `.psc_provenance.json` sidecar so
the manifest's key set stays identical to a locally produced one.

**Rows produced on PSC are not bit-comparable with rows produced on the Mac**
(`docs/grid5_design.md:108`, "Map-vs-local diffs are cross-platform FP").
Aggregate metrics are comparable; per-row values are not. Any merged c1 dataset
should record which slices came from which machine.

---

## 6. Open items

- **`grid5/analysis/load5.py:162` requires `protocol == "grid5-v1"`, and every
  current manifest says `grid5-v2`.** This blocks c3 and c8 as much as the PSC
  data — it is not something the PSC work introduced. Not urgent while the v2
  analysis layer is still unwritten, but the v2 loader needs to accept v2.
- **`docs/grid5/RUN_GRID5.md` line 3 says 2,142,720 rows/config**, while the
  code, `count`, and every manifest say **4,147,200**. Probably stale relative
  to the 2026-08-26 axis changes (the {150..190} hip_phi trim being dropped and
  leg_amp 65 being removed). Worth reconciling — it is the kind of number that
  gets quoted later.
- **`QUIET_MAX_T` tail risk.** If a config stands without falling but never
  quiesces (`max|qvel|` stays ≥ 0.3), `t0` falls back to 10.0 s and the trial
  runs 29,000 steps instead of 21,001 — **38% more expensive**. Never triggered
  so far: `t_start` is 2.001 on c3 and 2.118 on c1. Check the distribution when
  moving to a new COM variant.
- **hip_phi 0 and 10** were produced elsewhere and are not in the PSC output.
  They are needed for a complete c1 dataset.

## 7. Where things live

- `../PenguMujoco_psc/` — isolated tree outside this repo (so nothing here can
  be picked up by the fleet's revivers or committed by accident), with
  `PSC_MEMO.md`, the driver/controller/slurm scripts, and the retrieved data
- `/ocean/projects/cis250009p/bgu/pengu/` — the same tree on Bridges-2
