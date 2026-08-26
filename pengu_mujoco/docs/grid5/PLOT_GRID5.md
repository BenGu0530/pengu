# PLOT_GRID5 — brief for a plot-only analysis session

Companion to `RUN_GRID5.md` (which is about *producing* the data). This memo is
about *reading* it. It is written to be the first thing a fresh session opens:
everything needed to load a GRID-5 map CSV and make a defensible figure is here,
including the traps that already cost us a wrong conclusion in round 1.

Protocol of record: `docs/grid5_design.md`. Round-1 analysis lessons that this
memo encodes: `docs/lessons_2026-08-25.md` §4.

---

## 0. Bootstrapping a plot session

Paste this into a fresh session as the opening message:

> Read `pengu_mujoco/docs/grid5/PLOT_GRID5.md` and follow it. This is a
> plot-only session: read the GRID-5 map CSVs, build figures into
> `results/grid5_report/`. Do not start, stop, or touch any sweep.

Then, first three commands:

    cd ~/Documents/ben_gu/ben_pengu/pengu/pengu_mujoco
    git status -sb                                    # expect friction-experiments
    ls -la results/gait_sweep/sweep_grid5_*.csv*      # what data exists here

Interpreter: `/home/rml2/anaconda3/envs/pengu_sim/bin/python` (numpy 2.4,
matplotlib 3.10). **`pandas` and `pyarrow` are NOT installed in either
`pengu_sim` or `.sweep_venv`** — do not write pandas code and do not install it.
The round-1 precedent (`physics/grid4_report.py`) is stdlib `csv` + a dense
numpy array, and that is the pattern to follow.

---

## 1. Scope — what a plot session may and may not do

**May:** read `results/gait_sweep/sweep_grid5_*.csv[.gz]` and their manifests;
build arrays, tables, figures, markdown reports into `results/grid5_report/`;
write NEW analysis scripts under `grid5/analysis/`; commit figures with
`git add -f`.

**Must not:**

1. **Never edit the sweep's own modules.** `grid5/grid5_sweep.py`,
   `grid5/gait_sweep.py`, `grid5/gait_config.py`, `grid5/torso_control.py`,
   `grid5/friction_utils.py` and every `grid5/*.sh` are live: shards are running
   and the watchdog restarts them, so an edit lands mid-run as silent protocol
   drift. (Lesson 5a — one such edit cost 18.5 h.) New analysis code goes in
   **new files** under `grid5/analysis/`.
2. **Never edit `physics/`.** That is the frozen GRID-4 pipeline and backup.
   Copy from it, don't modify it.
3. **Never run a simulation.** Anything that imports `mujoco` and steps a model
   (confirmation runs, fine scans, demos) is sweep-session work and competes for
   the CPU the shards are using. A plot session's only input is CSV.
4. **Never change a frozen definition to make a figure nicer.** Gates, tiers and
   the neighborhood rule are fixed in §5. If a definition needs to change, say
   so and ask; don't quietly redefine.
5. No Claude attribution anywhere — commits, figure captions, file headers.

---

## 2. Where the data is, and whether it is usable

    results/gait_sweep/sweep_grid5_<cfg>_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
                       ... .manifest.json          # ships WITH the csv, always
                       ... .csv.gz                 # committed snapshot (ship-back form)
                       ... .csv.snap.gz            # 6-hourly local watchdog snapshot
                       ... .csv.done               # marker: all shards finished
                       grid5_<cfg>_run.log         # shard log
                       machine_<name>.log          # queue log

**Completeness.** A finished config has **2,142,721 lines** (2,142,720 rows +
header) and a `.csv.done` marker beside it. Anything less is partial.

**A partial config is still plottable — but must be labelled.** Rows are written
in shard-interleaved cell order, so a partial file is a *strided subsample of the
hip_off blocks*, not a uniform random sample: cells iterate `hip_off` outermost
(`grid5_sweep.py:cells()`), so at 22% done you have roughly the `hip_off=0` and
`hip_off=10` blocks and nothing else. Never compare a partial config against a
complete one on any whole-grid statistic. Either restrict both to the axis range
actually present, or wait.

**Reading a live CSV.** Shards append with a flush per row, so a concurrent read
is safe in practice, but the last line can be torn. Count malformed lines and
drop them (grid4_report's `bad` counter is the pattern); if `bad > 5`, stop and
look — that is not tearing.

**The manifest is a gate, not decoration.** Before using a CSV, load its
`.manifest.json` and check `protocol`, `config`, `kappa`, `com_target`, `K`,
`axes`, `start`, `slip`, `gates`. If two CSVs disagree on any of those, they are
not comparable and must not appear on the same axes. Record `K` and
`repo_commit` in every figure caption or report header. (Lesson 5h.)

---

## 3. Configs

| cfg | kappa | COM ratio | round-1 twin |
|---|---|---|---|
| c1 | 0 | 1.05 | GRID-4 c1 |
| c2 | 0 | 1.20 | GRID-4 c2 |
| c3 | 0 | 1.31 | GRID-4 c3 |
| c4 | 2 | 1.05 | GRID-4 c4 |
| c5 | 2 | 1.20 | GRID-4 c5 |
| c6 | 2 | 1.31 | GRID-4 c6 |
| c7 | 0 | 1.10 | new |
| c8 | 0 | 1.40 | new |
| c9 | 2 | 1.10 | new |
| c10 | 2 | 1.40 | new |

Two families to plot against each other: **kappa** (0 = no torso follow, 2 =
torso follows) and **COM ratio** (1.05 low → 1.40 high). The interesting
comparisons are matched-COM (c1 vs c4, c2 vs c5, c3 vs c6, c7 vs c9, c8 vs c10)
and matched-kappa ladders (c1,c7,c2,c3,c8 at kappa=0; c4,c9,c5,c6,c10 at kappa=2).

Do **not** write species labels ("penguin gait", "human COM") into a figure
without Ben confirming the mapping for that figure. It is an interpretation, not
a column.

---

## 4. The grid

| axis | values | n |
|---|---|---|
| freq | 1.21 … 2.00 step 0.01 | 80 |
| hip_phi | 0…140 and 200…350 step 10 (**150–190 removed**) | 31 |
| leg_amp | 85, 95, 105, 115, 125, 135 | 6 |
| hip_amp | 12, 16, 20, 24, 28, 32 | 6 |
| hip_off | 0, 10, 20, 30, 40, 50 | 6 |
| mu | 0.1, 0.3, 0.5, 0.7 | 4 |

Cells per mu = **535,680**; rows per config = **2,142,720**.

Natural array shape, matching round 1: `A[mu, freq, phi, leg, hip, off]` =
`(4, 80, 31, 6, 6, 6)`. One float32 plane is 8.6 MB; all 24 numeric metrics is
~206 MB, which fits comfortably in RAM.

### 4.1 The hip_phi seam — read this before writing any neighborhood code

GRID-4's phi axis was contiguous 0–350, so `np.roll(..., axis=phi)` was a correct
circular neighborhood everywhere. **GRID-5 cut the 150–190 dead band**, so array
index 14 (`phi=140`) and index 15 (`phi=200`) are adjacent *in the array* and 60°
apart *in reality*. A rolled neighborhood silently averages two unrelated regions
across that seam.

Round-1 code copied unchanged will produce this bug. The fix:

```python
PHI = np.array([0,10,...,140, 200,...,350])          # from the manifest
adj = (np.abs(PHI[:,None] - PHI[None,:]) % 360)
adj = np.minimum(adj, 360 - adj) <= 10               # true circular adjacency
# -> 350 and 0 ARE neighbours; 140 and 200 are NOT.
```

Wrap at 350→0 is genuine and must be kept.

### 4.2 Comparing GRID-5 against GRID-4

Robust-volume **counts** are not comparable across rounds — the grids differ in
size (454,500 vs 535,680 cells per mu, +17.9%). Two valid options, and the
figure must say which:

- report a **fraction** of the config's own grid, not a count; or
- restrict both rounds to the shared subgrid: freq ≥ 1.21, phi excluding
  150–190, leg ≤ 125, hip ≤ 28, off ≥ 10 → **310,000 cells per mu**.

Even on the shared subgrid the start protocol differs (staged + rest lean vs
abrupt hip_off step), which is the whole point of round 2, so any round-1
overlay is a *protocol* comparison, not a config comparison. Label it that way.

---

## 5. Column dictionary and frozen definitions

31 columns. The first 12 are byte-identical in meaning to GRID-4; the last 19
are new.

### Axes (0–5)
`freq, hip_phi, leg_amp, hip_amp, hip_off, mu`

`mu` is the **nominal** value; each trial actually ran at `mu * U(0.95, 1.05)`.

### Aggregates over the K repeats (6–11)

| column | meaning | trap |
|---|---|---|
| `pass_rate` | share of K repeats meeting the pass gate | **use this, don't recompute** |
| `surv_rate` | share that stayed upright (root z ≥ 0.05 m throughout) | the filter for everything below |
| `net_fwd_mean` | mean forward speed [m/s] over K | can be **negative** (retrograde) |
| `net_fwd_min` | worst repeat | equals mean while K=1 |
| `slip_mean` | mean of legacy `slip_ratio` = `slip_dist / path` | kinematic-only; **counts rolling as slip** — that is why the dual criterion exists |
| `head_mean` | mean `heading_align` = (body +y facing) · (travel direction) | facing-vs-travel, **not** travel-vs-world. NaN if the trial never reached the measurement window |

**Pass gate (frozen, identical to GRID-4):**
`survived AND heading_align > 0.5 AND net_fwd > 0.05 m/s`. Already applied —
`pass_rate` is exactly this.

**Post-hoc tiers** (recomputed from the map, never re-run):

| tier | rule |
|---|---|
| surv-only | `surv_rate > 0` |
| pass | `pass_rate > 0` |
| strict-heading | pass ∧ `head_mean >= 0.9` |
| clean-pass | pass ∧ low slip — **see the open item in §8**, the design's `slip_ratio2` is not a column |

**While K = 1**, `pass_rate` and `surv_rate` are binary and
`net_fwd_min == net_fwd_mean`. Say "K=1" in the caption; these become real
fractions only after the K=5 topup.

### Extended metrics (12–30)

All 19 are **nan-means over the K repeats**, except `fall_phase`.

| column | meaning | notes |
|---|---|---|
| `t_start` | time [s] the staged start reached quiescence (max abs qvel < 0.3), min 2 s | **`t_start >= 9.9` means it hit the 10 s cap and never settled** — a first-class failure readout |
| `t_fall` | time [s] of the fall | NaN if it survived |
| `fall_phase` | string tally `"hold:1"` / `"trans:1"` / `"settle:1"` / `"walk:1"`, `""` if survived | not a float — see §6 |
| `slip_dist2` | dual-criterion slide distance [m], force-weighted integral | the round-2 slip number |
| `roll_dist` | contact-centroid travel while sticking [m] = rolling | pairs with `slip_dist2` |
| `slip_frac` | share of loaded samples classified slipping | dimensionless 0–1 |
| `cone_util_p50/p95` | percentiles of `|Ft| / (mu * Fn)` | the GRF readout; 1.0 = pegged at the cone |
| `fn_peak`, `fn_mean` | normal force [body weights] | |
| `lat_disp` | lateral displacement [m] in the **initial heading frame** | the c6 sideways-walk metric, now a column |
| `lat_vel_rms` | lateral velocity RMS [m/s] | |
| `e_pos` | positive mechanical work [J] | |
| `cot_net` | `e_pos / (m g * |net displacement|)` | **NaN when net < 0.02 m** — a stalled gait has no COT, not a bad one |
| `cot_path` | `e_pos / (m g * path length)` | defined for loopers too |
| `power_mean` | `e_pos / walk_time` [W] | |
| `imu_roll_mean` | torso lateral lean [deg], gravity method, relative to per-trial rest pose | signed |
| `imu_roll_rms` | torso roll RMS [deg] | |
| `imu_pitch_rms` | torso pitch RMS [deg] | |

A slip decomposition is the point of round 2: `slip_dist2` vs `roll_dist`
separates sliding from rolling, which `slip_mean` cannot. Calibration evidence is
in `docs/grid5_design.md` (rolling regime: 100% rejected by the cone leg; sliding
regime: 94.4% classified).

**Robust neighborhood (frozen):** mean of `pass_rate` over freq ±2 and phi ±1
(true circular adjacency, §4.1); freq edges NaN; **divide by the count of valid
contributors**, not a fixed 15, because the seam columns have fewer. A cell is
*robust* iff its neighborhood mean ≥ 0.8. Note in the caption that this differs
from GRID-4's fixed `/15`, so the two rounds' volumes are not identical
constructions.

**Region thickness (frozen):** robust cells / raw passing cells, per (config, mu).
It is computed over freq × phi only — a thin region on the other three axes will
not show up. Say so wherever it is used.

---

## 6. The loader

Write it once, in `grid5/analysis/load5.py`, and have every figure script import
it. Contract:

1. Take a config name, resolve `.csv` or `.csv.gz` (prefer `.gz` — it is the
   shipped, complete form).
2. Load and validate the manifest; return it alongside the data.
3. Parse with stdlib `csv`, index into `A[mu, freq, phi, leg, hip, off]` planes
   built from the **manifest's** axis lists, not hardcoded ones.
4. `fall_phase` is a string: expand it into four int8 planes
   `nfall_hold / nfall_trans / nfall_settle / nfall_walk` by parsing
   `"name:count|name:count"`. Never coerce it to float.
5. Count and report `rows_loaded`, `malformed`, and `complete = rows == 2142720`.
6. **Cache.** Parsing 2.1M rows is ~40–60 s per config; ten configs re-parsed on
   every tweak is ten minutes a figure. Save
   `results/grid5_report/cache/<cfg>.npz` keyed on the manifest's `repo_commit`
   plus the CSV's size and mtime, and reload from it when the key matches. Every
   later figure then loads in under a second.

---

## 7. Figure catalog

Ordered so that anything buildable from a partial config comes first. Output to
`results/grid5_report/`; per-config figures under `results/grid5_report/<cfg>/`,
cross-config under `results/grid5_report/cross/`.

**Map level — needs only the CSV.**

| # | figure | question it answers |
|---|---|---|
| F1 | `<cfg>/heatmap.png` — 4 mu panels, neighborhood-mean pass over freq × phi at the best (leg, hip, off) slice | where does this config walk at all? |
| F2 | `cross/volume_vs_mu.png` — robust fraction vs mu, one line per config | which configs keep a usable region as grip returns? |
| F3 | `cross/passfrac_vs_mu.png` — raw `pass_rate > 0` share vs mu | same, without the neighborhood filter |
| F4 | `cross/speed_vs_mu.png` — best `net_fwd_mean` among passers vs mu | peak capability (**best-of-best, not a mean** — label it) |
| F5 | `cross/thickness_vs_mu.png` — robust / raw passing | is the region a plateau or a spike? |
| F6 | `cross/com_ladder.png` — x = COM ratio (1.05, 1.10, 1.20, 1.31, 1.40), two lines (kappa 0 and 2), one panel per mu | **the round-2 headline**: where is the COM cliff, and does kappa move it? |
| F7 | `cross/overlap_mu01.png` — Jaccard overlap of passing cell sets between configs | do the configs walk in the *same* place or different places? |

**Failure decomposition — new in round 2, this is what round 1 could not do.**

| # | figure | question |
|---|---|---|
| F8 | `cross/fall_phase_stack.png` — stacked share of hold/trans/settle/walk falls per (config, mu) | are the falls *walking* failures or *start* failures? Round 1's c6 was 91.7% pre-measurement |
| F9 | `cross/tstart_vs_mu.png` — share of cells with `t_start >= 9.9` | which configs cannot even stand still |
| F10 | `<cfg>/nonpasser_breakdown.png` — of surviving non-passers, split heading-fail vs net-fail vs both | is scarcity a steering problem or a propulsion problem? |

**Physics readouts.**

| # | figure | question |
|---|---|---|
| F11 | `cross/slip_vs_roll.png` — `slip_dist2` vs `roll_dist` scatter (passers only) per mu | is a low-mu "walk" real stepping or slide-walking? |
| F12 | `cross/cone_util.png` — `cone_util_p95` distribution per (config, mu) | how close to the friction cone does each family operate? |
| F13 | `cross/cot_frontier.png` — `cot_net` vs `net_fwd_mean` Pareto front, passers only, per mu | the speed/economy tradeoff, and who owns the frontier |
| F14 | `cross/lat_disp.png` — `lat_disp` and `lat_vel_rms` vs mu | the sideways-walking question round 1 could only chase with one-off probes |
| F15 | `cross/imu_roll.png` — `imu_roll_mean` vs `imu_roll_rms` | separates a held lean from a waddle (see trap T5) |

Start with F2, F4, F6, F8. Those four carry the round-2 story; the rest are
support.

---

## 8. Selection tables — how far a plot session may go

`docs/grid5_design.md` freezes a five-step selection. Steps **1 and 2 are pure
post-hoc reads of the map** and belong to this session:

1. eligibility = pass (report the strict and clean tiers alongside);
2. three **independent** top-20 tracks per **(config, mu)**:
   - **T-speed** — `net_fwd_mean` descending
   - **T-cot** — `cot_net` ascending, floored at `net_fwd_mean >= 50%` of that
     cell's T-speed #1
   - **T-slip** — slip ascending, same floor

Write these to `results/grid5_report/<cfg>/select_mu<XX>.csv`, one file per mu.
**One file per mu is not optional** — see trap T1.

Steps 3–5 (topup K=5, confirmation on offset seeds, champion neighborhood fine
scan) all run simulations and are **not** plot-session work. Hand them back.

### OPEN ITEM — the T-slip track cannot be built as written

The design specifies T-slip and clean-pass in terms of `slip_ratio2`. **There is
no such column**, and it cannot be derived from the map: it would be
`slip_dist2 / path`, and `path` is not carried in the aggregated CSV. What the
map does have:

- `slip_frac` — share of loaded time classified slipping (0–1);
- `slip_dist2 / (slip_dist2 + roll_dist)` — share of contact travel that slid;
- `slip_mean` — the legacy kinematic ratio the dual criterion was built to replace.

Recommendation: use **`slip_frac`** as the T-slip key and report
`slip_dist2/(slip_dist2+roll_dist)` beside it as a cross-check. But note the
clean-pass threshold `<= 0.05` was Ben's cut on the *legacy* `slip_ratio`; it has
not been calibrated for `slip_frac` and must not be carried over by name. Pick
the threshold from the observed `slip_frac` distribution (the calibration probe
puts rolling near 0 and sliding near 0.94, so the gap is wide), show the
histogram that justifies it, and get Ben's sign-off before the track is frozen.

Do not silently substitute a column and keep the old threshold. That is exactly
the shape of the round-1 `valid`-vs-`pass` error.

---

## 9. Traps — each of these already produced a wrong published conclusion

**T1. Selecting at one mu and scoring at all mu measures the selection.**
Round 1's "high-friction collapse" was an artifact of a top-20 chosen at mu=0.1
and re-scored elsewhere; giving each mu its own top-20 moved pass from 0.00 to
0.85–1.00. GRID-5's selection is per-(config, mu) *by design* — keep it that
way, and never build a cross-mu comparison on a single-mu selection.
(`results/grid4_report/INDEX.md:31`, lesson 4a.)

**T2. Check what a column means in the code that writes it.** Round 1 read
`valid` (a stepping gate: ≥2 steps/leg ∧ single_frac ≥ 0.3) as the pass
criterion and published an inverted c1 conclusion. Grep `grid5/gait_sweep.py`
for the column name before trusting it. (Lesson 4b.)

**T3. Never average an extended metric over non-survivors.** Every ext column is
integrated over the whole episode including the fall. A real row from c6:
`surv_rate=0`, `imu_roll_mean = -116.1 deg`, `imu_roll_rms = 118.8` — that is a
robot lying on the floor, and it will drag any mean it enters. **Filter
`surv_rate > 0` (or `pass_rate > 0`) first, and print n per bar.** Where a config
has no survivors, draw a gap and label it, never a zero.

**T4. A scalar can hide the behaviour.** "Nothing moves at mu=0.7" turned out to
be walking in circles: 70 steps, 5.74 m of path, 0.61 m net, straightness 0.107.
When a summary says nothing happened, check the path integral before believing
it. (Lesson 4c.)

**T5. RMS cannot separate a waddle from a lean** — `RMS² = mean² + var`, so a
±30° oscillation and a held 30° lean give the same RMS. That is why
`imu_roll_mean` and `imu_roll_rms` are separate columns; plot them **together**
(F15), never RMS alone. (Lesson 4d.)

**T6. `cot_net` NaN is a stall, not a good score.** It is NaN whenever net
displacement < 0.02 m. `nanmin` over a config will happily hand you the most
efficient of the gaits that actually moved while silently dropping every gait
that didn't. Always report the n behind a COT number.

**T7. `net_fwd_mean` goes negative.** Retrograde walking is common. Colormaps
must be diverging around 0 or explicitly clipped with the clip stated.

**T8. Best-of-best is not typical.** `max(net_fwd)` among passers is one cell,
often a spike one grid step wide. Always pair a peak speed with its region
thickness (F5). Round 1's fastest config on ice had the thinnest region of all
five.

**T9. Look at the artifact.** Round 1's most convincing false positive agreed
with itself across every summary column and was a body parked at −47° and held
there. Anything with a shape gets rendered and watched — but that is a sim task,
so flag it for the sweep session rather than doing it here. (Lesson 4e.)

---

## 10. House style

- **Visual encoding is frozen in `docs/grid5/PLOT_STYLE.md`** (Ben, 2026-08-26),
  implemented as `grid5/analysis/style5.py`: colour + linestyle = gait (κ),
  marker shape = COM ratio, greyscale twin mandatory. This SUPERSEDES the
  round-1 per-config colours (`physics/grid4_support_figs.py:24`) — do not
  reuse them.
- Config labels: `c4 (κ=2, COM 1.05)` — never a bare `c4` in a legend
  (`style5.label_for`).
- Every figure states, in caption or title: **K**, the tier used, and whether the
  quantity is a mean or a best-of-best — enforced by `style5.finish()`, the
  mandatory save path for every figure.
- Missing data is a gap plus a label ("no survivors"), never an implicit zero.
- `matplotlib.use("Agg")`, `dpi=130`, `tight_layout()` (all inside style5).
- Regenerate `results/grid5_report/REPORT.md` and `INDEX.md` alongside the
  figures; the index carries the caveats so a reader cannot miss them (that is
  what saved the round-1 diagonal note from being lost).

## 11. Shipping

`results/` and `*.png` are gitignored — figures enter the repo only via
`git add -f` (112 round-1 artifacts are tracked that way). Branch is
**`friction-experiments`**; confirm with Ben before pushing. Commit message
describes the finding, not the tooling, and carries no AI attribution.

## 12. Data status at the time of writing (2026-08-26)

| cfg | machine | state |
|---|---|---|
| c4 → c2 → c9 | naomio | c4 running |
| c6 → c7 | rml2 | c6 running, 474,490 / 2,142,720 rows |
| c5 → c10 | rml3 | c5 running |
| c3 → c8 | mac | to deploy |
| c1 | laptop | to deploy |

Nothing is complete yet. The first real plot session should therefore build and
validate the loader, the cache, and the F1–F8 scripts against **partial** c6 —
restricting every statistic to the `hip_off` blocks actually present and saying
so — so that the package runs end-to-end the moment the first `.done` lands.
Validate the harness on one config before queueing figures behind it (lesson 5d).
