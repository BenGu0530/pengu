# penguV3 Gait Sweep — Full Analysis & Findings

> Purpose of this document: a self-contained, data-first record of the open-loop
> gait parameter sweeps on the `penguV3` biped, written so a future agent (or human)
> can reconstruct WHY we swept, WHAT we swept, WHAT the data says, and WHETHER we got
> what we wanted — without re-reading the whole chat history. All numbers below are
> extracted directly from the sweep CSVs in `results/gait_sweep/`.

Last updated: 2026-07-05 (fine3c 100% complete).

---

## 1. Goal & motivation

Mentor's plan ("For Ben TODO"): study bio-inspired bipedal walking on a **slippery
floor** (floor friction μ = 0.7). The scientific questions:

1. Can the robot walk **naturally** (penguin-like) rather than just fast?
2. What gait minimizes the **required friction** `μ_req = |F_tangential| / F_normal`
   at the feet (i.e. is least likely to slip)?
3. How do **torso strategy** (active torso swing) and, later, **mass distribution /
   COM height** (penguin vs. human) change ground reaction forces and μ_req?

We (Ben) define "best / most natural" by TWO eye-level criteria, NOT by raw speed:

- **A. Distance actually covered forward** over a fixed time — measured per stride and
  as *net* progress, because a gait that turns/circles can rack up path length while
  going nowhere. This forced a metric redesign (see §4).
- **B. Clean, motivated foot clearance**: the swing foot must fully leave the ground
  (clearance > 0, no minimum height imposed), while the stance foot stays planted —
  a conventional walking gait, not shuffling.

Everything else (COM regularity, weight transfer, stance width) is **eval-only /
secondary** — used to judge grace after the fact, never to filter during the sweep.
Ben's explicit rule: *"优美只做 eval 不能当作筛选"* (grace is for eval, not a filter).

---

## 2. System under test

- **Model**: `penguV3` (`pengu_mujoco/penguV3/robot.xml`), a hardened CAD export.
  - Actuators trimmed to **5 core DOF** to match the V2 CPG controller: `hip-L`,
    `hip-R`, two leg-extension cranks (`crank1-L`, `crank1-R`, closed-loop
    crank-slider), and `torso`.
  - Motor limits: `forcerange = ±4.1 N·m` (XM430 stall). Position actuators.
  - `ctrlrange`: hip ±90°, torso ±45°. `timestep = 0.001`, `implicitfast`.
- **Controller**: open-loop CPG (`gait_config.py`). Sinusoidal patterns; swing/stance
  share ONE frequency but may be phase-shifted (treated as phase-offset DOF).
- **Posture**: ~25° forward pitch, achieved the STABLE way — a **symmetric hip offset**
  `hip_off = 30°` eased in over `T_HOLD=5s` + `T_TRANSITION=4s` after spawn (teleporting
  the root pitch caused immediate falls; the legs must be allowed to hold the lean).
- **Floor**: μ = 0.7. **Critical fix**: MuJoCo combines contact friction by
  element-wise *max* at equal geom priority, so a slippery floor (0.7) was being
  ignored in favor of the foot friction (0.9). We set `geom_priority[floor] = 1` in
  `friction_utils.set_floor_friction` so the floor dictates contact — otherwise the
  entire slippery-surface study would be invalid.

### Swept DOF (6)
`freq`, `hip_phi` (hip↔leg phase offset), `leg_amp`, `hip_amp`, `torso_amp`,
`torso_phi` (torso↔stance phase offset). Frequency band chosen around the **penguin
~1.27 Hz**.

---

## 3. Metrics (definitions)

Measured after an 11 s settle window over a 24 s sim:

| metric | meaning |
|---|---|
| `survived` | did not fall (root height stayed up) |
| `valid` | survived AND ≥2 clean steps/foot AND clearance>0 AND single_frac≥0.3 |
| `path_speed` | **total 2-D path length / time** — inflated by turning/circling |
| `net_fwd_speed` | **net forward displacement / time** — the real "distance covered" |
| `straightness` | net displacement / path length (1.0 = straight line) |
| `single_frac` | fraction of time in single support (1.0 = perfect L/R alternation) |
| `mu_req_p95` | 95th-pct of `|Ft|/Fn`, **stance-gated** (only when Fn>4N) = friction demand |
| `cadence` | steps/s |
| grace (eval): `lat_sep`, `COM_regularity`, `weight_transfer` | stance width, COM spectral purity, how fully COM leans onto the stance foot |

**Footfall de-bounce**: Schmitt trigger on normal force (enter stance >4N, exit <1N) +
`TD_REFRACTORY=0.25s` + `CLEAR_MIN=0.003m`, so contact chatter does not inflate cadence
or corrupt stride/clearance.

---

## 4. Why the metric had to change (the single most important lesson)

Early sweeps ranked by `path_speed`. This is **actively misleading**: the fastest
"path" gaits spin/curl in place. Hard evidence from the data:

- **fine2** global best by `path_speed`: `freq 1.96, hip_phi 210, torso_phi 0` →
  `path_speed = 0.461` but **`net_fwd_speed = −0.120`** — i.e. the "fastest" gait is
  net-**backward**.
- **fine3c** top-6 by `path_speed`: all `hip_phi 90, torso_phi ~320` at high cadence
  (~5 steps/s), `path_speed ≈ 0.50–0.52` but `net_fwd_speed ≈ 0.13–0.14`,
  `single_frac ≈ 0.77`. Of the **top-100 by path_speed, 82 have net_fwd < 0.15** —
  path_speed systematically selects curling gaits.

⇒ We added `net_fwd_speed` + `straightness`, and **all conclusions below rank by
`net_fwd_speed`**. `path_speed` is retained only as a diagnostic.

---

## 5. Sweep history (coarse → fine, resumable, sharded)

| tag | cells | resolution | purpose | outcome |
|---|---|---|---|---|
| coarse 6-DOF | ~thousands | coarse everything | find robust "highlands" via heatmaps | pointed (misleadingly, via path_speed) to hip_phi≈270, freq 1.5–2.0 |
| **fine1** | 22,440 | freq 0.01, hip_phi 250–300 | refine the coarse highland | had **no net_fwd metric yet**; best path_speed `f1.97/hip_phi290/torso_phi180` (single 0.73 — actually poor). Window **missed hip_phi=180**. |
| **fine2** | 26,928 | freq 0.01, hip_phi 30–330 (WIDE) | re-sweep with full phase incl. ±90 | **net_fwd best = `f1.59, hip_phi180, torso_phi0, leg110` → net_fwd 0.226, single 0.999, μ 0.531`**. Wide phase was essential (fine1 missed 180). |
| **fine3c** | **3,965,760** | freq 0.01, both phases 10°, amps densified (leg×5, hip×4, torso×3), full phase | penguin LOW-FREQ band 1.00–1.50, every feasible DOF fine | **complete**; see §6. |

Infrastructure notes (for reproducibility / auto-resume):
- fine3c runs **16 shards** (`SHARD_ID`, `N_SHARDS` env; cell `i` handled by shard
  `i % 16` → disjoint, zero-overlap). ~0.29 s/cell single-thread, but ~1 s/cell under
  16-way contention (24 threads ≈ 12 physical cores) → ~2.7 days wall.
- `run_grid.sh` = `flock`-serialized launcher; per-shard pidfile + `/proc` liveness;
  per-shard sentinel `<csv>.shard<i>of16.done`; master `<csv>.done` written by the last
  shard; cron `@reboot` + `*/10`. **Survived a real mid-run reboot** (resumed from
  checkpoint, no data loss).
- Header written once via `gait_sweep.py initcsv` so shards never race the header.
- Data integrity: 263 duplicate rows total (from one stray non-sharded worker that was
  killed early); de-duplicated by 6-tuple in all analysis. Final unique cells =
  3,965,760 (100%).

---

## 6. fine3c results (3,965,760 cells, penguin band 1.00–1.50 Hz)

### 6.1 Population
- `survived` = 2,770,104 (69.9%). `valid` = 2,381,622 (60.1%).
- **38.6% of "valid" gaits have `net_fwd_speed ≤ 0`** (they curl/back up while still
  passing the validity gate). Median net_fwd of valid gaits = only **0.009 m/s**.
  ⇒ Good forward gaits are rare islands; this justifies the exhaustive fine sweep.

### 6.2 Global top by net_fwd_speed
```
 freq  hip_phi leg hip torso torso_phi | path  net_fwd straight single cad  mu
 1.50   200    115  22   20     0       0.429  0.2278  0.536   1.000  2.92 0.474
 1.48   200    110  22   20    350      0.436  0.2272  0.522   1.000  2.92 0.498
 1.50   210    110  22   20     0       0.419  0.2259  0.540   1.000  2.92 0.514
 1.49   200    115  22   20     0       0.444  0.2250  0.508   1.000  3.00 0.476
 1.48   210    115  22   20     0       0.429  0.2218  0.523   1.000  2.92 0.460
 ...
 1.21   110    110  22   20    260      0.411  0.2192  0.534   0.846  2.77 0.597   <- "family A" interloper
```
The top is **one coherent gait family**: `hip_phi ≈ 200–210, torso_phi = 0,
leg 110–115, hip_amp 22, torso_amp 20`, all with **single_frac = 1.0** and **μ ≈ 0.47**.

### 6.3 The winning family is sharply frequency-selective (why 0.01 resolution mattered)
Fixing the winning family (`hip_phi∈{200,210}, torso_phi=0, leg115, hip22, torso20`)
and sweeping freq reveals **narrow, nonlinear regimes** — invisible at 0.05 spacing:

```
 freq   net_fwd  single  mu       regime
 1.00-1.23  ~0 (neg)  0.49-0.88 ~0.65   DOES NOT WALK (rocks in place / backs up)
 1.24    0.167   1.000  0.459   <-- SHARP ONSET (bifurcation between 1.23 and 1.24)
 1.24-1.31  0.17-0.215 1.000  ~0.47   PENGUIN PLATEAU (clean walking)
 1.32-1.41  0.07-0.13  1.000  ~0.48   DIP (partial regime, less progress)
 1.42-1.50  0.19-0.228 1.000  ~0.45-0.47  HIGH-FREQ PLATEAU (fastest)
```
Takeaway: forward walking in this family **switches on abruptly at 1.24 Hz** and has a
mid-band dip at 1.32–1.41. A coarse freq grid would have reported a flat/false picture.

### 6.4 Phase sensitivity is extreme
- **`torso_phi` must be 0** (torso swings *in phase* with stance). At `f1.27/hip_phi210`,
  `torso_phi=0` → net_fwd **0.215**; ANY other torso phase → net_fwd ≤ 0.04. The active
  torso helps ONLY when in phase; a phase-offset torso destroys progress.
- **`hip_phi` must be ≈ 200–210**. At `f1.27/torso_phi0`, hip_phi 200→0.178, 210→0.215;
  a weak secondary lobe exists at hip_phi 270–300 (net_fwd ~0.10, single 0.64–0.77);
  everything else ≈ 0. This is the antisymmetric swing/stance offset that produces a
  true step.

### 6.5 Amplitude trends (marginal means within the winning phase family)
- `hip_amp`: 16→−0.003, 18→0.000, 20→0.005, **22→0.011** (more hip swing = more progress).
- `torso_amp`: 12→−0.008, 16→0.002, **20→0.015** (active torso swing clearly helps →
  supports the mentor's "torso strategy" hypothesis).
- `leg_amp`: top individual gaits use 115, but the marginal mean slightly favors 95–100
  (interaction effect; leg extension trades off with the phase regime).

### 6.6 A second, less-clean family ("family A")
`freq 1.21, hip_phi 110, torso_phi 260, leg 110` → net_fwd **0.219** (as high as the
best!) BUT `single_frac 0.846`, `μ 0.597`. Higher friction demand and imperfect
alternation ⇒ faster-but-scrappier. Kept on record as an alternative regime, not the
recommended gait.

---

## 7. The two marked gaits (recorded in `BEST_GAITS.md`)

| | **A: fine2 high-band** | **B: fine3c PENGUIN** |
|---|---|---|
| freq | 1.59 Hz | **1.27 Hz (penguin natural freq)** |
| hip_phi / torso_phi | 180 / 0 | 210 / 0 |
| leg / hip / torso amp | 110 / 20 / 20 | 115 / 22 / 20 |
| net_fwd_speed | 0.226 | 0.215 (−5%) |
| single_frac | 0.999 | **1.000** |
| μ_req p95 | 0.55 | **0.469** (least likely to slip) |
| COM_regularity | 0.70 | **0.82** (most periodic) |
| lat_sep (stance width) | 0.086 m | **0.139 m** (wider, penguin-like waddle) |
| weight_transfer | **+0.94** | +0.57 |
| videos | `results/gait_sweep/fine2_best_netfwd.mp4` | `results/gait_sweep/fine3c_penguin_f1.27.mp4` |

Both live in the SAME gait family (`hip_phi 180–210, torso_phi 0`), confirming a robust
optimum spanning 1.27→1.59 Hz. **B (1.27 Hz)** is the recommended "natural penguin" gait:
it sits at the true penguin frequency, has perfect alternation, the **lowest friction
demand** (best for the slippery-floor question), the most regular COM, and a wider
penguin-like stance — at the cost of only 5% forward speed and a less complete weight
transfer than A.

---

## 8. Did we get what we wanted?

| requirement | status | evidence |
|---|---|---|
| Walk on μ=0.7 floor without slipping | **Yes** | best gaits have μ_req p95 ≈ 0.47 < 0.7 floor; friction fix validated |
| Rank by real forward distance, not circling | **Yes** | net_fwd_speed added; path_speed shown to select backward/curling gaits (82/100) |
| Clean foot clearance + planted stance (conventional gait) | **Yes** | winning family has single_frac = 1.0, clearance>0, debounced steps |
| Natural / penguin-like at ~1.27 Hz | **Yes** | gait B: f=1.27, single 1.0, wide stance, COM_regularity 0.82 |
| Exhaustive multi-DOF fine sweep (no locked DOF, 0.01 freq) | **Yes** | 3.97M cells, full phase coverage; revealed the 1.24 Hz bifurcation & 1.32–1.41 dip |
| Torso strategy affects gait/GRF | **Yes (partial)** | torso_amp 20 >> 12 for progress; torso_phi must be 0 — active *in-phase* torso helps, offset torso hurts |
| Mass-distribution / species (COM height) comparison | **NOT YET** | needs new Onshape exports (human-like COM); pipeline (this same sharded sweep) is ready to run per variant |

**Bottom line**: for the penguin-mass / active-torso / μ=0.7 condition we now have a
clear, robust, low-slip, natural gait at the penguin frequency (gait B), plus a full
map of how forward progress depends nonlinearly on frequency and (very sharply) on the
two phase offsets. The remaining open item is the COM-height/species comparison, which
is a data-generation task waiting on model variants, not a methodology gap.

---

## 9. Open items / next steps
1. **Human / mid / upper COM variants**: import Onshape exports, run the identical
   fine3c sharded sweep per COM, compare μ_req and the freq/phase gait map vs. COM
   height (the mentor's species comparison).
2. **CMA-ES cross-check** (`cma_search.py`): continuous 6-DOF optimization to confirm no
   between-grid-node optimum beats gait B.
3. Render `family A` (hip_phi 110) for eye comparison; decide if the scrappier-but-fast
   regime is ever preferable.
4. Regenerate/inspect fine3c heatmaps for the write-up.

## 10. Key files
- Sweep engine: `physics/gait_sweep.py` (sharding + `initcsv` + metrics + resume).
- Launcher/watchdog: `physics/run_grid.sh` (flock, per-shard pidfiles, sentinels).
- Report/video + grace metrics: `physics/gait_report.py`.
- Friction fix: `friction_utils.py` (repo root) (`geom_priority[floor]=1`).
- Data: `results/gait_sweep/sweep_v3_p25_fine{1,2,3c}_*.csv` (+ `.done` sentinels).
- Marked gaits: `results/BEST_GAITS.md`.
