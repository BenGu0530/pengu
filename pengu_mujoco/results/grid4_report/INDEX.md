# GRID-4 analysis package — index

Generated 2026-08-21 on machine D. **All 6 configs complete** (c2 landed from rml3).
Every μ level now has its own selection, so the gait × COM table is complete and
selection-matched.

```
config   kappa   COM ratio        config   kappa   COM ratio
c1       0       1.05             c4       2       1.05
c2       0       1.20             c5       2       1.20
c3       0       1.31             c6       2       1.31
```

---

## Read this first: selection matters more than anything else here

Ranking gaits at μ=0.1 and re-running them elsewhere measures **transfer**, not
capability. It produced a "collapse at high μ" that is not real. Re-selecting with the
same rule at each μ (`nbhd>=0.8 AND pass>0`, sorted by `net_fwd_mean`):

| config | pass @μ=0.7, selected@μ=0.1 | pass @μ=0.7, selected@μ=0.7 |
|---|---|---|
| c1 | 0.00 | 0.85 |
| c2 | — | 1.00 |
| c3 | 0.00 | 1.00 |
| c4 | 0.00 | 1.00 |
| c5 | 0.00 | 0.90 |
| c6 | 0.00 | 0.95 |

Use the **diagonal** files (`finalists_mu<XX>.csv`) for any cross-friction comparison.
`finalists.csv` is the μ=0.1 selection and is correct only at μ=0.1.

Also: finalists `valid` is a **stepping gate**
(`survived AND n_steps[L]>=2 AND n_steps[R]>=2 AND single_frac>=0.3`), NOT the map's
pass criterion (`survived AND heading_align>0.5 AND net_fwd>0.05`). Since
`heading_align` is not carried in finalists.csv, the proxy used throughout is
**`survived AND net_fwd > 0.05`**.

---

## The result: on the diagonal, pass stops discriminating — speed is the axis

All 24 config/μ cells pass 0.85–1.00 when gaits are chosen for the surface. What
separates the configs is speed of the passers (`cross/diag_matrix.png`):

| config | κ, COM | μ=0.1 | μ=0.3 | μ=0.5 | μ=0.7 |
|---|---|---|---|---|---|
| c1 | 0, 1.05 | 0.1659 | 0.2363 | 0.3071 | 0.1418 |
| c2 | 0, 1.20 | 0.1707 | 0.2479 | 0.2532 | 0.1198 |
| c3 | 0, 1.31 | 0.1516 | 0.2402 | 0.1036 | 0.0836 |
| **c4** | **2, 1.05** | 0.2930 | **0.4825** | **0.3139** | **0.2447** |
| c5 | 2, 1.20 | **0.3969** | 0.3426 | 0.1519 | 0.1481 |
| c6 | 2, 1.31 | 0.3131 | 0.2185 | 0.0993 | 0.0892 |

Fastest per μ: c5 (0.1), then **c4 at 0.3, 0.5 and 0.7**. c4 is the only cell strong
across the whole friction range; c5 and c6 are fast on ice and fall away steeply; the
κ=0 row is flatter but lower.

### Robust volume tells a different story — and answers a different question

Robust volume (`nbhd>=0.8` cells) measures how **big** the working region is, not how
fast the best gaits are. Matched κ=0 vs κ=2, now possible at all three COM levels:

| COM | μ=0.1 | μ=0.3 | μ=0.5 | μ=0.7 |
|---|---|---|---|---|
| 1.05 (c1/c4) | 35,135 vs 97,468 → κ=2 | 63,823 vs 101,474 → κ=2 | 24,163 vs 22,063 → **κ=0** | 11,471 vs 14,224 → κ=2 |
| 1.20 (c2/c5) | 26,258 vs 60,625 → κ=2 | 17,708 vs 18,812 → κ=2 | 10,398 vs 883 → **κ=0** | 4,388 vs 602 → **κ=0** |
| 1.31 (c3/c6) | 2,948 vs 3,697 → κ=2 | 1,468 vs 125 → **κ=0** | 252 vs 36 → **κ=0** | 130 vs 47 → **κ=0** |

**κ=2 wins 3/3 at μ=0.1, 2/3 at μ=0.3, 0/3 at μ=0.5, 1/3 at μ=0.7.** The crossover sits
between μ=0.3 and μ=0.5. So κ=2 buys a larger tolerant region on low friction and loses
it on high friction, while the diagonal speed table says κ=2 at COM 1.05 is still the
fastest everywhere. Those are not in conflict — a big region is not the same as a fast
best gait.

The COM ladder is monotonic within both gaits at every μ: 1.05 > 1.20 > 1.31, with an
order-of-magnitude cliff between 1.20 and 1.31 (κ=2 at μ=0.3: 101,474 → 18,812 → 125).

---

## Layout

```
INDEX.md          this file
REPORT.md         map-level tables, all 6 configs x 4 mu
cross/
  diag_matrix.png        2x3 gait x COM grid, selection-matched speed   <- the deliverable
  diag_speed_vs_mu.png   selection-matched speed, all 6 overlaid
  diag_pass_vs_mu.png    selection-matched pass fraction (all 0.85-1.00)
  diag_support_vs_mu.png single_frac and ds_move_frac, selection-matched
  ds_move_mu07.png       shuffle vs stepping, mu=0.1 vs mu=0.7, both own-selection
  volume_vs_mu.png  passfrac_vs_mu.png  speed_vs_mu.png  overlap_mu01.png   (map-level)
  roll_to_speed.png  ds_move_mu01.png                                       (mu=0.1 finalists)
  start_pass_vs_mu.png  start_speed_vs_mu.png  start_roll_vs_mu.png  start_delta.png
cN/
  heatmap.png                nbhd-mean pass over freq x hip_phi, one panel per mu
  top_gaits.csv              top 50 ranked at mu=0.1
  top_gaits_mu03/05/07.csv   top 50 ranked at that mu
  finalists.csv              top-20 selected at mu=0.1, re-run at all 4 mu
  finalists_mu03/05/07.csv   top-20 selected at that mu, re-run at all 4 mu
  finalists_staged.csv       mu=0.1 selection with a slow start
  demos/                     mp4 clips
c6/com_wiper_mu01.png        COM sway vs height in the body frame
```

Eligible cells per selection (how much room each config has on that surface):

| config | μ=0.3 | μ=0.5 | μ=0.7 |
|---|---|---|---|
| c1 | 63,323 | 23,502 | 11,061 |
| c2 | 16,105 | 9,406 | 3,950 |
| c3 | 1,218 | 209 | 110 |
| c4 | 100,055 | 21,114 | 13,595 |
| c5 | 16,923 | 796 | 551 |
| c6 | 108 | 29 (only 29 rows) | 39 (only 39 rows) |

c6 is down to 29–39 eligible cells at μ≥0.5, so its top-20 there is nearly the entire
qualifying population, not a selection from a broad field. Treat its diagonal numbers
accordingly.

---

## Supporting findings

**Traction is the precondition for stepping, not the result.** Same gaits, μ varied:
slip_ratio collapses 14–77× (c4: 7.031 → 0.092), foot clearance rises 1.8–5.2×
(c5: 5.0 → 26.1 mm), and `single_frac` rises 2–3× (c1: 0.257 → 0.798) while cadence and
n_steps stay flat. The commanded duty cycle never changes; what changes is whether the
foot actually leaves the ground. At μ=0.1 the feet skate (slip 7× the body's travel),
the leg cannot unload, and both feet stay down.

**"No forward progress at μ=0.7" was walking in circles.** c1's μ=0.1-selected rank-1 at
μ=0.7: 70 steps, 5.74 m of path, 0.61 m net, straightness 0.107. `net_fwd` only measures
world +y, so a loop cancels. With the μ=0.7 selection the same config goes 3.07 m forward
at 7.4° off axis.

**Heading drift is the waddle, not an external bias.** For c6's champion at μ=0.1, roll
(about its own heading) and yaw are one motion: corr +0.960, slope 0.729° yaw per ° roll,
in phase to 8 ms of a 510 ms cycle. Net drift is the residue of roll oscillating about
**+2.73° instead of 0**. Ruled out: landing transient (15 s hold changes 75.1° → 75.7°),
lead leg (42.0° vs 41.7°), model asymmetry (feet mirror to 0.006 mm, COM x = 4.9e-7),
hip-offset handedness (world hip axes are already opposite).

Worth noting against that: **c2's μ=0.1 champion walks nearly straight** — +5.4° off
axis, 2.68 m forward against 0.25 m lateral, the straightest of any config measured.

**Slow start does not change the picture**, but costs some gaits speed: c6 at μ=0.1 gains
45% and c5 at μ=0.3 gains 59% when started gently.

---

## Caveats

1. **All K=1.** `physics/topup_all.sh` upgrades hot regions to K=5.
2. **`nbhd>=0.8` covers freq × hip_phi only.** `leg_amp`, `hip_amp` and `hip_off`
   contribute nothing to the width metric. A 1° scan of `hip_off` on c6 shows 7.35 m at
   20 dropping to 2.60 m at 21 — 65% per degree — while the sweep samples it in 10°
   steps. Protocol is frozen; flagged, not changed.
3. **`HEAD_MIN = 0.5`** admits a 75° course error, which is how c6's champion passes.
4. Metrics on fallen trials are not meaningful; the figures average survivors only and
   draw gaps otherwise.
5. `gait_sweep.com_lat` uses **world x** while the robot yaws; `physics/com_wiper.py`
   supersedes it with a body-frame lateral axis.

## How to regenerate

```bash
cd pengu_mujoco
.sweep_venv/bin/python physics/grid4_report.py                    # map tables + heatmaps + top_gaits
.sweep_venv/bin/python physics/grid4_finalists.py --no-demo       # mu=0.1 finalists + cross figs
for m in 0.3 0.5 0.7; do
  .sweep_venv/bin/python physics/grid4_top_at_mu.py --mu $m
  .sweep_venv/bin/python physics/grid4_finalists_at.py --mu $m
done
.sweep_venv/bin/python physics/grid4_diagonal_figs.py             # the diagonal figures
.sweep_venv/bin/python physics/grid4_dsmove_fig.py --mu 0.7 --compare 0.1
MUJOCO_GL=egl .sweep_venv/bin/python physics/grid4_demos.py       # demos
```
