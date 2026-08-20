# GRID-4 analysis session — 2026-08-20 (machine D)

Scope: the 5 complete configs (c1, c3, c4, c5, c6). c2 was partial at 264,949 /
1,818,000 rows and is auto-skipped everywhere. Everything below is measurement;
conclusions are Ben's to draw.

Package: `results/grid4_report/` (see its `INDEX.md`). 13 commits, `d5b2b46..4e6d71b`.

---

## 0. Corrections to earlier claims — read this first

Three things reported earlier in this session were **wrong** and were corrected in
place. They are listed first so nobody works off the retracted version.

| claim | status | where corrected |
|---|---|---|
| finalist `valid` = the pass criterion | **wrong** — it is a stepping gate | `1a41c4d` |
| gaits depend on the start transient, cannot self-start | **wrong** — all 5 self-start | `d6afb09` |
| the robot collapses at mu=0.5/0.7 | **wrong** — selection artifact | `42e5e61` |

### 0.1 `valid` is not `pass`

```
map    pass  = survived AND heading_align > 0.5 AND net_fwd > 0.05
final  valid = survived AND n_steps[L]>=2 AND n_steps[R]>=2 AND single_frac >= 0.3
```

`valid` is a **stepping** gate. A robot sliding forward without stepping scores
pass=1, valid=0. The apparent "c1 inversion" in the first version of INDEX.md was
this mismatch, not physics: c1's top-20 score 20/20 on the map at mu=0.1 and 0/20 on
valid, because at mu=0.1 c1 advances by shuffling (`ds_move_frac` 0.766).

Since `heading_align` is not carried in `finalists.csv`, the comparable proxy used
throughout is **`survived AND net_fwd > 0.05`**.

### 0.2 The robot does self-start

`compute_gait` applies `hip_off` **without** the alpha blend:

```python
hip_L = hip_off - hip_lean + alpha * hip_amp * max(0.0, sC)
```

so at the first instant of the blend the hip command steps by the full `hip_off`.
A 30 deg offset is a 30 deg shove, not a walking test (Ben's catch).

`staged_start_probe.py` separates them: ramp `hip_off` in over 4 s, hold 6 s until
the rocking decays, then blend the oscillation. All five configs' rank-1 gaits
self-start fine:

```
cfg  gait freq/phi/leg/hip/off   ABRUPT            STAGED
c1   1.8/260/125/20/50           2.50 m UPRIGHT    2.52 m UPRIGHT
c3   1.9/280/105/28/10           2.49 m UPRIGHT    2.48 m UPRIGHT
c4   1.8/340/115/28/10           5.01 m UPRIGHT    4.87 m UPRIGHT
c5   1.94/320/105/28/40          5.34 m UPRIGHT    7.28 m UPRIGHT
c6   1.96/240/105/28/20          8.62 m UPRIGHT    8.06 m UPRIGHT
```

The one counterexample (c6 freq=1.37 phi=0 off=20) is not a self-start failure
either — it is non-monotonic in the settle duration: 2 s and 4 s walk 6.3 / 6.1 m,
0.5 s, 6 s and 10 s fall. At `hip_off=20` the rock rate at the end of settle is still
0.082 rad/s, so what decides it is the **phase** of that residual rocking when the
oscillation blends in, not whether the robot has settled.

### 0.3 The high-friction collapse was a selection artifact

Every finalist was ranked at mu=0.1 and then re-run elsewhere, so "nothing survives
at mu=0.5/0.7" only ever meant "gaits chosen for ice do not transfer". Re-selecting
at mu=0.7 with the **same rule** (`nbhd>=0.8 AND pass>0`, sorted by `net_fwd_mean`):

| config | eligible cells @mu=0.7 | pass, selected@mu0.1 | pass, selected@mu0.7 |
|---|---|---|---|
| c1 (k=0, COM 1.05) | 11,061 | 0.00 | **0.85** |
| c3 (k=0, COM 1.31) | 110 | 0.00 | **1.00** |
| c4 (k=2, COM 1.05) | 13,595 | 0.00 | **1.00** |
| c5 (k=2, COM 1.20) | 551 | 0.00 | **0.90** |
| c6 (k=2, COM 1.31) | 39 | 0.00 | **0.95** |

All five walk on mu=0.7. Any cross-friction comparison must use each mu's own
selection; mu=0.3 and mu=0.5 have **not** been re-selected yet.

---

## 1. Checking three readings of the data

### 1.1 "kappa=2 is dominantly better than kappa=0" — NOT supported

Matched-COM head-to-head on robust volume (`nbhd>=0.8` cells) flips with mu:

| mu | COM 1.05: c4 vs c1 | COM 1.31: c6 vs c3 |
|---|---|---|
| 0.1 | 97,468 > 35,135 — k=2 | 3,697 > 2,948 — k=2 |
| 0.3 | 101,474 > 63,823 — k=2 | 125 < 1,468 — **k=0** |
| 0.5 | 22,063 < 24,163 — **k=0** | 36 < 252 — **k=0** |
| 0.7 | 14,224 > 11,471 — k=2 | 47 < 130 — **k=0** |

kappa=2 sweeps both matchups only at mu=0.1. At mu=0.5 kappa=0 takes both. The
supportable statement is "kappa=2 leads on low friction", not "dominantly better".

### 1.2 "c6 is fastest on ice" — confirmed

Best single cell (`best_net_fwd`, i.e. best-of-the-best) at mu=0.1:
c6 **0.602** > c5 0.574 > c4 0.370 > c3 0.171 > c1 0.170.

### 1.3 "kappa=2 on ice is more genuinely walking" — confirmed, and it is the GAIT

At mu=0.1, top-20 means:

| config | single_frac | ds_move_frac | clearance |
|---|---|---|---|
| c4 (k=2) | **0.517** | 0.323 | 0.011 |
| c6 (k=2) | 0.495 | **0.126** | 0.011 |
| c5 (k=2) | 0.404 | 0.240 | 0.008 |
| c3 (k=0) | 0.362 | 0.591 | 0.007 |
| c1 (k=0) | **0.229** | **0.766** | 0.004 |

Driven by kappa, not COM: c4 and c6 differ in COM (1.05 vs 1.31) but have nearly the
same single_frac; c1 and c3 are both kappa=0 and both low.

---

## 2. Peak is not width

With K=1 the map is binary (`pass_rate` is only 0.0 or 1.0 — verified), so width has
to come from neighbourhood structure. Define **thickness = robust cells (nbhd>=0.8) /
raw passing cells**.

| config | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|
| c1 | 0.613 | 0.698 | 0.436 | 0.305 |
| c3 | 0.156 | 0.080 | 0.028 | 0.027 |
| c4 | 0.677 | 0.704 | 0.381 | 0.316 |
| c5 | 0.565 | 0.392 | 0.079 | 0.065 |
| c6 | 0.146 | 0.013 | 0.009 | 0.013 |

At mu=0.1 peak speed and thickness rank almost oppositely:

| config | best net_fwd | raw pass | robust | thickness |
|---|---|---|---|---|
| c6 | **0.602** (fastest) | 25,376 | 3,697 | **0.146** (thinnest) |
| c5 | 0.574 | 107,243 | 60,625 | 0.565 |
| c4 | 0.370 | 144,043 | 97,468 | 0.677 |
| c3 | 0.171 | 18,952 | 2,948 | 0.156 |
| c1 | 0.170 (slowest) | 57,327 | 35,135 | **0.613** |

**Caveat on the width metric itself**: `nbhd>=0.8` averages over **freq x hip_phi
only**. `leg_amp`, `hip_amp` and `hip_off` contribute nothing to it. A 1-deg scan of
`hip_off` on c6 (freq 1.37, phi 0, leg 95, hip 28) at mu=0.1:

```
off 20 -> 7.35 m      off 21 -> 2.60 m (-65%)      off 22 -> 2.59 m
off 23 -> 1.44 m      off 24 -> 0.34 m (fell)      off 25 -> 1.56 m
```

The GRID-4 `hip_off` axis is sampled in 10-deg steps {10,20,30,40,50}, so it cannot
see a structure that changes 65% per degree, and 20 lands on an isolated spike. The
map's own data agrees: of the five `hip_off` values for that gait, **only 20
survives** — 10/30/40/50 all have `surv=0`. This is a protocol question for Ben;
GRID-4 is frozen and nothing was changed.

---

## 3. Why nothing "moved forward" at mu=0.7 — it was walking in circles

c1's mu=0.1-selected rank-1, run at mu=0.7:

```
n_steps 70   cadence 5.385   single_frac 0.616     <- actively stepping
path length      5.7432 m
net displacement 0.6127 m
straightness     0.107                             <- net/path, ~0 = looping
```

Not standing still: 70 steps and 5.74 m of path, ending 0.61 m from the start.
`net_fwd` measures only the world +y component, so a loop cancels out.

c1's straightness by mu: 0.504 (mu=0.1), **0.992** (0.3), 0.347 (0.5), 0.294 (0.7).
High friction converts the open-loop yaw bias into real turning; low friction lets
the feet slip so the path stays straighter.

With the mu=0.7 selection the same config goes **3.07 m forward at 7.4 deg off axis**
(c4: 3.95 m at 6.8 deg).

---

## 4. Traction is what enables stepping, not the other way round

Ben's question: at mu=0.7 the duty cycle is more single-support — is the passive
dynamics "refusing" to hold on two feet? It is the reverse.

**Same gaits** (the mu=0.7-selected top-20), only mu changed:

| config | metric | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|---|
| c1 | single_frac | 0.257 | 0.454 | 0.651 | **0.798** |
| c1 | ds_move_frac | 0.677 | 0.452 | 0.243 | **0.170** |
| c1 | slip_ratio | **3.076** | 0.961 | 0.413 | 0.216 |
| c4 | single_frac | 0.381 | 0.609 | 0.847 | **0.930** |
| c4 | slip_ratio | **7.031** | 0.460 | 0.208 | **0.092** |

Slip ratio (slip distance / path) collapses 14-77x; foot clearance rises 1.8-5.2x
(c5 5.0 -> 26.1 mm, c1 9.7 -> 25.3 mm, c6 16.5 -> 30.2 mm). Cadence and n_steps are
roughly constant, so the **commanded** duty cycle never changes — what changes is
whether the foot actually leaves the ground.

At mu=0.1, c4's slip_ratio of 7.03 means the feet slide 7x the distance the body
travels. Without grip there is nothing to push against, the leg cannot unload, the
swing foot clears only 5-10 mm, and both feet stay down — so the travel that happens
is sliding. Traction is the precondition for push-off.

Note `ds_move_frac` also falls partly because sliding is impossible at high mu. But
`single_frac` is a pure **time** measure and it still rises 2-3x, so the duty cycle
genuinely changes; it is not only a sliding artifact.

c6 is the exception worth noting: at mu=0.1 it already clears 16.5 mm and holds
single_frac 0.699, the only config still genuinely stepping on ice — at the cost of
16/20 surviving.

---

## 5. The heading drift: what it is not, and what it is

c6's champion (freq 1.96, phi 240, leg 105, hip 28, off 20) travels **75 deg off the
world +y axis** at mu=0.1 — 2.34 m forward, 8.81 m lateral. It is not crab-walking:
base-trajectory straightness is 0.954, i.e. a clean straight line aimed wrong. It
passes the map gate because `pass` only needs `heading_align > 0.5` and this gait
scores 0.716. **Whether HEAD_MIN=0.5 is too permissive is a decision for Ben** — it
admits a 75-deg course error.

Ruled out as the cause:

- **Landing / rocking transient.** With a 15 s hold the rock rate falls to
  0.00213 rad/s (from 0.2845) and yaw at end of hold is -0.019 deg, yet the final
  travel angle is +75.7 deg vs +75.1 deg. No effect. Yaw is ~0 for the entire hold
  and appears the instant walking starts (t=5.0 -> 5.5 s: +0.01 -> -12.63 deg).
- **Which leg swings first.** Verified the lead leg flips (hip_phi=240 -> L first,
  +180 shift -> R first). Drift: 42.0 deg vs 41.7 deg. No effect. The two probed
  gaits have opposite lead legs and drift the same way.
- **Model asymmetry.** The two feet are true mirrors in world frame to within
  0.001-0.006 mm; COM x = +4.9e-7 m; paired masses identical to 6 decimals; inertia
  diagonals to 7.
- **Hip-offset handedness.** World-frame hip axes are already opposite
  (`hip-L` [-1,0,0], `hip-R` [+1,0,0]), so equal-signed commands ARE mirror
  symmetric. The antisymmetric knob (`WALK_HIP_LEAN_DEG`) does not help: lean=0 is
  already near-optimal and +/-3 deg destroys the gait.

What it is: **roll and yaw are one motion.** Measuring roll about the robot's own
heading (walking window, 120 Hz):

```
roll  mean +2.73  RMS 25.30  range -35.6..+41.5  amplitude 38.6 deg
yaw   mean +32.75            range  -3.0..+72.4  amplitude 37.7 deg

Pearson corr(roll, yaw) = +0.960
slope = +0.729 deg yaw per deg roll
peak cross-correlation +0.970 at lag -8.3 ms   (gait period 510 ms)
```

Each roll cycle swings the heading by ~0.73x the lean, essentially in phase (8 ms of
a 510 ms cycle). The net course drift is the residue of the roll oscillating about
**+2.73 deg instead of 0**. So it is not an external bias pushing the robot over — the
waddle itself is off-centre.

### COM in the robot's own frame (the "wiper")

```
lateral sway  mean +4.5  RMS 37.7  range -52.6..+64.6 mm   peak-to-peak 117.2
COM height    mean 217.8           range 204.1..234.8 mm   peak-to-peak  30.8
```

Sway is nearly 4x the vertical bob. The trace is a **closed hysteresis loop, not a
line swept back and forth**: the left-going stroke (right foot lower) and the
right-going stroke (left foot lower) follow different paths, and the loop is not
centred. Same asymmetry as the +2.73 deg roll offset, made visible.

Note `gait_sweep.com_lat` measures lateral as `com[0] - contact[0]`, i.e. **world x**,
commented "forward travel is +y". That assumption fails once the robot yaws, so the
existing `com_lat` / `com_lat_rms` mix sway with heading.

---

## 6. Slow start does not change the result

All five configs' top-20 re-run at every mu with a staged start
(`finalists_staged.csv`). Pass fraction moves by at most 0.05 in 18 of 20 config/mu
cells; every zero at mu=0.5/0.7 stays zero except two cells going 0.00 -> 0.05. The
two real movers are both at mu=0.3: c5 drops 0.40 -> 0.10, c4 drops 0.50 -> 0.40.

Speed does move where pass/fail does not: c6 at mu=0.1 gains 45% (0.3131 -> 0.4536
m/s), c5 at mu=0.3 gains 59% (0.1413 -> 0.2250). The abrupt start was costing some
gaits speed.

Caveat: one schedule only (4 s ramp / 6 s settle / 4 s blend). Given the settle-phase
sensitivity in 0.2, c5's -0.30 may be a bad-phase hit rather than a property of slow
starting.

---

## 7. Bugs found

| bug | status |
|---|---|
| `grid4_report.py` wrote flat `top_gaits_cN.csv`, `grid4_finalists.py` read `cN/top_gaits.csv` — silent no-op, exit 0, empty figures | fixed upstream in `aeab0a3` |
| `sweep_watchdog.sh` has no `.done` check — every machine that completes a config spawns 14 shards every 10 min forever until someone touches `WATCHDOG_OFF` | **OPEN** — one-line fix, `[ -f "$CSV.done" ] && exit 0` |
| `compute_gait` applies `hip_off` without the alpha blend (step input at blend start) | documented, not changed (protocol frozen) |
| `com_lat` uses world x while the robot yaws | documented, `com_wiper.py` supersedes |
| `SETTLE` comment says "5+4+2 = 11s" while `T_TRANSITION` reads 2.0 at line 75; runtime is 4.0 | cosmetic |

---

## 8. Tools added (all in `physics/`)

| script | purpose |
|---|---|
| `grid4_demos.py` | demos for every config at every mu |
| `gait_probe.py` | one gait: forward vs lateral breakdown, heading angle, optional demo |
| `yaw_probe.py` | yaw/xy through hold -> blend -> walk, `--hold` override |
| `staged_start_probe.py` | ramp hip_off, settle, then walk; `--abrupt` for the original |
| `grid4_finalists_staged.py` | full rich eval with a staged start |
| `grid4_cross_staged.py` | abrupt vs staged cross figures |
| `grid4_support_figs.py` | single-support duty and double-support travel |
| `grid4_top_at_mu.py` | **select top-N at any mu**, same rule as grid4_report |
| `grid4_finalists_at.py` | rich eval for an arbitrary-mu selection |
| `grid4_dsmove_fig.py` | ds_move violin at any mu, `--compare` for two panels |
| `gait_liveplot.py` | render + live roll/yaw plot, self-calibrated axes |
| `com_wiper.py` | COM sway vs height in the body frame, `--video` |

Two axis gotchas these encode, both of which silently corrupt results if assumed:
`easytorso`'s body **+y points straight DOWN** (`[0,0,-1]` at neutral), so it cannot
be used as a heading axis; the root body `leftthighmotor` has the horizontal +y that
`heading_align` uses. `gait_liveplot.py` and `com_wiper.py` self-calibrate both axes
at the neutral stance rather than assuming.

---

## 9. Open items

1. **mu=0.3 and mu=0.5 need their own top-20** (`grid4_top_at_mu.py --mu 0.3` etc).
   Only mu=0.1 and mu=0.7 are currently selection-matched, so the cross-friction
   comparison has two of four columns done.
2. **c2 is still missing**, so the kappa=0 arm covers only COM 1.05 and 1.31 —
   kappa=0 vs kappa=2 is unmatched at COM 1.20 and c5 has no counterpart. The whole
   package needs a rerun when c2 lands.
3. **Everything is K=1.** `physics/topup_all.sh` upgrades hot regions to K=5.
4. **`HEAD_MIN = 0.5`** admits a 75-deg course error. Ben's call.
5. **`hip_off` at 10-deg steps** aliases a structure that changes 65% per degree.
   Ben's call; protocol is frozen.
6. **Watchdog `.done` check** — will hit every machine that finishes a config.
7. Suggested next diagnostic: wiper loops for all 5 configs at mu=0.1 and mu=0.7 side
   by side. Loop symmetry would be a direct readout of heading stability, which is
   currently only inferred from the +2.73 deg roll offset.
