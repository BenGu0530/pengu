# GRID-6 hardware-realism plan — 2026-09-08

Two jobs, one rollout function. Both re-use `grid6/ff_sweep.rollout` (protocol of the
2026-09-02 feedforward study) so every number below is comparable with `ffact_mu05.csv`.
Nothing here touches the GRID-5 maps or `ff_sweep.py`; the map files' md5 are recorded in
`results/grid6_hw/map_md5_before.txt`.

## Why

The hardware sessions (c6 2026-08-29, c1 2026-08-30 / 09-03) disagree with the GRID-5
maps in a friction-dependent way:

| | sim (GRID-5 champion) | hardware | sim / hw |
|---|---|---|---|
| c6, mu 0.45 vs 0.5 | 0.158 | 0.097–0.122 | 1.3–1.6× |
| c6, mu 0.12 vs 0.1 | 0.512 (A rms 11.4°) | 0.058 (A rms 4.0°) | 6.5–9× |
| c1 vs c6 on ice | c6 faster (paper claim) | c1 faster: 0.066 vs 0.051 (net), 0.133 vs 0.075 (path), 0.144 vs 0.058 (their heading-projected v_fwd) | — |

Three suspects can be quantified in simulation; the fourth (MoI following the ballast)
is frozen as an experimental assumption at Ben's decision.

## Job A — realism filter of the GRID-5 maps (`grid6/realism_check.py`)

**Configs.** c1 (kappa 0, COM 1.05) and c6 (kappa 2, COM 1.31): the two builds that
exist. Model exactly as `grid5_sweep`: base `models/pengu1_31`, torso inertial COM slid
along world-up with `apply_com_variant` (c6: +8.73 mm → 1.3100), kappa from the config
table, torso PID kp 2.0 / ki 0.1 / ±45° (the same law and gains as firmware
`pengu_champ`).

**Cells.** Every (cell, mu) row of the GRID-5 map with `pass_rate = 1` and
1.20 ≤ freq ≤ 1.70, de-duplicated over mu → unique five-axis cells:

| | passing rows in 1.2–1.7 Hz | unique cells | over the 354 deg/s ceiling |
|---|---|---|---|
| c1 | 311,044 | 192,152 | 96% |
| c6 | 141,341 | 117,670 | 83% |

All hip_off values 0–50 are kept: this is a re-score of existing data, not a new grid.
Lists: `results/grid6_hw/filter_cells_c{1,6}.csv`.

**Friction.** mu = 0.12 (ice) and 0.45 (lab floor): the drag-test values (break static
friction, 20 pulls averaged, N / weight). These are static coefficients; MuJoCo's `mu`
is sliding, and mu_k ≤ mu_s, so the sim values are an upper bound on the surface. The
GRID-5 grid 0.1 / 0.3 / 0.5 / 0.7 is not used again.

**Variants.**

| config | variants | why |
|---|---|---|
| c1 | `act` | the robot now runs the feedforward torso, in which the sensor delay cannot enter; the map's kappa = 0 PID plus delay is the failure mode already documented (pengu-12/13), not something we would flash |
| c6 | `act`, `both` | kappa = 2 feedback is what the hardware ran on 08-29, with the delay in the loop; `both` is the robot, `act` isolates the actuator layer |

One rollout per (cell, mu, variant): 384k (c1) + 471k (c6). Output per rollout: `fell`,
`v_net`, `clear`, `drift`, `rollrms`, `axisrms`, `sat`, `fore`, `rearp5`.

**What comes out.** For each config and mu: how much of the passing region survives the
actuator model, where the surviving speed sits, and — the paper question — whether c1 or
c6 is faster at mu 0.12 once both are scored under the actuators they actually have.
Smoke test on c6's three fastest cells: two fall under `act` at both mu, one
(1.70/260/105/28/20) runs 0.516 m/s at mu 0.12 under `act` and falls under `both`.

## Job B — hardware_c1 sweep (`grid6/hw_sweep.py`)

A copy of `ff_sweep.py` with three changes: mu, no PID rollout, a straightness column.

**Model.** `models/hardware_c1` — the as-built CAD (2.2724 kg, COM ratio 1.050, five
actuators), kappa 0. Not the GRID-5 c1 (which is the 1.31 body with the COM slid).

**Grid** (unchanged from ff_sweep, Ben's choices of 09-03):

| axis | values | n |
|---|---|---|
| freq | 1.20 … 1.70 step 0.02 | 26 |
| hip_phi | 200 … 300 step 10 | 11 |
| leg_amp | 70 … 130 step 5 | 13 |
| hip_amp | 12 16 20 24 28 32 | 6 |
| hip_off | 20 25 30 35 40 | 5 |

111,540 cells × 2 mu. Above 1.70 Hz every cell is deep past the crank ceiling.

**Per cell.** (1) HELD rollout, torso at home → fit the hip-axis roll at the gait
frequency → A0, phi0. (2) Three FF rollouts with torso = A0·sin(phase + phi0 + 180° +
lead), lead ∈ {30, 50, 70}°; keep the one with the lowest torso roll rms. Rank on the FF
rollout's `v_net`. 4 rollouts per cell, 446k per mu.

**Columns.** As ff_sweep plus `straight` (see Lessons, 4).

## How the three layers are modelled (both jobs)

**1. Actuator limit (`act`).** A hard velocity cap on the four leg actuators (crank L/R,
hip L/R), applied to the position command every control step:

    held += clip(cmd − held, −354 deg/s · dt, +354 deg/s · dt)

354 ± 4 deg/s from twelve bench points (2026-08-30, air and ground pooled). Ben
2026-09-08: the cap only — the 29 ms one-pole ff_sweep also carried is dropped. A cell
that demands more than 354 deg/s is no longer excluded; the command simply cannot keep
up and the executed waveform is whatever the cap leaves. This is a *velocity* model;
MuJoCo's ±4.1 N·m force range is unchanged and the 2.000 A supply limit is not modelled.
c6 champion at mu 0.12: ideal 0.465 → cap 0.280 m/s, clearance 16.2 → 8.0 mm.

**Recorded per rollout** (both jobs, so nothing is re-simulated after the download):
`v_net` = whole-body COM (mass-weighted `xipos`) net displacement / window; `straight`;
`clear` (minimum over cycles of the per-cycle foot-clearance apex, mm) and `clear_ok`
(`clear` ≥ 10 mm — a constant, re-threshold from the `clear` column); roll drift; torso
and axis roll rms; clamp saturation; fore-aft CoM margins.

**Torso clamp.** As flashed for the build being scored: hardware_c1 sweep 25° (current
firmware `TORSO_CLAMP_DEG`); c6 filter 45° (`pengu_champ`, 2026-08-29); c1 filter 25°.

**2. Sensor/servo delay (`lag`).** The torso command is delayed 56 ms through a FIFO,
whatever produced it (held, FF, PID). Measured on the robot: corr(J[k], goal[k−2]) =
0.984 at 28 ms per sample (pengu-A/-B/-10); the torso peaks 56 ms after the axis, by
which time the axis is already returning in 76–90% of events.

**3. Feedforward torso (`ff`).** torso = A0·sin(2πf t + phi0 + 180° + lead), locked to
the leg phase. No measurement in the loop, so layer 2 cannot feed back; the servo's own
lag is cancelled by the lead. Mirrors firmware `TM_FF` (`ff_amp`, `ff_phi`); the
firmware's 2 Hz low-pass on the feedforward path is not modelled.

## What GRID-5 taught, and where it is applied

1. **The champions live above the actuator.** 96% / 83% of passing cells exceed
   354 deg/s; the first fix (a ceiling gate) threw them away and left 0.081 m/s as the
   best "safe" c1 cell. Now the clipping is simulated instead. Already measured: c6
   champion 0.504 → 0.351 under `act` alone.
2. **Knife edges.** 1.46/250/75/32/10 was a spike one grid point wide in two axes; a
   0.01° change in ff_amp flips 1.42/… between 0.6336 and a fall at 5.3 s; the GRID-5
   robustness filter ran at 0.05 Hz and never re-checked the 0.01 Hz champion. Job B
   keeps 0.02 Hz resolution and gets a stage 2: for the top 40 FF cells, the 3×3 box
   leg_amp ±1 / freq ±0.01, ranked on the *minimum* over the box (`plateau.py`).
3. **Friction mismatch.** GRID-5 at 0.1 / 0.3 / 0.5 / 0.7 vs a floor measured at 0.12
   and 0.45; both jobs use the measured values.
4. **Heading.** Half of the c1 ice takes walked in arcs (net / path 0.18–0.43); the only
   heading metric in the maps is the pass gate `heading_align > 0.5`. Job B records
   `straight` = net displacement / path length of the root, the path taken on the
   trajectory low-passed over one gait period so the waddle does not count. The same
   quantity was computed on the hardware mocap today (c1 0.50, c6 0.68 at mu 0.12).
5. **Speed definition.** `v_net` = |end − start| / T over the 13 s window — the same net
   displacement the hardware check used, not the heading-projected `v_fwd` that gives
   c1 its 2.5× on ice.
6. **K = 1 order statistics.** Every GRID-5 champion is one deterministic rollout; the
   box in (2) is the substitute for repeats.
7. **The torso loop.** GRID-5's loop reads true state at 1 kHz. c6's champion with 56 ms
   in that loop falls in 4–5 s at mu 0.1 / 0.12 while the hardware walked 17 takes at
   4° axis roll — the sim solution and the robot are in different regimes. That is why
   c6 runs both `act` and `both`: the filter will say whether the fall is general or a
   property of the knife-edge cells.

## Not modelled (say so in the paper)

Supply current limit and voltage sag (2.000 A hit on 3–34% of burst samples, 16 V
sagging to 9.94 V); daisy-chain current sharing; the firmware's ~36 Hz loop and 2 Hz
feedforward low-pass; IMU noise; static-vs-kinetic friction; contact compliance /
penetration (hardware sole height oscillates 30 mm p-p with no stance plateau,
`v_foot_min` 0.098 m/s — the sim equivalent has not been computed); torso clamp ±25°
(firmware) vs ±45° (sim).

## Execution

Bridges-2 RM-shared, 256 single-core array tasks per job, resumable by six-axis tuple.
Isolated non-git tree `$PROJECT/pengu_hw` built by `psc/make_run_tree.sh` from the
pushed branch (login node, zero SU).

| job | rollouts | est. RM SU |
|---|---|---|
| realism c1 `act` | 384k | ~110 |
| realism c6 `act,both` | 471k | ~135 |
| hw_sweep mu 0.12 | 446k | ~130 |
| hw_sweep mu 0.45 | 446k | ~130 |

Then: `--merge` each, verify the map md5 unchanged, and (Job B) the stage-2 box on the
top 40 per mu before anything is flashed.
