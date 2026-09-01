# Hardware session 2026-08-29 — mocap + power supply: A measured directly, and COT

OptiTrack + Rigol DP800 capture on the c6 robot (kappa=2, COM 1.31), two surfaces
(mu = 0.12 and 0.45). 26 mocap takes, 7 of them paired with a power log.

**The one result that changes what we know:** the hip-axis roll **A**, which the robot
has only ever *reconstructed* as `T − s·J`, was measured directly from the thigh markers.
It lands inside the reconstructed range. The 2.3x sim2real gap in body roll is therefore
not a measurement artefact.

Raw data: `/Users/ben/Downloads/归档/{c6_COMdata, c6_powersupply_data}` (176 MB of mocap
CSV kept outside the repo; point `PENGU_MOCAP_DIR` at it).
Code: `pengu_mujoco/hardware/`. Outputs: `HardwareData/cot_mocap_0829/analysis/`.

---

## 1. A, measured directly for the first time

Every take has its first and last 10 s discarded outright (Ben, 2026-08-29): the head is
the staged start (T_RAMP 4 + T_SETTLE 6) and the tail is where the robot was picked up or
fell. That costs two takes shorter than 20 s but removes the argument at the edges. All
headline numbers below are from the trimmed data; §5 shows the untrimmed run as a
robustness check.

| A rms | source |
|---|---|
| **4.01 deg** (8 bouts, mu=0.12, trimmed) | **this session — thigh rigid bodies, direct** |
| 3.64–5.63 deg (6 gain/rate conditions) | 2026-08-28 — reconstructed as `T − s·J` |
| 11.40 deg | sim, c6 champion at mu=0.1 |

The direct measurement sits inside the reconstructed range (the untrimmed run gives 4.31,
IQR 3.87-4.58, over 12 bouts -- see §5). Two consequences:

1. **`docs/hardware_session_2026-08-28.md` §6 open item 1 is closed.** The reconstruction
   is sound while walking, not only in the static bench check (which gave `d(roll)/dJ` =
   +1.09/+1.02, R² > 0.99 with the lower body held).
2. **The residual gap is real.** A on hardware is **35%** of sim (38% untrimmed). Everything upstream of it
   — actuators (crank 0.865, hip 1.006), the roll definition, the sign chain, the
   mechanical range, the discrete-loop stability — has now been measured and transfers.
   What does not transfer is how much the body itself rocks.

## 2. Attitude, by surface

Quality gate: a bout is used only if the two independent measurements of A (left thigh vs
right thigh) agree to better than 2.5 deg, the recovered gait frequency is within 10% of
the command, and the bout is at least 5 s. **12 of 18 bouts pass**; the six that fail are
listed in §5.

| | n | f_meas | A rms | T rms | **T/A** | T–A corr | speed |
|---|---|---|---|---|---|---|---|
| mu = 0.12 | 8 | 1.687 | **4.01** | 8.76 | **2.42** | 0.74 | 0.054 |
| mu = 0.45 | 4 | 1.941 | **2.62** | 8.19 | **2.98** | 0.62 | 0.097 |
| sim c6 @ mu=0.1 | — | 1.670 | 11.40 | 23.30 | **2.02** | — | 0.497 |

`f_meas` confirms which firmware ran: 1.672 vs a commanded 1.67 on the mu=0.12 surface
(`pengu_champ_k2_mu01`), 1.946 vs 1.92 on mu=0.45 (`pengu_champ_k2_mu05`, the boot-default
`n` gait — the `m` alternative at 1.21 Hz was not used).

**T/A = 2.98 at mu=0.45 is a finding, not a conclusion.** It cannot be attributed to
friction: that surface also ran a different gait (1.92 Hz, hip_amp 20, hip_off 10 vs
1.67/24/20), and it rests on 4 bouts. Two variables moved at once.

**T–A correlation is 0.62–0.74, not ~1.** The torso is not tightly phase-locked to the
body's lean. The obvious candidate is lag in the 57–80 Hz control loop; a cross-correlation
would settle it and has not been run.

## 3. COT

**Distance must be path length, not net displacement.** The robot curves hard: net
displacement is only **0.11-0.56** of the distance it actually walked (median ~0.3), which
is the roll-yaw coupling already on record. Dividing energy by net displacement therefore
charges the gait for a heading-control defect, and the result is unstable -- it swings
13.9-21.6 depending on where the window is cut, and one bout produced a COT of 526 because
the robot walked back on itself and the net displacement collapsed to 3 cm.

Against path length the number is stable to the point of being boring, which is what a
real measurement should look like:

| | n | mean power | path | **COT (dimensionless)** | energy/path |
|---|---|---|---|---|---|
| **mu = 0.12** | 4 | 21.0 ± 1.0 W | 12.05 m | **5.3** | **118 J/m** |
| **mu = 0.45** | 3 | 18.9 ± 1.6 W | 6.66 m | **3.9** | **86 J/m** |

Per bout: mu=0.12 gives 5.3 / 5.0 / 5.4 / 5.3 (spread 1.08x), mu=0.45 gives 3.8 / 3.8 /
3.9 (spread 1.03x).

**And it is insensitive to how the takes are trimmed**, because path length and energy both
grow linearly with walking time so a trim removes proportional amounts of each:

| trim | COT_path mu=0.12 | COT_path mu=0.45 | COT_net mu=0.12 | COT_net mu=0.45 |
|---|---|---|---|---|
| none | **5.3** | **3.8** | 21.6 | 17.1 |
| head 10 s, tail 10 s | **5.3** | **4.1** | 16.8 | 13.9 |
| head 10 s, tail 5 s | **5.3** | **3.9** | 17.4 | 20.7 |

The trim was worth doing anyway -- it is what let the *attitude* numbers be quoted without
arguing about the edges, and it cut the net-displacement spread from 2.8x to 1.18x -- but
the path-length COT never needed it.

Idle draw with the motors holding is 5.0 W; torque-off is 3.2 W (`test4.ROF`).
`m·g` = 22.29 N. Power is the mean over the whole current burst: the burst is already
current-threshold-defined so it excludes the standing part, and walking power is flat
enough (±1 W) that the exact sub-window does not matter.

**Electrical power is nearly constant across both surfaces** (21.0 ± 1.0 vs 18.9 ± 1.6 W,
under 10% apart), so the COT difference between surfaces is almost entirely a *speed*
difference, not a power difference.

Three limits that travel with every number above:

1. **These are lower bounds.** The supply enters constant-current at 2.000 A during
   3-26% of the samples inside a burst, with voltage sagging to 9.94 V. Peak power is
   clipped by the supply, not by the robot. `COT_railfix_lb` in the CSV replaces railed
   samples with the burst's un-railed p99 and is a tighter lower bound, still a bound.
   The only fix is to raise the current limit and re-run.
2. **The recorder undersamples the load.** The period is ~0.8 s against a 1.67 Hz gait
   whose power swings 5->32 W. The mean stays unbiased (the periods are incommensurate)
   but the standard error is 0.76-1.64 W, i.e. ±4-8% on ~20 W.
3. **Two variables separate the surfaces.** mu=0.45 also ran a different gait (1.92 Hz,
   hip_amp 20 vs 1.67, 24), so the 5.3 -> 3.9 difference cannot be assigned to friction.

**The sampling period is ~0.8 s, not the 1.0 s default.** Three single-burst files give
0.79 / 0.80 / 0.90 s against their mocap walking duration. COT is immune to this (power is
a segment mean; distance and time come from the mocap's own 360 Hz clock), so the period
only affects where burst boundaries fall.

## 4. Is the mocap any good?

For the torso and thighs, yes, and the evidence is checkable rather than impressionistic:

| check | value |
|---|---|
| Kabsch fit residual, median | 0.35–1.10 mm (p99 ≤ 2.3 mm) |
| **A from left thigh vs A from right thigh** | **0.76–1.5 deg rms, r = 0.97** |
| recovered gait frequency vs command | 1.672 vs 1.67 (0.1%); 1.946 vs 1.92 |
| hip axle single-axis quality s2/s1 | 0.081 |
| torso marker occlusion | ≤ 3.3% in all 26 takes; 10 takes have none at all |

The left-vs-right agreement is the strongest of these: it is the same physical quantity
measured twice through independent markers, agreeing to 1.5 deg on a signal whose own rms
is 4.0–4.3 deg — roughly 3:1.

Weaker parts: foot markers are occluded up to 80% in some takes (used only for
segmentation, never for attitude); no rigid-body poses were exported so every attitude is
a fit; and `markerconfig` changed mid-session, so **absolute zeros are not comparable
across takes** — only AC quantities are.

## 5. Method notes that matter

**Roll must be the tilt about a common hinge axis, not each body's own Euler roll.** On the
same data:

| definition | A_L vs A_R disagreement |
|---|---|
| tilt about the shared hinge axis | **0.76 deg rms** (r = 0.971) |
| each body's own z-x-y roll | **5.10 deg** (each is only 6.7 rms) |

The hip joints rotate about a *lateral* axis, so the thighs swing antiphase and each
thigh's own forward axis swings with it; a roll referred to that axis picks the swing up as
a spurious antiphase signal. This is the same trap `torso_control.py`'s docstring already
records for the sim, and the mocap side inherits the decision so the numbers stay
comparable to `pid.torso_roll` / `pid.axis_roll`.

**The hinge axis is built geometrically, not fitted.** Fitting torso-vs-thigh relative
rotation gives s2/s1 = 0.21 because that motion contains the hip DOF as well as the torso
DOF — two joints. Instead the hip axle is fitted (s2/s1 = 0.081, clean), and the hinge is
`normalize(cross(axle, z_world))`, the horizontal direction perpendicular to it.

**The DP800 ASCII `.ROF` must not be read with `rof2csv.py`.** Its line regex requires bare
V/A/W units, so the samples written while the output is off — `0.000mV,0.000uA,0.000uW,` —
fail to match and are skipped, while its time column is `row_index × period` over the
*surviving* rows. `p_mu0-12_1.ROF` drops 146 rows from the middle of the file, shifting
everything after it 146 periods earlier; `p_mu0-45_3.ROF` drops 4075 of 4177. Mixed rows
exist too (`15.999V, 0.290A,-28.926mW,`), so unit prefixes must be read per field.
`hardware/rof.py` keeps every physical line and uses the line index as the time base.

**Per-sample `W` is not `V×A`** (rms difference 0.9–5.8 W) but the means agree to
0.01–0.26 W. The meter samples the three fields at different instants inside one record
period while the load swings at the gait frequency. Means are used; both are reported.

**Six of 18 bouts failed the self-consistency gate** and are excluded from §2:
`mu012_take12` (L/R 7.5), `mu012_take2` (4.8), `mu012_take5` twice (6.7, 7.5),
`mu012_take3` (1.4, failed on frequency), `mu045_take5` (2.5).

**Robustness check — the trim changes nothing qualitatively.** Untrimmed, with the
segmenter alone doing the work, 22 bouts give 16 through the gate:

| | n | A rms | T rms | T/A | T–A corr | speed |
|---|---|---|---|---|---|---|
| mu = 0.12 untrimmed | 12 | 4.31 | 9.82 | 2.28 | 0.75 | 0.062 |
| mu = 0.12 trimmed | 8 | 4.01 | 8.76 | 2.42 | 0.74 | 0.054 |
| mu = 0.45 untrimmed | 4 | 3.44 | 10.21 | 3.09 | 0.61 | 0.099 |
| mu = 0.45 trimmed | 4 | 2.62 | 8.19 | 2.98 | 0.62 | 0.097 |

Every quantity moves by under 20% and no conclusion turns on it. Where the trim *does*
matter is COT, because that divides by a distance (§3).

**Segmentation was mandatory, and it found what Ben said was there**: `mu012_take4` is 26%
fallen, `take11/12` 12%, and all three `mu045_COT` takes end with the robot carried
(4–13%). Walking is detected from band-limited power of the hip differential angle rather
than from speed — a carried robot moves fast with its legs still, a slipping one is still
with its legs swinging.

## 6. What this does not support

- **Which surface is better.** The path-length COT does separate them (5.3 vs 3.9, tight
  within each), but the gait changed with the surface, so the comparison is confounded.
- **Absolute COT.** Lower bound, because of the supply clipping.
- **T/A = 2.98 being a friction effect.** The gait changed too.
- **Anything about the 5–20x speed gap** (0.023–0.091 m/s measured vs 0.497 sim). It is
  larger than the attitude gap and untouched by this analysis.

## 7. Open

1. **Why is A only 38% of sim?** Everything else now transfers. This is the remaining gap
   and it sits in contact / foot geometry / mass distribution.
2. **T–A correlation 0.62–0.74** — run the cross-correlation and get the lag.
3. **Re-run COT with the current limit raised.** Path length already makes the distance
   usable, so the clipping is now the only thing standing between this and a real number.
4. **No serial telemetry was captured during these takes**, so the mocap-measured T could
   not be compared sample-by-sample against the BNO055's `imu_roll`. That comparison would
   answer the 08-28 open item on dynamic IMU error and costs one more session.
5. `markerconfig1/2` and `newstart/newstart2` are undocumented; absolute zeros are not
   comparable across the change.

## Files

```
pengu_mujoco/hardware/rof.py            DP800 ASCII .ROF, no row dropped
pengu_mujoco/hardware/mocap.py          Motive 1.25 loader, Y-up -> sim axes, rightthugh alias
pengu_mujoco/hardware/rigid.py          Kabsch, reference shape, shared-axis fit, tilt_about
pengu_mujoco/hardware/attitude.py       T / A / J for one take
pengu_mujoco/hardware/run_attitude.py   driver over all takes + segmentation
HardwareData/cot_mocap_0829/analysis/attitude_0829_trim10.csv   18 bouts (headline)
HardwareData/cot_mocap_0829/analysis/attitude_0829.csv          22 bouts (untrimmed check)
HardwareData/cot_mocap_0829/analysis/cot_0829.csv               paired runs
```

Trim is `PENGU_TRIM_S` (default 10 s); `PENGU_TAG` names the output.
```
```
