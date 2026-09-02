# The torso controller was the sim2real gap — and a phase-locked feedforward closes it

**2026-09-02. This is a HARDWARE study of the c1 build — the COM 1.05 robot, model
`models/hardware_c1`. It is not a simulation result and none of it changes a swept map.
It was necessary because every gait the maps selected failed on the robot, and this is
what the failure turned out to be.**

## Where the gap actually was

Everything below is measured, on the robot, with both gait amplitudes at zero or with one
component at a time, so the pieces separate.

**The plant is right.** With the torso passive — commanded to its home angle, no controller
running — the model and the robot agree on body roll to within a few percent:

| gait | model, torso held | robot, torso held |
|---|---|---|
| `1.39/240/80/16/30` | 20.0 deg p2p | 21.1 |
| `1.39/240/80/32/30` | 42.3 | 42.8 |
| axis roll amplitude at the gait frequency | 7.55 deg | 7.49 |

So the rigid bodies, the contacts, the crank-slider linkage and the mass distribution are
not the problem. Two things that looked like plant errors earlier in the day were not:
the "leg extension rolls the robot 59x more than the model" reading came from a bout with
the torso loop CLOSED, and so did the `|axis/J|` 0.97-against-0.67 discrepancy. Both are
controller artefacts.

**The torso loop is the gap.** Closing it makes the roll worse, not better:

| gait | loop held | loop closed |
|---|---|---|
| `hip_amp 16` | 21.0 deg p2p | 66.7 |
| `hip_amp 32` | 42.5 | 59.2 |

And it does so by throwing the lower body around. Holding the torso level is paid for by
the reaction on the legs, and the torso is 49% of the robot's mass:

| | hip-axis roll rms | torso world roll rms |
|---|---|---|
| loop held | 5.39 | 5.93 |
| loop closed | **23.47** | 13.34 |

**The mechanism is 56 ms of delay.** Across three independent closed-loop recordings the
torso joint reaches its extreme 56 ms after the hip axis reaches its own, by which time
the axis is already returning in 76-90% of events, and `corr(dJ, d_axis)` peaks at -0.94 at
exactly -56 ms. The correction has the right sign and the wrong timing, so it pushes the
lower body the way it is already going. 56 ms is also what the servo's own tracking shows
(`corr(J[k], goal[k-2]) = 0.984`, two samples at 28 ms).

Putting 56 ms into the model reproduces it: `torso_roll_rms` goes 0.67 -> 4.57 and the hip
axis gets worse, 5.13 -> 6.31 rms. With no delay the PID is free; with the robot's delay it
is destructive. The maps were built in the first world.

An earlier, separate defect — a 3.1-3.3 Hz limit cycle while standing perfectly still — was
a different consequence of the same delay and was fixed on 2026-09-02 by low-passing only
the feedforward path of the kappa law at 2 Hz (roll 31 deg p2p -> 1.3, current 1000 mA ->
60). That fix holds; it does not touch the walking case, where the loop saturates.

## The fix: drive the torso off the gait phase

The disturbance is periodic and the board generates the period itself, so the torso does
not need to measure anything:

    torso_deg = ff_gain * ff_amp * sin(phase + ff_phi)

`phase` is the same variable the legs use, so a bumpless frequency change carries the torso
with it. No measurement is in the loop, so the sensor delay cannot enter; the servo's own
lag is cancelled by leading `ff_phi` instead of by closing a faster loop.

**kappa survives.** kappa is defined by `torso_world_roll = kappa * axis_roll`, and the
feedforward reaches any value of it by choosing the amplitude: `ff_gain` 1.0 is kappa 0 and
0 is the torso held. The research axis is unchanged; only the implementation is.

`ff_amp` and `ff_phi` come from a torso-held recording — the axis roll there is exactly what
the torso has to cancel — and only the phase offset has to be trimmed on the robot. The
model predicted that offset would be 45-70 deg past the naive value; on the robot it is 43.

## Calibration, ten bouts at `1.39/240/80/16/30`

That gait was chosen because it is the only bout that never fell (max total tilt 11.1 deg
over 15 s) and because its axis roll is clean enough to fit — 16% residual, against 64% for
the same gait at `hip_amp 32`, where a single-frequency feedforward should not be expected
to work.

| torso | roll rms | roll p2p | cycles walked | first fall | torso on the clamp |
|---|---|---|---|---|---|
| held | 7.35 deg | 23.3 | 20/21 | none | 0% |
| kappa PID | 34.09 | 105.9 | 11/21 | 11.5 s | 32% |
| **feedforward, phi 119, A 7.5** | **1.46** | **7.9** | **21/21** | **none** | **0%** |

The phi scan: 4.83 / 1.54 / 1.46 / 3.03 / 4.89 / 5.34 at 94 / 114 / 119 / 129 / 144 / 159.
The amplitude scan at phi 119: 2.32 / 1.46 / 1.59 at 6.0 / 7.5 / 9.0.

**What is and is not established.** Splitting each bout into thirds puts the run-to-run
floor at 0.44 deg rms. phi 114 (1.54) and A 9.0 (1.59) are therefore *not* distinguishable
from the flashed values: the region phi 114-119 / A 7.5-9.0 is established, the exact point
is not. Separating them needs three repeats per setting. What is clearly established is
that phi 94, 129, 144, 159 and A 6.0 are worse, by 2 to 12 times the floor.

The PID number is also not a steady-state measurement: its thirds read 5.77 / 14.40 / 11.69,
it went down at 11.5 s, and its Euler pitch wrapped, so its backward falls are not even
counted. The honest comparison is not "23x better" but: over the same 21 cycles the
feedforward walked all of them without the torso command ever reaching its clamp, and the
PID fell at 11.5 s with the command on the clamp a third of the time.

**The calibration is for kappa = 0 only.** No other kappa has been measured.

## Instrument fixes this depended on

Several readings had to be corrected before any of the above could be trusted, and each
mattered:

- **`cycle_table` was blind to the axis the robot falls about.** DOWN required roll past
  25 deg AND a stalled torso joint. A record ending with five cycles at +70 to +85 deg of
  *pitch* — flat on its back — was scored 20 walk / 0 down. Now a cycle is DOWN if the
  pitch sits more than 35 deg off its walking value, or if the torso command is on the
  clamp for the whole cycle.
- **Euler pitch wraps** through +-180 whenever the roll gets large, which is exactly when
  the interesting gaits run; it cost three recordings their fall counts. The firmware now
  logs the BNO's gravity vector, which never wraps and carries both tilts. Recovering the
  attitude from it correlates with the Euler roll at -0.999.
- **The ring buffer was spending itself on the staged start.** 14 s of ramp-and-settle
  against a 15 s ring left 53 of 570 rows at full amplitude. Recording now begins one
  second before the blend.
- **Three web buttons never worked.** The page sends keys through `encodeURIComponent` and
  the board read the raw character, so `[`, `]` and `,` arrived as `%`. Frequency could not
  be changed at all and `hip_phi` only went up. Every hand-tuning record made before
  2026-09-02 17:00 ran at whatever frequency its preset set.
- **Importing `gait_quality` changed four protocol globals** — stand lean, hip-offset ramp,
  staged start, extended metrics — because `grid6_sweep` sets them at module level. The
  same cell read a rear margin of 12.9 mm from one script and 16.8 from another. Both
  measurement scripts now set the protocol explicitly and print it.

## What this does not settle

- Simulation with 56 ms of lag reproduces the *direction* of the PID's failure but not its
  severity: 4.57 rms against the robot's 34.09. Something beyond transport delay is in the
  real loop — saturation, the servo's inner loop, discretisation — and none of it is
  identified. It stops mattering once the measurement leaves the loop, which is why the
  feedforward works, but it is not explained.
- The whole-robot CoM sits behind the loaded feet 45-75% of the time in **every** one of the
  1570 passing cells at mu 0.5, and the excursion peaks at the hip's swing apex. Reducing
  `hip_amp` fixes it and costs the same amount of forward speed; all 36 `hip_phi` values lie
  on that one trade-off curve; moving mass forward does not help. This is a property of the
  gait family, not a controller artefact, and it is unresolved.
- Every map in GRID-4/5/6 was scored with a torso loop that reads true state instantly at
  1 kHz. Those maps are not wrong, but they describe a robot whose torso behaves in a way
  this one does not. Re-scoring the passing set under the feedforward torso is the obvious
  next campaign.
