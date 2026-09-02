# Getting one walking gait, and measuring the gap on the way

2026-09-02. Written after the two WiFi dumps of preset 1 (`1.46/250/75/32/10`) and
preset 2 (`1.95/350/125/32/0`).

The goal is **one gait that walks on the robot**. Everything below is ordered by how fast
it gets there; the gap decomposition is what falls out of the same runs, not a separate
campaign.

## What the two dumps already establish

**The crank ceiling turns a commanded cell into a different cell.** Preset 2 asked for
762 deg/s of crank and got 353 and 358 -- the same 354 measured on 2026-08-30, now from
two more points. The hips only needed 379 and tracked at 0.97. Because the crank lags
90 ms and the hip 31 ms, the *difference* lands on hip_phi:

| | commanded | executed |
|---|---|---|
| hip_phi | 350 | **31** |
| leg_amp | 125 | **67** |

hip_phi's grid step is 10 and leg_amp's is 10, so the robot ran a cell four and six steps
away from the one selected. Simulating what it actually executed gives 0.0769 m/s against
the commanded cell's 0.3974 -- 81% of the loss.

**Inside the envelope, a 1% tracking error is still enough to change cells.** Preset 1's
crank demand is 343, inside the ceiling, and it tracked at 0.99 -- but 0.99 of 75 is 74.3,
and in simulation:

```
leg_amp   73      74      75      76      80
net     .0659   .0665  .1079   .0717   .1180
freq     1.44    1.45    1.46    1.47    1.48
net     .0649   .0647  .1079   .0669   .0683
```

The champion is a spike one grid point wide in two axes. It survives five different
settle/window combinations (0.107-0.114 at leg_amp 75, 0.055-0.082 at 73/74/76), so it is
in the model, not the readout.

**Why the robustness filter did not catch it.** The filter ran on stage 1 at 0.05 Hz and
tested freq +-0.05 and hip_phi +-10. The champion came out of stage 2 at 0.01 Hz and its
neighbours were never re-tested at that resolution. leg_amp's grid step is 10, so +-1 was
never simulated anywhere in the campaign. The "robust region" is robust at 0.05 Hz and
says nothing at 0.01.

**What is left over.** Simulating preset 2's *executed* cell still walks -- drift 2.3 deg,
43% air, 8.5 mm clearance. On the robot that segment has 3 walking cycles out of 9, with
no run of 4 consecutive. Preset 1 is three build-up-and-collapse episodes in 15.7 s. So
after both layers above are accounted for, simulation walks and the robot does not.

## Fix the instruments first (~30 min, before any run)

1. **`cycle_table` mislabels pinned cycles.** DOWN requires roll > 25 AND torso travel
   < 6 deg. Preset 1's cycle 6 (roll 22.8, travel 1.1, command on the rail 100% of the
   cycle) and cycle 16 (roll 49.8, travel 9.9, rail 100%) both escaped and were counted as
   walking. Add: rail == 100% for a whole cycle is DOWN regardless of roll.
2. **The drift number needs a floor.** Phases are being differenced between cycles whose
   roll fundamental is 0.88 deg and cycles where it is 15.34. Below ~2 deg report "not
   determined" instead of a number. On preset 1 that leaves zero cycles -- which is the
   honest answer for that record.
3. **Report the drift's sensitivity to the cycle-grid origin.** Same data, same rules,
   t0 from 16.0 to 19.0 gives 60 / 69 / 78 / 93 / 102. One number hides that.
4. **`hardware/tune_report.py <csv>`** -- three numbers in one command: longest run of
   consecutive walking cycles, roll drift (or "not determined"), rail %. The tuning loop
   has to be faster than a full phase_probe read.
5. **Measure distance.** The robot has no displacement sensor, so `net_fwd` -- the quantity
   the entire map ranks on -- has never been measured on hardware at all. Two strips of
   tape and a fixed 15 s bout is enough to end that.

## Experiments

E1 and E2 need no reflash; both amplitudes and hip_phi are buttons.

**E1 -- leg_amp ladder.** Fix 1.46 / 250 / hip 32 / off 10. Step leg_amp 0, 5, 10 ... 100,
dumping each. Read: longest walking run, and distance covered. Asks whether the robot has
anything like the model's two modes, and where its own best amplitude is. This is the run
most likely to produce a walker on its own.

**E2 -- hip_phi ladder.** At E1's best leg_amp, sweep hip_phi 200 to 300 in 10s. This is
the axis the ceiling corrupts, and 10 is exactly the map's grid step, so the ladder is
directly comparable to the map.

**E3 -- frequency ladder.** +-0.05 across 1.2-1.7 at the settings E1 and E2 chose.

Each rung is ~20 s of walking plus a dump; 30-40 rungs is an afternoon.

**E4 -- torso clamp.** Preset 1 spends 16% of its cycles with the torso command pinned at
+-25 while simulation never reaches the clamp at all. Raise `TORSO_CLAMP_DEG` 25 -> 30
(the mechanical stop measured +27.8 / -31.0, so 35 would hit it). Read: does rail % fall
and does the walking run lengthen? Needs a reflash.

**E5 -- kappa.** 0 -> 0.5 at the best gait from E1-E3. One constant.

**E6 -- loop rate.** The telemetry period comes out 28 ms against `TEL_MS = 20`; the main
loop is running at 36 Hz. Find what is spending the time. Not a gap explanation, but every
lag number above is quantised by it.

## In simulation, in parallel

- **`grid6/plateau.py`** (running) re-ranks the top 33 stage-2 cells at mu=0.5 by the
  *minimum* v_net over leg_amp +-3 and freq +-0.03, not by the centre. A cell that keeps
  walking across that whole box is one the robot can be aimed at; the current champion is
  not. 1617 simulations.
- **Rate-limit the simulated actuators.** `gait_quality` drives ideal waveforms, but the
  robot's crank is flattened -- fundamental 33.4 deg yet peak rate only 353 where a sine of
  that amplitude at 1.95 Hz would need 409. Adding the measured 354 deg/s limiter is the
  only way to make layer one quantitative rather than inferred.

## Decision point

If E1 produces a gait that walks continuously, take it and stop -- tune from there with
E2/E3 and leave the rest of this document for later. If E1 produces nothing that walks at
any amplitude, the problem is not in the gait parameters and E4/E5 become the next move.
