# imu_probe — bench test for the torso IMU's roll definition

One question, answered with hardware data: **once the torso is pitched ~20–25° forward
(the working pose set by `hip_off`), does the BNO055's Euler `roll` — the number the
κ-PID currently feeds on — still read lateral lean?**

Sim says a gravity-vector roll is pitch-immune by construction and matches the quantity
the κ-PID actually nulls to within RMS 0.23° on the c6 champion trajectory
(`pengu_mujoco/grid5/imu_frame_probe.py`). Whether the *device's* Euler output is mixing
is a claim about the BNO055 that has never been checked on the robot. This is that check.

The sketch logs the **quaternion, device Euler, fused gravity vector and raw
accelerometer** together, so every decomposition can be recomputed offline from a single
capture — no re-running the bench if a convention turns out to be wrong.

Nothing is commanded: no DynamixelShield, no WiFi. **Motors do not need power**; USB from
the laptop runs the board and the IMU. The robot can be held in your hands.

## Run it

1. Upload `imu_probe.ino` (Arduino IDE → open this folder → upload).
2. `bash Arduino/capture.sh bench_2026-08-28.csv`
   - keys: `z` = 2 s static average, `1`..`9` = label the pose, space = pause, `h` = header
   - **quit: Ctrl-A then k then y** (not Ctrl-C)
3. Watch `cal_acc` in the last columns — it must reach **3** before the gravity vector is
   trustworthy. Move the robot slowly through a few orientations, pausing a second in
   each, until it does.

## The bench protocol (~10 minutes, robot in hand)

Hold each pose still and press `z` (2 s average). Set the pose label first with the digit
key so the rows are self-identifying.

| key | pose | what it establishes |
|---|---|---|
| `1` | **level**: torso upright, hips at 0, robot standing normally on the table | gravity baseline — which component is "down" |
| `2` | **pure lateral tilt +10°** (lean the robot sideways, no forward pitch) | which component is the lateral axis, and its sign |
| `3` | **pure lateral tilt −10°** (the other way) | confirms the sign, checks symmetry |
| `4` | **pure forward pitch ~20–25°**, zero lateral tilt — the walking pose | **the decisive one.** A correct roll reads ≈0 here. Whatever reads non-zero is mixing pitch in |
| `5` | **pitch ~20° AND lateral +10° together** | which formula recovers the 10° |

Use a protractor or a wedge for poses 2–5 if you can; if not, eyeball it and note the
angle you were aiming for — the comparison between the two formulas holds either way,
since both see the same true attitude.

Optional: `6` = slow hand-rock about the forward axis while the robot is pitched forward,
streaming (no `z`). That gives a dynamic trace of both roll definitions on the same motion.

## Reading the result

Each `z` prints a `#AVG` line with mean ± sd of every channel, including two candidate
gravity rolls:

- `roll_gx = atan2(gx, −gz)`
- `roll_gy = atan2(gy, −gz)`

Poses 1–3 identify which candidate is the real lateral lean and its sign (the mounting
orientation is not assumed anywhere in this sketch). Pose 4 is the verdict: whichever of
`eul_roll` / `roll_gx` / `roll_gy` stays at ≈0 under pure forward pitch is the one the
κ-PID should be using.

Send the CSV back and the offline decomposition (heading-frame yaw→pitch→roll, and the
comparison against the sim's hinge-frame roll) gets done from the quaternion column.

## Not in scope here

Motor telemetry — crank goal-vs-present (the velocity-saturation question) and the torso
joint angle — needs the DynamixelShield and therefore lives in the walking sketches, not
this one. That is a separate diagnostic stream.
