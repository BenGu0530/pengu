# Hardware session 2026-08-28/29 — locating the sim2real gap

One day of bench work and walking tests on the COM-1.31 robot, aimed at one question:
**where does the sim2real gap actually live?** Corrections first: three of the leading
hypotheses died, and one of my own predictions failed twice before the data pinned the
right model.

All raw captures: `HardwareData/motor_probe_0828/`, `HardwareData/pid_walk_0828/`.
Sketches: `Arduino/imu_probe`, `motor_probe`, `hip_probe`, `torso_rom`, `pengu_kappa_ab`.
The original `Arduino/pengu_champ*` sketches were left byte-identical (`git checkout`);
everything new lives in its own folder, serial-only, no WiFi tabs.

---

## 0. Corrections to earlier claims

1. **"The BNO055's Euler roll mixes pitch, so the firmware must switch to a gravity-vector
   roll."** Inferred from a sim identity (`grid5/imu_frame_probe.py`), never checked on the
   device. Measured on the bench: at 27 deg of forward pitch, device Euler roll and
   `atan2(g_x,-g_y)` agree to **0.30 deg**. There is no mixing. The planned firmware roll
   fix (approved plan Part B) is **not needed**.
2. **"Crank velocity saturation explains the 3-5x speed gap."** Built on the XM430-W350
   datasheet no-load figure of 46 rpm = 276 deg/s. Measured on this robot: the cranks reach
   **420-440 deg/s** and track the champion command at **0.92 hanging / 0.909 loaded**.
   The datasheet number is not the practical ceiling. Worth at most ~10% of speed.
3. **"The torso collides with the legs beyond +-10 deg"** (comment in
   `pengu_champ_k0_105.ino`). Measured: **+30.9/-31.9 deg at hip_off=0, +27.8/-31.0 at
   hip_off=20**. Off by 3x.
4. **My own model of the loop instability was wrong twice.** First I predicted raising the
   control rate would stabilise kp=2 (it did not: 76% sign flips at 80 Hz vs 77% at 25 Hz).
   Then I argued kp=0 would be optimal because the servo is deadbeat (it was not: kp=0.25
   tracked worse than kp=0.5). The data pins the servo at **g ~= 0.67** of the gap per
   period, which makes kp~0.5 optimal and kp=2 marginal — both observed.
5. **Friction:** the ice surface used on 2026-08-25 measures **mu = 0.11**, so that session
   really was comparable to the sim's mu=0.1. The current lab floor is ~0.4, which is why
   this session's gait was selected at mu=0.3.

---

## 1. What was measured, and what it ruled out

| subsystem | measurement | result |
|---|---|---|
| IMU roll definition | 5 static bench poses, gravity vector + device Euler + quaternion logged | device Euler == gravity roll to **0.30 deg** at 27 deg pitch; axis map: **gy vertical, gx lateral, gz fore/aft**; level reading repeatable to **1.9 deg**; roll changes **1.06 deg** over an 87 deg yaw |
| accelerometer trust | \|g\| across 4 attitudes | 9.797-9.800 (true 9.807) — sound regardless of the `cal_acc` flag |
| crank speed | freq and amplitude ladders, hanging and loaded | ceiling **420-440 deg/s**; amplitude retention **0.92 / 0.909** at the c6 champion; loading costs only 1-2 points |
| hip speed + torque | same ladders, hanging and standing, with PRESENT_CURRENT | retention **0.96 / 0.97-0.99**; torque rises 1.0 -> 2.0 N.m under body weight and retention does **not** drop |
| torso range of motion | current-capped ramp (800 mA) to the mechanical stop | **+30.9/-31.9 deg** (hip_off=0), **+27.8/-31.0** (hip_off=20); position pinned dead with current at the cap = hard stops |
| A reconstruction | slow torso sweep with the lower body held | **d(roll)/dJ = +1.09 / +1.02, R2 = 0.9944 / 0.9914** — `T = A + s*J` holds, and `S_TILT=+1` (hardcoded, never verified before) is correct. Residual 4 deg is an upper bound, confounded with hand motion |

So: actuators, sensing, geometry and the sign chain all transfer. None of them is the gap.

**Method note:** the A-check only works if the *lower body* is the fixed side. The first
attempt hung the robot by the torso, which fixes the wrong side — the legs absorbed 93% of
the joint rotation (`d(roll)/dJ = 0.067`). Holding the hips with the robot standing gave
the clean +1.0.

---

## 2. The torso controller: a discrete-time instability, and its fix

The walking test (`Arduino/pengu_kappa_ab`, gait 1.06/220/105/28/10 at mu=0.3, chosen for
robustness rather than speed — see §4) reproduced the symptom immediately: the torso
trembled, then the motor quit. The log says why.

The torso command **reversed sign on 50% of consecutive samples** and railed to the +-25 deg
clamp on 38% of them, while current alternated to +-2300 mA (stall) until the XM430 tripped
`hwErr=32` (Overload) and disabled its own torque — 37 deg of position error with ~0 current.

Substituting the exact kinematic relation `T = A + s*J` into the control law:

```
axis = T - s*J = A                     (exact; verified R2 > 0.99)
e    = kappa*A - T = (kappa-1)*A - J
cmd  = (kappa-1)*A + kp*e + ki*int     =  (kappa-1)(1+kp)*A  -  kp*J
```

so `d(cmd)/dJ = -kp`. Closing the loop once per control period with the servo covering a
fraction `g` of the gap:

```
dJ_{n+1} = dJ_n * [1 - g*(1+kp)]      stable iff  g*(1+kp) < 2
                                       fastest when g*(1+kp) = 1
```

**The equilibrium `J* = (kappa-1)*A` does not depend on kp** — kp sets only how fast the
torso converges, not what it converges to. Lowering it is therefore *not* a change of
controller.

Measured (kp swept live, same walk, legs untouched):

| kp | rate | sign-flip | clamp | tracking err | mean mA | peak mA |
|---|---|---|---|---|---|---|
| 0.25 | 80 Hz | 11% | 0% | 1.68 deg | 181 | 648 |
| **0.5** | 57 Hz | **9%** | 0% | 1.93 deg | **188** | 785 |
| 0.5 | 80 Hz | 14% | 0% | 1.79 deg | 207 | 551 |
| 1.0 | 25 Hz | 32% | 0% | 3.73 deg | 480 | 2330 |
| 2.0 | 25 Hz | 77% | 11% | 18.19 deg | 1084 | 2760 |
| 2.0 | 80 Hz | 76% | 1% | 12.88 deg | 922 | 2292 |

(A clean 1.06 Hz sinusoid sampled at 57 Hz crosses zero on ~3.7% of samples, so 9-14% is
essentially a clean waveform and 76% is the per-sample limit cycle.)

kp=0.5 being the optimum implies `g ~= 0.67`, which in turn predicts the stability limit
`kp < 2/g - 1 = 2.0` — and kp=2 is exactly where it breaks. One model, both observations.

**Why sim never sees this:** sim's actuator is a compliant position servo
(`kp=50 dampratio=1 forcerange +-4.1`) stepped at 1 kHz, so it covers ~2% of the gap per
step (g ~ 0.02) and kp=2 is comfortably stable. The hardware Dynamixel with
`PROFILE_VELOCITY=0` covers two thirds of it in 12-17 ms. **The mismatch is the actuator
model, not the sample rate.**

### Loop rate

The inherited `delay(20)` (nominally "50 Hz") sat on top of the real work, giving a
measured **37-42 ms (25 Hz)**. Removing it and moving the two telemetry-only I2C reads
(`VECTOR_LINEARACCEL`, `getCalibration`) out of the control path to 1 Hz gives **12-18 ms
(57-80 Hz)**. The gait waveform is unaffected — phase comes from `millis()`, absolute time.
The integral now uses the measured period instead of a hardcoded 0.02 s (it had been
accumulating 3.7x slower than designed).

---

## 3. The result: the kappa law transfers

Absolute torso roll is the wrong scoreboard — kappa=2 *amplifies the body's own lean*, so if
the legs rock less, the torso rolls less no matter how good the loop is. The controller's
own objective is the ratio.

| config | T rms | A rms | **T/A** | J/A | mean mA |
|---|---|---|---|---|---|
| kp=0.5 @ 25 Hz | 7.59 | 3.64 | 2.08 | 1.11 | 229 |
| kp=2.0 @ 25 Hz | 9.47 | 5.63 | 1.68 | 1.07 | 1084 |
| **kp=0.5 @ 57 Hz** | 11.25 | 5.58 | **2.02** | **1.03** | 188 |
| kp=2.0 @ 80 Hz | 8.56 | 4.48 | 1.91 | 1.14 | 922 |
| kp=0.25 @ 80 Hz | 8.56 | 3.88 | 2.21 | 1.23 | 181 |
| kp=0.5 @ 80 Hz | 8.22 | 3.69 | 2.23 | 1.25 | 207 |
| **sim kappa=2** | 13.10 | 6.50 | **2.02** | **1.03** | — |

At kp=0.5 and 57 Hz the hardware hits **T/A = 2.02 and J/A = 1.03 — the sim's values to
three digits.** kp=2 degrades the ratio (the oscillation is eating the tracking) and costs
4-5x the current; kp=0.25-0.5 at 80 Hz overshoots the ratio by ~11%.

The earlier reading that "hardware torso roll is 61% of sim" dissolves: T varied 7.6-11.3
across runs because **A** varied 3.6-5.6, not because the loop varied.

### Where the gap is now

| layer | status |
|---|---|
| actuators (crank, hip: speed and torque) | transfers (0.91 / 0.97) |
| sensing (roll definition, A reconstruction, sign chain) | verified |
| mechanical range | sufficient (+-28 deg available, +-12.7 needed) |
| **torso kappa control law** | **transfers: T/A 2.02 vs 2.02**, at kp~0.5 and >=57 Hz |
| **lower-body roll A** | **hardware is 56-86% of sim (3.6-5.6 vs 6.5), and repeats to only ~1.5x across runs** |

The residual is **A** — how much the legs themselves rock — which is a gait/contact/dynamics
question, and the same quantity that drives forward speed. That is the next thing to chase.

---

## 4. Gait selection for gap measurement

The speed champions are knife edges: c6's 1.67/95 walks 0.379 m/s while 1.66/95 and 1.67/85
both fall. Landing on such a cell demands precision the hardware does not have (0.91
amplitude retention plus phase lag), so any sim-vs-hardware difference is swamped by which
neighbouring cell the robot actually landed in.

Selected instead (`grid5/pick_robust_gait.py`, filters: mu=0.3, freq <= 1.3, head >= 0.9,
crank demand <= 420 deg/s, 5-D neighbourhood score):

> **1.06 / 220 / 105 / 28 / 10** — walks under *both* kappa=0 and kappa=2, neighbourhood
> 5/8 and 3/8, crank demand 350 deg/s.

Same gait under both torso laws means the legs are identical (sim crank amplitude 104.2 vs
104.5) and pressing '0' or '2' changes exactly one thing. Sim prediction for this pair
(`grid5/walk_prediction.py`, 10 s window, mu=0.3):

| | kappa=2 | kappa=0 |
|---|---|---|
| speed | 0.172 | 0.062 m/s |
| torso world roll rms | 12.1 | 0.8 deg |
| torso torque p95 | 1.81 | 0.61 N.m (stall is 4.1; **0% of the window near it**) |

The champions, by contrast, put the kappa=2 torso at stall torque 13.2% of the time — under
which no controller could produce the commanded lean, so they cannot test a control law at
all. Demos rendered by `grid5/render_gait_demo.py` into `results/grid5_probes/`.

**Open item on kappa=0:** it does not hold the torso level on hardware (T rms 6.07 vs sim
0.9, with 49.8 deg swings) — measured before the kp fix, so it needs re-running at kp=0.5.

---

## 5. Firmware settled on

`Arduino/pengu_kappa_ab` (serial only; `pengu_champ*` untouched):

- **kp = 0.5**, ki = 0.1, kd = 0, kappa switchable live ('0'/'2'), kp switchable live ('4'-'7')
- **no pacing delay**; telemetry-only IMU reads at 1 Hz -> **57-80 Hz** control
- integral uses the **measured** period
- `TORSO_CLAMP_DEG = 25` (nearest measured stop is 27.8)
- 20 Hz CSV telemetry on 't': `w,t,alpha,goal/pos x5,mA_torso,imu_roll,imu_pitch,axis,dt_ms`
- 1 Hz line reports kp, kappa, measured period and a loop counter

Do **not** raise kp toward the sim's 2.0 on this hardware: it is above the measured
stability limit and trips the motor's overload protection.

---

## 6. Open

1. **A is smaller on hardware than in sim** (56-86%) and varies 1.5x between runs. Is that
   contact, foot geometry, the floor, or measurement? Independent measurement of the lower
   body (mocap markers or a second IMU) would settle whether the reconstructed A is real.
2. **kappa=0 re-run at kp=0.5** — the only pre-fix result still on the books.
3. **Speed is not the hardware metric** (Ben, 2026-08-29). The robot is open loop and is
   not being optimised on hardware, and the sim's forward measure is built from per-step
   foot contact rather than anything a stopwatch reproduces. Torso attitude and the T/A
   ratio are what the hardware comparison rests on. The 2026-08-25 speed numbers stay as
   a record, not as a target.
4. **Dynamic IMU error while walking** is still unmeasured; every bench check was static or
   slow, and any error there feeds A with gain `d(cmd)/dA = (kappa-1) + kp*kappa`.
5. GroupSyncWrite for the five goal positions (step 3 of the loop-rate work) was never done;
   57-80 Hz was enough.
