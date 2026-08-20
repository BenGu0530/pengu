#!/usr/bin/env python
"""Staged start: ramp hip_off in slowly, let the rocking die at that offset, THEN walk.

Why: gait_config.compute_gait applies hip_off WITHOUT the alpha blend --
    hip_L = hip_off - hip_lean + alpha * hip_amp * max(0, sC)
-- so at the first instant of the blend the hip command jumps by the full hip_off
(a 30 deg step). Large offsets therefore fail on the step input, not on the walking.

Schedule here:
  [0, t1)    plain stand at STAND_HIP_DEG
  [t1, t2)   ramp hip_off 0 -> full (smoothstep), no oscillation
  [t2, t3)   hold at full offset, let rocking decay
  [t3, t4)   blend the oscillation in (alpha 0 -> 1), phase counts from t3
  [t4, end)  full walk

usage:
  python physics/staged_start_probe.py c6 --freq 1.37 --phi 0 --leg 95 --hip 28 \
      --off 30 --ramp 4 --offsettle 6 [--abrupt] [--demo out.mp4]
"""
import os, sys, math, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, mujoco
import gait_config as gc, gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
import grid4_sweep as g4

CONF = {"c1": (0.0, 1.05), "c3": (0.0, 1.31), "c4": (2.0, 1.05),
        "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
ap = argparse.ArgumentParser()
ap.add_argument("cfg")
for f in ["freq", "phi", "leg", "hip", "off"]:
    ap.add_argument(f"--{f}", type=float, required=True)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--hold", type=float, default=5.0)
ap.add_argument("--ramp", type=float, default=4.0, help="seconds to ramp hip_off in")
ap.add_argument("--offsettle", type=float, default=6.0, help="seconds to settle AT the offset")
ap.add_argument("--blend", type=float, default=4.0)
ap.add_argument("--walk", type=float, default=15.0, help="seconds of walking to measure")
ap.add_argument("--abrupt", action="store_true", help="original behaviour: hip_off as a step")
ap.add_argument("--demo", default=None)
a = ap.parse_args()

t1 = a.hold
t2 = t1 + (0.0 if a.abrupt else a.ramp)
t3 = t2 + (0.0 if a.abrupt else a.offsettle)
t4 = t3 + a.blend
T_END = t4 + a.walk

kappa, com = CONF[a.cfg]
model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model); g4.apply_com_variant(model, com)
pid = TorsoKappaPID(model, kappa=kappa, measure_after=t4); gc.TORSO_CONTROLLER = pid
gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None; gs.CONDITION["hip_off"] = a.off
set_floor_friction(model, a.mu)
gs._set_gait(dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg, hip_amp=a.hip))
pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)

def smoothstep(t, lo, hi):
    if hi <= lo: return 1.0
    x = min(1.0, max(0.0, (t - lo) / (hi - lo)))
    return x * x * (3 - 2 * x)

mode = "ABRUPT (original)" if a.abrupt else "STAGED"
print(f"{a.cfg} {mode}  mu={a.mu}  hip_off={a.off}")
print(f"  stand[0,{t1}) ramp[{t1},{t2}) settle[{t2},{t3}) blend[{t3},{t4}) walk[{t4},{T_END})")
print(f"  gait freq={a.freq} phi={a.phi} leg={a.leg} hip={a.hip}\n")

stand = math.radians(gc.STAND_HIP_DEG)
frames = []
if a.demo:
    import imageio.v2 as imageio
    cams = []
    for dist, elev, az in [(1.4, -10, 0), (1.1, -12, -90)]:
        c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.elevation, c.azimuth = dist, elev, az
        cams.append(c)
    ren = mujoco.Renderer(model, height=480, width=640)

log, nxt_s, nxt_f = [], 0.0, 0.0
while data.time < T_END:
    t = data.time
    off_frac = 0.0 if t < t1 else (smoothstep(t, t1, t2) if t < t2 else 1.0)
    alpha = 0.0 if t < t3 else (smoothstep(t, t3, t4) if t < t4 else 1.0)
    gc.WALK_HIP_OFFSET_DEG = a.off * off_frac
    phase = 2 * math.pi * gc.WALK_FREQ * (t - t3) if t >= t3 else 0.0
    hip_L, hip_R, crank_L, crank_R, torso = gc.compute_gait(phase, alpha)
    if t < t3:
        # match apply_ctrl's HOLD branch: torso commanded to 0 and the PID is NOT
        # called, so its integrator does not wind up during ramp/settle.
        torso = 0.0
        if hasattr(gc.TORSO_CONTROLLER, "reset"):
            gc.TORSO_CONTROLLER.reset()
    elif gc.TORSO_CONTROLLER is not None:
        torso = gc.TORSO_CONTROLLER(data, t, alpha)
    data.ctrl[act["hip-L"]] = stand * (1 - alpha) + hip_L
    data.ctrl[act["hip-R"]] = stand * (1 - alpha) + hip_R
    data.ctrl[act["crank1-L"]] = crank_L
    data.ctrl[act["crank1-R"]] = crank_R
    data.ctrl[act["torso"]] = torso
    mujoco.mj_step(model, data)
    if data.time >= nxt_s:
        nxt_s += 0.5
        rock = float(np.hypot(data.qvel[3], data.qvel[4]))
        log.append((data.time, float(data.qpos[0]), float(data.qpos[1]),
                    float(data.qpos[2]), rock, off_frac, alpha))
    if a.demo and data.time >= nxt_f:
        nxt_f += 1.0 / 30
        pair = []
        for c in cams:
            c.lookat[:] = data.xpos[ids[3]]
            ren.update_scene(data, c); pair.append(ren.render().copy())
        frames.append(np.hstack(pair))

def phase_of(t):
    if t < t1: return "stand"
    if t < t2: return "ramp"
    if t < t3: return "settle"
    if t < t4: return "blend"
    return "walk"

print(f"{'t':>6}{'x':>9}{'y':>9}{'z':>8}{'rock':>9}{'off%':>7}{'alpha':>7}  phase")
for t, x, y, z, rock, of, al in log:
    if t % 2.0 < 0.51 or phase_of(t) != phase_of(t - 0.5):
        print(f"{t:>6.1f}{x:>9.4f}{y:>9.4f}{z:>8.4f}{rock:>9.4f}{of*100:>6.0f}%{al:>7.2f}  {phase_of(t)}")

pre = [r for r in log if r[0] < t3]
walk = [r for r in log if r[0] >= t4]
if pre:
    print(f"\nrock rate in last 2 s before blend: "
          f"{max(r[4] for r in pre if r[0] > t3 - 2):.5f} rad/s")
    print(f"height z at end of settle: {pre[-1][3]:.4f} m  (fallen if << 0.1)")
if walk:
    dx = walk[-1][1] - walk[0][1]; dy = walk[-1][2] - walk[0][2]
    print(f"\nWALK PHASE ONLY ({t4:.1f} -> {T_END:.1f} s):")
    print(f"  dx={dx:+.4f}  dy={dy:+.4f}  disp={math.hypot(dx,dy):.4f} m  "
          f"angle off +y = {math.degrees(math.atan2(dx,dy)):+.1f} deg")
    print(f"  final height z = {walk[-1][3]:.4f} m  "
          f"({'UPRIGHT' if walk[-1][3] > 0.1 else 'FALLEN'})")
if a.demo:
    ren.close()
    imageio.mimsave(a.demo, frames, fps=30, macro_block_size=1)
    print(f"\ndemo -> {a.demo} ({len(frames)} frames)")
