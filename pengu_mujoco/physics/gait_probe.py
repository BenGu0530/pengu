#!/usr/bin/env python
"""Probe ONE gait: trajectory breakdown (forward vs lateral) + optional demo render.

usage:
  python physics/gait_probe.py c6 --freq 1.96 --phi 240 --leg 105 --hip 28 --off 20 --mu 0.1
  ... --demo out.mp4        also render side+back video

Reports where the travel actually goes: net forward (y), net lateral (x), path length,
straightness, and the walk-quality metrics (single_frac, ds_move_frac, clearances, slip).
"""
import os, sys, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
import grid4_sweep as g4

CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}

ap = argparse.ArgumentParser()
ap.add_argument("cfg"); ap.add_argument("--freq", type=float, required=True)
ap.add_argument("--phi", type=float, required=True); ap.add_argument("--leg", type=float, required=True)
ap.add_argument("--hip", type=float, required=True); ap.add_argument("--off", type=float, required=True)
ap.add_argument("--mu", type=float, default=0.1); ap.add_argument("--demo", default=None)
a = ap.parse_args()

kappa, com = CONF[a.cfg]
model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model)
slide, got = g4.apply_com_variant(model, com)
pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
gc.TORSO_CONTROLLER = pid

p = dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg, hip_amp=a.hip)
gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None
gs.CONDITION["hip_off"] = a.off
set_floor_friction(model, a.mu)

print(f"{a.cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm  mu={a.mu}")
print(f"gait: freq={a.freq} hip_phi={a.phi} leg_amp={a.leg} hip_amp={a.hip} hip_off={a.off}\n")

# --- metrics via the standard trial (same numbers the sweep records) ---
pid.reset()
r = gs.run_trial(model, data, ids, dict(p))
print("run_trial metrics:")
for k in ["survived", "valid", "path_speed", "net_fwd_speed", "straightness", "single_frac",
          "ds_move_frac", "ss_move_frac", "clear_L", "clear_R", "n_steps", "cadence", "slip_ratio"]:
    if k in r:
        print(f"  {k:<16}{r[k]}")

# --- trajectory: where did it actually go? ---
gs._set_gait(dict(p)); pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)
xs, ys, ts = [], [], []
frames = []
if a.demo:
    import imageio.v2 as imageio
    cams = []
    for dist, elev, az in [(1.4, -10, 0), (1.1, -12, -90)]:
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = dist, elev, az
        cams.append(cam)
    ren = mujoco.Renderer(model, height=480, width=640)
nxt = 0.0
while data.time < gs.SIM_DURATION:
    gc.apply_ctrl(data, act, data.time)
    mujoco.mj_step(model, data)
    if data.time >= nxt:
        nxt += 1.0 / 30
        xs.append(float(data.qpos[0])); ys.append(float(data.qpos[1])); ts.append(data.time)
        if a.demo:
            pair = []
            for cam in cams:
                cam.lookat[:] = data.xpos[ids[3]]
                ren.update_scene(data, cam); pair.append(ren.render().copy())
            frames.append(np.hstack(pair))

x0, y0 = xs[0], ys[0]
dx, dy = xs[-1] - x0, ys[-1] - y0
path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
net = float(np.hypot(dx, dy))
ang = np.degrees(np.arctan2(dx, dy))
print("\ntrajectory (base freejoint xy, 24 s):")
print(f"  net forward  (y): {dy:+.4f} m")
print(f"  net lateral  (x): {dx:+.4f} m      <- +x = one side, -x = the other")
print(f"  net displacement: {net:.4f} m   path length: {path:.4f} m")
print(f"  straightness (net/path): {net/path if path>1e-9 else float('nan'):.3f}")
print(f"  heading off forward-axis: {ang:+.1f} deg")
print(f"  lateral/forward ratio: {abs(dx)/abs(dy) if abs(dy)>1e-9 else float('inf'):.2f}")

if a.demo:
    ren.close()
    imageio.mimsave(a.demo, frames, fps=30, macro_block_size=1)
    print(f"\ndemo -> {a.demo}  ({len(frames)} frames)")
