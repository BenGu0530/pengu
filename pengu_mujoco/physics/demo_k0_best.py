#!/usr/bin/env python
"""Render the BEST swept GRID-3 k0 (Gait 1, kappa=0 torso-upright) gait.

Uses the exact winning cell from the complete 454,500 sweep (no re-search):
  freq=2.0  hip_phi=250  leg_amp=125  hip_amp=28  hip_off=10  (net_fwd=0.291)
Back view (the axis-vs-torso roll is only legible from behind) + live telemetry.

usage: MUJOCO_GL=egl python physics/demo_k0_best.py
out:   results/gait_sweep/demo_k0_best.mp4
"""
import os, sys
os.environ.setdefault("PENGU_MODEL", "v3")
os.environ.setdefault("MUJOCO_GL", "egl")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import imageio.v2 as imageio
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction

FPS = 30
P = dict(freq=2.0, hip_phi=250.0, leg_amp=125.0, hip_amp=28.0, hip_off=10.0)

model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model); root = ids[3]
gs.FLOOR_MU = 0.7

pid = TorsoKappaPID(model, kappa=0.0, measure_after=gs.SETTLE)
set_floor_friction(model, gs.FLOOR_MU)
gs.CONDITION["hip_off"] = P["hip_off"]
gs._set_gait({k: v for k, v in P.items() if k != "hip_off"})
gc.TORSO_CONTROLLER = pid
pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)

cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 1.2, -12.0, -90.0    # behind, looking +y
frames, tele = [], []
with mujoco.Renderer(model, height=480, width=640) as ren:
    nxt = 0.0
    while data.time < gs.SIM_DURATION:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            print(f"FELL at t={data.time:.2f}"); break
        if data.time >= nxt:
            nxt += 1.0 / FPS
            h = pid.hinge(data)
            a = np.degrees(pid.axis_roll(data, h))
            t = np.degrees(pid.torso_roll(data, h))
            cam.lookat[:] = data.xpos[root]
            ren.update_scene(data, cam)
            frames.append(ren.render())
            tele.append((data.time, a, t))

out = os.path.join(_ROOT, "results", "gait_sweep", "demo_k0_best.mp4")
imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
tele = np.array(tele)
w = tele[tele[:, 0] > gs.SETTLE]
print(f"wrote {out}  ({len(frames)} frames)")
print(f"  axis  tilt: rms {np.sqrt((w[:,1]**2).mean()):5.2f}d  "
      f"range [{w[:,1].min():+6.1f},{w[:,1].max():+6.1f}]")
print(f"  torso tilt: rms {np.sqrt((w[:,2]**2).mean()):5.2f}d  "
      f"range [{w[:,2].min():+6.1f},{w[:,2].max():+6.1f}]  (kappa=0 target: 0)")
gc.TORSO_CONTROLLER = None
