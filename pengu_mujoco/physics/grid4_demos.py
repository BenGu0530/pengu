#!/usr/bin/env python
"""GRID-4 stage-4: demo videos for every config at EVERY mu level.

grid4_finalists.py renders only the mu=0.1 demo. This renders the full matrix:
for each config with results/grid4_report/cN/top_gaits.csv, take the #1 gait and
render it at each mu -> results/grid4_report/cN/demos/demo_mu{01,03,05,07}.mp4

Each clip is side + back cameras hstacked (1280x480, 30 fps, 24 s), nominal
conditions (exact mu, no pose jitter) — same setup grid4_finalists.py uses, so
the clips correspond 1:1 with the rows in cN/finalists.csv at rank=1.

usage: python physics/grid4_demos.py [cN ...]      # default: all configs found
"""
import os, sys, csv
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import imageio.v2 as imageio
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
import grid4_sweep as g4

OUT = os.path.join(_ROOT, "results", "grid4_report")
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
MUS = [0.1, 0.3, 0.5, 0.7]
CAMS = [(1.4, -10, 0), (1.1, -12, -90)]          # (distance, elevation, azimuth)

wanted = [a for a in sys.argv[1:] if a in CONF] or list(CONF)


def render(cfg):
    top_csv = os.path.join(OUT, cfg, "top_gaits.csv")
    if not os.path.exists(top_csv):
        return 0
    gd = next(iter(csv.DictReader(open(top_csv))), None)
    if gd is None:
        return 0

    kappa, com = CONF[cfg]
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = g4.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid

    p = dict(freq=float(gd["freq"]), hip_phi=float(gd["hip_phi"]),
             leg_amp=float(gd["leg_amp"]), hip_amp=float(gd["hip_amp"]))
    hip_off = float(gd["hip_off"])
    print(f"{cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm  "
          f"gait freq={p['freq']} hip_phi={p['hip_phi']} "
          f"leg_amp={p['leg_amp']} hip_amp={p['hip_amp']} off={hip_off}")

    ddir = os.path.join(OUT, cfg, "demos")
    os.makedirs(ddir, exist_ok=True)
    n = 0
    for mu in MUS:
        set_floor_friction(model, mu)
        gs.FLOOR_MU = mu
        gs.POSE_JITTER = None
        gs.CONDITION["hip_off"] = hip_off
        gs._set_gait(dict(p))
        pid.reset()
        act, jadr = gc.build_ids(model)
        gc.set_initial_pose(model, data, act, jadr)

        cams = []
        for dist, elev, az in CAMS:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.elevation, cam.azimuth = dist, elev, az
            cams.append(cam)

        frames = []
        with mujoco.Renderer(model, height=480, width=640) as ren:
            nxt = 0.0
            while data.time < gs.SIM_DURATION:
                gc.apply_ctrl(data, act, data.time)
                mujoco.mj_step(model, data)
                if data.time >= nxt:
                    nxt += 1.0 / 30
                    pair = []
                    for cam in cams:
                        cam.lookat[:] = data.xpos[ids[3]]
                        ren.update_scene(data, cam)
                        pair.append(ren.render().copy())
                    frames.append(np.hstack(pair))

        tag = f"{mu:.1f}".replace("0.", "0")          # 0.1 -> 01
        out = os.path.join(ddir, f"demo_mu{tag}.mp4")
        imageio.mimsave(out, frames, fps=30, macro_block_size=1)
        print(f"  mu={mu}: {len(frames)} frames -> {os.path.relpath(out, _ROOT)}")
        n += 1

    gc.TORSO_CONTROLLER = None
    return n


total = 0
for cfg in wanted:
    total += render(cfg)
print(f"\nrendered {total} demo clips for: {', '.join(wanted)}")
