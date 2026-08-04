#!/usr/bin/env python
"""Render several k0 (Gait 1, kappa=0) gaits side by side for visual comparison.

Each gait gets its own MP4 with a consistent tracked 3/4 back view and an on-frame
label: its swept metrics (net_fwd, slip_ratio) + a LIVE cumulative slide distance so
you can watch slip pile up. All at floor mu=0.7 with the kappa=0 PID torso.

usage: MUJOCO_GL=egl python physics/demo_compare.py
out:   results/gait_sweep/demo_cmp_<name>.mp4
"""
import os, sys
os.environ.setdefault("PENGU_MODEL", "v3")
os.environ.setdefault("MUJOCO_GL", "egl")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import math
import numpy as np
import mujoco
import imageio.v2 as imageio
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False

FPS = 30
# FORWARD-FACING gaits only (heading_align>0.5, from the clean 29-col k0 sweep).
# name, params, swept net_fwd, slip_ratio, heading_align
GAITS = [
    ("fwd_fastest",    dict(freq=2.00, hip_phi=250, leg_amp=125, hip_amp=28, hip_off=10), 0.291, 0.104, 0.99),
    ("fwd_fast_lowslip", dict(freq=1.99, hip_phi=260, leg_amp=125, hip_amp=28, hip_off=10), 0.235, 0.085, 0.99),
    ("fwd_minslip",    dict(freq=1.47, hip_phi=310, leg_amp=105, hip_amp=28, hip_off=10), 0.155, 0.081, 1.00),
    ("fwd_highpitch",  dict(freq=1.12, hip_phi=220, leg_amp=125, hip_amp=24, hip_off=50), 0.258, 0.121, 1.00),
]

model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model); floor_id, foot_geom, foot_bid, root = ids
gs.FLOOR_MU = 0.7
DT = float(model.opt.timestep)
pid = TorsoKappaPID(model, kappa=0.0, measure_after=gs.SETTLE)


def slip_speed_now(f6, vf6):
    """total force-weighted stance slip speed this step (same defn as run_trial)."""
    num = den = 0.0
    for c in range(data.ncon):
        ct = data.contact[c]
        fg = ct.geom2 if ct.geom1 == floor_id else (ct.geom1 if ct.geom2 == floor_id else -1)
        if fg not in foot_geom:
            continue
        mujoco.mj_contactForce(model, data, c, f6)
        fn = abs(f6[0])
        if fn <= gs.F_HI:
            continue
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, fg, vf6, 0)
        v_pt = vf6[3:6] + np.cross(vf6[0:3], ct.pos - data.geom_xpos[fg])
        n = ct.frame[0:3]
        v_tan = v_pt - np.dot(v_pt, n) * n
        num += fn * float(np.linalg.norm(v_tan)); den += fn
    return num / den if den > 0 else 0.0


def label(frame, lines):
    if not HAVE_PIL:
        return frame
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 300, 12 + 15 * len(lines)], fill=(0, 0, 0))
    for i, ln in enumerate(lines):
        d.text((6, 4 + 15 * i), ln, fill=(255, 255, 255))
    return np.asarray(im)


# --- single renderer instance (reuse across gaits) ---
def run():
    ren = mujoco.Renderer(model, height=480, width=640)
    outs = []
    for name, p, net_swept, slip_swept, head_swept in GAITS:
        set_floor_friction(model, gs.FLOOR_MU)
        gs.CONDITION["hip_off"] = p["hip_off"]
        gs._set_gait({k: v for k, v in p.items() if k != "hip_off"})
        gc.TORSO_CONTROLLER = pid; pid.reset()
        act, jadr = gc.build_ids(model)
        gc.set_initial_pose(model, data, act, jadr)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.elevation, cam.azimuth = 1.4, -14.0, -60.0
        f6 = np.zeros(6); vf6 = np.zeros(6)
        frames = []; nxt = 0.0; slip_cum = 0.0; y0 = None; fell = False
        pstr = f"f{p['freq']:.2f} phi{p['hip_phi']:.0f} leg{p['leg_amp']:.0f} hip{p['hip_amp']:.0f} off{p['hip_off']:.0f}"
        while data.time < gs.SIM_DURATION:
            gc.apply_ctrl(data, act, data.time); mujoco.mj_step(model, data)
            if data.time >= gs.SETTLE:
                if y0 is None:
                    y0 = float(data.xpos[root][1])
                slip_cum += slip_speed_now(f6, vf6) * DT
            if data.xpos[root][2] < 0.05:
                fell = True; break
            if data.time >= nxt:
                nxt += 1.0 / FPS
                cam.lookat[:] = data.xpos[root]
                ren.update_scene(data, cam)
                fwd = (float(data.xpos[root][1]) - y0) if y0 is not None else 0.0
                lines = [name, pstr,
                         f"swept: net_fwd={net_swept:.3f}  slip={slip_swept*100:.1f}%  heading={head_swept:+.2f}",
                         f"live:  fwd={fwd:+.2f}m  slid={slip_cum:.2f}m"]
                frames.append(label(ren.render().copy(), lines))
        out = os.path.join(_ROOT, "results", "gait_sweep", f"demo_cmp_{name}.mp4")
        imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
        outs.append((name, out, len(frames), slip_cum, fell))
        print(f"wrote {out}  ({len(frames)} frames, slid {slip_cum:.2f}m, fell={fell})")
    ren.close(); gc.TORSO_CONTROLLER = None
    return outs


if __name__ == "__main__":
    run()
