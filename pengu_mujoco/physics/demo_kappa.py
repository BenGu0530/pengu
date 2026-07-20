#!/usr/bin/env python
"""
Demo: walk with the hip axis rolling while the torso is held upright (kappa=0),
side-by-side against kappa=1 (torso rides the axis) and kappa=2 (torso leans further).

Gait B's leg params were co-tuned with the OPEN-LOOP torso sinusoid, so they do not walk
under a PID torso. This script therefore (1) CMA-searches leg/hip/freq/hip_off that walk
under the requested kappa, then (2) renders a BACK view (the roll is only visible from
behind) with live telemetry: hip-axis tilt vs torso tilt.

usage: MUJOCO_GL=egl python physics/demo_kappa.py [kappa ...]      e.g. 0 1 2
out:   results/gait_sweep/demo_kappa{K}.mp4
"""
import os, sys

os.environ.setdefault("PENGU_MODEL", "v3")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import cma
import imageio.v2 as imageio
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction

KAPPAS = [float(x) for x in sys.argv[1:]] or [0.0]
MAXFEV = 220
# hip_off (forward-pitch) is HELD CONSTANT across kappa so the only thing that differs
# between the demos is the torso roll strategy — not the forward lean. Only leg/hip/freq
# are optimized per kappa.
FIXED_HIP_OFF = 30.0
BOUNDS = {"leg_amp": (40.0, 120.0), "hip_amp": (5.0, 30.0), "freq": (1.0, 2.0)}
FREE = list(BOUNDS)
SEED = dict(leg_amp=100.0, hip_amp=20.0, freq=1.4, hip_off=FIXED_HIP_OFF, hip_phi=210.0)
FPS = 30

model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model)
gs.FLOOR_MU = 0.7


def build_p(x):
    p = dict(SEED)
    for n, v in zip(FREE, [BOUNDS[k][0] + xi * (BOUNDS[k][1] - BOUNDS[k][0])
                           for k, xi in zip(FREE, np.clip(x, 0, 1))]):
        p[n] = float(v)
    return p


def trial(p):
    gs.CONDITION["hip_off"] = p["hip_off"]
    return gs.run_trial(model, data, ids, {k: v for k, v in p.items() if k != "hip_off"})


def search(kappa):
    """find leg params that actually WALK under this kappa (demo needs a real gait)."""
    pid = TorsoKappaPID(model, kappa=kappa)
    gc.TORSO_CONTROLLER = pid
    best = {"J": 1e9, "p": None, "r": None}

    def obj(x):
        p = build_p(x)
        r = trial(p)
        if not r["survived"]:
            return 2.0
        # want real forward progress AND clean single-support alternation
        J = -r["net_fwd_speed"] - 0.3 * r["single_frac"]
        if J < best["J"]:
            best.update(J=J, p=p, r=r)
        return J

    x0 = np.array([(SEED[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in FREE])
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"bounds": [0, 1], "maxfevals": MAXFEV,
                                            "verb_disp": 0, "seed": 1})
    es.optimize(obj)
    return best, pid


def render(kappa, p, pid):
    """back view; the torso-vs-axis roll is only legible from behind."""
    set_floor_friction(model, gs.FLOOR_MU)
    gs.CONDITION["hip_off"] = p["hip_off"]
    gs._set_gait({k: v for k, v in p.items() if k != "hip_off"})
    gc.TORSO_CONTROLLER = pid
    pid.reset()
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 1.1, -12.0, -90.0   # behind, looking +y
    frames, tele = [], []
    with mujoco.Renderer(model, height=480, width=640) as ren:
        nxt = 0.0
        while data.time < gs.SIM_DURATION:
            gc.apply_ctrl(data, act, data.time)
            mujoco.mj_step(model, data)
            if data.time >= nxt:
                nxt += 1.0 / FPS
                h = pid.hinge(data)
                a = np.degrees(pid.axis_roll(data, h))
                t = np.degrees(pid.torso_roll(data, h))
                cam.lookat[:] = data.xpos[ids[3]]
                ren.update_scene(data, cam)
                frames.append(ren.render())
                tele.append((data.time, a, t))
    out = os.path.join(_ROOT, "results", "gait_sweep", f"demo_kappa{kappa:g}.mp4")
    imageio.mimsave(out, frames, fps=FPS, macro_block_size=1)
    tele = np.array(tele)
    w = tele[tele[:, 0] > gs.SETTLE]
    print(f"  wrote {out}")
    print(f"    axis  tilt: rms {np.sqrt((w[:,1]**2).mean()):5.2f}d  range [{w[:,1].min():+6.1f},{w[:,1].max():+6.1f}]")
    print(f"    torso tilt: rms {np.sqrt((w[:,2]**2).mean()):5.2f}d  range [{w[:,2].min():+6.1f},{w[:,2].max():+6.1f}]")


for k in KAPPAS:
    print(f"=== kappa {k:g}: searching a gait that walks under this torso mode ...")
    best, pid = search(k)
    p, r = best["p"], best["r"]
    print(f"  best: leg={p['leg_amp']:.1f} hip={p['hip_amp']:.1f} f={p['freq']:.3f} "
          f"hip_off={p['hip_off']:.1f} | net_fwd={r['net_fwd_speed']:.3f} "
          f"single={r['single_frac']:.3f} mu={r['mu_req_p95']:.3f}")
    render(k, p, pid)
gc.TORSO_CONTROLLER = None
