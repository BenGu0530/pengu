#!/usr/bin/env python
"""Calibrate the GRID-5 slip dual-criterion deadband (gait_sweep.SLIP_V0, SLIP_C).

Three reference regimes, measured with the same contact instrumentation as the sweep:
  STATIC   stand only (no gait) on mu=0.7      -> |v_tan| noise floor  => SLIP_V0
  ROLLING  slow modest gait on mu=0.9          -> cone util << 1, contact motion is
           rolling; patch-edge |v_tan| vs |omega|*r_patch bounds       => SLIP_C
  SLIDING  aggressive gait on mu=0.05          -> cone pegged, true sliding; the
           deadband must NOT swallow these velocities (separation check)

Prints distributions; Ben freezes the constants, which then go into the manifest.
Run from grid5/:  ../.sweep_venv/bin/python slip_calib_probe.py
"""
import os, sys, math
os.environ["PENGU_MODEL"] = "1.31"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, mujoco
import gait_config as gc, gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
from grid5_sweep import apply_com_variant

def collect(mu, gait, kappa, com, dur=20.0, stand_only=False):
    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)
    _lean = gc.STAND_HIP_DEG; gc.STAND_HIP_DEG = 0.0
    apply_com_variant(model, com)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.STAND_HIP_DEG = 5.0
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    if gait is not None:
        gs.CONDITION["hip_off"] = gait[4]
        gs._set_gait(dict(freq=gait[0], hip_phi=gait[1], leg_amp=gait[2], hip_amp=gait[3]))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    gc.T_HOLD = 1e9 if stand_only else 3.0
    f6 = np.zeros(6); vf6 = np.zeros(6)
    rows = []   # (fn, ft, vtan, util, omega, r_patch, ncon_foot)
    t_meas = 2.0 if stand_only else 9.0    # gait: skip hold+transition
    while data.time < dur:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            print(f"   (fell at {data.time:.2f}s)")
            break
        if data.time < t_meas: continue
        pts = {"L": [], "R": []}; om = {"L": 0.0, "R": 0.0}
        for c in range(data.ncon):
            ct = data.contact[c]
            fg = ct.geom2 if ct.geom1 == floor_id else (ct.geom1 if ct.geom2 == floor_id else -1)
            ft = foot_geom.get(fg)
            if ft:
                mujoco.mj_contactForce(model, data, c, f6)
                fn = abs(f6[0]); ftm = math.hypot(f6[1], f6[2])
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, fg, vf6, 0)
                v_pt = vf6[3:6] + np.cross(vf6[0:3], ct.pos - data.geom_xpos[fg])
                n = ct.frame[0:3]
                vt = float(np.linalg.norm(v_pt - np.dot(v_pt, n) * n))
                pts[ft].append((fn, ftm, vt, ct.pos[:2].copy()))
                om[ft] = float(np.linalg.norm(vf6[0:3]))
        for s in ("L", "R"):
            P = pts[s]
            fn_tot = sum(p[0] for p in P)
            if fn_tot <= gs.F_HI or not P: continue
            r_patch = 0.0
            if len(P) > 1:
                for i in range(len(P)):
                    for j in range(i+1, len(P)):
                        d = P[i][3] - P[j][3]
                        r_patch = max(r_patch, 0.5*math.hypot(d[0], d[1]))
            for fn, ftm, vt, _ in P:
                if fn > 1e-9:
                    rows.append((fn, ftm, vt, ftm/(mu*fn), om[s], r_patch, len(P)))
    gc.T_HOLD = 5.0
    return rows

def report(name, rows):
    if not rows:
        print(f"-- {name}: no loaded-contact samples"); return
    vt = np.array([r[2] for r in rows]); util = np.array([r[3] for r in rows])
    om = np.array([r[4] for r in rows]); rp = np.array([r[5] for r in rows])
    multi = rp > 0
    print(f"-- {name}  (n={len(rows)}, multi-contact share={multi.mean()*100:.1f}%)")
    print(f"   |v_tan| m/s : p50={np.percentile(vt,50):.5f} p90={np.percentile(vt,90):.5f} "
          f"p99={np.percentile(vt,99):.5f} max={vt.max():.4f}")
    print(f"   cone util   : p50={np.percentile(util,50):.3f} p90={np.percentile(util,90):.3f} "
          f"p99={np.percentile(util,99):.3f}")
    if multi.any():
        ratio = vt[multi] / np.maximum(om[multi]*rp[multi], 1e-9)
        print(f"   multi-contact vt/(omega*r_patch): p50={np.percentile(ratio,50):.2f} "
              f"p90={np.percentile(ratio,90):.2f} p99={np.percentile(ratio,99):.2f}")
    return vt, util

print("=== STATIC (stand only, mu=0.7, rest lean 5) — v_tan noise floor -> SLIP_V0 ===")
s_static = collect(0.7, None, 0.0, 1.31, dur=14.0, stand_only=True)
report("static", s_static)

print("\n=== ROLLING (slow modest gait, mu=0.9: freq1.3 phi270 leg95 hip16 off10, k0 1.05) ===")
s_roll = collect(0.9, (1.3, 270.0, 95.0, 16.0, 10.0), 0.0, 1.05, dur=22.0)
report("rolling", s_roll)

print("\n=== SLIDING (aggressive gait, mu=0.05: freq1.9 phi270 leg135 hip28 off20, k2 1.05) ===")
s_slide = collect(0.05, (1.9, 270.0, 135.0, 28.0, 20.0), 2.0, 1.05, dur=22.0)
report("sliding", s_slide)

print("\n=== separation with candidate constants ===")
for v0 in (0.002, 0.005, 0.01, 0.02):
    for cc in (0.5, 1.0, 2.0):
        def frac_slipping(rows, mu):
            k = 0
            for fn, ftm, vt, util, om, rp, nc in rows:
                cone = ftm >= (1-0.05)*mu*fn
                kin = vt >= cc*om*rp + v0
                k += int(cone and kin)
            return k/max(1,len(rows))
        print(f"  v0={v0:<6} c={cc:<4} -> slipping%%: static={100*frac_slipping(s_static,0.7):5.1f} "
              f"rolling={100*frac_slipping(s_roll,0.9):5.1f}  sliding={100*frac_slipping(s_slide,0.05):5.1f}")
