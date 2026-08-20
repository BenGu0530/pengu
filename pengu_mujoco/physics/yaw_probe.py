#!/usr/bin/env python
"""When does the yaw appear? Track base yaw/xy through hold -> transition -> walk.

The open-loop command is L/R symmetric (hip_phi shifts both cranks equally, so left
and right stay 180 deg apart), so a persistent yaw has to come from somewhere else:
the landing/rocking transient, or a symmetry-breaking instability.

usage:
  python physics/yaw_probe.py c6 --freq 1.96 --phi 240 --leg 105 --hip 28 --off 20
  ... --hold 15        override T_HOLD (longer settle before walking starts)
"""
import os, sys, argparse, math
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
ap.add_argument("cfg"); ap.add_argument("--freq", type=float, required=True)
ap.add_argument("--phi", type=float, required=True); ap.add_argument("--leg", type=float, required=True)
ap.add_argument("--hip", type=float, required=True); ap.add_argument("--off", type=float, required=True)
ap.add_argument("--mu", type=float, default=0.1)
ap.add_argument("--hold", type=float, default=None, help="override T_HOLD (settle time)")
ap.add_argument("--dur", type=float, default=None, help="override SIM_DURATION")
a = ap.parse_args()

if a.hold is not None:
    gc.T_HOLD = a.hold
    gs.SETTLE = gc.T_HOLD + gc.T_TRANSITION + 2.0
if a.dur is not None:
    gs.SIM_DURATION = a.dur

kappa, com = CONF[a.cfg]
model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model); g4.apply_com_variant(model, com)
pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE); gc.TORSO_CONTROLLER = pid
gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None
gs.CONDITION["hip_off"] = a.off
set_floor_friction(model, a.mu)
gs._set_gait(dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg, hip_amp=a.hip))
pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)

print(f"{a.cfg} mu={a.mu}  T_HOLD={gc.T_HOLD}  T_TRANSITION={gc.T_TRANSITION}  "
      f"SETTLE={gs.SETTLE}  SIM_DURATION={gs.SIM_DURATION}")
print(f"gait freq={a.freq} phi={a.phi} leg={a.leg} hip={a.hip} off={a.off}\n")

def yaw_of(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

marks = [gc.T_HOLD, gc.T_HOLD + gc.T_TRANSITION, gs.SETTLE]
rows, nxt, mi = [], 0.0, 0
peak_rockrate = 0.0
while data.time < gs.SIM_DURATION:
    gc.apply_ctrl(data, act, data.time)
    mujoco.mj_step(model, data)
    if data.time >= nxt:
        nxt += 0.5
        yaw = yaw_of(data.qpos[3:7])
        wz = float(data.qvel[5])            # yaw rate
        rock = float(np.hypot(data.qvel[3], data.qvel[4]))   # roll+pitch rate magnitude
        rows.append((data.time, data.qpos[0], data.qpos[1], yaw, wz, rock))

print(f"{'t':>6}{'x':>9}{'y':>9}{'yaw_deg':>9}{'yaw_rate':>10}{'rock_rate':>11}  phase")
for t, x, y, yaw, wz, rock in rows:
    ph = "hold" if t < gc.T_HOLD else ("blend" if t < gc.T_HOLD + gc.T_TRANSITION else "walk")
    if t % 1.0 < 0.51 or t < 3:
        print(f"{t:>6.1f}{x:>9.4f}{y:>9.4f}{yaw:>9.2f}{wz:>10.4f}{rock:>11.4f}  {ph}")

t0 = [r for r in rows if r[0] <= gc.T_HOLD]
tw = [r for r in rows if r[0] >= gs.SETTLE]
print(f"\nyaw at end of HOLD (t={gc.T_HOLD}): {t0[-1][3]:+.3f} deg" if t0 else "")
print(f"yaw at start of measured walk (t={gs.SETTLE}): {tw[0][3]:+.3f} deg" if tw else "")
print(f"yaw at end (t={rows[-1][0]:.1f}): {rows[-1][3]:+.3f} deg")
print(f"max |rock rate| during hold: {max((r[5] for r in t0), default=0):.5f} rad/s")
print(f"max |rock rate| in last 2 s of hold: "
      f"{max((r[5] for r in t0 if r[0] > gc.T_HOLD-2), default=0):.5f} rad/s")
dx, dy = rows[-1][1] - rows[0][1], rows[-1][2] - rows[0][2]
print(f"\nnet travel: dx={dx:+.4f}  dy={dy:+.4f}  "
      f"angle off +y = {math.degrees(math.atan2(dx, dy)):+.1f} deg")
