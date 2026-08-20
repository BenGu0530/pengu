#!/usr/bin/env python
"""COM motion in the robot's OWN frame: lateral sway vs height (the "wiper" plot).

gait_sweep's com_lat uses WORLD x ("forward travel is +y"), which is wrong once the
robot yaws -- and it yaws ~70 deg in open loop. Here the lateral axis is rebuilt every
frame from the robot's own heading:

  fwd    = root body forward axis, projected horizontal, normalised
  left   = z_world x fwd                      -> +lateral is to the robot's LEFT
  footmid= midpoint of the two foot bodies
  lat    = (COM - footmid) . left             -> sway, signed left/right
  height = COM_z                              -> above the floor

Plotting lat (x) against height (y) traces the inverted-pendulum arc: the COM vaults
left-right-left-right over alternating stance feet.

usage:
  python physics/com_wiper.py c6 --freq 1.96 --phi 240 --leg 105 --hip 28 --off 20 \
      --mu 0.1 --out results/grid4_report/c6/com_wiper_mu01.png
"""
import os, sys, math, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
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
ap.add_argument("--out", required=True)
ap.add_argument("--video", default=None,
                help="also render an mp4: MuJoCo view | animated wiper | sway & height")
a = ap.parse_args()

kappa, com_ratio = CONF[a.cfg]
model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model)
slide, got = g4.apply_com_variant(model, com_ratio)
pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
gc.TORSO_CONTROLLER = pid
gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None; gs.CONDITION["hip_off"] = a.off
set_floor_friction(model, a.mu)
gs._set_gait(dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg, hip_amp=a.hip))
pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)

RID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
FOOT = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n, s in gs.FOOT_BODIES.items()}
mujoco.mj_forward(model, data)
_R0 = data.xmat[RID].reshape(3, 3)
_AX = [np.array(v, float) for v in ([1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1])]
FWD = max(_AX, key=lambda v: float((_R0 @ v) @ np.array([0., 1., 0.])))
Z = np.array([0.0, 0.0, 1.0])
print(f"{a.cfg}: kappa={kappa} com={got:.4f} mu={a.mu}  fwd axis {FWD} "
      f"-> world {np.round(_R0 @ FWD, 3)}")

T, LAT, HGT, STANCE, FWDPOS = [], [], [], [], []
F_HI = getattr(gs, "F_HI", 4.0)
VF = []                       # video frames (30 Hz) when --video
vren = vcam = None
if a.video:
    vcam = mujoco.MjvCamera(); vcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    vcam.distance, vcam.elevation, vcam.azimuth = 1.5, -12, -60
    vren = mujoco.Renderer(model, height=480, width=560)
vnxt = 0.0
nxt = 0.0
while data.time < gs.SIM_DURATION:
    gc.apply_ctrl(data, act, data.time)
    mujoco.mj_step(model, data)
    if data.time >= nxt:
        nxt += 1.0 / 120
        R = data.xmat[RID].reshape(3, 3)
        f = R @ FWD
        fh = np.array([f[0], f[1], 0.0]); n = np.linalg.norm(fh)
        fh = fh / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        left = np.cross(Z, fh)
        com = np.asarray(data.subtree_com[0], float)
        pL = np.asarray(data.xpos[FOOT["L"]], float)
        pR = np.asarray(data.xpos[FOOT["R"]], float)
        mid = 0.5 * (pL + pR)
        rel = com - mid
        LAT.append(float(rel @ left))
        HGT.append(float(com[2]))
        FWDPOS.append(float(rel @ fh))
        # which foot is lower = carrying weight (cheap stance proxy, no contact query)
        STANCE.append(1 if pL[2] < pR[2] else -1)
        T.append(data.time)
    if a.video and data.time >= vnxt:
        vnxt += 1.0 / 30
        vcam.lookat[:] = data.xpos[ids[3]]
        vren.update_scene(data, vcam)
        VF.append((data.time, vren.render().copy()))

T = np.array(T); LAT = np.array(LAT) * 1000.0; HGT = np.array(HGT) * 1000.0
STANCE = np.array(STANCE); FWDPOS = np.array(FWDPOS) * 1000.0
k = T >= gs.SETTLE
print(f"walking window {gs.SETTLE}-{gs.SIM_DURATION}s, {k.sum()} samples")
print(f"  lateral sway : mean {LAT[k].mean():+.1f}  RMS {np.sqrt((LAT[k]**2).mean()):.1f}  "
      f"range {LAT[k].min():+.1f}..{LAT[k].max():+.1f} mm   peak-to-peak {np.ptp(LAT[k]):.1f} mm")
print(f"  COM height   : mean {HGT[k].mean():.1f}  range {HGT[k].min():.1f}..{HGT[k].max():.1f} mm"
      f"   peak-to-peak {np.ptp(HGT[k]):.1f} mm")

fig = plt.figure(figsize=(13.4, 4.9))
gsx = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.5, wspace=0.28)
ax_w = fig.add_subplot(gsx[:, 0])
ax_l = fig.add_subplot(gsx[0, 1]); ax_h = fig.add_subplot(gsx[1, 1])

pts = np.column_stack([LAT[k], HGT[k]]).reshape(-1, 1, 2)
segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segs, cmap="coolwarm", linewidth=1.3, alpha=0.85)
lc.set_array(STANCE[k][:-1].astype(float))
ax_w.add_collection(lc)
ax_w.set_xlim(LAT[k].min() - 6, LAT[k].max() + 6)
ax_w.set_ylim(HGT[k].min() - 4, HGT[k].max() + 4)
ax_w.axvline(0, color="gray", ls=":", lw=1)
ax_w.set_xlabel("lateral sway in BODY frame [mm]   (+ = robot's left)")
ax_w.set_ylabel("COM height [mm]")
ax_w.set_title(f"{a.cfg} COM in its own frame, $\\mu$={a.mu} — blue: right foot lower, "
               f"red: left foot lower", fontsize=9)
ax_w.grid(alpha=0.3)

ax_l.plot(T[k], LAT[k], color="#c0392b", lw=0.9)
ax_l.axhline(0, color="gray", ls=":", lw=1)
ax_l.set_ylabel("lateral [mm]", fontsize=8); ax_l.set_title("sway vs time", fontsize=9)
ax_h.plot(T[k], HGT[k], color="#2471a3", lw=0.9)
ax_h.set_ylabel("height [mm]", fontsize=8); ax_h.set_xlabel("t [s]", fontsize=8)
ax_h.set_title("COM height vs time", fontsize=9)
for ax in (ax_l, ax_h):
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(a.out, dpi=130); plt.close()
print(f"wrote {a.out}")

if a.video:
    vren.close()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import imageio.v2 as imageio
    f2 = plt.figure(figsize=(7.2, 4.8), dpi=100)
    cv = FigureCanvasAgg(f2)
    g2 = f2.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.5, wspace=0.32)
    aw = f2.add_subplot(g2[:, 0]); al = f2.add_subplot(g2[0, 1]); ah = f2.add_subplot(g2[1, 1])
    aw.plot(LAT[k], HGT[k], color="#d5d8dc", lw=0.8)
    aw.axvline(0, color="gray", ls=":", lw=1)
    aw.set_xlim(LAT[k].min() - 6, LAT[k].max() + 6)
    aw.set_ylim(HGT[k].min() - 4, HGT[k].max() + 4)
    aw.set_xlabel("lateral sway, BODY frame [mm]  (+ = left)", fontsize=8)
    aw.set_ylabel("COM height [mm]", fontsize=8)
    aw.set_title("COM wiper in the robot's own frame", fontsize=9)
    aw.grid(alpha=0.3); aw.tick_params(labelsize=7)
    al.plot(T[k], LAT[k], color="#d5d8dc", lw=0.8); al.axhline(0, color="gray", ls=":", lw=1)
    al.set_ylabel("lateral [mm]", fontsize=8); al.set_title("sway vs t", fontsize=9)
    ah.plot(T[k], HGT[k], color="#d5d8dc", lw=0.8)
    ah.set_ylabel("height [mm]", fontsize=8); ah.set_xlabel("t [s]", fontsize=8)
    ah.set_title("COM height vs t", fontsize=9)
    for ax in (al, ah): ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
    trail, = aw.plot([], [], color="#c0392b", lw=1.6)
    dot,   = aw.plot([], [], "o", color="black", ms=6)
    dl,    = al.plot([], [], "o", color="black", ms=4)
    dh,    = ah.plot([], [], "o", color="black", ms=4)
    tt = f2.text(0.01, 0.965, "", fontsize=8, family="monospace")
    out = []
    for tv, mjf in VF:
        i = int(np.searchsorted(T, tv))
        i = min(max(i, 0), len(T) - 1)
        if T[i] >= gs.SETTLE:
            j = max(0, i - 120)
            trail.set_data(LAT[j:i + 1], HGT[j:i + 1])
            dot.set_data([LAT[i]], [HGT[i]])
            dl.set_data([T[i]], [LAT[i]]); dh.set_data([T[i]], [HGT[i]])
            ph = "walking"
        else:
            ph = "settling"
        tt.set_text(f"t={T[i]:5.2f}s  lat={LAT[i]:+6.1f}mm  h={HGT[i]:6.1f}mm  {ph}")
        cv.draw()
        buf = np.asarray(cv.buffer_rgba())[:, :, :3]
        hh = max(mjf.shape[0], buf.shape[0])
        def pad(im):
            if im.shape[0] == hh: return im
            return np.vstack([im, np.full((hh - im.shape[0], im.shape[1], 3), 255, np.uint8)])
        out.append(np.hstack([pad(mjf), pad(buf)]))
    plt.close(f2)
    imageio.mimsave(a.video, out, fps=30, macro_block_size=1)
    print(f"wrote {a.video}  ({len(out)} frames, {out[0].shape[1]}x{out[0].shape[0]})")
