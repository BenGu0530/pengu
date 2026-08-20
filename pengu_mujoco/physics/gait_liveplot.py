#!/usr/bin/env python
"""Render a gait with a live roll / yaw plot beside it.

Roll is measured about the robot's OWN heading axis, not a world axis -- the robot
yaws substantially in open loop, so world-frame roll would mix in the turn. For each
frame:

  heading comes from the ROOT body (leftthighmotor), whose +y is horizontal and is the
  axis heading_align uses. Roll comes from the TORSO. Both axes are self-calibrated at
  the neutral stance rather than assumed, because the torso's +y actually points DOWN
  ([0,0,-1] at t=0) -- using it as a heading makes yaw and roll thrash through +-180.

  f_h   = root forward axis, projected horizontal, normalised   -> heading
  yaw   = atan2(f_h.x, f_h.y)                                   -> deg off world +y
  left  = z_world x f_h                                         -> left of heading
  u     = torso up axis in world                                -> body up
  roll  = atan2(u . left, u . z_world)                          -> lean about the heading

Panels: MuJoCo view | roll & yaw vs time | roll-vs-yaw phase plot with COM ground track.

usage:
  python physics/gait_liveplot.py c6 --freq 1.96 --phi 240 --leg 105 --hip 28 \
      --off 20 --mu 0.1 --out demo_liveplot.mp4
"""
import os, sys, math, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as imageio
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
ap.add_argument("--body", default="easytorso")
a = ap.parse_args()

kappa, com = CONF[a.cfg]
model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model)
slide, got = g4.apply_com_variant(model, com)
pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
gc.TORSO_CONTROLLER = pid
gs.FLOOR_MU = a.mu; gs.POSE_JITTER = None; gs.CONDITION["hip_off"] = a.off
set_floor_friction(model, a.mu)
gs._set_gait(dict(freq=a.freq, hip_phi=a.phi, leg_amp=a.leg, hip_amp=a.hip))
pid.reset()
act, jadr = gc.build_ids(model)
gc.set_initial_pose(model, data, act, jadr)

BID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a.body)
if BID < 0:
    BID = ids[3]
    print(f"body '{a.body}' not found, using body id {BID}")
RID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)

# self-calibrate at the neutral stance: which body axis of each body points along
# world +y (forward) and world +z (up)?  Do not assume -- the torso's +y points DOWN.
mujoco.mj_forward(model, data)
_R0r = data.xmat[RID].reshape(3, 3); _R0t = data.xmat[BID].reshape(3, 3)
_AX = [np.array(v, float) for v in
       ([1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1])]
FWD_R = max(_AX, key=lambda v: float((_R0r @ v) @ np.array([0., 1., 0.])))
UP_T = max(_AX, key=lambda v: float((_R0t @ v) @ np.array([0., 0., 1.])))
print(f"calibrated: root fwd axis {FWD_R} -> world {np.round(_R0r @ FWD_R, 3)} ; "
      f"torso up axis {UP_T} -> world {np.round(_R0t @ UP_T, 3)}")
print(f"{a.cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm mu={a.mu}")
print(f"gait freq={a.freq} phi={a.phi} leg={a.leg} hip={a.hip} off={a.off}")
print(f"roll/yaw measured on body '{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, BID)}'")

Z = np.array([0.0, 0.0, 1.0])
T, ROLL, YAW, CX, CY = [], [], [], [], []
frames_mj = []
cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 1.5, -12, -60

ren = mujoco.Renderer(model, height=480, width=560)
nxt = 0.0
while data.time < gs.SIM_DURATION:
    gc.apply_ctrl(data, act, data.time)
    mujoco.mj_step(model, data)
    if data.time >= nxt:
        nxt += 1.0 / 30
        Rr = data.xmat[RID].reshape(3, 3)
        R = data.xmat[BID].reshape(3, 3)
        f = Rr @ FWD_R
        fh = np.array([f[0], f[1], 0.0])
        n = np.linalg.norm(fh)
        if n < 1e-9:
            fh = np.array([0.0, 1.0, 0.0]); n = 1.0
        fh /= n
        left = np.cross(Z, fh)
        u = R @ UP_T
        YAW.append(math.degrees(math.atan2(fh[0], fh[1])))
        ROLL.append(math.degrees(math.atan2(float(u @ left), float(u @ Z))))
        mtot = model.body_mass.sum()
        c = (model.body_mass[:, None] * data.xipos).sum(0) / mtot
        CX.append(float(c[0])); CY.append(float(c[1])); T.append(data.time)
        cam.lookat[:] = data.xpos[ids[3]]
        ren.update_scene(data, cam)
        frames_mj.append(ren.render().copy())
ren.close()
print(f"simulated {len(T)} frames; roll range {min(ROLL):+.1f}..{max(ROLL):+.1f} deg, "
      f"yaw range {min(YAW):+.1f}..{max(YAW):+.1f} deg")

T = np.array(T); ROLL = np.array(ROLL); YAW = np.array(YAW)
CX = np.array(CX); CY = np.array(CY)
# Yaw accumulates (the robot turns), so unwrap it to read as one continuous curve.
# Roll must NOT be unwrapped: a lean is bounded, so unwrapping it just accumulates a
# spurious full turn. It stays in (-180, 180].
YAW = np.degrees(np.unwrap(np.radians(YAW)))
print(f"yaw after unwrap: {YAW.min():+.1f}..{YAW.max():+.1f} deg  "
      f"(total turn {YAW[-1]-YAW[0]:+.1f})")
SETTLE = gs.SETTLE
rm = ROLL[T >= SETTLE]
print(f"after settle: roll mean {rm.mean():+.2f} deg, RMS {np.sqrt((rm**2).mean()):.2f} deg, "
      f"amplitude {(rm.max()-rm.min())/2:.2f} deg")

fig = plt.figure(figsize=(7.2, 4.8), dpi=100)
canvas = FigureCanvasAgg(fig)
gsx = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.25, 1],
                       hspace=0.45, wspace=0.35)
ax_r = fig.add_subplot(gsx[0, 0]); ax_y = fig.add_subplot(gsx[1, 0])
ax_p = fig.add_subplot(gsx[:, 1])

ax_r.plot(T, ROLL, color="#c0392b", lw=1.0)
ax_r.axvline(SETTLE, color="gray", ls=":", lw=1)
ax_r.set_ylabel("roll [deg]", fontsize=8); ax_r.set_title(
    "torso roll about its OWN heading", fontsize=9)
ax_y.plot(T, YAW, color="#2471a3", lw=1.0)
ax_y.axvline(SETTLE, color="gray", ls=":", lw=1)
ax_y.set_ylabel("yaw [deg]", fontsize=8); ax_y.set_xlabel("t [s]", fontsize=8)
ax_y.set_title("heading yaw off world +y", fontsize=9)
for ax in (ax_r, ax_y):
    ax.grid(alpha=0.3); ax.tick_params(labelsize=7)
ax_p.plot(YAW, ROLL, color="#7f8c8d", lw=0.7, alpha=0.7)
ax_p.set_xlabel("yaw [deg]", fontsize=8); ax_p.set_ylabel("roll [deg]", fontsize=8)
ax_p.set_title("roll vs yaw", fontsize=9)
ax_p.grid(alpha=0.3); ax_p.tick_params(labelsize=7)

cur_r, = ax_r.plot([], [], "o", color="black", ms=5)
cur_y, = ax_y.plot([], [], "o", color="black", ms=5)
cur_p, = ax_p.plot([], [], "o", color="black", ms=6)
trail_p, = ax_p.plot([], [], color="#c0392b", lw=1.4)
txt = fig.text(0.01, 0.965, "", fontsize=8, family="monospace")

out_frames = []
for i in range(len(T)):
    cur_r.set_data([T[i]], [ROLL[i]])
    cur_y.set_data([T[i]], [YAW[i]])
    cur_p.set_data([YAW[i]], [ROLL[i]])
    j = max(0, i - 90)
    trail_p.set_data(YAW[j:i + 1], ROLL[j:i + 1])
    ph = "settling" if T[i] < SETTLE else "walking"
    txt.set_text(f"t={T[i]:5.2f}s  roll={ROLL[i]:+6.1f}°  yaw={YAW[i]:+6.1f}°  {ph}")
    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
    mj = frames_mj[i]
    h = max(mj.shape[0], buf.shape[0])
    def pad(im):
        if im.shape[0] == h: return im
        p = np.full((h - im.shape[0], im.shape[1], 3), 255, np.uint8)
        return np.vstack([im, p])
    out_frames.append(np.hstack([pad(mj), pad(buf)]))
plt.close(fig)

imageio.mimsave(a.out, out_frames, fps=30, macro_block_size=1)
print(f"wrote {a.out}  ({len(out_frames)} frames, "
      f"{out_frames[0].shape[1]}x{out_frames[0].shape[0]})")
