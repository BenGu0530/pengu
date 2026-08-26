#!/usr/bin/env python
"""IMU frame probe — validate the torso-attitude decoupling math in sim, and render a
side-by-side demo (robot walking | live triads for the whole-body base and the torso).

The hardware problem (Ben, 2026-08-25): the BNO055 sits on the torso; once the torso
pitches ~20 deg forward its Euler 'roll' no longer reads the lateral lean — the mocap
metric and the on-robot kappa-PID both feed on a mixed quantity. The fix direction is a
gravity-vector roll. This probe establishes, in sim where every rotation is known:

  1. z-x-y intrinsic decomposition (yaw about world z, pitch about lateral x, roll
     about forward y) recovers synthetic (yaw,pitch,roll) exactly;
  2. gravity-vector roll == the z-x-y roll ANALYTICALLY (both come from the rotation
     matrix bottom row) — i.e. a *correctly decomposed* Euler roll is already
     pitch-immune; the hardware mixing must come from the BNO055's own fusion/output
     conventions, so the firmware fix should compute roll from the RAW gravity/accel
     vector, not from the device's Euler output;
  3. against the hinge-axis roll (torso_control._tilt_about — what the kappa-PID
     nulls): reports the gap between world-lateral lean and hinge-frame roll over a
     real walking trajectory (grows with yaw/pitch; this is the documented reason the
     PID measures about the hinge).

usage (from grid5/):
  ../.sweep_venv/bin/python imu_frame_probe.py                 # numeric validation
  MUJOCO_GL=egl ../.sweep_venv/bin/python imu_frame_probe.py --demo [out.mp4]
"""
import os, sys, math
os.environ.setdefault("PENGU_MODEL", "1.31")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID, _tilt_about
import grid5_sweep as g5              # sets staged/ramp/lean/extended switches

# demo gait: the c6 champion (kappa=2, COM 1.31) on mu=0.1 — big torso roll, clear story
GAIT = dict(freq=1.67, hip_phi=340.0, leg_amp=95.0, hip_amp=24.0)
HIP_OFF = 20.0
MU = 0.1
KAPPA, COM = 2.0, 1.31


# ---------------- rotation helpers (the quantities under test) ----------------------
def euler_zxy(R):
    """Intrinsic z-x-y: R = Rz(yaw) @ Rx(pitch) @ Ry(roll). World y = forward, so
    pitch = forward dip (about lateral x), roll = lateral lean (about forward y)."""
    pitch = math.asin(max(-1.0, min(1.0, float(R[2, 1]))))
    roll = math.atan2(-float(R[2, 0]), float(R[2, 2]))
    yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return yaw, pitch, roll


def grav_roll(R):
    """Gravity-vector roll: g in the frame's own coords, lean about the forward axis.
    Pitch-immune by construction — the firmware-fix form (raw accel, not device Euler)."""
    g = -R[2, :]                      # R^T @ (0,0,-1)
    return math.atan2(float(g[0]), -float(g[2]))


def compose_zxy(yaw, pitch, roll):
    cz, sz = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(roll), math.sin(roll)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    Rx = np.array([[1.0, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1.0, 0], [-sy, 0, cy]])
    return Rz @ Rx @ Ry


# ---------------- shared trial setup -------------------------------------------------
def build():
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    _lean = gc.STAND_HIP_DEG; gc.STAND_HIP_DEG = 0.0
    g5.apply_com_variant(model, COM)
    pid = TorsoKappaPID(model, kappa=KAPPA, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = _lean
    from friction_utils import set_floor_friction
    set_floor_friction(model, MU)
    gs.FLOOR_MU = MU
    gs.CONDITION["hip_off"] = HIP_OFF
    gs._set_gait(dict(GAIT))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
    R0t = data.xmat[tid].reshape(3, 3).copy()     # torso mount calibration (rest pose)
    R0r = data.xmat[rid].reshape(3, 3).copy()     # base calibration
    return model, data, ids, act, pid, tid, rid, R0t, R0r


def staged_step(model, data, act, state):
    """One physics step under the grid5 staged schedule; returns (t0, alive)."""
    t = data.time
    if state["t0"] is None:
        if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                or t >= gs.QUIET_MAX_T:
            state["t0"] = t
            gc.T_HOLD = t
    gc.apply_ctrl(data, act, data.time)
    mujoco.mj_step(model, data)
    return state["t0"], data.xpos[3][2] if False else True


def main_validate():
    print("=== 1. synthetic round-trip: compose R = Rz(yaw)Rx(pitch)Ry(roll), decompose ===")
    worst = 0.0; worst_g = 0.0
    for yaw in np.radians([-150, -60, 0, 30, 120]):
        for pitch in np.radians([-60, -20, 0, 20, 60]):
            for roll in np.radians([-70, -25, 0, 25, 70]):
                R = compose_zxy(yaw, pitch, roll)
                y2, p2, r2 = euler_zxy(R)
                worst = max(worst, abs(y2-yaw), abs(p2-pitch), abs(r2-roll))
                worst_g = max(worst_g, abs(grav_roll(R) - roll))
    print(f"   125 poses (yaw+-150, pitch+-60, roll+-70): max decomposition error = "
          f"{math.degrees(worst):.2e} deg;  max |grav_roll - roll| = {math.degrees(worst_g):.2e} deg")

    print("\n=== 2. walking trajectory (c6 champion, mu=0.1, grid5 staged protocol) ===")
    gc.T_HOLD = 1e9
    model, data, ids, act, pid, tid, rid, R0t, R0r = build()
    root = ids[3]
    state = {"t0": None}
    rows = []
    while data.time < (state["t0"] + 16.0 if state["t0"] is not None else 1e9):
        t = data.time
        if state["t0"] is None:
            if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                state["t0"] = t; gc.T_HOLD = t
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            print(f"   fell at {data.time:.2f}s"); break
        if state["t0"] is None or data.time < state["t0"] + 6.0:
            continue
        Rt = data.xmat[tid].reshape(3, 3)
        Rrel = Rt @ R0t.T
        yw, pt, rl = euler_zxy(Rrel)
        gr = grav_roll(Rrel)
        h = data.xaxis[pid.jid].copy()
        hinge_roll = _tilt_about(data, tid, pid.u_torso, h)       # what the PID nulls
        axis_roll = pid.axis_roll(data, h)
        rows.append((data.time, yw, pt, rl, gr, hinge_roll, axis_roll))
    gc.T_HOLD = 5.0
    a = np.array(rows)
    deg = np.degrees
    print(f"   samples: {len(a)}  window: [{a[0,0]:.2f}, {a[-1,0]:.2f}] s")
    print(f"   torso attitude: yaw rms={deg(np.sqrt(np.mean(a[:,1]**2))):6.2f}  "
          f"pitch rms={deg(np.sqrt(np.mean(a[:,2]**2))):6.2f}  roll rms={deg(np.sqrt(np.mean(a[:,3]**2))):6.2f} deg")
    d_ge = deg(np.max(np.abs(a[:,4] - a[:,3])))
    print(f"   |grav_roll - euler_roll|: max = {d_ge:.2e} deg   -> identical (same matrix row); "
          f"the hardware Euler mixing is a DEVICE property, use raw gravity on firmware")
    corr = np.corrcoef(a[:,5], a[:,3])[0, 1]
    m, b = np.polyfit(a[:,3], a[:,5], 1)            # hinge ~ m*euler + b (sign convention)
    resid = deg(a[:,5] - (m*a[:,3] + b))
    print(f"   hinge-axis roll vs world-lateral (euler) roll: corr={corr:.4f}  slope={m:+.3f} "
          f"(hinge axis sign vs forward axis)")
    print(f"   sign/offset-corrected residual: rms={np.sqrt(np.mean(resid**2)):.2f} deg  "
          f"max={np.max(np.abs(resid)):.2f} deg   <- the true yaw/pitch coupling gap;")
    print(f"   (this residual is the extra layer the mocap analysis must model when converting")
    print(f"    IMU/marker attitude to lean; the kappa-PID sidesteps it by measuring about the hinge)")
    k = np.polyfit(a[:,6], a[:,5], 1)[0]
    print(f"   sanity: hinge torso-roll vs hip-axis roll slope = {k:.2f} (kappa cmd = {KAPPA})")


# ---------------- demo video ---------------------------------------------------------
def main_demo(out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import imageio.v2 as imageio

    gc.T_HOLD = 1e9
    model, data, ids, act, pid, tid, rid, R0t, R0r = build()
    root = ids[3]
    R = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.azimuth = 135; cam.elevation = -10; cam.distance = 1.05

    fig = plt.figure(figsize=(12.8, 4.8), dpi=100)
    gsp = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], hspace=0.16, wspace=0.02,
                           left=0.0, right=0.99, top=0.93, bottom=0.03)
    ax_im = fig.add_subplot(gsp[:, 0]); ax_im.axis("off")
    ax_b = fig.add_subplot(gsp[0, 1], projection="3d")
    ax_t = fig.add_subplot(gsp[1, 1], projection="3d")

    def draw_triad(ax, Rrel, title, extra):
        ax.cla()
        cols = ("tab:red", "tab:green", "tab:blue"); labs = ("x lat", "y fwd", "z up")
        for i in range(3):     # faint world reference axes
            e = np.eye(3)[:, i]
            ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color="0.8", lw=1.0, zorder=1)
        for i in range(3):     # live frame
            v = Rrel[:, i]
            ax.quiver(0, 0, 0, v[0], v[1], v[2], color=cols[i], lw=2.5,
                      arrow_length_ratio=0.12)
            ax.text(v[0]*1.18, v[1]*1.18, v[2]*1.18, labs[i], color=cols[i], fontsize=8)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=16, azim=-128)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, y=0.97)
        ax.text2D(-0.42, 0.52, extra, transform=ax.transAxes, fontsize=8,
                  family="monospace", va="center", ha="left")

    state = {"t0": None}
    frames = []
    RENDER_DT = 0.04                              # 25 fps
    next_render = 0.0
    print("rendering ...")
    while True:
        t = data.time
        if state["t0"] is None:
            if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                state["t0"] = t; gc.T_HOLD = t
        else:
            if t >= state["t0"] + 16.0:
                break
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            print(f"   fell at {data.time:.2f}s"); break
        if data.time + 1e-9 < next_render:
            continue
        next_render += RENDER_DT
        cam.lookat[:] = data.xpos[root]
        R.update_scene(data, camera=cam)
        img = R.render()
        Rb = data.xmat[rid].reshape(3, 3) @ R0r.T
        Rt = data.xmat[tid].reshape(3, 3) @ R0t.T
        yb, pb, rb = euler_zxy(Rb)
        yt, pt, rt = euler_zxy(Rt)
        grt = grav_roll(Rt)
        t0 = state["t0"]
        phase = ("HOLD (settling)" if t0 is None else
                 "TRANSITION (ramp)" if data.time < t0 + 4.0 else
                 "SETTLE" if data.time < t0 + 6.0 else "WALK")
        ax_im.cla(); ax_im.axis("off")
        ax_im.imshow(img)
        ax_im.text(0.02, 0.965, f"t={data.time:5.2f}s  {phase}   c6 champion  mu={MU}  "
                   f"kappa={KAPPA} COM={COM}", transform=ax_im.transAxes, fontsize=10,
                   color="w", family="monospace",
                   bbox=dict(facecolor="black", alpha=0.55, pad=3))
        d = np.degrees
        draw_triad(ax_b, Rb, "whole body (base frame)",
                   f"yaw   {d(yb):+6.1f}\npitch {d(pb):+6.1f}\nroll  {d(rb):+6.1f} deg")
        draw_triad(ax_t, Rt, "torso (IMU frame)",
                   f"yaw   {d(yt):+6.1f}\npitch {d(pt):+6.1f}\nroll  {d(rt):+6.1f} deg\n"
                   f"grav-roll\n      {d(grt):+6.1f}")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        if len(frames) % 100 == 0:
            print(f"   {len(frames)} frames (t={data.time:.1f}s)")
    gc.T_HOLD = 5.0
    plt.close(fig)
    imageio.mimsave(out_path, frames, fps=25, quality=8)
    print(f"wrote {out_path}  ({len(frames)} frames)")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        i = sys.argv.index("--demo")
        out = sys.argv[i+1] if len(sys.argv) > i+1 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "results",
            "grid5_probes", "imu_frame_demo.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        main_demo(os.path.abspath(out))
    else:
        main_validate()
