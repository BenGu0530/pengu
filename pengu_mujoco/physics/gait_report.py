"""
gait_report.py - STANDARD per-gait report video. Three panels in one captioned
video: elevated-side view | back view | live COM trajectory + footsteps.
Caption states the combination (COM / pitch / torso mode / params) and which
mentor-plan experiment requirement it is.

Reusable template: edit G (gait params) + COMBO / EXPERIMENT labels.

Run from pengu_mujoco/:
  PENGU_MODEL=v3 MUJOCO_GL=egl python physics/gait_report.py
"""
import os
import sys
import math
import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_config as gc
from gait_config import build_ids, set_initial_pose
from friction_utils import set_floor_friction

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "..", "penguV3", "scene.xml")
FLOOR_MU = 0.7
FOOT_BODIES = {"right_foot0080": "R", "right_foot0080___fillet13": "L"}

# ---- gait under test (GRID global best-by-distance, freq=1.3 penguin band) ----
G = dict(freq=1.27, leg_amp=115.0, hip_amp=22.0, torso_amp=20.0,
         hip_phi=210.0, torso_phi=0.0, hip_off=30.0)
COMBO = "penguV3  |  COM/leg=1.05  |  25 deg fwd pitch  |  fine3c PENGUIN-band best (freq 1.27, single=1.0)"
EXPERIMENT = "Mentor plan cell:  torso active-swing  x  penguin mass (COM/leg 1.05)  x  floor mu=0.7"

T_HOLD = 9.0          # longer settle (reduce start-up fore/aft rocking)
T_TRANS = 4.0
SIM_T = 22.0
TAG = "fine3c_penguin_f1.27"


def main():
    m = mujoco.MjModel.from_xml_path(XML); set_floor_friction(m, FLOOR_MU)
    d = mujoco.MjData(m); act, jadr = build_ids(m)
    gc.T_HOLD = T_HOLD; gc.T_TRANSITION = T_TRANS
    gc.set_crank_amp(G["leg_amp"]); gc.set_hip_amp(G["hip_amp"]); gc.set_torso_amp(G["torso_amp"])
    gc.set_walk_freq(G["freq"]); gc.WALK_HIP_OFFSET_DEG = G["hip_off"]; gc.WALK_HIP_LEAN_DEG = 0.0
    gc.PHASE_OFFSET_A_DEG = 0; gc.PHASE_OFFSET_B_DEG = 0
    gc.PHASE_OFFSET_C_DEG = G["hip_phi"]; gc.PHASE_OFFSET_D_DEG = G["hip_phi"]; gc.PHASE_OFFSET_E_DEG = G["torso_phi"]
    set_initial_pose(m, d, act, jadr)

    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_bid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n): s for n, s in FOOT_BODIES.items()}
    foot_geom = {g: foot_bid[m.geom_bodyid[g]] for g in range(m.ngeom)
                 if m.geom_bodyid[g] in foot_bid and m.geom_contype[g]}
    bm = m.body_mass.copy(); M = float(bm[1:].sum())

    R = mujoco.Renderer(m, height=480, width=640)
    cam_s = mujoco.MjvCamera(); cam_s.azimuth = 135; cam_s.elevation = -20; cam_s.distance = 1.5  # elevated side
    cam_b = mujoco.MjvCamera(); cam_b.azimuth = 270; cam_b.elevation = -12; cam_b.distance = 1.4  # back
    re = max(1, int(round(0.04 / m.opt.timestep)))   # ~25 fps capture
    SETTLE = T_HOLD + T_TRANS + 1.0
    f6 = np.zeros(6)
    side, back, com_hist = [], [], []
    falls = []               # (frame_idx, foot, x, y)
    ws_cx, ws_sx = [], []    # single-support COM-x & stance-foot-x (for weight transfer; EVAL ONLY)
    prevc = {"L": True, "R": True}; mu_all = []
    k = 0
    while d.time < SIM_T:
        gc.apply_ctrl(d, act, d.time); mujoco.mj_step(m, d)
        if d.xpos[root][2] < 0.05:
            break
        if d.time >= SETTLE:                                  # friction accounting
            for c in range(d.ncon):
                ct = d.contact[c]
                ft = foot_geom.get(ct.geom2) if ct.geom1 == floor_id else (
                     foot_geom.get(ct.geom1) if ct.geom2 == floor_id else None)
                if ft:
                    mujoco.mj_contactForce(m, d, c, f6)
                    if abs(f6[0]) > 1.0:
                        mu_all.append(math.hypot(f6[1], f6[2]) / abs(f6[0]))
        if d.time >= gc.T_HOLD and k % re == 0:
            cam_s.lookat[:] = d.xpos[root]; R.update_scene(d, camera=cam_s); side.append(R.render().copy())
            cam_b.lookat[:] = d.xpos[root]; R.update_scene(d, camera=cam_b); back.append(R.render().copy())
            com = (bm[1:, None] * d.xipos[1:]).sum(0) / M
            com_hist.append(com[:2].copy())
            con = {"L": False, "R": False}
            for c in range(d.ncon):
                ct = d.contact[c]
                ft = foot_geom.get(ct.geom2) if ct.geom1 == floor_id else (
                     foot_geom.get(ct.geom1) if ct.geom2 == floor_id else None)
                if ft:
                    con[ft] = True
            for bid, s in foot_bid.items():
                if con[s] and not prevc[s]:
                    falls.append((len(side) - 1, s, float(d.xpos[bid][0]), float(d.xpos[bid][1])))
                prevc[s] = con[s]
            scon = [s for s in ("L", "R") if con[s]]          # single-support weight transfer
            if len(scon) == 1:
                sb = [b for b, ss in foot_bid.items() if ss == scon[0]][0]
                ws_cx.append(float(com[0])); ws_sx.append(float(d.xpos[sb][0]))
        k += 1
    R.close()
    com_hist = np.array(com_hist)
    mu_all = np.array(mu_all) if mu_all else np.array([0.0])
    mu95 = float(np.percentile(mu_all, 95)); mupk = float(mu_all.max())

    # ---- GRACE metrics (EVAL ONLY -- reported, never used to filter/rank/optimize) ----
    Lx = [f[2] for f in falls if f[1] == "L"]; Rx = [f[2] for f in falls if f[1] == "R"]
    lat_sep = abs(np.mean(Lx) - np.mean(Rx)) if (Lx and Rx) else float("nan")   # clean L/R separation
    cxs = com_hist[:, 0] - com_hist[:, 0].mean() if len(com_hist) else np.array([0.0])
    if cxs.size > 8:
        P = np.abs(np.fft.rfft(cxs)) ** 2
        com_reg = float(P[1:].max() / (P[1:].sum() + 1e-12))   # spectral purity of COM rock (1=pure sine)
    else:
        com_reg = float("nan")
    if ws_cx:
        cxm = float(np.mean(ws_cx)); a = np.array(ws_cx) - cxm; b = np.array(ws_sx) - cxm
        wtrans = float(np.mean(np.sign(a) * np.sign(b)))       # +1 = COM leans onto the stance foot
    else:
        wtrans = float("nan")
    print(f"# GRACE (eval only): L/R lat_sep={lat_sep:.3f}m  COM_regularity={com_reg:.2f}  weight_transfer={wtrans:+.2f}")

    # fixed traj limits
    cx, cy = com_hist[:, 0], com_hist[:, 1]
    xlim = (cx.min() - 0.1, cx.max() + 0.1); ylim = (cy.min() - 0.1, cy.max() + 0.1)
    out = []
    for i in range(len(side)):
        fig = plt.figure(figsize=(15, 5.2))
        gsp = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.9])
        a0 = fig.add_subplot(gsp[0]); a0.imshow(side[i]); a0.axis("off"); a0.set_title("elevated side", fontsize=10)
        a1 = fig.add_subplot(gsp[1]); a1.imshow(back[i]); a1.axis("off"); a1.set_title("back", fontsize=10)
        a2 = fig.add_subplot(gsp[2])
        a2.plot(cx[:i + 1], cy[:i + 1], "-", color="0.5", lw=1.2, label="COM path")
        for s, c, mk in (("L", "tab:red", "o"), ("R", "tab:blue", "s")):
            xs = [f[2] for f in falls if f[1] == s and f[0] <= i]; ys = [f[3] for f in falls if f[1] == s and f[0] <= i]
            a2.plot(xs, ys, mk, color=c, ms=5, label=f"{s} step")
        if i < len(cx):
            a2.plot(cx[i], cy[i], "*", color="k", ms=12)
        a2.set_xlim(*xlim); a2.set_ylim(*ylim); a2.set_aspect("equal")
        a2.set_xlabel("x lat [m]"); a2.set_ylabel("y fwd [m]"); a2.set_title("COM trajectory + footsteps", fontsize=10)
        a2.legend(loc="upper left", fontsize=7); a2.grid(alpha=.3)
        fig.suptitle(f"{COMBO}\n{EXPERIMENT}\nf={G['freq']}Hz leg={G['leg_amp']:.0f}d hip={G['hip_amp']:.0f}d "
                     f"torso={G['torso_amp']:.0f}d | friction needed: mu_req p95={mu95:.2f} (peak {mupk:.2f})"
                     f"\nGRACE (eval only): L/R sep={lat_sep:.3f}m  COM regularity={com_reg:.2f}  "
                     f"weight-transfer={wtrans:+.2f}",
                     fontsize=10, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    outdir = os.path.join(_HERE, "..", "results", "gait_sweep")
    imageio.mimsave(os.path.join(outdir, f"{TAG}.mp4"), out, fps=25, quality=8)
    imageio.mimsave(os.path.join(outdir, f"{TAG}.gif"), out[::2], fps=12)
    print(f"# wrote {TAG}.mp4/.gif ({len(out)} frames)  mu_req p95={mu95:.2f} peak={mupk:.2f}")


if __name__ == "__main__":
    main()
