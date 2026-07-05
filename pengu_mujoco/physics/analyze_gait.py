"""
analyze_gait.py - deep look at ONE gait (default = the CMA-ES joint optimum):
  - back-view render (gif/mp4),
  - top-down COM trajectory + L/R footstep diagram,
  - friction cone: per-foot normal/tangential GRF, mu_req = |Ft|/Fn (peak/p95/mean)
    + Ft-vs-Fn scatter against the floor mu.

Run from pengu_mujoco/:
  PENGU_MODEL=v3 MUJOCO_GL=egl python physics/analyze_gait.py
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

# CMA-ES joint optimum (25deg pitch fixed via hip_off)
G = dict(freq=2.0, leg_amp=106.8, hip_amp=19.9, torso_amp=20.0,
         hip_phi=190.5, torso_phi=18.2, hip_off=30.0)
SIM_T = 18.0


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    set_floor_friction(m, FLOOR_MU)
    d = mujoco.MjData(m)
    act, jadr = build_ids(m)
    gc.T_HOLD = 5.0; gc.T_TRANSITION = 4.0
    gc.set_crank_amp(G["leg_amp"]); gc.set_hip_amp(G["hip_amp"]); gc.set_torso_amp(G["torso_amp"])
    gc.set_walk_freq(G["freq"])
    gc.WALK_HIP_OFFSET_DEG = G["hip_off"]; gc.WALK_HIP_LEAN_DEG = 0.0
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = G["hip_phi"]; gc.PHASE_OFFSET_D_DEG = G["hip_phi"]
    gc.PHASE_OFFSET_E_DEG = G["torso_phi"]
    set_initial_pose(m, d, act, jadr)

    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_bid = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n): s for n, s in FOOT_BODIES.items()}
    foot_geom = {g: foot_bid[m.geom_bodyid[g]] for g in range(m.ngeom)
                 if m.geom_bodyid[g] in foot_bid and m.geom_contype[g]}
    bm = m.body_mass.copy(); M = float(bm[1:].sum())

    R = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.azimuth = 270; cam.elevation = -10; cam.distance = 1.3  # BACK view
    re = max(1, int(round(0.03 / m.opt.timestep)))
    SETTLE = gc.T_HOLD + gc.T_TRANSITION + 1.0
    f6 = np.zeros(6)
    frames = []
    COM = []; FN = {"L": [], "R": []}; FT = {"L": [], "R": []}; MU = []
    falls = []                       # (foot, x, y) touchdowns
    prevc = {"L": False, "R": False}
    k = 0; fell = False
    while d.time < SIM_T:
        gc.apply_ctrl(d, act, d.time); mujoco.mj_step(m, d)
        if d.xpos[root][2] < 0.05:
            fell = True; break
        if d.time >= gc.T_HOLD and k % re == 0:
            cam.lookat[:] = d.xpos[root]; R.update_scene(d, camera=cam); frames.append(R.render().copy())
        if d.time >= SETTLE:
            com = (bm[1:, None] * d.xipos[1:]).sum(0) / M
            COM.append(com.copy())
            Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}; con = {"L": False, "R": False}
            for c in range(d.ncon):
                ct = d.contact[c]
                ft = foot_geom.get(ct.geom2) if ct.geom1 == floor_id else (
                     foot_geom.get(ct.geom1) if ct.geom2 == floor_id else None)
                if ft:
                    con[ft] = True
                    mujoco.mj_contactForce(m, d, c, f6)
                    Fn[ft] += abs(f6[0]); Ft[ft] += math.hypot(f6[1], f6[2])
            for s in ("L", "R"):
                if Fn[s] > 1.0:
                    FN[s].append(Fn[s]); FT[s].append(Ft[s]); MU.append(Ft[s] / Fn[s])
                if con[s] and not prevc[s]:
                    falls.append((s, float(d.xpos[[b for b, ss in foot_bid.items() if ss == s][0]][0]),
                                  float(d.xpos[[b for b, ss in foot_bid.items() if ss == s][0]][1])))
                prevc[s] = con[s]
        k += 1
    R.close()
    COM = np.array(COM); MU = np.array(MU)
    outdir = os.path.join(_HERE, "..", "results", "gait_sweep")
    imageio.mimsave(os.path.join(outdir, "cma_back.gif"), frames[::2], fps=20)
    imageio.mimsave(os.path.join(outdir, "cma_back.mp4"), frames, fps=40, quality=8)

    # ---- friction cone numbers ----
    allFn = np.array(FN["L"] + FN["R"]); allFt = np.array(FT["L"] + FT["R"])
    print(f"survived={not fell}  mass={M:.2f}kg  floor_mu={FLOOR_MU}")
    print(f"=== FRICTION CONE (mu_req = |Ft|/Fn) ===")
    print(f"  peak={MU.max():.3f}  p99={np.percentile(MU,99):.3f}  p95={np.percentile(MU,95):.3f}  "
          f"mean={MU.mean():.3f}  median={np.median(MU):.3f}")
    print(f"  -> needs floor mu >= ~{np.percentile(MU,95):.2f} (p95) to not slip; "
          f"peak demand {MU.max():.2f}")

    # ---- plots: COM top-down + footsteps  |  friction cone ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    ax[0].plot(COM[:, 0], COM[:, 1], "-", color="0.5", lw=1.2, label="COM path")
    for s, c, mk in (("L", "tab:red", "o"), ("R", "tab:blue", "s")):
        xs = [f[1] for f in falls if f[0] == s]; ys = [f[2] for f in falls if f[0] == s]
        ax[0].plot(xs, ys, mk, color=c, ms=5, label=f"{s} step")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x lateral [m]"); ax[0].set_ylabel("y forward [m]")
    ax[0].set_title("COM trajectory + footsteps (top-down)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].scatter(allFn, allFt, s=4, alpha=.3, color="tab:purple", label="contacts")
    fnmax = allFn.max() if allFn.size else 1
    ax[1].plot([0, fnmax], [0, FLOOR_MU * fnmax], "g--", label=f"floor mu={FLOOR_MU} (slip line)")
    ax[1].plot([0, fnmax], [0, MU.max() * fnmax], "r--", label=f"mu_req peak={MU.max():.2f}")
    ax[1].set_xlabel("normal force Fn [N]"); ax[1].set_ylabel("tangential force |Ft| [N]")
    ax[1].set_title("friction cone (each foot-floor contact)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"CMA optimum: f={G['freq']} leg={G['leg_amp']:.0f} hip={G['hip_amp']:.0f} "
                 f"torso={G['torso_amp']:.0f} pitch~25deg | mu_req p95={np.percentile(MU,95):.2f}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cma_traj_friction.png"), dpi=130, bbox_inches="tight")
    print(f"wrote cma_back.gif/.mp4 and cma_traj_friction.png in {outdir}")


if __name__ == "__main__":
    main()
