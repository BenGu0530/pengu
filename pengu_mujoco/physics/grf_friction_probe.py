"""
grf_friction_probe.py - DEBUG probe for the slippery-surface study.

Runs ONE open-loop walk on the crank-revolute model (penguV2/scene.xml, via
gait_config) and measures the ground-reaction-force / friction-cone signature
that the Summer-2026 penguin plan asks for:
  - per-foot normal (Fn) and tangential (Ft) ground reaction force,
  - mu_req = |Ft| / Fn  (the minimum floor friction needed not to slip; peak
    and 95th-pct over the walk window are the headline numbers),
  - forward speed, COM height, foot roll-vs-pitch amplitude.

This is the measurement core that the 7-config sweep will reuse. NOT RL.

Run from pengu_mujoco/:
  python physics/grf_friction_probe.py [mu] [torso_amp] [phase_e] [freq] [hip] [crank]
defaults reproduce a known walking gait (5dof: hip12 crank73 torso9 @1.25Hz).
"""
import os
import sys
import math
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_config as gc
from friction_utils import set_floor_friction

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "..", "penguV2", "scene.xml")
ROOT_BODY = "leftthighmotor"
FOOT_BODIES = {"right_foot0080": "R", "right_foot0080___fillet13": "L"}
FN_MIN = 1.0   # N; ignore micro-contacts below this normal force for mu_req

# ---- args ----
mu       = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
torso_a  = float(sys.argv[2]) if len(sys.argv) > 2 else 9.0
phase_e  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
freq     = float(sys.argv[4]) if len(sys.argv) > 4 else 1.25
hip_a    = float(sys.argv[5]) if len(sys.argv) > 5 else 12.0
crank_a  = float(sys.argv[6]) if len(sys.argv) > 6 else 73.0
SIM_T = 20.0


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    floor_id = set_floor_friction(model, mu)
    data = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    foot_bid = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): s
                for n, s in FOOT_BODIES.items()}
    # collision geoms that belong to a foot body
    foot_geom = {}
    for g in range(model.ngeom):
        b = model.geom_bodyid[g]
        if b in foot_bid and model.geom_contype[g]:
            foot_geom[g] = foot_bid[b]

    bm = model.body_mass.copy()
    M = float(bm[1:].sum())

    # gait config
    gc.set_hip_amp(hip_a); gc.set_crank_amp(crank_a); gc.set_torso_amp(torso_a)
    gc.set_walk_freq(freq)
    global_E = gc.PHASE_OFFSET_E_DEG
    gc.PHASE_OFFSET_E_DEG = phase_e
    gc.set_initial_pose(model, data, act, jadr)

    R0 = data.xmat[root_id].reshape(3, 3).copy()
    up_local = R0.T @ np.array([0.0, 0.0, 1.0])
    foot_R0 = {g: data.xmat[model.geom_bodyid[g]].reshape(3, 3).copy() for g in foot_geom}
    foot_uploc = {g: foot_R0[g].T @ np.array([0.0, 0.0, 1.0]) for g in foot_geom}

    walk_start = gc.T_HOLD + gc.T_TRANSITION
    settle = walk_start + 2.0

    f6 = np.zeros(6)
    mu_req = {"L": [], "R": []}
    fn_peak = {"L": 0.0, "R": 0.0}
    com_z = []
    froll = {"L": [], "R": []}; fpitch = {"L": [], "R": []}
    pos_ws = None; last = data.xpos[root_id][:2].copy()
    fell = False

    while data.time < SIM_T:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        p = data.xpos[root_id]
        last = p[:2].copy()
        if p[2] < 0.05:
            fell = True; break
        if data.time < settle:
            continue
        if pos_ws is None:
            pos_ws = p[:2].copy()

        # per-foot GRF this step
        Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}
        for c in range(data.ncon):
            con = data.contact[c]
            g1, g2 = con.geom1, con.geom2
            foot = None
            if g1 == floor_id and g2 in foot_geom: foot = foot_geom[g2]
            elif g2 == floor_id and g1 in foot_geom: foot = foot_geom[g1]
            if foot is None:
                continue
            mujoco.mj_contactForce(model, data, c, f6)
            Fn[foot] += abs(f6[0])
            Ft[foot] += math.hypot(f6[1], f6[2])
        for s in ("L", "R"):
            if Fn[s] > FN_MIN:
                mu_req[s].append(Ft[s] / Fn[s])
                fn_peak[s] = max(fn_peak[s], Fn[s])

        com = (bm[1:, None] * data.xipos[1:]).sum(0) / M
        com_z.append(com[2])
        for g, s in foot_geom.items():
            up = data.xmat[model.geom_bodyid[g]].reshape(3, 3) @ foot_uploc[g]
            froll[s].append(math.degrees(math.atan2(up[0], up[2])))
            fpitch[s].append(math.degrees(math.atan2(up[1], up[2])))

    gc.PHASE_OFFSET_E_DEG = global_E  # restore

    # ---- report ----
    survived = not fell
    dfwd = float(last[1] - pos_ws[1]) if pos_ws is not None else 0.0
    dlat = float(last[0] - pos_ws[0]) if pos_ws is not None else 0.0
    wt = max(1e-6, (SIM_T if survived else data.time) - settle)
    print(f"\n=== GRF / friction-cone probe  (crank model) ===")
    print(f"mu_floor={mu}  freq={freq}  hip={hip_a}  crank={crank_a}  torso={torso_a}  phase_E={phase_e}")
    print(f"survived={survived}  fwd={dfwd:+.3f} m  lat={dlat:+.3f} m  speed={dfwd/wt:.3f} m/s  M={M:.2f} kg")
    if com_z:
        print(f"COM height: mean={np.mean(com_z):.3f}  range=[{min(com_z):.3f},{max(com_z):.3f}] m")
    print(f"{'foot':<5}{'n_contact_steps':>16}{'Fn_peak[N]':>12}{'mu_req_peak':>13}{'mu_req_p95':>12}{'mu_req_mean':>12}")
    for s in ("L", "R"):
        a = np.array(mu_req[s])
        if a.size:
            print(f"{s:<5}{a.size:>16}{fn_peak[s]:>12.1f}{a.max():>13.3f}{np.percentile(a,95):>12.3f}{a.mean():>12.3f}")
        else:
            print(f"{s:<5}{0:>16}{'-':>12}{'-':>13}{'-':>12}{'-':>12}")
    allmu = np.array(mu_req["L"] + mu_req["R"])
    if allmu.size:
        print(f"\n>>> min floor friction required (both feet): peak={allmu.max():.3f}  p95={np.percentile(allmu,95):.3f}")
    for s in ("L", "R"):
        if froll[s]:
            ra = (max(froll[s]) - min(froll[s])) / 2
            pa = (max(fpitch[s]) - min(fpitch[s])) / 2
            print(f"foot {s}: roll_amp={ra:.1f} deg  pitch_amp={pa:.1f} deg  (roll/pitch={ra/max(pa,1e-3):.2f})")


if __name__ == "__main__":
    main()
