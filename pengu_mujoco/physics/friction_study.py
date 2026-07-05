"""
friction_study.py - Summer-2026 penguin plan, "For Ben to do" sim experiments.

Open-loop, crank-revolute model (penguV2/scene.xml). NOT RL. For each gait
config we OPTIMIZE the motor amplitudes + frequency (CMA-ES, phases are the
CONTROLLED variable and held fixed) for clean forward walking, then MEASURE the
ground-reaction-force / friction signature at the optimum:

  (a) min floor friction to WALK  : lowest floor mu (priority-fixed) at which the
      optimized gait still survives and makes forward progress  [operational]
  (b) friction-cone demand        : 95th-pct of |Ft|/Fn on a high-mu (no-slip)
      reference surface                                          [mechanistic]
  + forward speed, COM height, foot roll-vs-pitch amplitude, and a torso-vs-stance
    correlation check (verifies the torso actually leans the intended way).

Configs implemented here are the ones that need NO new model (penguin mass =
current model): torso upright / over-stance (penguin) / over-swing
(nonplantigrade). The human-mass and no-torso cells need a model variant (TODO).

Run from pengu_mujoco/:
  python physics/friction_study.py [maxfev]
"""
import os
import sys
import csv
import math
import time
import numpy as np
import mujoco
import cma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_config as gc
from gait_config import build_ids, set_initial_pose
from friction_utils import set_floor_friction
from optimize_gait import evaluate

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "..", "penguV2", "scene.xml")
ROOT_BODY = "leftthighmotor"
FOOT_BODIES = {"right_foot0080": "R", "right_foot0080___fillet13": "L"}
FN_MIN = 1.0
SIM_DURATION = 20.0

# fixed (controlled) phases. torso_phi is the experimental variable.
HIP_PHI = 250.0
PHI_STANCE = 150.0   # torso over stance foot (penguin)        -- from calibration
PHI_SWING = 240.0    # torso over swing foot (nonplantigrade)  -- weak on this bot
REF_MU = 1.0                                   # high-mu no-slip reference (opt + (b))
MU_LADDER = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]   # for (a)
DIST_OK = 0.15                                 # m fwd to count as "still walking"

BOUNDS = {"leg_amp": (10.0, 120.0), "hip_amp": (0.0, 30.0),
          "torso_amp": (2.0, 30.0), "freq": (1.0, 2.2), "torso_phi": (0.0, 360.0)}
LAMBDA_CORR = 0.40   # soft reward weight: J += LAMBDA_CORR * corr_sign * realized_corr
ROLL_CAP = 15.0; PITCH_CAP = 25.0
# MATCHED-SPEED comparison: every config is driven to the SAME target walking
# speed (not "as fast as possible"), then we compare friction demand + how much
# torso-over-stance/swing lean each mode can realize at that speed. This removes
# the walk-fast vs lean-hard confound so mu_req is compared fairly.
V_TARGET = 0.08      # m/s target forward speed for all configs
W_SPEED = 4.0        # penalty weight for missing the target speed

# torso amplitude + phase are the CONTROLLED treatment (held fixed); only the
# leg/hip drive amplitudes + frequency are optimized so it walks under each torso
# condition. (Letting torso_amp optimize freely collapses every mode to ~0 swing,
# since torso swinging does not help raw forward speed on this robot.)
TORSO_TREAT_DEG = 15.0   # penguin-like visible roll swing for the two swing modes

# corr_sign: +1 reward torso-over-stance, -1 reward torso-over-swing, 0 = no torso.
# torso phase is a FREE variable for the swing modes so the optimizer can co-find
# a phase that both walks AND realizes the intended lean (the soft corr reward).
CONFIGS = [
    dict(name="upright",     corr_sign=0,  free=["leg_amp", "hip_amp", "freq"],
         fixed={"torso_amp": 0.0, "torso_phi": 0.0}),
    dict(name="over_stance", corr_sign=+1, free=["leg_amp", "hip_amp", "freq", "torso_phi"],
         fixed={"torso_amp": TORSO_TREAT_DEG}),
    dict(name="over_swing",  corr_sign=-1, free=["leg_amp", "hip_amp", "freq", "torso_phi"],
         fixed={"torso_amp": TORSO_TREAT_DEG}),
]


def base_params(cfg):
    p = {"leg_amp": 70.0, "hip_amp": 8.0, "torso_amp": 12.0, "freq": 1.4,
         "hip_phi": HIP_PHI, "torso_phi": PHI_STANCE, "hip_lean": 0.0, "hip_off": 0.0}
    p.update(cfg["fixed"])
    return p


def cfg_objective(m, cfg):
    """Matched-speed objective: hit V_TARGET (not max speed) - root tilt overshoot
    + soft reward for realizing the intended torso-over-stance/swing lean. Falls
    get partial credit to guide the search."""
    if not m["survived"]:
        return -2.0 + 0.3 * max(0.0, m["dist"])
    J = -W_SPEED * abs(m["speed"] - V_TARGET)          # drive to the common speed
    if np.isfinite(m["root_roll_amp"]):
        J -= 0.02 * max(0.0, m["root_roll_amp"] - ROLL_CAP)
    if np.isfinite(m["root_pitch_off"]):
        J -= 0.02 * max(0.0, abs(m["root_pitch_off"]) - PITCH_CAP)
    if cfg["corr_sign"] != 0 and np.isfinite(m["torso_stance_corr"]):
        J += LAMBDA_CORR * cfg["corr_sign"] * m["torso_stance_corr"]
    return J


def optimize_cfg(model, data, aid, jadr, root, ids, cfg, maxfev):
    # optimize ON the reference surface (mu + floor priority) so the best gait is
    # consistent with where we later measure friction demand (b).
    set_floor_friction(model, REF_MU)
    free = cfg["free"]
    lo = np.array([BOUNDS[n][0] for n in free]); hi = np.array([BOUNDS[n][1] for n in free])
    base = base_params(cfg)
    best = {"J": -1e9, "params": dict(base), "metrics": {}}

    def obj(x):
        p = dict(base)
        for n, v in zip(free, lo + np.clip(x, 0, 1) * (hi - lo)):
            p[n] = float(v)
        m = measure(model, data, aid, jadr, root, ids, p, REF_MU)
        J = cfg_objective(m, cfg)
        if J > best["J"]:
            best["J"] = J; best["params"] = p; best["metrics"] = m
        return -J

    x0 = np.array([(base[n] - BOUNDS[n][0]) / (BOUNDS[n][1] - BOUNDS[n][0]) for n in free])
    es = cma.CMAEvolutionStrategy(x0, 0.25,
                                  {"bounds": [0, 1], "maxfevals": maxfev, "verb_disp": 0, "seed": 1})
    es.optimize(obj)
    return best


def _set_gait(p):
    gc.set_crank_amp(p["leg_amp"]); gc.set_hip_amp(p["hip_amp"]); gc.set_torso_amp(p["torso_amp"])
    gc.WALK_HIP_OFFSET_DEG = p.get("hip_off", 0.0); gc.WALK_HIP_LEAN_DEG = p.get("hip_lean", 0.0)
    gc.set_walk_freq(p["freq"])
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = p["hip_phi"]; gc.PHASE_OFFSET_D_DEG = p["hip_phi"]
    gc.PHASE_OFFSET_E_DEG = p["torso_phi"]


def measure(model, data, aid, jadr, root, ids, p, mu):
    """Run optimized gait at floor friction mu; return GRF/friction/COM/foot metrics."""
    floor_id, foot_geom, tid, rfid, lfid = ids
    set_floor_friction(model, mu)
    _set_gait(p)
    set_initial_pose(model, data, aid, jadr)
    bm = model.body_mass.copy(); M = float(bm[1:].sum())
    R0 = data.xmat[root].reshape(3, 3).copy()
    up_local = R0.T @ np.array([0.0, 0.0, 1.0])         # root tilt reference
    foot_uploc = {g: data.xmat[model.geom_bodyid[g]].reshape(3, 3).copy().T @ np.array([0., 0., 1.])
                  for g in foot_geom}
    ws = gc.T_HOLD + gc.T_TRANSITION + 2.0
    f6 = np.zeros(6)
    mu_req = []; com_z = []; corr = []
    froll = {"L": [], "R": []}; fpitch = {"L": [], "R": []}
    roll_min = roll_max = pit_min = pit_max = None
    pos_ws = None; last = data.xpos[root][:2].copy(); fell = False
    while data.time < SIM_DURATION:
        gc.apply_ctrl(data, aid, data.time); mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            fell = True; break
        last = data.xpos[root][:2].copy()
        if data.time < ws:
            continue
        if pos_ws is None:
            pos_ws = data.xpos[root][:2].copy()
        up = data.xmat[root].reshape(3, 3) @ up_local   # root roll (lateral) / pitch (fwd)
        rr = math.degrees(math.atan2(up[0], up[2])); pp_ = math.degrees(math.atan2(up[1], up[2]))
        roll_min = rr if roll_min is None else min(roll_min, rr)
        roll_max = rr if roll_max is None else max(roll_max, rr)
        pit_min = pp_ if pit_min is None else min(pit_min, pp_)
        pit_max = pp_ if pit_max is None else max(pit_max, pp_)
        Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}
        for c in range(data.ncon):
            con = data.contact[c]
            ft = foot_geom.get(con.geom2) if con.geom1 == floor_id else (
                 foot_geom.get(con.geom1) if con.geom2 == floor_id else None)
            if ft:
                mujoco.mj_contactForce(model, data, c, f6)
                Fn[ft] += abs(f6[0]); Ft[ft] += math.hypot(f6[1], f6[2])
        for s in ("L", "R"):
            if Fn[s] > FN_MIN:
                mu_req.append(Ft[s] / Fn[s])
        com = (bm[1:, None] * data.xipos[1:]).sum(0) / M
        com_z.append(com[2])
        # torso-vs-stance: +1 if torso COM leans toward the loaded foot
        if max(Fn["L"], Fn["R"]) > 2.0:
            sx = data.xpos[rfid][0] if Fn["R"] > Fn["L"] else data.xpos[lfid][0]
            corr.append(np.sign(data.xipos[tid][0] - data.xpos[root][0]) * np.sign(sx - data.xpos[root][0]))
        for g, s in foot_geom.items():
            up = data.xmat[model.geom_bodyid[g]].reshape(3, 3) @ foot_uploc[g]
            froll[s].append(math.degrees(math.atan2(up[0], up[2])))
            fpitch[s].append(math.degrees(math.atan2(up[1], up[2])))
    survived = not fell
    dist = float(last[1] - pos_ws[1]) if pos_ws is not None else 0.0
    wt = max(1e-6, (SIM_DURATION if survived else data.time) - ws)
    roll_a = np.mean([(max(froll[s]) - min(froll[s])) / 2 for s in ("L", "R") if froll[s]]) if froll["L"] else float("nan")
    pitch_a = np.mean([(max(fpitch[s]) - min(fpitch[s])) / 2 for s in ("L", "R") if fpitch[s]]) if fpitch["L"] else float("nan")
    root_roll_amp = (roll_max - roll_min) / 2.0 if roll_min is not None else float("nan")
    root_pitch_off = (pit_max if (pit_max is not None and abs(pit_max) >= abs(pit_min)) else pit_min) \
        if pit_min is not None else float("nan")
    return dict(survived=survived, dist=dist, speed=dist / wt,
                mu_req_p95=float(np.percentile(mu_req, 95)) if mu_req else float("nan"),
                com_z=float(np.mean(com_z)) if com_z else float("nan"),
                foot_roll=float(roll_a), foot_pitch=float(pitch_a),
                root_roll_amp=float(root_roll_amp), root_pitch_off=float(root_pitch_off),
                torso_stance_corr=float(np.mean(corr)) if corr else float("nan"))


def lock_torso_phase(model, data, aid, jadr, root, ids, p, target):
    """After amplitudes/freq are set, pick the torso phase that actually realizes
    the intended lean: max torso-vs-stance corr for 'stance', min for 'swing'.
    Fixed torso_phi numbers drift once the gait changes, so we DEFINE the mode by
    the realized lean. Returns (best_phi, best_corr)."""
    best_phi, best_c = None, None
    for phi in range(0, 360, 20):
        pp = dict(p); pp["torso_phi"] = float(phi)
        r = measure(model, data, aid, jadr, root, ids, pp, REF_MU)
        if not r["survived"] or math.isnan(r["torso_stance_corr"]):
            continue
        c = r["torso_stance_corr"]
        if best_c is None or (target == "stance" and c > best_c) or (target == "swing" and c < best_c):
            best_phi, best_c = float(phi), c
    return best_phi, best_c


def min_mu_to_walk(model, data, aid, jadr, root, ids, p):
    """(a) lowest floor mu where the optimized gait still walks (survives + dist>DIST_OK)."""
    last_ok = None
    for mu in MU_LADDER:
        r = measure(model, data, aid, jadr, root, ids, p, mu)
        if r["survived"] and r["dist"] > DIST_OK:
            last_ok = mu
        else:
            break
    return last_ok


def main():
    maxfev = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_bid = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): s for n, s in FOOT_BODIES.items()}
    foot_geom = {g: foot_bid[model.geom_bodyid[g]] for g in range(model.ngeom)
                 if model.geom_bodyid[g] in foot_bid and model.geom_contype[g]}
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    rfid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080")
    lfid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13")
    ids = (floor_id, foot_geom, tid, rfid, lfid)

    outdir = os.path.join(_HERE, "..", "results", "friction_study")
    os.makedirs(outdir, exist_ok=True)
    rows = []
    print(f"# friction_study  maxfev={maxfev}/config  configs={[c['name'] for c in CONFIGS]}")
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        best = optimize_cfg(model, data, aid, jadr, root, ids, cfg, maxfev)
        bp = best["params"]
        ref = measure(model, data, aid, jadr, root, ids, bp, REF_MU)   # (b) on no-slip ref
        mmu = min_mu_to_walk(model, data, aid, jadr, root, ids, bp)     # (a) threshold
        dt = time.perf_counter() - t0
        row = dict(config=cfg["name"], torso_phi=round(bp["torso_phi"], 0),
                   leg_amp=round(bp["leg_amp"], 1), hip_amp=round(bp["hip_amp"], 1),
                   torso_amp=round(bp["torso_amp"], 1), freq=round(bp["freq"], 3),
                   opt_speed=round(best["metrics"].get("speed", 0.0), 3),
                   min_mu_to_walk=mmu, mu_req_p95=round(ref["mu_req_p95"], 3),
                   speed=round(ref["speed"], 3), com_z=round(ref["com_z"], 3),
                   foot_roll=round(ref["foot_roll"], 1), foot_pitch=round(ref["foot_pitch"], 1),
                   torso_stance_corr=round(ref["torso_stance_corr"], 2))
        rows.append(row)
        print(f"\n## {cfg['name']}  ({dt:.0f}s)")
        print(f"   best gait: leg={row['leg_amp']} hip={row['hip_amp']} torso={row['torso_amp']} "
              f"freq={row['freq']}  opt_speed={row['opt_speed']}")
        print(f"   (a) min_mu_to_walk = {mmu}")
        print(f"   (b) mu_req_p95 @ref = {row['mu_req_p95']}   speed={row['speed']} com_z={row['com_z']}")
        print(f"   foot roll={row['foot_roll']} pitch={row['foot_pitch']}  torso-stance corr={row['torso_stance_corr']}")

    csv_path = os.path.join(outdir, "penguin_configs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n# wrote {csv_path}")
    print("\n=== SUMMARY (penguin mass; headline = min floor friction to walk) ===")
    print(f"{'config':<13}{'min_mu':>8}{'mu_req_p95':>12}{'speed':>8}{'com_z':>8}{'roll/pitch':>12}{'corr':>7}")
    for r in rows:
        rp = f"{r['foot_roll']}/{r['foot_pitch']}"
        print(f"{r['config']:<13}{str(r['min_mu_to_walk']):>8}{r['mu_req_p95']:>12}{r['speed']:>8}"
              f"{r['com_z']:>8}{rp:>12}{r['torso_stance_corr']:>7}")


if __name__ == "__main__":
    main()
