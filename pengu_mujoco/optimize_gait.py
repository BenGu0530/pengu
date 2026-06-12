# CMA-ES gait optimization for Pengu. Maximizes CLEAN forward speed.
#
#   objective J (maximize):
#     survived & clean:  mean_speed_fwd [m/s]
#                        - 0.02*max(0, roll_amp-15) - 0.02*max(0,|pitch_off|-25)
#     fell:              -1.0 + 0.3*dist_before_fall   (partial credit, guides search)
#   energy (mean |sum tau*omega|, W) is recorded but NOT optimized.
#
# Variables (normalized [0,1] internally, mapped to these bounds):
#   leg_amp [10,120] hip_amp [0,30] torso_amp [0,30]
#   hip_phi [0,360]  torso_phi [0,360]  freq [1.0,2.2]
# Leg phase offsets are held at 0; hip_phi/torso_phi shift hip/torso vs leg.
#
# Usage (from pengu_mujoco/):
#   python optimize_gait.py legtorso   # 4D: leg+torso, hip locked 0
#   python optimize_gait.py full       # 6D: leg+hip+torso
"""optimize_gait.py - CMA-ES clean-speed gait optimization."""
import os
import sys
import csv
import math
import time
import numpy as np
import mujoco
import cma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gait_config as gc
from gait_config import build_ids, set_initial_pose
from friction_utils import set_floor_friction, SURFACES

SURFACE = "mocap_floor"
ROOT_BODY = "leftthighmotor"
FALL_Z = 0.05
ROLL_SETTLE = 2.0
SIM_DURATION = 20.0
ROLL_CAP = 15.0
PITCH_CAP = 25.0

# (name, lo, hi)
VARS_FULL = [
    ("leg_amp", 10.0, 120.0), ("hip_amp", 0.0, 30.0), ("torso_amp", 0.0, 30.0),
    ("hip_phi", 0.0, 360.0), ("torso_phi", 0.0, 360.0), ("freq", 1.0, 2.2),
]
VARS_LEGTORSO = [
    ("leg_amp", 10.0, 120.0), ("torso_amp", 0.0, 30.0),
    ("torso_phi", 0.0, 360.0), ("freq", 1.0, 2.2),
]
# add antisymmetric hip lean as a variable (direction A)
VARS_LEGTORSO_LEAN = VARS_LEGTORSO + [("hip_lean", -15.0, 15.0)]
VARS_FULL_LEAN = VARS_FULL + [("hip_lean", -15.0, 15.0)]

# ROBUST objective: average CLEAN speed over a small frequency band so the
# optimizer cannot sit on a chaotic razor-thin spike (real robot can't hold
# freq to 0.001 Hz). Rewards broad plateaus, punishes spikes and fragility.
ROBUST_OFFSETS = [-0.03, -0.015, 0.0, 0.015, 0.03]


def robust_metrics(model, data, aid, jadr, root_id, p):
    cs = []
    center = None
    for off in ROBUST_OFFSETS:
        pp = dict(p); pp["freq"] = p["freq"] + off
        m = evaluate(model, data, aid, jadr, root_id, pp)
        clean = (m["survived"] and np.isfinite(m["roll_amp"]) and m["roll_amp"] < ROLL_CAP
                 and np.isfinite(m["pitch_off"]) and abs(m["pitch_off"]) < PITCH_CAP)
        cs.append(m["speed"] if clean else 0.0)
        if abs(off) < 1e-9:
            center = m
    out = dict(center)
    out["J"] = float(np.mean(cs))      # robust objective = mean clean speed over band
    out["Jworst"] = float(np.min(cs))  # worst-case in band
    return out


def make_model():
    m = mujoco.MjModel.from_xml_path("penguV2/scene.xml")
    set_floor_friction(m, SURFACES[SURFACE])
    return m


def evaluate(model, data, aid, jadr, root_id, p):
    """p: dict of real-valued gait params. Returns metrics dict."""
    gc.set_crank_amp(p.get("leg_amp", 0.0))
    gc.set_hip_amp(p.get("hip_amp", 0.0))
    gc.set_torso_amp(p.get("torso_amp", 0.0))
    gc.WALK_HIP_OFFSET_DEG = p.get("hip_off", 0.0)   # symmetric hip offset
    gc.WALK_HIP_LEAN_DEG = p.get("hip_lean", 0.0)    # antisymmetric postural lean
    gc.set_walk_freq(p["freq"])
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = p.get("hip_phi", 0.0); gc.PHASE_OFFSET_D_DEG = p.get("hip_phi", 0.0)
    gc.PHASE_OFFSET_E_DEG = p.get("torso_phi", 0.0)

    set_initial_pose(model, data, aid, jadr)
    R0 = data.xmat[root_id].reshape(3, 3).copy()
    up_local = R0.T @ np.array([0.0, 0.0, 1.0])
    walk_start = gc.T_HOLD + gc.T_TRANSITION
    roll_min = roll_max = pit_min = pit_max = None
    pos_ws = None
    last = data.xpos[root_id][:2].copy()
    e_sum = 0.0; e_n = 0
    fell = False; fall_t = float("nan")

    while data.time < SIM_DURATION:
        gc.apply_ctrl(data, aid, data.time)
        mujoco.mj_step(model, data)
        p3 = data.xpos[root_id]
        last = p3[:2].copy()
        if p3[2] < FALL_Z:
            fell = True; fall_t = data.time; break
        if pos_ws is None and data.time >= walk_start:
            pos_ws = p3[:2].copy()
        if data.time >= walk_start + ROLL_SETTLE:
            R = data.xmat[root_id].reshape(3, 3)
            up = R @ up_local
            roll = math.degrees(math.atan2(up[0], up[2]))
            pit = math.degrees(math.atan2(up[1], up[2]))
            roll_min = roll if roll_min is None else min(roll_min, roll)
            roll_max = roll if roll_max is None else max(roll_max, roll)
            pit_min = pit if pit_min is None else min(pit_min, pit)
            pit_max = pit if pit_max is None else max(pit_max, pit)
            pw = data.actuator_force * data.actuator_velocity
            e_sum += float(np.abs(pw).sum()); e_n += 1

    survived = not fell
    end_t = (data.time if survived else fall_t)
    walk_time = max(0.0, end_t - walk_start)
    dist = float(last[1] - pos_ws[1]) if pos_ws is not None else 0.0
    speed = dist / walk_time if walk_time > 1e-6 else 0.0
    roll_amp = (roll_max - roll_min) / 2.0 if roll_min is not None else float("nan")
    pit_off = (pit_max if (pit_max is not None and abs(pit_max) >= abs(pit_min)) else pit_min) \
        if pit_min is not None else float("nan")
    energy = e_sum / e_n if e_n else float("nan")

    if survived:
        J = speed
        if np.isfinite(roll_amp):
            J -= 0.02 * max(0.0, roll_amp - ROLL_CAP)
        if np.isfinite(pit_off):
            J -= 0.02 * max(0.0, abs(pit_off) - PITCH_CAP)
    else:
        J = -1.0 + 0.3 * max(0.0, dist)
    return dict(J=J, survived=survived, dist=dist, speed=speed, roll_amp=roll_amp,
                pitch_off=pit_off, energy=energy, walk_time=walk_time)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "legtorso"
    robust = (len(sys.argv) > 2 and sys.argv[2] == "robust")
    VARS = {"legtorso": VARS_LEGTORSO, "full": VARS_FULL,
            "legtorso_lean": VARS_LEGTORSO_LEAN, "full_lean": VARS_FULL_LEAN}[mode]
    names = [v[0] for v in VARS]
    lo = np.array([v[1] for v in VARS]); hi = np.array([v[2] for v in VARS])

    model = make_model()
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)

    log_path = f"results/opt_{mode}_log.csv"
    os.makedirs("results", exist_ok=True)
    logf = open(log_path, "w", newline="")
    writer = csv.writer(logf)
    writer.writerow(names + ["J", "survived", "dist", "speed", "roll_amp", "pitch_off", "energy"])

    nfev = [0]
    best = {"J": -1e9}

    def denorm(x):
        xc = np.clip(x, 0, 1)
        real = lo + xc * (hi - lo)
        return {n: float(r) for n, r in zip(names, real)}

    def obj(x):
        p = denorm(x)
        if robust:
            m = robust_metrics(model, data, aid, jadr, root_id, p)
        else:
            m = evaluate(model, data, aid, jadr, root_id, p)
            m["Jworst"] = m["J"]
        nfev[0] += 1
        writer.writerow([round(p[n], 4) for n in names] +
                        [round(m["J"], 4), int(m["survived"]), round(m["dist"], 4),
                         round(m["speed"], 4), round(m["roll_amp"], 2) if np.isfinite(m["roll_amp"]) else "",
                         round(m["pitch_off"], 2) if np.isfinite(m["pitch_off"]) else "",
                         round(m["energy"], 3) if np.isfinite(m["energy"]) else ""])
        if m["J"] > best["J"]:
            best.update(m); best["params"] = p
            print(f"  [{nfev[0]:4d}] NEW BEST J={m['J']:.4f} (worst={m.get('Jworst',m['J']):.3f}) "
                  f"speed={m['speed']:.3f} roll={m['roll_amp']:.1f} pitch={m['pitch_off']:.1f} "
                  f"E={m['energy']:.1f}W  " + " ".join(f"{n}={p[n]:.1f}" for n in names))
        return -m["J"]

    # x0: reasonable guess in normalized space
    guess = {"leg_amp": 70, "hip_amp": 8, "torso_amp": 12, "hip_phi": 250,
             "torso_phi": 180, "freq": 1.4, "hip_lean": 0.0}
    x0 = np.array([(guess[n] - l) / (h - l) for n, l, h in zip(names, lo, hi)])
    base_fev = 600 if mode.startswith("legtorso") else 900
    maxfev = (base_fev // 2) if robust else base_fev
    print(f"# CMA-ES mode={mode} robust={robust} vars={names} maxfev={maxfev}"
          f"{' (x5 sims/eval)' if robust else ''}")
    t0 = time.perf_counter()
    es = cma.CMAEvolutionStrategy(x0, 0.25, {
        "bounds": [0, 1], "maxfevals": maxfev, "verb_disp": 0, "seed": 1})
    es.optimize(obj)
    logf.close()
    dt = time.perf_counter() - t0
    bp = best["params"]
    print(f"\n# DONE mode={mode}  evals={nfev[0]}  wall={dt:.1f}s  log={log_path}")
    print(f"# BEST J={best['J']:.4f}  speed={best['speed']:.3f} m/s  dist={best['dist']:.3f} m  "
          f"roll={best['roll_amp']:.1f}  pitch_off={best['pitch_off']:.1f}  energy={best['energy']:.1f} W")
    print("# BEST PARAMS: " + "  ".join(f"{n}={bp[n]:.2f}" for n in names))


if __name__ == "__main__":
    main()
