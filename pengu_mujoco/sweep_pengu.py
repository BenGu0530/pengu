"""
sweep_walk.py  –  Sweep crank amplitude (and optionally hip amp)

Crank logic (corrected):
  - Stance leg (hip going back): crank at MAX (leg extended, supporting)
  - Swing leg (hip going forward): crank at 0 (leg retracted, clearing ground)

  Left hip  = offset + hip_amp * sin(phase)
  Right hip = offset - hip_amp * sin(phase)

  When sin > 0: left hip forward (swing) → left crank = 0
                right hip backward (stance) → right crank = max
  When sin < 0: opposite

  So: left crank  = amp * 0.5 * (1 - sin)   →  sin=+1: 0,    sin=-1: amp
      right crank = amp * 0.5 * (1 + sin)   →  sin=+1: amp,  sin=-1: 0

  (Negate for right side per your actuator convention)

Usage (run from ~/Documents/ben_gu/ben_pengu/pengu):
  python sweep_walk.py
"""

import math
import csv
import numpy as np
import mujoco

# ─── Config ───────────────────────────────────────────────────────
XML_PATH = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE
# ═════════════════════════════════════════════════════════════════
INIT_Z         = 0.20
INIT_PITCH_DEG = -30.0
STAND_HIP_DEG  = -25.0

# ═════════════════════════════════════════════════════════════════
#  FIXED WALK PARAMS
# ═════════════════════════════════════════════════════════════════
WALK_FREQ           = 1.62   # [Hz] — from freq sweep
WALK_HIP_OFFSET_DEG = 0.0    # [deg]
TORSO_PASSIVE       = True

T_HOLD       = 3.0
T_TRANSITION = 2.0

# ═════════════════════════════════════════════════════════════════
#  SWEEP RANGES
# ═════════════════════════════════════════════════════════════════
# Sweep crank amplitude
CRANK_AMP_VALUES = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]  # degrees

# Sweep hip amplitude alongside
HIP_AMP_VALUES = [10]  # degrees — keep short, just 2 values

MAX_SIM_TIME  = 30.0
FALL_Z_THRESH = 0.05

# ─── Helpers ──────────────────────────────────────────────────────

def get_act_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def get_jnt_qposadr(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[j]

def world_com(model, data):
    masses = model.body_mass
    total = float(np.sum(masses))
    return (masses[:, None] * data.xipos).sum(axis=0) / total

def smoothstep(t, t0, t1):
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


# ─── Run one trial ───────────────────────────────────────────────

def run_trial(hip_amp_deg, crank_amp_deg):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    act_ids = {}
    for name in ACTUATORS:
        act_ids[name] = get_act_id(model, name)
    jnt_adr = {}
    for jname in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]:
        jnt_adr[jname] = get_jnt_qposadr(model, jname)

    # Initial pose
    mujoco.mj_resetData(model, data)
    data.qpos[2] = INIT_Z
    pitch_rad = math.radians(INIT_PITCH_DEG)
    half = pitch_rad / 2.0
    data.qpos[3] = math.cos(half)
    data.qpos[4] = math.sin(half)
    stand_hip = math.radians(STAND_HIP_DEG)
    data.qpos[jnt_adr["hip-L"]] = stand_hip
    data.qpos[jnt_adr["hip-R"]] = stand_hip
    mujoco.mj_forward(model, data)

    com_start = world_com(model, data).copy()

    # Precompute
    walk_hip_off = math.radians(WALK_HIP_OFFSET_DEG)
    walk_hip_amp = math.radians(hip_amp_deg)
    crank_amp    = math.radians(crank_amp_deg)
    t_walk_start = T_HOLD + T_TRANSITION

    while data.time < MAX_SIM_TIME:
        t = data.time

        if t < T_HOLD:
            data.ctrl[act_ids["hip-L"]]    = stand_hip
            data.ctrl[act_ids["hip-R"]]    = stand_hip
            # Cranks at max during standing (legs extended)
            data.ctrl[act_ids["crank1-L"]] =  crank_amp
            data.ctrl[act_ids["crank1-R"]] = -crank_amp

        elif t < t_walk_start:
            alpha = smoothstep(t, T_HOLD, t_walk_start)
            phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
            s = math.sin(phase)

            hip_center = stand_hip * (1.0 - alpha) + walk_hip_off * alpha
            hip_osc = alpha * walk_hip_amp * s
            data.ctrl[act_ids["hip-L"]] = hip_center + hip_osc
            data.ctrl[act_ids["hip-R"]] = hip_center - hip_osc

            # Crank: stance=max, swing=0
            # Left swing when sin>0 → left crank low when sin>0
            # Left crank  = amp * 0.5 * (1 - sin)
            # Right crank = amp * 0.5 * (1 + sin)
            cL = alpha * crank_amp * 0.5 * (1.0 - s) + (1.0 - alpha) * crank_amp
            cR = alpha * crank_amp * 0.5 * (1.0 + s) + (1.0 - alpha) * crank_amp
            data.ctrl[act_ids["crank1-L"]] =  cL
            data.ctrl[act_ids["crank1-R"]] = -cR

        else:
            phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
            s = math.sin(phase)

            # Hip: anti-phase
            hip_osc = walk_hip_amp * s
            data.ctrl[act_ids["hip-L"]] = walk_hip_off + hip_osc
            data.ctrl[act_ids["hip-R"]] = walk_hip_off - hip_osc

            # Crank: stance leg = max, swing leg = 0
            #   Left swinging (sin>0) → left crank = 0, right crank = max
            #   Right swinging (sin<0) → right crank = 0, left crank = max
            crank_L = crank_amp * 0.5 * (1.0 - s)   # sin=+1→0, sin=-1→amp
            crank_R = crank_amp * 0.5 * (1.0 + s)   # sin=+1→amp, sin=-1→0
            data.ctrl[act_ids["crank1-L"]] =  crank_L
            data.ctrl[act_ids["crank1-R"]] = -crank_R

        mujoco.mj_step(model, data)

        com_now = world_com(model, data)
        if com_now[2] < FALL_Z_THRESH:
            break

    com_end = world_com(model, data)
    survival = data.time
    dist_fwd = abs(com_end[1] - com_start[1])
    dx = com_end[0] - com_start[0]
    dy = com_end[1] - com_start[1]

    return survival, dist_fwd, dx, dy


# ─── Main ─────────────────────────────────────────────────────────

def main():
    total = len(HIP_AMP_VALUES) * len(CRANK_AMP_VALUES)

    print(f"{'='*75}")
    print(f"  CRANK + HIP SWEEP  ({total} trials)")
    print(f"  freq={WALK_FREQ:.2f}Hz  hip_offset={WALK_HIP_OFFSET_DEG:.1f}°")
    print(f"  hip_amps={HIP_AMP_VALUES}°  crank_amps={CRANK_AMP_VALUES}°")
    print(f"  Crank logic: stance=max, swing=0")
    print(f"{'='*75}\n")

    results = []
    trial = 0

    for hip_amp in HIP_AMP_VALUES:
        for crank_amp in CRANK_AMP_VALUES:
            trial += 1
            survival, dist_fwd, dx, dy = run_trial(hip_amp, crank_amp)
            walk_time = max(0, survival - T_HOLD - T_TRANSITION)

            results.append({
                'hip_amp': hip_amp,
                'crank_amp': crank_amp,
                'survival': round(survival, 2),
                'walk_time': round(walk_time, 2),
                'dist_fwd': round(dist_fwd, 4),
                'dx': round(dx, 4),
                'dy': round(dy, 4),
            })

            status = "SURVIVED" if survival >= MAX_SIM_TIME else f"fell@{survival:.1f}s"
            print(f"  [{trial:3d}/{total}]  hip={hip_amp:4.0f}°  crank={crank_amp:4.0f}°  "
                  f"surv={survival:6.1f}s  walk={walk_time:5.1f}s  "
                  f"dist={dist_fwd:6.3f}m  dy={dy:+.3f}m  ({status})")

    # Save CSV
    csv_path = "sweep_walk_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] {csv_path}")

    # Top 10 by survival
    print(f"\n{'='*75}")
    print("  TOP 10 by SURVIVAL TIME:")
    print(f"{'='*75}")
    by_survival = sorted(results, key=lambda r: (r['survival'], r['dist_fwd']), reverse=True)
    for i, r in enumerate(by_survival[:10]):
        print(f"  #{i+1}  hip={r['hip_amp']:4.0f}°  crank={r['crank_amp']:4.0f}°  "
              f"surv={r['survival']:6.1f}s  walk={r['walk_time']:5.1f}s  "
              f"dist={r['dist_fwd']:6.3f}m  dy={r['dy']:+.3f}m")

    # Top 10 by distance
    print(f"\n{'='*75}")
    print("  TOP 10 by FORWARD DISTANCE:")
    print(f"{'='*75}")
    by_dist = sorted(results, key=lambda r: r['dist_fwd'], reverse=True)
    for i, r in enumerate(by_dist[:10]):
        print(f"  #{i+1}  hip={r['hip_amp']:4.0f}°  crank={r['crank_amp']:4.0f}°  "
              f"surv={r['survival']:6.1f}s  walk={r['walk_time']:5.1f}s  "
              f"dist={r['dist_fwd']:6.3f}m  dy={r['dy']:+.3f}m")

    best = by_survival[0]
    print(f"\n{'='*75}")
    print(f"  BEST: hip={best['hip_amp']:.0f}°  crank={best['crank_amp']:.0f}°")
    print(f"  survived {best['survival']:.1f}s, walked {best['dist_fwd']:.3f}m forward")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()