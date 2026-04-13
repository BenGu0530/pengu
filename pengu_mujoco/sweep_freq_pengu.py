"""
sweep_freq.py  –  Sweep walking frequency, measure survival time & distance

Runs headless (no viewer). For each frequency, spawns the penguin,
holds standing for 3s, transitions for 2s, then walks until it falls
or hits max time. Logs results to CSV + prints a summary.

Usage (run from ~/Documents/ben_gu/ben_pengu/pengu):
  python sweep_freq.py
"""

import math
import csv
import numpy as np
import mujoco

# ─── Config ───────────────────────────────────────────────────────
XML_PATH = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE (same as your working values)
# ═════════════════════════════════════════════════════════════════
INIT_Z         = 0.20
INIT_PITCH_DEG = -30.0
STAND_HIP_DEG  = -25.0

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS (fixed for this sweep — conservative)
# ═════════════════════════════════════════════════════════════════
WALK_HIP_OFFSET_DEG = -20.0    # hip center during walk
WALK_HIP_AMP_DEG    = 5.0   # keep low for frequency sweep
WALK_CRANK_AMP_DEG  = 0.0    # crank OFF
WALK_TORSO_AMP_DEG  = 0.0    # torso passive

T_HOLD       = 5.0    # shorter hold for faster sweeps
T_TRANSITION = 2.0

# ═════════════════════════════════════════════════════════════════
#  SWEEP RANGE
# ═════════════════════════════════════════════════════════════════
FREQ_MIN  = 1.0
FREQ_MAX  = 2.0
FREQ_STEP = 0.02

MAX_SIM_TIME = 30.0   # max seconds per trial
FALL_Z_THRESH = 0.05  # COM z below this = fallen

# ─── Helpers ──────────────────────────────────────────────────────

def get_act_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        raise RuntimeError(f"Actuator '{name}' not found.")
    return i

def get_jnt_qposadr(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise RuntimeError(f"Joint '{name}' not found.")
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

def run_trial(freq):
    """Run one walking trial at the given frequency. Returns (survival_time, distance_xy)."""
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Cache IDs
    act_ids = {}
    for name in ACTUATORS:
        act_ids[name] = get_act_id(model, name)
    jnt_adr = {}
    for jname in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]:
        jnt_adr[jname] = get_jnt_qposadr(model, jname)

    # Set initial pose
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = INIT_Z

    pitch_rad = math.radians(INIT_PITCH_DEG)
    half = pitch_rad / 2.0
    data.qpos[3] = math.cos(half)
    data.qpos[4] = math.sin(half)
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    stand_hip_rad = math.radians(STAND_HIP_DEG)
    data.qpos[jnt_adr["hip-L"]] = stand_hip_rad
    data.qpos[jnt_adr["hip-R"]] = stand_hip_rad

    mujoco.mj_forward(model, data)

    # Record start position
    com_start = world_com(model, data).copy()

    # Precompute
    walk_hip_off = math.radians(WALK_HIP_OFFSET_DEG)
    walk_hip_amp = math.radians(WALK_HIP_AMP_DEG)
    walk_crank_amp = math.radians(WALK_CRANK_AMP_DEG)
    t_walk_start = T_HOLD + T_TRANSITION

    # Simulate
    while data.time < MAX_SIM_TIME:
        t = data.time

        # Controller logic (inline for speed)
        if t < T_HOLD:
            data.ctrl[act_ids["hip-L"]]    = stand_hip_rad
            data.ctrl[act_ids["hip-R"]]    = stand_hip_rad
            data.ctrl[act_ids["crank1-L"]] = 0.0
            data.ctrl[act_ids["crank1-R"]] = 0.0

        elif t < t_walk_start:
            alpha = smoothstep(t, T_HOLD, t_walk_start)
            phase = 2 * math.pi * freq * (t - T_HOLD)
            hip_center = stand_hip_rad * (1.0 - alpha) + walk_hip_off * alpha
            hip_osc = alpha * walk_hip_amp * math.sin(phase)

            data.ctrl[act_ids["hip-L"]]    = hip_center + hip_osc
            data.ctrl[act_ids["hip-R"]]    = hip_center - hip_osc
            data.ctrl[act_ids["crank1-L"]] = 0.0
            data.ctrl[act_ids["crank1-R"]] = 0.0

        else:
            phase = 2 * math.pi * freq * (t - T_HOLD)
            hip_osc = walk_hip_amp * math.sin(phase)

            data.ctrl[act_ids["hip-L"]]    = walk_hip_off + hip_osc
            data.ctrl[act_ids["hip-R"]]    = walk_hip_off - hip_osc
            data.ctrl[act_ids["crank1-L"]] = 0.0
            data.ctrl[act_ids["crank1-R"]] = 0.0

        mujoco.mj_step(model, data)

        # Check fall
        com_now = world_com(model, data)
        if com_now[2] < FALL_Z_THRESH:
            break

    # Results
    com_end = world_com(model, data)
    survival = data.time
    dist_xy = math.sqrt((com_end[0] - com_start[0])**2 +
                         (com_end[1] - com_start[1])**2)
    # Forward distance (might be +y or -y, take magnitude)
    dist_forward = abs(com_end[1] - com_start[1])

    return survival, dist_xy, dist_forward, com_end[0] - com_start[0], com_end[1] - com_start[1]


# ─── Main ─────────────────────────────────────────────────────────

def main():
    freqs = np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP/2, FREQ_STEP)
    freqs = np.round(freqs, 2)

    print(f"{'='*70}")
    print(f"  FREQUENCY SWEEP: {FREQ_MIN:.1f} → {FREQ_MAX:.1f} Hz  "
          f"(step={FREQ_STEP:.2f}, {len(freqs)} trials)")
    print(f"  hip_amp={WALK_HIP_AMP_DEG:.1f}°  hip_offset={WALK_HIP_OFFSET_DEG:.1f}°  "
          f"crank=OFF  max_time={MAX_SIM_TIME:.0f}s")
    print(f"{'='*70}\n")

    results = []

    for i, freq in enumerate(freqs):
        survival, dist_xy, dist_fwd, dx, dy = run_trial(freq)
        walk_time = max(0, survival - T_HOLD - T_TRANSITION)

        results.append({
            'freq': freq,
            'survival': survival,
            'walk_time': walk_time,
            'dist_xy': dist_xy,
            'dist_fwd': dist_fwd,
            'dx': dx,
            'dy': dy,
        })

        status = "SURVIVED" if survival >= MAX_SIM_TIME else f"fell@{survival:.1f}s"
        print(f"  [{i+1:3d}/{len(freqs)}]  freq={freq:5.2f}Hz  "
              f"survival={survival:6.1f}s  walk={walk_time:5.1f}s  "
              f"dist={dist_fwd:6.3f}m  ({status})")

    # Save CSV
    csv_path = "sweep_freq_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[Saved] {csv_path}")

    # Summary — top 10 by survival time, then by distance
    print(f"\n{'='*70}")
    print("  TOP 10 by SURVIVAL TIME:")
    print(f"{'='*70}")
    by_survival = sorted(results, key=lambda r: r['survival'], reverse=True)
    for i, r in enumerate(by_survival[:10]):
        print(f"  #{i+1}  freq={r['freq']:5.2f}Hz  "
              f"survival={r['survival']:6.1f}s  "
              f"walk={r['walk_time']:5.1f}s  "
              f"dist_fwd={r['dist_fwd']:6.3f}m")

    print(f"\n{'='*70}")
    print("  TOP 10 by FORWARD DISTANCE:")
    print(f"{'='*70}")
    by_dist = sorted(results, key=lambda r: r['dist_fwd'], reverse=True)
    for i, r in enumerate(by_dist[:10]):
        print(f"  #{i+1}  freq={r['freq']:5.2f}Hz  "
              f"survival={r['survival']:6.1f}s  "
              f"walk={r['walk_time']:5.1f}s  "
              f"dist_fwd={r['dist_fwd']:6.3f}m")

    # Best overall
    best = by_survival[0]
    print(f"\n{'='*70}")
    print(f"  BEST FREQUENCY: {best['freq']:.2f} Hz")
    print(f"  (survived {best['survival']:.1f}s, walked {best['dist_fwd']:.3f}m forward)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()