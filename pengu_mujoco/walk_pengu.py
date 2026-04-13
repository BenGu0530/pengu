"""
walk_pengu.py  –  Spawn standing, stabilize, then walk

Hip: anti-phase sine (left = +sin, right = -sin)
Crank: in-phase with SAME-SIDE hip, range [0, amp] (extend only)
  - left crank peaks when left hip peaks
  - right crank peaks when right hip peaks (offset by π)

Usage (run from ~/Documents/ben_gu/ben_pengu/pengu):
  python walk_pengu.py

Viewer: Space=pause, Backspace=reset
"""

import math
import numpy as np
import mujoco
import mujoco.viewer

# ─── Config ───────────────────────────────────────────────────────
XML_PATH = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE
# ═════════════════════════════════════════════════════════════════
INIT_Z         = 0.20       # [m]
INIT_PITCH_DEG = -30.0      # [deg]
STAND_HIP_DEG  = -25.0      # [deg]

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS
# ═════════════════════════════════════════════════════════════════
WALK_FREQ           = 1.62  # [Hz] — optimal from sweep

# Hip: anti-phase swing
WALK_HIP_OFFSET_DEG = -5.0   # [deg] hip center during walk
WALK_HIP_AMP_DEG    = 15.0  # [deg] hip oscillation amplitude

# Crank (leg extension): in-phase with same-side hip
# Range: 0 (retracted) to WALK_CRANK_AMP_DEG (extended)
# NOT ±amp, but 0 to +amp using (1+sin)/2 mapping
WALK_CRANK_AMP_DEG  = -40.0  # [deg] max crank extension — TUNE THIS

# Torso
TORSO_PASSIVE = True        # True = don't actuate torso

# ═════════════════════════════════════════════════════════════════
#  TIMING
# ═════════════════════════════════════════════════════════════════
T_HOLD       = 5.0     # [s] hold standing
T_TRANSITION = 2.0     # [s] blend to walk

# ─── Global state ────────────────────────────────────────────────
_act_ids = {}
_jnt_qposadr = {}

# ─── Helpers ──────────────────────────────────────────────────────

def get_act_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        avail = [model.actuator(j).name for j in range(model.nu)]
        raise RuntimeError(f"Actuator '{name}' not found. Available: {avail}")
    return i

def get_jnt_qposadr(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise RuntimeError(f"Joint '{name}' not found.")
    return model.jnt_qposadr[j]

def smoothstep(t, t0, t1):
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def set_initial_pose(model, data):
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
    data.qpos[_jnt_qposadr["hip-L"]] = stand_hip_rad
    data.qpos[_jnt_qposadr["hip-R"]] = stand_hip_rad

    mujoco.mj_forward(model, data)
    print(f"[Pose] z={INIT_Z:.3f}m  pitch={INIT_PITCH_DEG:.1f}°  "
          f"stand_hip={STAND_HIP_DEG:.1f}°")


# ─── Controller ───────────────────────────────────────────────────

def controller(model, data):
    t = data.time

    stand_hip      = math.radians(STAND_HIP_DEG)
    walk_hip_off   = math.radians(WALK_HIP_OFFSET_DEG)
    walk_hip_amp   = math.radians(WALK_HIP_AMP_DEG)
    walk_crank_amp = math.radians(WALK_CRANK_AMP_DEG)

    t_walk_start = T_HOLD + T_TRANSITION

    if t < T_HOLD:
        # ── Phase 1: Hold standing ──
        data.ctrl[_act_ids["hip-L"]]    = stand_hip
        data.ctrl[_act_ids["hip-R"]]    = stand_hip
        data.ctrl[_act_ids["crank1-L"]] = 0.0
        data.ctrl[_act_ids["crank1-R"]] = 0.0

    elif t < t_walk_start:
        # ── Phase 2: Transition ──
        alpha = smoothstep(t, T_HOLD, t_walk_start)
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)

        # Hip: blend center + ramp up anti-phase oscillation
        hip_center = stand_hip * (1.0 - alpha) + walk_hip_off * alpha

        # Left hip: +sin(phase), Right hip: -sin(phase) = sin(phase + π)
        hip_L = hip_center + alpha * walk_hip_amp * math.sin(phase)
        hip_R = hip_center - alpha * walk_hip_amp * math.sin(phase)

        # Crank: in-phase with same-side hip, range [0, amp]
        # Left crank follows left hip phase: (1 + sin(phase)) / 2
        # Right crank follows right hip phase: (1 + sin(phase + π)) / 2 = (1 - sin(phase)) / 2
        crank_L = alpha * walk_crank_amp * 0.5 * (1.0 + math.sin(phase))
        crank_R = alpha * walk_crank_amp * 0.5 * (1.0 - math.sin(phase))

        data.ctrl[_act_ids["hip-L"]]    = hip_L
        data.ctrl[_act_ids["hip-R"]]    = hip_R
        data.ctrl[_act_ids["crank1-L"]] = crank_L
        data.ctrl[_act_ids["crank1-R"]] = crank_R

    else:
        # ── Phase 3: Full walk ──
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)

        # Hip: anti-phase
        hip_L = walk_hip_off + walk_hip_amp * math.sin(phase)
        hip_R = walk_hip_off - walk_hip_amp * math.sin(phase)

        # Crank: in-phase with same-side hip, range [0, amp]
        # Left peaks when sin(phase) = 1, right peaks when sin(phase) = -1
        crank_L = walk_crank_amp * 0.5 * (1.0 + math.sin(phase))
        crank_R = walk_crank_amp * 0.5 * (1.0 - math.sin(phase))

        data.ctrl[_act_ids["hip-L"]]    = hip_L
        data.ctrl[_act_ids["hip-R"]]    = hip_R
        data.ctrl[_act_ids["crank1-L"]] = crank_L
        data.ctrl[_act_ids["crank1-R"]] = crank_R


# ─── Main ─────────────────────────────────────────────────────────

def main():
    global _act_ids, _jnt_qposadr

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    for name in ACTUATORS:
        _act_ids[name] = get_act_id(model, name)
    for jname in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]:
        _jnt_qposadr[jname] = get_jnt_qposadr(model, jname)

    set_initial_pose(model, data)

    print(f"\n[Walk] Sequence:")
    print(f"  0-{T_HOLD:.0f}s     : Hold standing")
    print(f"  {T_HOLD:.0f}-{T_HOLD+T_TRANSITION:.0f}s   : Transition")
    print(f"  {T_HOLD+T_TRANSITION:.0f}s+     : Walk")
    print(f"\n[Gait] freq={WALK_FREQ:.2f}Hz  hip_amp={WALK_HIP_AMP_DEG:.1f}°  "
          f"crank_amp={WALK_CRANK_AMP_DEG:.1f}° (0→{WALK_CRANK_AMP_DEG:.1f}°)")
    print(f"  Hip: L=+sin  R=-sin (anti-phase)")
    print(f"  Crank: L=(1+sin)/2  R=(1-sin)/2 (in-phase with same-side hip)")

    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()