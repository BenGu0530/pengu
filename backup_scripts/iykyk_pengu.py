"""
walk_pengu.py  –  Spawn standing, stabilize, then walk

Sequence:
  0-5s    : Hold standing pose (hips at lean offset)
  5-7s    : Smoothly transition hips toward walk offset + start oscillation
  7s+     : Full walking gait

Usage (run from ~/Documents/ben_gu/ben_pengu/pengu):
  python walk_pengu.py

Viewer: Space=pause, Backspace=reset, drag=rotate camera
"""

import math
import numpy as np
import mujoco
import mujoco.viewer

# ─── Config ───────────────────────────────────────────────────────
XML_PATH = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE  (same as your working stand_pengu values)
# ═════════════════════════════════════════════════════════════════

INIT_Z         = 0.20       # [m]
INIT_PITCH_DEG = -30.0      # [deg]
STAND_HIP_DEG  = -25.0      # [deg] hip offset for standing

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS — derived from your Arduino code
#  Tune these if the gait doesn't look right!
# ═════════════════════════════════════════════════════════════════

WALK_FREQ       = 1.29      # [Hz] oscillation frequency (Arduino: 1.29)

# Hip: in Arduino, hips oscillate ±15° around a 21° lean offset
# In MuJoCo we found -25° = standing. Walk lean might differ.
WALK_HIP_OFFSET_DEG = 0.0   # [deg] hip offset during walk (0 = let torso lean forward)
WALK_HIP_AMP_DEG    = 15.0  # [deg] hip oscillation amplitude (Arduino: 15°)

# Crank (= "slider" in Arduino): oscillate anti-phase
# Arduino uses 74.5° amplitude but that's for Dynamixel range.
# MuJoCo crank might need different range. Start conservative.
WALK_CRANK_AMP_DEG  = 30.0  # [deg] crank amplitude — TUNE THIS

# Torso roll: small oscillation helps shift weight
WALK_TORSO_AMP_DEG  = 0.0   # [deg] torso roll amplitude

# ═════════════════════════════════════════════════════════════════
#  TIMING
# ═════════════════════════════════════════════════════════════════

T_HOLD      = 5.0     # [s] hold standing pose
T_TRANSITION = 2.0    # [s] smooth transition to walk
# Walk starts at T_HOLD + T_TRANSITION

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
    """Smooth interpolation from 0 to 1 over [t0, t1]."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def set_initial_pose(model, data):
    """Spawn in standing pose."""
    mujoco.mj_resetData(model, data)

    # Root position
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = INIT_Z

    # Root pitch
    pitch_rad = math.radians(INIT_PITCH_DEG)
    half = pitch_rad / 2.0
    data.qpos[3] = math.cos(half)
    data.qpos[4] = math.sin(half)
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0

    # Hips at standing offset
    stand_hip_rad = math.radians(STAND_HIP_DEG)
    data.qpos[_jnt_qposadr["hip-L"]] = stand_hip_rad
    data.qpos[_jnt_qposadr["hip-R"]] = stand_hip_rad

    mujoco.mj_forward(model, data)
    print(f"[Pose] z={INIT_Z:.3f}m  pitch={INIT_PITCH_DEG:.1f}°  "
          f"hip={STAND_HIP_DEG:.1f}°")


# ─── Controller ───────────────────────────────────────────────────

def controller(model, data):
    """Called every substep by MuJoCo."""
    t = data.time

    stand_hip  = math.radians(STAND_HIP_DEG)
    walk_hip_offset = math.radians(WALK_HIP_OFFSET_DEG)
    walk_hip_amp    = math.radians(WALK_HIP_AMP_DEG)
    walk_crank_amp  = math.radians(WALK_CRANK_AMP_DEG)
    walk_torso_amp  = math.radians(WALK_TORSO_AMP_DEG)

    t_walk_start = T_HOLD + T_TRANSITION

    if t < T_HOLD:
        # ── Phase 1: Hold standing pose ──
        data.ctrl[_act_ids["hip-L"]]    = stand_hip
        data.ctrl[_act_ids["hip-R"]]    = stand_hip
        data.ctrl[_act_ids["crank1-L"]] = 0.0
        data.ctrl[_act_ids["crank1-R"]] = 0.0
        data.ctrl[_act_ids["torso"]]    = 0.0

    elif t < t_walk_start:
        # ── Phase 2: Smooth transition to walk pose ──
        alpha = smoothstep(t, T_HOLD, t_walk_start)

        # Blend hip from standing offset to walk offset
        hip_base = stand_hip * (1.0 - alpha) + walk_hip_offset * alpha

        # Ramp up oscillation amplitude
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
        hip_osc   = alpha * walk_hip_amp * math.sin(phase)
        crank_osc = alpha * walk_crank_amp * math.sin(phase)
        torso_osc = alpha * walk_torso_amp * math.sin(phase)

        # Arduino pattern: both hips get same sign oscillation
        # (because physical mechanism is symmetric)
        data.ctrl[_act_ids["hip-L"]]    = hip_base + hip_osc
        data.ctrl[_act_ids["hip-R"]]    = hip_base + hip_osc
        data.ctrl[_act_ids["crank1-L"]] =  crank_osc
        data.ctrl[_act_ids["crank1-R"]] = -crank_osc
        data.ctrl[_act_ids["torso"]]    =  torso_osc

    else:
        # ── Phase 3: Full walk ──
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)

        hip_cmd   = walk_hip_offset + walk_hip_amp * math.sin(phase)
        crank_cmd = walk_crank_amp * math.sin(phase)
        torso_cmd = walk_torso_amp * math.sin(phase)

        data.ctrl[_act_ids["hip-L"]]    =  hip_cmd
        data.ctrl[_act_ids["hip-R"]]    =  hip_cmd
        data.ctrl[_act_ids["crank1-L"]] =  crank_cmd
        data.ctrl[_act_ids["crank1-R"]] = -crank_cmd
        data.ctrl[_act_ids["torso"]]    =  torso_cmd


# ─── Main ─────────────────────────────────────────────────────────

def main():
    global _act_ids, _jnt_qposadr

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Cache IDs
    for name in ACTUATORS:
        _act_ids[name] = get_act_id(model, name)
    for jname in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]:
        _jnt_qposadr[jname] = get_jnt_qposadr(model, jname)

    # Spawn
    set_initial_pose(model, data)

    print(f"\n[Walk] Sequence:")
    print(f"  0-{T_HOLD:.0f}s     : Hold standing pose")
    print(f"  {T_HOLD:.0f}-{T_HOLD+T_TRANSITION:.0f}s   : Transition to walk")
    print(f"  {T_HOLD+T_TRANSITION:.0f}s+     : Walking")
    print(f"\n[Gait] freq={WALK_FREQ:.2f}Hz  hip_amp={WALK_HIP_AMP_DEG:.1f}°  "
          f"crank_amp={WALK_CRANK_AMP_DEG:.1f}°  hip_offset={WALK_HIP_OFFSET_DEG:.1f}°")

    # Launch
    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()