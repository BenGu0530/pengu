"""
walk_pengu.py  –  Spawn standing, stabilize, then walk (hip swing only)

Sequence:
  0-5s    : Hold standing pose (hips at lean offset)
  5-7s    : Smoothly transition hips to walking oscillation
  7s+     : Full walking — hips anti-phase, crank off, torso free

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
#  INIT POSE  (your working stand values)
# ═════════════════════════════════════════════════════════════════

INIT_Z         = 0.20       # [m]
INIT_PITCH_DEG = -30.0      # [deg]
STAND_HIP_DEG  = -25.0      # [deg] hip offset for standing

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS
# ═════════════════════════════════════════════════════════════════

WALK_FREQ           = 1.62  # [Hz]

# Hip swing: anti-phase (left = +sin, right = -sin)
WALK_HIP_OFFSET_DEG = -10.0   # [deg] hip center during walk (0 = torso leans forward)
WALK_HIP_AMP_DEG    = 5.0  # [deg] hip oscillation amplitude

# Crank: 0 = off for now — TUNE THIS if you want to add crank motion
WALK_CRANK_AMP_DEG  = 0.0   # [deg] crank amplitude — disabled

# Torso: NOT actuated during walk — let it swing passively
# (we just won't command it, so the position actuator holds at 0
#  with whatever compliance the default kp gives)
TORSO_PASSIVE = True         # True = don't command torso actuator at all

# ═════════════════════════════════════════════════════════════════
#  TIMING
# ═════════════════════════════════════════════════════════════════

T_HOLD       = 5.0     # [s] hold standing pose
T_TRANSITION = 2.0     # [s] smooth blend to walk

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
    """Smooth 0→1 over [t0, t1]."""
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def set_initial_pose(model, data):
    """Spawn in standing pose."""
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
        if not TORSO_PASSIVE:
            data.ctrl[_act_ids["torso"]] = 0.0

    elif t < t_walk_start:
        # ── Phase 2: Transition ──
        alpha = smoothstep(t, T_HOLD, t_walk_start)
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)

        # Blend hip center from standing offset to walk offset
        hip_center = stand_hip * (1.0 - alpha) + walk_hip_off * alpha

        # Ramp up oscillation — ANTI-PHASE: left = +sin, right = -sin
        hip_osc = alpha * walk_hip_amp * math.sin(phase)

        data.ctrl[_act_ids["hip-L"]]    = hip_center + hip_osc
        data.ctrl[_act_ids["hip-R"]]    = hip_center - hip_osc

        # Crank (off for now)
        crank_osc = alpha * walk_crank_amp * math.sin(phase)
        data.ctrl[_act_ids["crank1-L"]] =  crank_osc
        data.ctrl[_act_ids["crank1-R"]] = -crank_osc

        if not TORSO_PASSIVE:
            data.ctrl[_act_ids["torso"]] = 0.0

    else:
        # ── Phase 3: Full walk ──
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)

        # Anti-phase hip swing
        hip_osc = walk_hip_amp * math.sin(phase)
        data.ctrl[_act_ids["hip-L"]]    = walk_hip_off + hip_osc
        data.ctrl[_act_ids["hip-R"]]    = walk_hip_off - hip_osc

        # Crank (off for now)
        crank_osc = walk_crank_amp * math.sin(phase)
        data.ctrl[_act_ids["crank1-L"]] =  crank_osc
        data.ctrl[_act_ids["crank1-R"]] = -crank_osc

        if not TORSO_PASSIVE:
            data.ctrl[_act_ids["torso"]] = 0.0


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
    print(f"  {T_HOLD:.0f}-{T_HOLD+T_TRANSITION:.0f}s   : Transition (hips anti-phase ramp)")
    print(f"  {T_HOLD+T_TRANSITION:.0f}s+     : Walk — hip only, torso {'passive' if TORSO_PASSIVE else 'active'}")
    print(f"\n[Gait] freq={WALK_FREQ:.2f}Hz  hip_amp={WALK_HIP_AMP_DEG:.1f}°  "
          f"hip_offset={WALK_HIP_OFFSET_DEG:.1f}°  crank=OFF")

    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()