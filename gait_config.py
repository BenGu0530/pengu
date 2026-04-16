"""
gait_config.py  –  All tunable parameters + gait controller

This is the ONLY file you edit to tune the walk.
Other scripts import from here.
"""

import math
import mujoco

# ─── Model ────────────────────────────────────────────────────────
XML_PATH  = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]
JOINTS    = ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE
# ═════════════════════════════════════════════════════════════════
INIT_Z         = 0.20       # [m]
INIT_PITCH_DEG = -30.0      # [deg]
STAND_HIP_DEG  = -25.0      # [deg]

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS
# ═════════════════════════════════════════════════════════════════
WALK_FREQ           = 1.64  # [Hz]

# Amplitudes
WALK_HIP_AMP_DEG    = 10.0  # [deg] hip swing (C and D)
WALK_HIP_OFFSET_DEG = 0.0   # [deg] hip center during walk
WALK_CRANK_AMP_DEG  = 30.0  # [deg] leg extension (A and B)
WALK_TORSO_AMP_DEG  = 10.0  # [deg] torso roll (E)

# ═════════════════════════════════════════════════════════════════
#  PHASE OFFSETS — extra shift on top of built-in phasing [deg]
#
#  Built-in:  A=0°  B=180°  C=180°  D=0°  E=0°
#  These add to that.
# ═════════════════════════════════════════════════════════════════
PHASE_OFFSET_A_DEG = 0.0    # left leg (crank-L)
PHASE_OFFSET_B_DEG = 0.0    # right leg (crank-R)
PHASE_OFFSET_C_DEG = 0.0    # left hip swing
PHASE_OFFSET_D_DEG = 0.0    # right hip swing
PHASE_OFFSET_E_DEG = 0.0    # torso roll

# ═════════════════════════════════════════════════════════════════
#  TIMING
# ═════════════════════════════════════════════════════════════════
T_HOLD       = 5.0     # [s] hold standing
T_TRANSITION = 2.0     # [s] blend to walk


# ─── Helpers (don't edit below) ───────────────────────────────────

def _get_act_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if i < 0:
        avail = [model.actuator(j).name for j in range(model.nu)]
        raise RuntimeError(f"Actuator '{name}' not found. Available: {avail}")
    return i

def _get_jnt_qposadr(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise RuntimeError(f"Joint '{name}' not found.")
    return model.jnt_qposadr[j]

def _smoothstep(t, t0, t1):
    if t <= t0: return 0.0
    if t >= t1: return 1.0
    x = (t - t0) / (t1 - t0)
    return x * x * (3.0 - 2.0 * x)


def compute_gait(phase, alpha=1.0):
    """
    Compute all 5 gait signals given a base phase [rad] and blend alpha.

    Returns (hip_L, hip_R, crank_L, crank_R, torso) in radians.
    """
    hip_off    = math.radians(WALK_HIP_OFFSET_DEG)
    hip_amp    = math.radians(WALK_HIP_AMP_DEG)
    crank_amp  = math.radians(WALK_CRANK_AMP_DEG)
    torso_amp  = math.radians(WALK_TORSO_AMP_DEG)

    po_a = math.radians(PHASE_OFFSET_A_DEG)
    po_b = math.radians(PHASE_OFFSET_B_DEG)
    po_c = math.radians(PHASE_OFFSET_C_DEG)
    po_d = math.radians(PHASE_OFFSET_D_DEG)
    po_e = math.radians(PHASE_OFFSET_E_DEG)

    sA = math.sin(phase + 0.0      + po_a)   # A: built-in 0°
    sB = math.sin(phase + math.pi  + po_b)   # B: built-in 180°
    sC = math.sin(phase + math.pi  + po_c)   # C: built-in 180°
    sD = math.sin(phase + 0.0      + po_d)   # D: built-in 0°
    sE = math.sin(phase + 0.0      + po_e)   # E: built-in 0°

    crank_L = alpha * crank_amp * 0.5 * (1.0 + sA)   # [0, amp]
    crank_R = alpha * crank_amp * 0.5 * (1.0 + sB)   # [0, amp]
    hip_L   = hip_off + alpha * hip_amp * max(0.0, sC)  # half-rectified
    hip_R   = hip_off + alpha * hip_amp * max(0.0, sD)  # half-rectified
    torso   = alpha * torso_amp * sE                     # full sine

    return hip_L, hip_R, crank_L, crank_R, torso


def set_initial_pose(model, data, act_ids, jnt_adr):
    """Reset and set standing pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = INIT_Z
    pitch_rad = math.radians(INIT_PITCH_DEG)
    data.qpos[3] = math.cos(pitch_rad / 2.0)
    data.qpos[4] = math.sin(pitch_rad / 2.0)
    data.qpos[5] = 0.0
    data.qpos[6] = 0.0
    stand_hip_rad = math.radians(STAND_HIP_DEG)
    data.qpos[jnt_adr["hip-L"]] = stand_hip_rad
    data.qpos[jnt_adr["hip-R"]] = stand_hip_rad
    mujoco.mj_forward(model, data)


def apply_ctrl(data, act_ids, t):
    """
    Full controller: hold → transition → walk.
    Call this every timestep.
    """
    stand_hip    = math.radians(STAND_HIP_DEG)
    t_walk_start = T_HOLD + T_TRANSITION

    if t < T_HOLD:
        data.ctrl[act_ids["hip-L"]]    = stand_hip
        data.ctrl[act_ids["hip-R"]]    = stand_hip
        data.ctrl[act_ids["crank1-L"]] = 0.0
        data.ctrl[act_ids["crank1-R"]] = 0.0
        data.ctrl[act_ids["torso"]]    = 0.0

    elif t < t_walk_start:
        alpha = _smoothstep(t, T_HOLD, t_walk_start)
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
        hip_L_w, hip_R_w, crank_L, crank_R, torso = compute_gait(phase, alpha)

        data.ctrl[act_ids["hip-L"]]    = stand_hip * (1.0 - alpha) + hip_L_w
        data.ctrl[act_ids["hip-R"]]    = stand_hip * (1.0 - alpha) + hip_R_w
        data.ctrl[act_ids["crank1-L"]] = crank_L
        data.ctrl[act_ids["crank1-R"]] = crank_R
        data.ctrl[act_ids["torso"]]    = torso

    else:
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
        hip_L, hip_R, crank_L, crank_R, torso = compute_gait(phase)

        data.ctrl[act_ids["hip-L"]]    = hip_L
        data.ctrl[act_ids["hip-R"]]    = hip_R
        data.ctrl[act_ids["crank1-L"]] = crank_L
        data.ctrl[act_ids["crank1-R"]] = crank_R
        data.ctrl[act_ids["torso"]]    = torso


def build_ids(model):
    """Build actuator and joint ID dicts. Returns (act_ids, jnt_adr)."""
    act_ids = {n: _get_act_id(model, n) for n in ACTUATORS}
    jnt_adr = {n: _get_jnt_qposadr(model, n) for n in JOINTS}
    return act_ids, jnt_adr


def print_config():
    """Print current gait config."""
    t_ws = T_HOLD + T_TRANSITION
    print(f"[Pose] z={INIT_Z:.3f}m  pitch={INIT_PITCH_DEG:.1f}°  stand_hip={STAND_HIP_DEG:.1f}°")
    print(f"[Walk] 0-{T_HOLD:.0f}s hold | {T_HOLD:.0f}-{t_ws:.0f}s transition | {t_ws:.0f}s+ walk")
    print(f"[Gait] freq={WALK_FREQ:.2f}Hz")
    print(f"  A left leg:   amp={WALK_CRANK_AMP_DEG:.1f}°  built-in=0°    extra={PHASE_OFFSET_A_DEG:+.1f}°")
    print(f"  B right leg:  amp={WALK_CRANK_AMP_DEG:.1f}°  built-in=180°  extra={PHASE_OFFSET_B_DEG:+.1f}°")
    print(f"  C left hip:   amp={WALK_HIP_AMP_DEG:.1f}°   built-in=180°  extra={PHASE_OFFSET_C_DEG:+.1f}°")
    print(f"  D right hip:  amp={WALK_HIP_AMP_DEG:.1f}°   built-in=0°    extra={PHASE_OFFSET_D_DEG:+.1f}°")
    print(f"  E torso:      amp={WALK_TORSO_AMP_DEG:.1f}°  built-in=0°    extra={PHASE_OFFSET_E_DEG:+.1f}°")