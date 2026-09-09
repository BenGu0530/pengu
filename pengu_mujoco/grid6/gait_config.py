"""
gait_config.py  –  All tunable parameters + gait controller

This is the ONLY file you edit to tune the walk.
Other scripts import from here.
"""

import math
import mujoco

# ─── Model ────────────────────────────────────────────────────────
# Model switch via env var PENGU_MODEL (default v2). penguV3 is the upright
# re-export (pitch fixed) with the SAME 5 actuator names (hip-L/R, crank1-R,
# crank1-L, torso) so this controller works on both unchanged.
import os as _os
_MODEL = _os.environ.get("PENGU_MODEL", "v2")
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]
# COM-ladder models (repo-root models/, hardened to the same 5-actuator convention
# as penguV3; penguV3 == the 1.05 rung). Paths are relative to pengu_mujoco/.
_COM_MODELS = {
    "1.10": "../models/pengu1_10/scene.xml",
    "1.20": "../models/pengu1_20/scene.xml",
    "1.31": "../models/pengu1_31/scene.xml",
    "1.40": "../models/pengu1_40/scene.xml",
    # as-built export of the physical lowered-counterweight robot, hardened to the
    # same 5-actuator convention and ballasted to 2.2724 kg / ratio 1.0500.
    # Its COM ratio comes from the geometry, so it is NOT slid at load time.
    "hardware_c1": "../models/hardware_c1/scene.xml",
    # 2026-09-08 hardware CAD re-exports (Ben), hardened, no hand ballast: 1.05 / 1.20
    "pengu1_05_hw_updated": "../models/pengu1_05_hw_updated/scene.xml",
    "pengu1_20_hw_updated": "../models/pengu1_20_hw_updated/scene.xml",
}
if _MODEL in _COM_MODELS:
    XML_PATH = _COM_MODELS[_MODEL]
    JOINTS   = ["hip-L", "hip-R", "torso"]
elif _MODEL == "v3":
    XML_PATH = "penguV3/scene.xml"
    JOINTS   = ["hip-L", "hip-R", "torso"]
else:
    XML_PATH = "penguV2/scene.xml"
    JOINTS   = ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE
# ═════════════════════════════════════════════════════════════════
if _MODEL == "v3" or _MODEL in _COM_MODELS:
    INIT_Z         = 0.18    # [m]  upright re-export stands ~0.16-0.18
    INIT_PITCH_DEG = 0.0     # [deg] pitch fixed in CAD -> natively upright
    STAND_HIP_DEG  = 0.0     # [deg] stands with hips at neutral
else:
    INIT_Z         = 0.20    # [m]
    INIT_PITCH_DEG = -30.0   # [deg]
    STAND_HIP_DEG  = -25.0   # [deg]

# ═════════════════════════════════════════════════════════════════
#  WALK PARAMS
# ═════════════════════════════════════════════════════════════════
WALK_FREQ           = 1.64  # [Hz]

# Amplitudes
WALK_HIP_AMP_DEG    = 0.0  # [deg] hip swing (C and D)
WALK_HIP_OFFSET_DEG = 0.0   # [deg] hip center during walk (symmetric, both hips)
WALK_HIP_LEAN_DEG   = 0.0   # [deg] ANTISYMMETRIC hip lean: hip_L -= lean, hip_R += lean
                            # (postural lean, mirrors real-robot p_leanAngle). 0 = none.
WALK_CRANK_AMP_DEG  = 30.0  # [deg] leg extension (A and B)
WALK_TORSO_AMP_DEG  = 15.0  # [deg] torso roll (E)

# ═════════════════════════════════════════════════════════════════
#  PHASE OFFSETS — extra shift on top of built-in phasing [deg]
#
#  Built-in:  A=0°  B=180°  C=180°  D=0°  E=0°
#  These add to that.
# ═════════════════════════════════════════════════════════════════
PHASE_OFFSET_A_DEG = 45.0    # left leg (crank-L)
PHASE_OFFSET_B_DEG = 45.0    # right leg (crank-R)
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

    hip_lean = math.radians(WALK_HIP_LEAN_DEG)

    crank_L = alpha * crank_amp * 0.5 * (1.0 + sA)   # [0, amp]
    crank_R = alpha * crank_amp * 0.5 * (1.0 + sB)   # [0, amp]
    if RAMP_HIP_OFFSET:                 # GRID-5: offset blends in with the transition
        hip_off = alpha * hip_off
    hip_L   = hip_off - hip_lean + alpha * hip_amp * max(0.0, sC)  # half-rectified
    hip_R   = hip_off + hip_lean + alpha * hip_amp * max(0.0, sD)  # half-rectified
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


# Optional reactive torso override. When None (default) the torso follows the open-loop
# sinusoid below and behaviour is bit-identical to before. When set to a callable
# f(data, t, alpha) -> torso joint command [rad] (see torso_control.TorsoKappaPID), it
# replaces the sinusoid during the transition and walk phases.
TORSO_CONTROLLER = None

# GRID-5 start protocol: when True, the symmetric hip offset is blended in with the
# transition alpha (command ramps stand_hip -> hip_off, mirroring the firmware's
# READY rest-lean -> walk-offset ramp). Default False = bit-identical legacy step
# (hip_off applied un-blended at the first transition instant).
RAMP_HIP_OFFSET = False


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
        if TORSO_CONTROLLER is not None:
            torso = TORSO_CONTROLLER(data, t, alpha)

        data.ctrl[act_ids["hip-L"]]    = stand_hip * (1.0 - alpha) + hip_L_w
        data.ctrl[act_ids["hip-R"]]    = stand_hip * (1.0 - alpha) + hip_R_w
        data.ctrl[act_ids["crank1-L"]] = crank_L
        data.ctrl[act_ids["crank1-R"]] = crank_R
        data.ctrl[act_ids["torso"]]    = torso

    else:
        phase = 2 * math.pi * WALK_FREQ * (t - T_HOLD)
        hip_L, hip_R, crank_L, crank_R, torso = compute_gait(phase)
        if TORSO_CONTROLLER is not None:
            torso = TORSO_CONTROLLER(data, t, 1.0)

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


def set_walk_freq(hz):
    """Override the global walk frequency [Hz] at runtime (used by sweeps).

    apply_ctrl/compute_gait read WALK_FREQ as a module global, so mutating it
    here takes effect immediately. The file default (1.64) is unchanged; callers
    that mutate this should restore the original value when done.
    """
    global WALK_FREQ
    WALK_FREQ = hz


def set_hip_amp(deg):
    """Override the global hip-swing amplitude [deg] at runtime (used by sweeps).

    compute_gait reads WALK_HIP_AMP_DEG as a module global, so mutating it here
    takes effect immediately. The file default is unchanged; callers that mutate
    this should restore the original value when done.
    """
    global WALK_HIP_AMP_DEG
    WALK_HIP_AMP_DEG = deg


def set_crank_amp(deg):
    """Override the global crank (leg-extension) amplitude [deg] at runtime.

    compute_gait reads WALK_CRANK_AMP_DEG as a module global, so mutating it here
    takes effect immediately. The file default is unchanged; callers that mutate
    this should restore the original value when done.
    """
    global WALK_CRANK_AMP_DEG
    WALK_CRANK_AMP_DEG = deg


def set_torso_amp(deg):
    """Override the global torso-roll amplitude [deg] at runtime (used by sweeps).

    compute_gait reads WALK_TORSO_AMP_DEG as a module global, so mutating it here
    takes effect immediately. The file default is unchanged; callers that mutate
    this should restore the original value when done.
    """
    global WALK_TORSO_AMP_DEG
    WALK_TORSO_AMP_DEG = deg