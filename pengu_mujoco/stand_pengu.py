"""
stand_pengu.py  –  Get the penguin standing & walking in MuJoCo

All tunable initial-pose parameters are in INIT POSE section.
All tunable gait parameters are in GAIT PARAMS section.

Usage (run from ~/Documents/ben_gu/ben_pengu/pengu):
  python stand_pengu.py                    # just stand
  python stand_pengu.py --test_hips        # gentle hip oscillation
  python stand_pengu.py --test_cranks      # gentle crank oscillation  
  python stand_pengu.py --test_torso       # gentle torso oscillation
  python stand_pengu.py --walk             # all together

Viewer: Space=pause, Backspace=reset, drag=rotate camera
"""

import math
import argparse
import numpy as np
import mujoco
import mujoco.viewer

# ─── Config ───────────────────────────────────────────────────────
XML_PATH = "penguV2/scene.xml"
ACTUATORS = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]

# ═════════════════════════════════════════════════════════════════
#  INIT POSE — tune these!
#  All initial pose numbers in one place.
# ═════════════════════════════════════════════════════════════════

INIT_Z         = 0.20     # root height above ground [m]
INIT_PITCH_DEG = -30       # root pitch [deg] — negative=forward lean

INIT_HIP_L_DEG   = -25.0   # left hip offset [deg] — swing legs to adjust lean
INIT_HIP_R_DEG   = -25.0   # right hip offset [deg] — match hip-L for symmetric
INIT_CRANK_L_DEG = 0.0   # left crank offset [deg]
INIT_CRANK_R_DEG = 0.0   # right crank offset [deg]
INIT_TORSO_DEG   = 0.0   # torso joint offset [deg]

# ═════════════════════════════════════════════════════════════════
#  GAIT PARAMS — tune these for walking!
# ═════════════════════════════════════════════════════════════════

FREQ      = 1.39    # oscillation frequency [Hz]
HIP_AMP   = 0.0     # hip amplitude [rad] (overridden by --flags)
CRANK_AMP = 0.0     # crank amplitude [rad]
TORSO_AMP = 0.0     # torso amplitude [rad]

# ─── Derived (don't edit) ────────────────────────────────────────
_INIT_HIP_L   = math.radians(INIT_HIP_L_DEG)
_INIT_HIP_R   = math.radians(INIT_HIP_R_DEG)
_INIT_CRANK_L = math.radians(INIT_CRANK_L_DEG)
_INIT_CRANK_R = math.radians(INIT_CRANK_R_DEG)
_INIT_TORSO   = math.radians(INIT_TORSO_DEG)

# ─── Global state (set in main, read by callback) ────────────────
_act_ids = {}
_freq = FREQ
_hip_amp = HIP_AMP
_crank_amp = CRANK_AMP
_torso_amp = TORSO_AMP
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


def set_initial_pose(model, data):
    """Set full initial pose from the INIT POSE constants above."""
    mujoco.mj_resetData(model, data)

    # Root position
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = INIT_Z

    # Root orientation — pitch around x-axis
    pitch_rad = math.radians(INIT_PITCH_DEG)
    half = pitch_rad / 2.0
    data.qpos[3] = math.cos(half)   # w
    data.qpos[4] = math.sin(half)   # qx
    data.qpos[5] = 0.0              # qy
    data.qpos[6] = 0.0              # qz

    # Joint offsets
    data.qpos[_jnt_qposadr["hip-L"]]    = _INIT_HIP_L
    data.qpos[_jnt_qposadr["hip-R"]]    = _INIT_HIP_R
    data.qpos[_jnt_qposadr["crank1-L"]] = _INIT_CRANK_L
    data.qpos[_jnt_qposadr["crank1-R"]] = _INIT_CRANK_R
    data.qpos[_jnt_qposadr["torso"]]    = _INIT_TORSO

    mujoco.mj_forward(model, data)

    # Diagnostics
    masses = model.body_mass
    total = float(np.sum(masses))
    com = (masses[:, None] * data.xipos).sum(axis=0) / total
    print(f"[Pose] z={INIT_Z:.3f}m  pitch={INIT_PITCH_DEG:.1f}°  "
          f"hip_L={INIT_HIP_L_DEG:.1f}°  hip_R={INIT_HIP_R_DEG:.1f}°")
    print(f"[Pose] COM = ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
    print(f"[Pose] Total mass = {total:.3f} kg")


# ─── Controller callback ─────────────────────────────────────────

def controller(model, data):
    """Called by MuJoCo every substep."""
    t = data.time
    phase = 2 * math.pi * _freq * t

    data.ctrl[_act_ids["hip-L"]]    = _INIT_HIP_L   + (-_hip_amp * math.sin(phase))
    data.ctrl[_act_ids["hip-R"]]    = _INIT_HIP_R   + ( _hip_amp * math.sin(phase))
    data.ctrl[_act_ids["crank1-L"]] = _INIT_CRANK_L  + ( _crank_amp * math.sin(phase))
    data.ctrl[_act_ids["crank1-R"]] = _INIT_CRANK_R  + (-_crank_amp * math.sin(phase))
    data.ctrl[_act_ids["torso"]]    = _INIT_TORSO    + ( _torso_amp * math.sin(phase))


# ─── Main ─────────────────────────────────────────────────────────

def main():
    global _act_ids, _hip_amp, _crank_amp, _torso_amp, _freq, _jnt_qposadr

    parser = argparse.ArgumentParser(description="Stand the penguin in MuJoCo")
    parser.add_argument("--test_hips", action="store_true")
    parser.add_argument("--test_cranks", action="store_true")
    parser.add_argument("--test_torso", action="store_true")
    parser.add_argument("--walk", action="store_true")
    parser.add_argument("--xml", default=XML_PATH)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # Cache IDs
    for name in ACTUATORS:
        _act_ids[name] = get_act_id(model, name)
    for jname in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]:
        _jnt_qposadr[jname] = get_jnt_qposadr(model, jname)

    # Gait flags
    if args.test_hips:
        _hip_amp = 0.15
        print("[Mode] Testing hips: amp=0.15 rad")
    if args.test_cranks:
        _crank_amp = 0.3
        print("[Mode] Testing cranks: amp=0.3 rad")
    if args.test_torso:
        _torso_amp = 0.05
        print("[Mode] Testing torso: amp=0.05 rad")
    if args.walk:
        _hip_amp = 0.1
        _crank_amp = 0.2
        _torso_amp = 0.01
        print("[Mode] Open-loop walk")
    if not any([args.test_hips, args.test_cranks, args.test_torso, args.walk]):
        print("[Mode] Standing still — actuators hold at init offsets")
        print("  Tune INIT_PITCH_DEG and INIT_HIP_*_DEG to find balance")
        print("  Try: --test_hips, --test_cranks, --test_torso, or --walk")

    # Set pose and launch
    set_initial_pose(model, data)
    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()