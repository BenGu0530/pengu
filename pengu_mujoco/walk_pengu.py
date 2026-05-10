"""
walk_pengu.py - MuJoCo viewer with per-joint static holds + decoded readout

Kinematic tree (root = leftthighmotor, an onshape-to-robot artifact):
  leftthighmotor                                   [freejoint]
    +-- easyaxis                                   [hip-L]
    |     +-- rightthighmotor                      [hip-R]
    |     |     +-- right_foot0080                 [slider-R]  = RIGHT foot
    |     |     +-- crank_circle -> crank_link     [crank1-R]
    |     +-- easytorso                            [torso]     = UPPER BODY
    +-- right_foot0080___fillet13                  [slider-L]  = LEFT foot
    +-- crank_circle_2 -> crank_link_2             [crank1-L]

WARNING: hip-L / hip-R are NOT "left hip" / "right hip" in a biomechanical
sense. They are internal hinges along the above chain. Always verify by
holding one and watching the feet in the viewer + the printout below.

World frame (inferred from gait_config.set_initial_pose):
  +x = lateral (robot's left-right)
  +y = forward (robot faces +y)
  +z = up
From behind the robot (camera at -y looking toward +y):
  robot's LEFT  = world -x
  robot's RIGHT = world +x
"""
import math
import numpy as np
import mujoco
import mujoco.viewer
from gait_config import (
    XML_PATH, build_ids, set_initial_pose, apply_ctrl, print_config,
    STAND_HIP_DEG,
)

# =====================================================================
# MODE
# =====================================================================
USE_STATIC_HOLD = False   # False -> normal walk; True -> hold values below

# ─── Static hold values (only used when USE_STATIC_HOLD = True) ──────
# hip-L / hip-R: stand pose is STAND_HIP_DEG = -25. "+10° from stand" = -15.
# torso: 0 = no roll. +15 = upper body leans to robot-LEFT (verified).
# crank: 0 = leg retracted (bottom of sine). WALK_CRANK_AMP_DEG = +10 = top.
STATIC_HIP_L_DEG   = -25.0
STATIC_HIP_R_DEG   = -25.0
STATIC_TORSO_DEG   =   0.0
STATIC_CRANK_L_DEG =   60.0
STATIC_CRANK_R_DEG =   0.0
# =====================================================================


# Module-level state
_act_ids = {}
_jnt_adr = {}
_torso_body_id = -1
_root_body_id  = -1
_lfoot_body_id = -1
_rfoot_body_id = -1
_last_print_t = -1.0
PRINT_INTERVAL = 0.5


def _apply_static_hold(data):
    """Hold every joint at its STATIC_*_DEG value."""
    data.ctrl[_act_ids["hip-L"]]    = math.radians(STATIC_HIP_L_DEG)
    data.ctrl[_act_ids["hip-R"]]    = math.radians(STATIC_HIP_R_DEG)
    data.ctrl[_act_ids["torso"]]    = math.radians(STATIC_TORSO_DEG)
    data.ctrl[_act_ids["crank1-L"]] = math.radians(STATIC_CRANK_L_DEG)
    data.ctrl[_act_ids["crank1-R"]] = math.radians(STATIC_CRANK_R_DEG)


def _decode_torso_tilt(torso_mat):
    """
    Extract the torso's 'up vector' in world and decode into side_lean /
    pitch_lean angles. The torso's geometric up direction in its own body
    frame is body -y (verified empirically: at rest ey_world ~ (0,-0.07,-1),
    so -ey_world ~ (0,+0.07,+1) points roughly up).

    Returns (up_world [3,], side_lean_deg, pitch_lean_deg).
      side_lean_deg  : + = leans robot-RIGHT (toward world +x)
                       - = leans robot-LEFT  (toward world -x)
      pitch_lean_deg : + = leans FORWARD (toward world +y)
                       - = leans BACKWARD
    """
    up = -torso_mat[:, 1]       # world-frame vector of torso "up"
    side_lean = math.degrees(math.atan2(up[0],  up[2]))
    pitch_lean = math.degrees(math.atan2(up[1], up[2]))
    return up, side_lean, pitch_lean


def controller(model, data):
    global _last_print_t

    if USE_STATIC_HOLD:
        _apply_static_hold(data)
    else:
        apply_ctrl(data, _act_ids, data.time)

    if data.time - _last_print_t < PRINT_INTERVAL:
        return
    _last_print_t = data.time

    # Commanded values (what we asked for)
    cmd_hipL  = math.degrees(data.ctrl[_act_ids["hip-L"]])
    cmd_hipR  = math.degrees(data.ctrl[_act_ids["hip-R"]])
    cmd_torso = math.degrees(data.ctrl[_act_ids["torso"]])
    cmd_cL    = math.degrees(data.ctrl[_act_ids["crank1-L"]])
    cmd_cR    = math.degrees(data.ctrl[_act_ids["crank1-R"]])

    # Actual joint angles (what the sim achieved)
    jnt_hipL  = math.degrees(data.qpos[_jnt_adr["hip-L"]])
    jnt_hipR  = math.degrees(data.qpos[_jnt_adr["hip-R"]])
    jnt_torso = math.degrees(data.qpos[_jnt_adr["torso"]])
    jnt_cL    = math.degrees(data.qpos[_jnt_adr["crank1-L"]])
    jnt_cR    = math.degrees(data.qpos[_jnt_adr["crank1-R"]])

    # World-frame body positions
    root   = data.xpos[_root_body_id]
    lfoot  = data.xpos[_lfoot_body_id]
    rfoot  = data.xpos[_rfoot_body_id]
    torso_pos = data.xpos[_torso_body_id]

    # Torso tilt decode
    torso_mat = data.xmat[_torso_body_id].reshape(3, 3)
    up, side_lean, pitch_lean = _decode_torso_tilt(torso_mat)

    # Foot split: which side is each foot on, vs forward?
    mid_x = 0.5 * (lfoot[0] + rfoot[0])
    mid_y = 0.5 * (lfoot[1] + rfoot[1])

    mode_str = "STATIC" if USE_STATIC_HOLD else "WALK"
    print(
        f"\n[{mode_str}] t = {data.time:5.2f} s"
        f"\n  CMD  hip-L={cmd_hipL:+6.1f}  hip-R={cmd_hipR:+6.1f}  "
        f"torso={cmd_torso:+6.1f}  crank-L={cmd_cL:+6.1f}  crank-R={cmd_cR:+6.1f}   [deg]"
        f"\n  JNT  hip-L={jnt_hipL:+6.1f}  hip-R={jnt_hipR:+6.1f}  "
        f"torso={jnt_torso:+6.1f}  crank-L={jnt_cL:+6.1f}  crank-R={jnt_cR:+6.1f}   [deg]"
        f"\n  --- world positions  (x=lateral, y=forward, z=up) [m] ---"
        f"\n    L foot  ({lfoot[0]:+.3f}, {lfoot[1]:+.3f}, {lfoot[2]:+.3f})   "
        f"offset from midfoot: dx={lfoot[0]-mid_x:+.3f}  dy={lfoot[1]-mid_y:+.3f}"
        f"\n    R foot  ({rfoot[0]:+.3f}, {rfoot[1]:+.3f}, {rfoot[2]:+.3f})   "
        f"offset from midfoot: dx={rfoot[0]-mid_x:+.3f}  dy={rfoot[1]-mid_y:+.3f}"
        f"\n    root    ({root[0]:+.3f}, {root[1]:+.3f}, {root[2]:+.3f})   (leftthighmotor)"
        f"\n    torso   ({torso_pos[0]:+.3f}, {torso_pos[1]:+.3f}, {torso_pos[2]:+.3f})"
        f"\n  --- torso orientation ---"
        f"\n    up vector in world = ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})  "
        f"(ideal upright = (0, 0, 1))"
        f"\n    side_lean  = {side_lean:+6.1f} deg    (+ = leans RIGHT, - = leans LEFT)"
        f"\n    pitch_lean = {pitch_lean:+6.1f} deg    (+ = leans FORWARD, - = leans BACK)"
    )


def main():
    global _act_ids, _jnt_adr, _last_print_t
    global _torso_body_id, _root_body_id, _lfoot_body_id, _rfoot_body_id
    _last_print_t = -1.0

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    _act_ids, _jnt_adr = build_ids(model)

    def _body_id(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise RuntimeError(f"Body '{name}' not found")
        return i

    _torso_body_id = _body_id("easytorso")
    _root_body_id  = _body_id("leftthighmotor")
    _lfoot_body_id = _body_id("right_foot0080___fillet13")  # slider-L host
    _rfoot_body_id = _body_id("right_foot0080")             # slider-R host

    set_initial_pose(model, data, _act_ids, _jnt_adr)

    print("=" * 70)
    if USE_STATIC_HOLD:
        print("  MODE: STATIC HOLD (USE_STATIC_HOLD = True)")
        print(f"    hip-L   = {STATIC_HIP_L_DEG:+6.1f} deg     "
              f"(stand = {STAND_HIP_DEG:+.1f})")
        print(f"    hip-R   = {STATIC_HIP_R_DEG:+6.1f} deg     "
              f"(stand = {STAND_HIP_DEG:+.1f})")
        print(f"    torso   = {STATIC_TORSO_DEG:+6.1f} deg     "
              f"(0 = no roll, +15 = leans robot-LEFT, verified)")
        print(f"    crank-L = {STATIC_CRANK_L_DEG:+6.1f} deg")
        print(f"    crank-R = {STATIC_CRANK_R_DEG:+6.1f} deg")
        print("  Change one value at a time, compare foot positions.")
    else:
        print("  MODE: WALK (USE_STATIC_HOLD = False, using gait_config)")
        print_config()
    print(f"  Tracking bodies: easytorso({_torso_body_id}), "
          f"L foot({_lfoot_body_id}), R foot({_rfoot_body_id}), "
          f"root({_root_body_id})")
    print("=" * 70)

    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()