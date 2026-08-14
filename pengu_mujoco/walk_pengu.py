"""
walk_pengu.py - MuJoCo passive viewer with side-tracking follow camera

Camera behavior:
  - Sits 2 m on the robot's LEFT side, looking at the torso.
  - Follows world position of easytorso every frame.
  - Tracks yaw extracted from torso xmat: when robot turns, camera orbits
    so the sagittal plane stays visible. Roll/pitch are ignored.
  - Yaw is derived robustly: at init we cache the body-frame direction that
    points world +y, then re-project it through xmat each frame. This works
    regardless of which torso body axis is "forward".

Hotkeys (passive viewer):
  Space  pause / resume physics (still able to free-look while paused)
  F      toggle follow-cam on/off (turn off, then drag with mouse)
  Esc    close window

Kinematic tree (root = leftthighmotor, an onshape-to-robot artifact):
  leftthighmotor                                   [freejoint]
    +-- easyaxis                                   [hip-L]
    |     +-- rightthighmotor                      [hip-R]
    |     |     +-- right_foot0080                 [slider-R]  = RIGHT foot
    |     |     +-- crank_circle -> crank_link     [crank1-R]
    |     +-- easytorso                            [torso]     = UPPER BODY
    +-- right_foot0080___fillet13                  [slider-L]  = LEFT foot
    +-- crank_circle_2 -> crank_link_2             [crank1-L]
"""
import math
import time
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

STATIC_HIP_L_DEG   = -25.0
STATIC_HIP_R_DEG   = -25.0
STATIC_TORSO_DEG   =   0.0
STATIC_CRANK_L_DEG =  60.0
STATIC_CRANK_R_DEG =   0.0

# =====================================================================
# FOLLOW CAMERA
# =====================================================================
CAM_DISTANCE   = 2.0       # meters from torso
CAM_ELEVATION  = -12.0     # deg. Negative = camera ABOVE lookat, looking down.
CAM_SIDE       = "left"    # "left" or "right" of the robot's heading
CAM_LOOKAT_ZMIN = 0.10     # clamp lookat z so camera doesn't dip into floor

# Module-level state
_act_ids = {}
_jnt_adr = {}
_torso_body_id = -1
_root_body_id  = -1
_lfoot_body_id = -1
_rfoot_body_id = -1
_last_print_t = -1.0
PRINT_INTERVAL = 0.5

# Mutable shared state for hotkeys & camera. dict so the closures can mutate.
_state = {
    "paused":          False,
    "follow_cam":      True,
    "fwd_in_body":     None,    # cached at init: world +y expressed in torso frame
    "side_offset_deg": 90.0 if CAM_SIDE == "left" else -90.0,
}


def _apply_static_hold(data):
    data.ctrl[_act_ids["hip-L"]]    = math.radians(STATIC_HIP_L_DEG)
    data.ctrl[_act_ids["hip-R"]]    = math.radians(STATIC_HIP_R_DEG)
    data.ctrl[_act_ids["torso"]]    = math.radians(STATIC_TORSO_DEG)
    data.ctrl[_act_ids["crank1-L"]] = math.radians(STATIC_CRANK_L_DEG)
    data.ctrl[_act_ids["crank1-R"]] = math.radians(STATIC_CRANK_R_DEG)


def _decode_torso_tilt(torso_mat):
    up = -torso_mat[:, 1]
    side_lean  = math.degrees(math.atan2(up[0], up[2]))
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

    cmd_hipL  = math.degrees(data.ctrl[_act_ids["hip-L"]])
    cmd_hipR  = math.degrees(data.ctrl[_act_ids["hip-R"]])
    cmd_torso = math.degrees(data.ctrl[_act_ids["torso"]])
    cmd_cL    = math.degrees(data.ctrl[_act_ids["crank1-L"]])
    cmd_cR    = math.degrees(data.ctrl[_act_ids["crank1-R"]])

    jnt_hipL  = math.degrees(data.qpos[_jnt_adr["hip-L"]])
    jnt_hipR  = math.degrees(data.qpos[_jnt_adr["hip-R"]])
    jnt_torso = math.degrees(data.qpos[_jnt_adr["torso"]])
    jnt_cL    = math.degrees(data.qpos[_jnt_adr["crank1-L"]])
    jnt_cR    = math.degrees(data.qpos[_jnt_adr["crank1-R"]])

    root      = data.xpos[_root_body_id]
    lfoot     = data.xpos[_lfoot_body_id]
    rfoot     = data.xpos[_rfoot_body_id]
    torso_pos = data.xpos[_torso_body_id]

    torso_mat = data.xmat[_torso_body_id].reshape(3, 3)
    up, side_lean, pitch_lean = _decode_torso_tilt(torso_mat)

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


# ---------------------------------------------------------------------
# Camera + hotkeys
# ---------------------------------------------------------------------
def _on_key(keycode):
    """Passive-viewer key callback. Receives ASCII keycode."""
    if keycode == 32:  # spacebar
        _state["paused"] = not _state["paused"]
        print(f"\n[KEY] paused = {_state['paused']}")
    elif keycode in (ord('F'), ord('f')):
        _state["follow_cam"] = not _state["follow_cam"]
        print(f"\n[KEY] follow_cam = {_state['follow_cam']}  "
              f"({'tracking robot' if _state['follow_cam'] else 'free mouse look'})")


def _update_follow_cam(viewer, data):
    """
    Re-aim the viewer's free camera at the torso each frame.
    Yaw is taken from the torso xmat by re-projecting the cached
    world-+y direction (in body frame) back into the world.
    """
    fwd_in_body = _state["fwd_in_body"]
    if fwd_in_body is None:
        return

    torso_pos = data.xpos[_torso_body_id].copy()
    R = data.xmat[_torso_body_id].reshape(3, 3)
    fwd_world = R @ fwd_in_body
    yaw_deg = math.degrees(math.atan2(fwd_world[1], fwd_world[0]))

    if torso_pos[2] < CAM_LOOKAT_ZMIN:
        torso_pos[2] = CAM_LOOKAT_ZMIN

    with viewer.lock():
        viewer.cam.lookat[0] = torso_pos[0]
        viewer.cam.lookat[1] = torso_pos[1]
        viewer.cam.lookat[2] = torso_pos[2]
        viewer.cam.distance  = CAM_DISTANCE
        viewer.cam.elevation = CAM_ELEVATION
        viewer.cam.azimuth   = yaw_deg + _state["side_offset_deg"]


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
    mujoco.mj_forward(model, data)  # populate xmat for the cached frame

    # Cache "world +y at t=0, expressed in torso body frame".
    # At runtime: fwd_world = xmat @ fwd_in_body. Yaw = atan2(fwd_world.y, fwd_world.x).
    R0 = data.xmat[_torso_body_id].reshape(3, 3).copy()
    _state["fwd_in_body"] = R0.T @ np.array([0.0, 1.0, 0.0])

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
    else:
        print("  MODE: WALK (USE_STATIC_HOLD = False, using gait_config)")
        print_config()
    print(f"  Tracking bodies: easytorso({_torso_body_id}), "
          f"L foot({_lfoot_body_id}), R foot({_rfoot_body_id}), "
          f"root({_root_body_id})")
    print(f"  Follow cam: side={CAM_SIDE}, dist={CAM_DISTANCE} m, "
          f"elev={CAM_ELEVATION:+.1f} deg")
    print("=" * 70)

    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause/resume | F=toggle follow-cam | "
          "drag=free-look (when follow-cam off) | Esc=close\n")

    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data, key_callback=_on_key) as viewer:
        # Apply camera once before first frame so the initial view is correct.
        _update_follow_cam(viewer, data)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not _state["paused"]:
                mujoco.mj_step(model, data)

            if _state["follow_cam"]:
                _update_follow_cam(viewer, data)

            viewer.sync()

            # Real-time pacing.
            if _state["paused"]:
                time.sleep(0.01)  # idle when paused, don't burn CPU
            else:
                slack = dt - (time.time() - step_start)
                if slack > 0:
                    time.sleep(slack)


if __name__ == "__main__":
    main()