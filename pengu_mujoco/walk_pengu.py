"""
walk_pengu.py  –  MuJoCo viewer for walking

All tuning is in gait_config.py. This file just runs the viewer.

Usage: python walk_pengu.py
Viewer: Space=pause, Backspace=reset
"""

import mujoco
import mujoco.viewer
from gait_config import (
    XML_PATH, build_ids, set_initial_pose, apply_ctrl, print_config
)

# Global IDs (set in main, used by callback)
_act_ids = {}

def controller(model, data):
    apply_ctrl(data, _act_ids, data.time)

def main():
    global _act_ids

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    _act_ids, jnt_adr = build_ids(model)
    set_initial_pose(model, data, _act_ids, jnt_adr)
    print_config()

    mujoco.set_mjcb_control(controller)
    print("\n[Viewer] Space=pause | Backspace=reset | Close window to quit\n")
    mujoco.viewer.launch(model, data)

if __name__ == "__main__":
    main()