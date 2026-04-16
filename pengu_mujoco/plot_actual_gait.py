"""
plot_actual_gait.py  –  Headless sim, plot commanded vs actual joint angles

All tuning is in gait_config.py. This file just runs headless + plots.

Usage: python plot_actual_gait.py
"""

import math
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from gait_config import (
    XML_PATH, JOINTS, T_HOLD, T_TRANSITION, WALK_FREQ,
    build_ids, set_initial_pose, apply_ctrl, print_config
)

SIM_TIME = 15.0  # total sim seconds

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    act_ids, jnt_adr = build_ids(model)
    set_initial_pose(model, data, act_ids, jnt_adr)
    print_config()

    t_walk_start = T_HOLD + T_TRANSITION

    # Logging
    log_t = []
    log_cmd = {k: [] for k in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]}
    log_act = {k: [] for k in ["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"]}

    # Map actuator names to joint names for reading actual qpos
    act_to_jnt = {
        "hip-L": "hip-L", "hip-R": "hip-R",
        "crank1-L": "crank1-L", "crank1-R": "crank1-R",
        "torso": "torso",
    }

    print(f"\n[Sim] Running headless for {SIM_TIME:.0f}s...")

    while data.time < SIM_TIME:
        apply_ctrl(data, act_ids, data.time)

        # Log every ~20ms
        if len(log_t) == 0 or data.time - log_t[-1] >= 0.02:
            log_t.append(data.time)
            for name in log_cmd:
                log_cmd[name].append(math.degrees(data.ctrl[act_ids[name]]))
                jn = act_to_jnt[name]
                log_act[name].append(math.degrees(data.qpos[jnt_adr[jn]]))

        mujoco.mj_step(model, data)

    # Plot walk phase only
    t_arr = np.array(log_t)
    mask = t_arr >= t_walk_start

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Hip
    ax = axes[0]
    ax.plot(t_arr[mask], np.array(log_cmd["hip-L"])[mask], 'r--', lw=1.5, label='cmd hip-L (C)')
    ax.plot(t_arr[mask], np.array(log_cmd["hip-R"])[mask], 'b--', lw=1.5, label='cmd hip-R (D)')
    ax.plot(t_arr[mask], np.array(log_act["hip-L"])[mask], 'r-',  lw=2,   label='actual hip-L')
    ax.plot(t_arr[mask], np.array(log_act["hip-R"])[mask], 'b-',  lw=2,   label='actual hip-R')
    ax.set_ylabel('Hip angle [deg]')
    ax.set_title('Hip commands (dashed) vs actual (solid)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Crank
    ax = axes[1]
    ax.plot(t_arr[mask], np.array(log_cmd["crank1-L"])[mask], 'r--', lw=1.5, label='cmd crank-L (A)')
    ax.plot(t_arr[mask], np.array(log_cmd["crank1-R"])[mask], 'b--', lw=1.5, label='cmd crank-R (B)')
    ax.plot(t_arr[mask], np.array(log_act["crank1-L"])[mask], 'r-',  lw=2,   label='actual crank-L')
    ax.plot(t_arr[mask], np.array(log_act["crank1-R"])[mask], 'b-',  lw=2,   label='actual crank-R')
    ax.set_ylabel('Crank angle [deg]')
    ax.set_title('Crank commands (dashed) vs actual (solid)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Torso + hip refs
    ax = axes[2]
    ax.plot(t_arr[mask], np.array(log_cmd["torso"])[mask], 'k--', lw=1.5, label='cmd torso (E)')
    ax.plot(t_arr[mask], np.array(log_act["torso"])[mask], 'k-',  lw=2.5, label='actual torso')
    ax.plot(t_arr[mask], np.array(log_cmd["hip-L"])[mask], 'r--', lw=0.8, alpha=0.5, label='cmd hip-L (ref)')
    ax.plot(t_arr[mask], np.array(log_cmd["hip-R"])[mask], 'b--', lw=0.8, alpha=0.5, label='cmd hip-R (ref)')
    ax.set_ylabel('Torso angle [deg]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Torso cmd vs actual (with hip refs)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('actual_gait_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[Saved] actual_gait_debug.png")

if __name__ == "__main__":
    main()