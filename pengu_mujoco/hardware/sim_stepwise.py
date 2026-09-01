"""sim_stepwise.py — measure the SIMULATION the same way the hardware is measured.

Ben caught this: the hardware speed came from summing the body's displacement between
consecutive footfalls (which counts turning), while the sim speed came from
||end - start|| / duration (which does not). Those are different quantities, and the
difference is not small -- one hardware bout covered 3.15 m of ground and finished 7 cm
from where it started.

So this script runs the sim rollout, detects footfalls from the sim's own foot bodies,
and hands the trajectories to the SAME footfall.step_speed() the hardware analysis uses.
Both numbers are printed side by side so the size of the mistake is visible.

    python hardware/sim_stepwise.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # pengu_mujoco/
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "grid5"))
sys.path.insert(0, HERE)
os.environ.setdefault("PENGU_MODEL", "1.31")

import mujoco                                      # noqa: E402
import footfall as ff                              # noqa: E402
import gait_config as gc                           # noqa: E402
import gait_sweep as gs                            # noqa: E402
import grid5_sweep as g5                           # noqa: E402
from torso_control import TorsoKappaPID            # noqa: E402

SETTLE = 6.0
WINDOW = float(os.environ.get("WINDOW", "16.0"))
FS = 90.0                                          # match the hardware's analysis rate

# the gaits actually flashed, per surface
CONFIGS = [
    # label, mu, kappa, com, freq, hip_phi, leg_amp, hip_amp, hip_off
    ("mu0.1  cmd 95 (as flashed)", 0.1, 2.0, 1.31, 1.67, 340.0, 95.0, 24.0, 20.0),
    ("mu0.1  exec 82 (measured)", 0.1, 2.0, 1.31, 1.67, 340.0, 82.0, 24.0, 20.0),
    ("mu0.5  cmd 115 (as flashed)", 0.5, 2.0, 1.31, 1.92, 240.0, 115.0, 20.0, 10.0),
    ("mu0.5  exec 99 (measured)", 0.5, 2.0, 1.31, 1.92, 240.0, 99.0, 20.0, 10.0),
]


def rollout(mu, kappa, com, freq, phi, leg, hip, off):
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _lean = gc.STAND_HIP_DEG
    gc.STAND_HIP_DEG = 0.0
    g5.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = _lean
    from friction_utils import set_floor_friction
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
    # foot body names are asymmetric in the export; take them from gait_sweep
    fR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080")
    fL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13")

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    T, B, L, R = [], [], [], []
    fell = None
    while True:
        if t0 is None:
            tt = data.time
            if (tt >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or tt >= gs.QUIET_MAX_T:
                t0 = tt
                gc.T_HOLD = tt
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05 and fell is None:
            fell = data.time - (t0 or 0.0)
            break
        if t0 is None or data.time < t0 + SETTLE:
            continue
        if data.time > t0 + SETTLE + WINDOW:
            break
        if data.time >= nxt:
            nxt = data.time + 1.0 / FS
            T.append(data.time)
            B.append(np.array(data.xpos[root][:3], float))
            L.append(np.array(data.xpos[fL][:3], float))
            R.append(np.array(data.xpos[fR][:3], float))
    gc.T_HOLD = 5.0
    return (np.array(T), np.array(B), np.array(L), np.array(R), fell,
            pid.roll_rms())


def main():
    print(f"{'config':30s}{'steps':>6s}{'step_len':>9s}{'v_step':>8s}{'v_net':>8s}"
          f"{'ratio':>7s}{'d_step':>8s}{'d_net':>7s}")
    print("-" * 90)
    for lab, mu, kap, com, f, phi, leg, hip, off in CONFIGS:
        t, B, L, R, fell, rr = rollout(mu, kap, com, f, phi, leg, hip, off)
        if len(t) < 100:
            print(f"{lab:30s}  fell at t0+{fell:.1f}s")
            continue
        st, steps = ff.step_speed(B[:, :2], L, R, t)
        dur = float(t[-1] - t[0])
        d_net = float(np.linalg.norm(B[-1, :2] - B[0, :2]))
        v_net = d_net / dur
        if not st["ok"]:
            print(f"{lab:30s}  footfalls not found ({st['n_steps']})  v_net={v_net:.3f}")
            continue
        print(f"{lab:30s}{st['n_steps']:6d}{st['step_d_median']:9.3f}"
              f"{st['v_pooled']:8.3f}{v_net:8.3f}{st['v_pooled'] / v_net:7.2f}"
              f"{st['dist_m']:8.2f}{d_net:7.2f}")
    print("\nv_step = per-footfall displacement summed / time  (counts turning; what the")
    print("         hardware analysis reports)")
    print("v_net  = ||end - start|| / time                     (what walk_prediction.py")
    print("         reported, and what the sim numbers in the memos were)")


if __name__ == "__main__":
    main()
