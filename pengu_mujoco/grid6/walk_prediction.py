"""walk_prediction.py — pre-registered sim predictions for a hardware PID walk.

Run BEFORE the hardware run, so the comparison is a prediction and not a story told
afterwards. For each config it reports the quantities the robot can actually report
back: joint angles (readable from PRESENT_POSITION), torso torque (readable from
PRESENT_CURRENT), gait frequency, and net speed.

The discriminator between kappa=0 and kappa=2 is the TORSO JOINT ANGLE, not the IMU:
the PID's output IS the joint command, so a working kappa=2 has to show up as a large
torso joint swing. That needs no IMU at all -- just the torso motor's own encoder.

    python grid5/walk_prediction.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PENGU_MODEL", "1.31")

import mujoco                                   # noqa: E402
import gait_config as gc                        # noqa: E402
import gait_sweep as gs                         # noqa: E402
import grid6_sweep as g5                        # noqa: E402
from torso_control import TorsoKappaPID         # noqa: E402

TAU_STALL = 4.1
MU = float(os.environ.get("MU", "0.1"))
WINDOW = 10.0        # a SHORT run, matching what the hardware will do
SETTLE = 6.0

CONFIGS = {
    # name: (kappa, com, freq, hip_phi, leg_amp, hip_amp, hip_off)
    "c6 (kappa=2, pengu_champ)":  (2.0, 1.31, 1.67, 340.0, 95.0, 24.0, 20.0),
    "c3 (kappa=0, pengu_champ_k0)": (0.0, 1.31, 1.61, 330.0, 115.0, 28.0, 10.0),
}

# GAITS env override: "label:kappa,com,freq,phi,leg,hip,off; label2:..."  -- used to screen
# robustness-selected candidates before committing one to hardware.
if os.environ.get("GAITS"):
    CONFIGS = {}
    for item in os.environ["GAITS"].split(";"):
        item = item.strip()
        if not item:
            continue
        lab, nums = item.split(":")
        CONFIGS[lab] = tuple(float(x) for x in nums.split(","))
ACT_JOINT = {"hip-L": "hip-L", "hip-R": "hip-R",
             "crank1-R": "crank2_R", "crank1-L": "crank1-L", "torso": "torso"}


def run(kappa, com, freq, phi, leg, hip, off):
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _lean = gc.STAND_HIP_DEG
    gc.STAND_HIP_DEG = 0.0
    g5.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = _lean
    from friction_utils import set_floor_friction
    set_floor_friction(model, MU)
    gs.FLOOR_MU = MU
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)

    dof, qad = {}, {}
    for n, jn in ACT_JOINT.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dof[n] = model.jnt_dofadr[jid]
        qad[n] = model.jnt_qposadr[jid]

    gc.T_HOLD = 1e9
    t0 = None
    q = {n: [] for n in ACT_JOINT}
    cmd_torso, tau_torso, tau_hipL, tau_hipR, tau_crank = [], [], [], [], []
    axis_roll, torso_roll = [], []
    p0 = p1 = None
    fell = None
    while True:
        if t0 is None:
            t = data.time
            if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                t0 = t
                gc.T_HOLD = t
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            fell = data.time - (t0 or 0.0)
            break
        if t0 is None or data.time < t0 + SETTLE:
            continue
        if data.time > t0 + SETTLE + WINDOW:
            break
        if p0 is None:
            p0 = np.array(data.xpos[root][:2], float)
        p1 = np.array(data.xpos[root][:2], float)
        for n in ACT_JOINT:
            q[n].append(math.degrees(data.qpos[qad[n]]))
        cmd_torso.append(math.degrees(float(data.ctrl[act["torso"]])))
        _h = pid.hinge(data)
        axis_roll.append(math.degrees(pid.axis_roll(data, _h)))
        torso_roll.append(math.degrees(pid.torso_roll(data, _h)))
        tau_torso.append(abs(float(data.actuator_force[act["torso"]])))
        tau_hipL.append(abs(float(data.actuator_force[act["hip-L"]])))
        tau_hipR.append(abs(float(data.actuator_force[act["hip-R"]])))
        tau_crank.append(abs(float(data.actuator_force[act["crank1-L"]])))
    gc.T_HOLD = 5.0

    if not q["torso"]:
        return dict(fell=fell, ok=False)
    net = float(np.linalg.norm(p1 - p0))
    dur = len(q["torso"]) * model.opt.timestep
    a = lambda v: float(np.max(v) - np.min(v))                      # noqa: E731
    r = lambda v: float(np.sqrt(np.mean(np.square(v))))             # noqa: E731
    return dict(
        ok=True, fell=fell, dur=dur, net_m=net, speed=net / dur,
        torso_cmd_amp=a(cmd_torso), torso_cmd_rms=r(cmd_torso),
        torso_joint_amp=a(q["torso"]), torso_joint_rms=r(q["torso"]),
        torso_world_roll_rms=pid.roll_rms(),
        axis_rms=r(axis_roll), axis_amp=a(axis_roll),
        T_rms=r(torso_roll), T_amp=a(torso_roll),
        hipL_amp=a(q["hip-L"]), hipR_amp=a(q["hip-R"]),
        crankL_amp=a(q["crank1-L"]), crankR_amp=a(q["crank1-R"]),
        tau_torso_p95=float(np.percentile(tau_torso, 95)), tau_torso_max=max(tau_torso),
        torso_stall_frac=float(np.mean(np.array(tau_torso) > 0.9 * TAU_STALL)),
        tau_hip_p95=float(np.percentile(tau_hipL + tau_hipR, 95)),
        tau_crank_p95=float(np.percentile(tau_crank, 95)),
    )


def main():
    print(f"=== sim predictions for a short PID walk, mu={MU}, {WINDOW:.0f} s measured ===")
    print("    (hardware tracks its command at ~0.91 on the cranks and ~0.97 on the hips,")
    print("     measured 2026-08-28, so treat leg amplitude as ~9% smaller on the robot)\n")
    out = {}
    for name, cfg in CONFIGS.items():
        res = run(*cfg)
        out[name] = res
        print(f"--- {name}   gait {cfg[2]}/{cfg[3]:.0f}/{cfg[4]:.0f}/{cfg[5]:.0f}/{cfg[6]:.0f}")
        if not res["ok"]:
            print(f"    FELL at t0+{res['fell']:.1f}s before the measurement window\n")
            continue
        if res["fell"]:
            print(f"    (fell at t0+{res['fell']:.1f}s)")
        print(f"    net displacement   {res['net_m']:.2f} m in {res['dur']:.1f} s"
              f"  -> speed {res['speed']:.3f} m/s")
        print(f"    TORSO joint angle  amplitude {res['torso_joint_amp']:.1f} deg,"
              f" rms {res['torso_joint_rms']:.1f} deg   <-- the kappa discriminator")
        print(f"    torso command      amplitude {res['torso_cmd_amp']:.1f} deg")
        print(f"    torso world roll   rms {res['torso_world_roll_rms']:.1f} deg")
        kap = cfg[0]
        sens = abs((kap - 1.0) + 2.0 * kap)
        print(f"    hip-axis roll A    rms {res['axis_rms']:.1f} deg, amplitude {res['axis_amp']:.1f} deg"
              f"   <-- NOT measurable on the robot, reconstructed as T - s*J")
        print(f"    torso tilt T       rms {res['T_rms']:.1f} deg, amplitude {res['T_amp']:.1f} deg")
        print(f"    d(cmd)/d(A) = |(k-1) + kp*k| = {sens:.0f}   -> a 1 deg error in the"
              f" reconstructed A moves the torso command {sens:.0f} deg")
        print(f"    hip joint amp      L {res['hipL_amp']:.1f}  R {res['hipR_amp']:.1f} deg")
        print(f"    crank joint amp    L {res['crankL_amp']:.1f}  R {res['crankR_amp']:.1f} deg")
        print(f"    torque p95         torso {res['tau_torso_p95']:.2f}  hip {res['tau_hip_p95']:.2f}"
              f"  crank {res['tau_crank_p95']:.2f}  N.m")
        print(f"    torso >90% stall   {100 * res['torso_stall_frac']:.1f}% of the window\n")
    if all(v.get("ok") for v in out.values()):
        k2 = out["c6 (kappa=2, pengu_champ)"]
        k0 = out["c3 (kappa=0, pengu_champ_k0)"]
        print("=== the one number that separates the two controllers ===")
        print(f"    torso joint amplitude:  kappa=2 -> {k2['torso_joint_amp']:.1f} deg"
              f"   vs   kappa=0 -> {k0['torso_joint_amp']:.1f} deg"
              f"   (ratio {k2['torso_joint_amp'] / max(k0['torso_joint_amp'], 1e-6):.1f}x)")
        print("    Readable straight off the torso motor's encoder -- no IMU needed.")


if __name__ == "__main__":
    main()
