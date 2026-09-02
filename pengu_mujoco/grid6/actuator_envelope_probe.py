"""actuator_envelope_probe.py — does the sim's champion gait fit inside a real XM430?

The sim models the motor's TORQUE limit (position actuators, forcerange +-4.1 N.m) but
has no VELOCITY limit at all, so nothing in the sweep prevents it from selecting gaits
that command joint rates a real XM430 cannot reach. This probe puts numbers on that.

Motor: XM430-W350-T at 12 V (Ben, 2026-08-28) -> datasheet stall 4.1 N.m, no-load
46 rpm = 276 deg/s. A brushed DC motor's operating points lie on the line between those
two corners, so the achievable speed at torque tau is

    w_max(tau) = W_NL * (1 - |tau| / TAU_STALL)

which is an UPPER bound: it ignores gearbox friction, supply sag and the servo's own
current limit. Anything the sim demands above that line is unreachable on hardware, and
the motor answers by running slower -- i.e. the real robot walks a different gait.

Usage:
    python grid5/actuator_envelope_probe.py                 # c6 champion, mu=0.1
    GAIT=c3 python grid5/actuator_envelope_probe.py         # c3 (kappa=0) champion
    MU=0.3 python grid5/actuator_envelope_probe.py
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

# ---- motor envelope (XM430-W350-T @ 12 V) -------------------------------------------
TAU_STALL = 4.1          # N.m
W_NL = 276.0             # deg/s no-load (46 rpm)

# ---- what to run --------------------------------------------------------------------
CHAMPIONS = {
    # name: (kappa, com, freq, hip_phi, leg_amp, hip_amp, hip_off)   -- K=5-verified picks
    "c6": (2.0, 1.31, 1.67, 340.0, 95.0, 24.0, 20.0),
    "c3": (0.0, 1.31, 1.61, 330.0, 115.0, 28.0, 10.0),
    "c1": (0.0, 1.05, 1.80, 270.0, 125.0, 28.0, 30.0),
    # c6 at freq 1.37 -- the gait machine-D used for the 1-deg hip_off scan (7.35 m @ mu=0.1)
    "c6f137": (2.0, 1.31, 1.37, 0.0, 95.0, 28.0, 20.0),
}
WHICH = os.environ.get("GAIT", "c6")
MU = float(os.environ.get("MU", "0.1"))
WINDOW = 16.0            # s of walking measured after the staged start settles
SETTLE = 6.0             # s skipped after t0 (same as imu_frame_probe)

KAPPA, COM, FREQ, PHI, LEG, HIP, OFF = CHAMPIONS[WHICH]

# actuator -> joint (the crank naming is asymmetric in the XML; copy it, do not guess)
ACT_JOINT = {"hip-L": "hip-L", "hip-R": "hip-R",
             "crank1-R": "crank2_R", "crank1-L": "crank1-L", "torso": "torso"}


def build():
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _lean = gc.STAND_HIP_DEG
    gc.STAND_HIP_DEG = 0.0
    g5.apply_com_variant(model, COM)
    pid = TorsoKappaPID(model, kappa=KAPPA, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = _lean
    from friction_utils import set_floor_friction
    set_floor_friction(model, MU)
    gs.FLOOR_MU = MU
    gs.CONDITION["hip_off"] = OFF
    gs._set_gait(dict(freq=FREQ, hip_phi=PHI, leg_amp=LEG, hip_amp=HIP))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    return model, data, act, pid


def main():
    print(f"=== actuator envelope probe: {WHICH} champion  (kappa={KAPPA} com={COM} mu={MU}) ===")
    print(f"    gait: freq={FREQ} phi={PHI} leg_amp={LEG} hip_amp={HIP} hip_off={OFF}")
    print(f"    motor: XM430-W350 @12V -> stall {TAU_STALL} N.m, no-load {W_NL:.0f} deg/s")
    print(f"    analytic peak crank rate = pi*f*A_leg = {math.pi * FREQ * LEG:.0f} deg/s"
          f"   (hips: 2*pi*f*A_hip = {2 * math.pi * FREQ * HIP:.0f} deg/s)\n")

    gc.T_HOLD = 1e9
    model, data, act, pid = build()
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)

    names = list(ACT_JOINT)
    aid = {n: act[n] for n in names}
    dof = {}
    for n, jn in ACT_JOINT.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        assert jid >= 0, f"joint {jn} not found"
        dof[n] = model.jnt_dofadr[jid]

    t0 = None
    rec = {n: [] for n in names}         # (|omega| deg/s, |tau| N.m)
    while True:
        t = data.time
        if t0 is None:
            if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                t0 = t
                gc.T_HOLD = t
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            print(f"    (fell at {data.time:.2f}s -- numbers below cover the walk up to the fall)")
            break
        if t0 is None:
            continue
        if data.time < t0 + SETTLE:
            continue
        if data.time > t0 + SETTLE + WINDOW:
            break
        for n in names:
            w = abs(math.degrees(data.qvel[dof[n]]))
            tau = abs(float(data.actuator_force[aid[n]]))
            rec[n].append((w, tau))

    gc.T_HOLD = 5.0
    n_s = len(rec[names[0]])
    if n_s == 0:
        print("    no measurement samples (fell during the settle window)")
        return
    print(f"    samples: {n_s} physics steps ({n_s * model.opt.timestep:.1f} s of walking)\n")

    print(f"{'actuator':10} {'peak |w|':>9} {'p95 |w|':>9} {'peak |tau|':>10} {'p95 |tau|':>10}"
          f" {'w_max@p95tau':>12} {'>90% stall':>11} {'outside':>8} {'worst x':>8}")
    print("-" * 95)
    for n in names:
        a = np.array(rec[n])
        w, tau = a[:, 0], a[:, 1]
        w_allow = W_NL * np.maximum(0.0, 1.0 - tau / TAU_STALL)
        outside = float(np.mean(w > w_allow))
        near_stall = float(np.mean(tau > 0.9 * TAU_STALL))
        # worst overshoot factor, guarding the near-stall points where w_allow -> 0
        ratio = w / np.maximum(w_allow, 1e-6)
        worst = float(np.max(ratio[w_allow > 20.0])) if np.any(w_allow > 20.0) else float("inf")
        print(f"{n:10} {w.max():9.0f} {np.percentile(w, 95):9.0f} {tau.max():10.2f}"
              f" {np.percentile(tau, 95):10.2f}"
              f" {W_NL * (1 - np.percentile(tau, 95) / TAU_STALL):12.0f}"
              f" {100 * near_stall:10.1f}% {100 * outside:7.1f}% {worst:8.2f}")
    print(f"\n    torso PID in this rollout: roll_rms = {pid.roll_rms():.1f} deg,"
          f"  ctrl saturation ({math.degrees(pid.limit):.0f} deg) = {100 * pid.saturation_frac():.1f}%")
    print("\n    outside = fraction of the measured walk where the demanded joint rate")
    print("    exceeds what the motor can deliver at that instant's torque.")
    print("    worst x = largest (demanded rate)/(available rate), over samples where the")
    print("    envelope still allows >20 deg/s (near stall the ratio blows up by construction).")


if __name__ == "__main__":
    main()
