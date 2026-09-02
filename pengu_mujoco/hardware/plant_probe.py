"""plant_probe.py — B(f) on the robot and in the model, from the same drive.

The kappa loop closes around one transfer: B(f) = torso world roll / torso joint angle,
what the body does when the torso motor moves it. With the loop closed and both gait
amplitudes at zero the robot sits in a 3.1-3.3 Hz limit cycle (pengu-3, pengu-4), and the
condition for that is |S(1-1.5B)| = 1 at angle 0, which those records satisfy to 1% and
1 deg. The model does not: at 3.11 Hz it gives 0.498 at +16.5 deg, and no floor friction
from 0.02 to 0.9 nor any lean from 0 to 30 deg moves it near unity. B is the term that
differs -- 0.789 at -64.6 deg measured, 0.584 at -40.0 in the model.

So B is measured directly. The firmware's 'P' mode holds the legs still, bypasses the
kappa loop, and drives the torso open-loop with a sine stepping through PROBE_HZ. The
drive is logged as goal_torso, so each segment's frequency is read out of the record
itself rather than assumed.

If arg(B) comes out linear in f, its slope is a transport delay -- the IMU reporting late
relative to the joint encoder, which the simulation has none of -- and the intercept is
mechanics, which is the part the model can be asked to reproduce. Separating them is the
point; the closed-loop records cannot, because there the two are multiplied together.

    python hardware/plant_probe.py ~/Downloads/pengu-5.csv        # robot
    python hardware/plant_probe.py ~/Downloads/pengu-5.csv --sim  # and the model, same f
"""
import argparse
import cmath
import csv
import math
import os
import sys

import numpy as np

PROBE_HZ  = [1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 6.0]        # must match the firmware table
PROBE_SEC = [2.0, 1.6, 1.4, 1.3, 1.2, 1.2, 1.2]
PROBE_AMP_DEG = 8.0
T_RAMP, T_SETTLE = 4.0, 6.0                            # the firmware's staged start
SKIP = 0.45                                            # of each segment, let the drive settle


def phasor(t, y, f):
    y = np.asarray(y, float) - np.mean(y)
    w = 2 * math.pi * f * np.asarray(t, float)
    M = np.column_stack([np.ones_like(w), np.cos(w), np.sin(w)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ c
    return cmath.rect(math.hypot(c[1], c[2]), math.atan2(c[1], c[2])), float(np.std(res))


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r.get("w") == "w"]
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    t = np.array([float(r["t"]) for r in rows])
    col = lambda n: np.array([float(r[n]) for r in rows])
    return t, col("goal_torso"), col("pos_torso"), col("imu_roll")


def measured(path):
    t, go, J, roll = load(path)
    if np.max(np.abs(go)) < 1e-9:
        raise SystemExit("goal_torso is identically zero -- the probe was not running")
    out = []
    acc = 0.0
    for k, f in enumerate(PROBE_HZ):
        a = T_RAMP + T_SETTLE + acc
        acc += PROBE_SEC[k]
        m = (t >= a + SKIP) & (t < T_RAMP + T_SETTLE + acc)
        if m.sum() < 15:
            out.append((f, None, None, None, m.sum()))
            continue
        # the frequency the record actually carries, not the one the table promises
        tt = t[m]
        fs = 1.0 / np.median(np.diff(tt))
        y = go[m] - go[m].mean()
        Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        ff = np.fft.rfftfreq(len(y), 1 / fs)
        sel = ff > 0.3
        f_seen = float(ff[sel][np.argmax(Y[sel])]) if sel.sum() else float("nan")
        g, _ = phasor(tt, go[m], f)
        j, rj = phasor(tt, J[m], f)
        r, rr = phasor(tt, roll[m], f)
        if abs(g) < 0.5 or abs(j) < 0.2:
            out.append((f, None, None, f_seen, m.sum()))
            continue
        out.append((f, j / g, r / j, f_seen, m.sum()))
    return out


def simulated(freqs, mu=0.5, amp_deg=PROBE_AMP_DEG):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    os.environ.setdefault("PENGU_MODEL", "hardware_c1")
    os.environ.setdefault("CONFIG", "c1")
    sys.path.insert(0, os.path.join(root, "grid6"))
    sys.path.append(root)
    import mujoco
    import gait_config as gc
    import gait_sweep as gs
    from torso_control import TorsoKappaPID
    from friction_utils import set_floor_friction

    out = []
    amp = math.radians(amp_deg)
    for f in freqs:
        model = mujoco.MjModel.from_xml_path(gs.XML)
        data = mujoco.MjData(model)
        lean = gc.STAND_HIP_DEG
        gc.STAND_HIP_DEG = 0.0
        probe = TorsoKappaPID(model, kappa=0.0, measure_after=0.0)
        gc.STAND_HIP_DEG = lean
        t_on = [None]
        drive = lambda d, t, a=1.0: (0.0 if t_on[0] is None
                                     else amp * math.sin(2 * math.pi * f * (t - t_on[0])))
        gc.TORSO_CONTROLLER = drive
        set_floor_friction(model, mu)
        gs.FLOOR_MU = mu
        gs.CONDITION["hip_off"] = 10.0
        gs._set_gait(dict(freq=1.45, hip_phi=250, leg_amp=0.0, hip_amp=0.0))
        act, jadr = gc.build_ids(model)
        gc.set_initial_pose(model, data, act, jadr)
        gc.T_HOLD = 1e9
        t0 = None
        T, GO, JJ, RO = [], [], [], []
        nxt = 0.0
        while data.time < 30.0:
            if t0 is None:
                if (data.time >= gs.QUIET_MIN_T
                        and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                        or data.time >= gs.QUIET_MAX_T:
                    t0 = data.time
                    gc.T_HOLD = data.time
            gc.apply_ctrl(data, act, data.time)
            mujoco.mj_step(model, data)
            if t0 is None:
                continue
            if t_on[0] is None and data.time > t0 + gc.T_TRANSITION + 1.0:
                t_on[0] = data.time
            if t_on[0] is None or data.time < t_on[0] + SKIP + 1.0:
                continue
            if data.time > t_on[0] + SKIP + 1.0 + 6.0:
                break
            if data.time >= nxt:
                nxt = data.time + 1 / 400.0
                h = probe.hinge(data)
                T.append(data.time)
                GO.append(drive(data, data.time))
                JJ.append(float(data.qpos[jadr["torso"]]))
                RO.append(probe.torso_roll(data, h))
        gc.T_HOLD = 5.0
        g, _ = phasor(T, GO, f)
        j, _ = phasor(T, JJ, f)
        r, _ = phasor(T, RO, f)
        out.append((f, j / g, r / j, f, len(T)))
    return out


def table(rows, label):
    print(f"\n=== {label} ===")
    print(f"{'f [Hz]':>7s}{'|S|':>7s}{'arg S':>8s}{'|B|':>7s}{'arg B':>8s}"
          f"{'|axis/J|':>10s}{'|S(1-1.5B)|':>13s}{'arg':>7s}{'n':>6s}")
    got = []
    for f, S, B, f_seen, n in rows:
        if S is None or B is None:
            print(f"{f:7.2f}{'--':>7s}{'--':>8s}{'--':>7s}{'--':>8s}"
                  f"{'--':>10s}{'--':>13s}{'--':>7s}{n:6d}   no usable segment")
            continue
        L = S * (1 - 1.5 * B)
        print(f"{f:7.2f}{abs(S):7.3f}{math.degrees(cmath.phase(S)):8.1f}"
              f"{abs(B):7.3f}{math.degrees(cmath.phase(B)):8.1f}{abs(B - 1):10.3f}"
              f"{abs(L):13.3f}{math.degrees(cmath.phase(L)):7.1f}{n:6d}")
        got.append((f, B))
    if len(got) >= 3:
        fs = np.array([g[0] for g in got])
        ph = np.unwrap([cmath.phase(g[1]) for g in got])
        sl, ic = np.polyfit(fs, ph, 1)
        print(f"\n  arg(B) fitted linear in f: slope {math.degrees(sl):.1f} deg/Hz "
              f"= {-sl / (2 * math.pi) * 1000:.0f} ms of transport delay, "
              f"intercept {math.degrees(ic):+.1f} deg")
        print("  (a pure delay gives a straight line through the origin; the intercept and\n"
              "   any curvature are mechanics, which is the part the model can reproduce)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="a dump recorded with the 'P' probe running")
    ap.add_argument("--sim", action="store_true", help="also run the model at the same f")
    ap.add_argument("--mu", type=float, default=0.5)
    a = ap.parse_args()
    if a.csv:
        table(measured(a.csv), f"ROBOT  {os.path.basename(a.csv)}")
    if a.sim or not a.csv:
        table(simulated(PROBE_HZ, mu=a.mu), f"MODEL  hardware_c1, mu {a.mu}")
    print("\nS = goal -> torso joint (the servo).  B = joint -> torso world roll (the body).")
    print("The kappa loop oscillates where |S(1-1.5B)| = 1 at angle 0.")


if __name__ == "__main__":
    main()
