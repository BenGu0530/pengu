"""gait_quality.py — what the pass gate does not look at.

A cell passes the sweep if the robot stays upright, faces where it travels, and advances
faster than 0.05 m/s. Nothing in that checks whether the feet ever leave the ground or
whether the body's motion is locked to the gait, so a gait that drags its feet and lurches
at some frequency of its own passes exactly like one that walks. This measures the two
things the gate misses, on any cell, so a shortlist can be re-ranked without changing what
"pass" means.

  CLEARANCE   per foot, the same definition as grid5/analysis/figs/foot_clearance.py:
              the minimum z of that foot's geoms, minus its mean height while loaded.
              Reads ~0 in stance and peaks at the swing apex. Reported as the median
              per-cycle apex, so one lucky step cannot carry it, plus the fraction of the
              window each foot is out of contact at all.

  PHASE LOCK  for roll, pitch and yaw: the amplitude at the gait frequency, the share of
              the signal's variance sitting there ("purity"), and how far the per-cycle
              phase wanders. On the hardware this last number separated a gait that walked
              from one that only looked like it did -- simulation locks to 1-6 deg per
              cycle, the robot drifted 50-60.

    python grid6/gait_quality.py 1.37/350/75/16/50 --mu 0.5
    python grid6/gait_quality.py cells.csv --mu 0.5        (a column of f/phi/leg/hip/off)
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.append(os.path.dirname(_HERE))
os.environ.setdefault("PENGU_MODEL", "hardware_c1")
os.environ.setdefault("CONFIG", "c1")

import mujoco                                    # noqa: E402
import grid6_sweep as g6                         # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402

SETTLE, WINDOW, FS = 2.0, 13.0, 200.0

# The protocol is set here, explicitly, rather than inherited from an import. grid6_sweep
# sets four values at module level, so merely importing something that imports it -- which
# gait_quality does and com_foreaft did not -- silently changed the rollout: the stand
# lean, whether the hip offset ramps in with the blend, and whether the staged start runs
# at all. The same cell then read a rear margin of 12.9 mm one way and 16.8 the other.
# These are the swept protocol's values; both scripts now state them and print them, so a
# table can never again be half one condition and half the other.
REST_LEAN = 5.0            # = grid6_sweep.REST_LEAN_DEG
def _protocol():
    gc.RAMP_HIP_OFFSET = True
    gs.STAGED_START = True
    gs.EXTENDED_METRICS = True



def fit(t, y, f):
    """Amplitude and phase of y at frequency f, plus the share of variance there."""
    w = 2 * math.pi * f * t
    M = np.column_stack([np.ones_like(t), np.cos(w), np.sin(w)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    amp = math.hypot(c[1], c[2])
    sd = float(np.std(y))
    return amp, math.degrees(math.atan2(c[1], c[2])), 100 * (amp / math.sqrt(2)) / max(sd, 1e-12)


def run(freq, phi, leg, hip, off, mu, kappa=0.0):
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _protocol()
    gc.STAND_HIP_DEG = 0.0               # the kappa PID calibrates its neutral at hips-0
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = REST_LEAN
    from friction_utils import set_floor_friction
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)

    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)
    zg = {s: [g for g, sd in foot_geom.items() if sd == s] for s in ("L", "R")}

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    T, Z, LOAD, RPY, POS = [], {"L": [], "R": []}, {"L": [], "R": []}, [], []
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
        if t0 is None:
            continue
        if data.xpos[root][2] < 0.05 and fell is None:
            fell = data.time - t0
            break
        if data.time < t0 + gc.T_TRANSITION + SETTLE:
            continue
        if data.time > t0 + gc.T_TRANSITION + SETTLE + WINDOW:
            break
        if data.time >= nxt:
            nxt = data.time + 1.0 / FS
            loaded = {"L": False, "R": False}
            for ci in range(data.ncon):
                c = data.contact[ci]
                for g in (c.geom1, c.geom2):
                    if g in foot_geom:
                        loaded[foot_geom[g]] = True
            T.append(data.time)
            for s in ("L", "R"):
                Z[s].append(min(data.geom_xpos[g][2] for g in zg[s]))
                LOAD[s].append(loaded[s])
            R = data.xmat[root].reshape(3, 3)
            RPY.append((math.degrees(math.atan2(R[2, 1], R[2, 2])),
                        math.degrees(-math.asin(max(-1.0, min(1.0, R[2, 0])))),
                        math.degrees(math.atan2(R[1, 0], R[0, 0]))))
            POS.append(data.xpos[root][:2].copy())
    gc.T_HOLD = 5.0
    if fell is not None or len(T) < 100:
        return None
    t = np.array(T)
    out = {"fell": fell, "n": len(t)}

    # ---- clearance, per foot -------------------------------------------------------
    for s in ("L", "R"):
        z = np.array(Z[s])
        ld = np.array(LOAD[s])
        base = float(z[ld].mean()) if ld.any() else float(z.min())
        clr = (z - base) * 1000.0
        # per-cycle apex, so a single lucky step cannot set the number
        apex = []
        for k in range(int((t[-1] - t[0]) * freq)):
            m = (t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)
            if m.sum() > 5:
                apex.append(float(clr[m].max()))
        out[f"clear_{s}"] = float(np.median(apex)) if apex else float("nan")
        out[f"clear_{s}_min"] = float(np.min(apex)) if apex else float("nan")
        out[f"air_{s}"] = 100.0 * float((~ld).mean())

    # ---- phase lock, per axis ------------------------------------------------------
    rpy = np.array(RPY)
    for i, nm in enumerate(("roll", "pitch", "yaw")):
        y = rpy[:, i] - rpy[:, i].mean()
        amp, _, pur = fit(t, y, freq)
        psis = []
        for k in range(int((t[-1] - t[0]) * freq)):
            m = (t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)
            if m.sum() > 5:
                psis.append(fit(t[m], y[m], freq)[1])
        d = [abs((psis[j] - psis[j - 1] + 180) % 360 - 180) for j in range(1, len(psis))]
        out[f"{nm}_amp"] = amp
        out[f"{nm}_pur"] = pur
        out[f"{nm}_drift"] = float(np.mean(d)) if d else float("nan")

    p = np.array(POS)
    out["v_net"] = float(np.linalg.norm(p[-1] - p[0])) / float(t[-1] - t[0])
    return out


def parse(spec):
    v = [float(x) for x in spec.split("/")]
    assert len(v) == 5, "want freq/hip_phi/leg_amp/hip_amp/hip_off"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gaits", nargs="+", help="freq/phi/leg/hip/off  (repeatable)")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.0)
    a = ap.parse_args()
    print(f"model {gc.XML_PATH}   mu {a.mu}   kappa {a.kappa}   rest lean {REST_LEAN}"
          f"   window {WINDOW:.0f} s\n")
    print(f"{'gait':24s}{'v_net':>7s}|{'clear L':>9s}{'clear R':>9s}{'min':>7s}"
          f"{'air L%':>8s}{'air R%':>7s}|{'roll amp':>9s}{'pur':>6s}{'drift':>7s}"
          f"|{'pitch amp':>10s}{'pur':>6s}")
    for g in a.gaits:
        f, phi, leg, hip, off = parse(g)
        r = run(f, phi, leg, hip, off, a.mu, a.kappa)
        if r is None:
            print(f"{g:24s}  fell"); continue
        print(f"{g:24s}{r['v_net']:7.4f}|{r['clear_L']:9.1f}{r['clear_R']:9.1f}"
              f"{min(r['clear_L_min'], r['clear_R_min']):7.1f}{r['air_L']:8.0f}{r['air_R']:7.0f}"
              f"|{r['roll_amp']:9.2f}{r['roll_pur']:5.0f}%{r['roll_drift']:7.1f}"
              f"|{r['pitch_amp']:10.2f}{r['pitch_pur']:5.0f}%")
    print("\nclearance in mm: median per-cycle swing apex above that foot's loaded height.")
    print("air%: share of the window with no floor contact on that foot.")
    print("pur: share of the signal's variance at the gait frequency. drift: mean change")
    print("in that axis's per-cycle phase, deg. Simulation normally locks to 1-6.")


if __name__ == "__main__":
    main()
