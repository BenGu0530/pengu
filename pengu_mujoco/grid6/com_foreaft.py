"""com_foreaft.py — where the CoM sits fore-and-aft of the feet, through the gait cycle.

The sweep records the CoM against the stance foot only laterally (gait_sweep.py:366,
`lx = com[0] - cxy[0]`, "forward travel is +y"), so nothing in the campaign has ever
looked at the fore-aft axis -- which is the one a robot falls backwards about.

Measured here, per sample and then averaged over the gait phase:

  fore   the whole-robot CoM ahead of the loaded feet, along the body's own +y axis, so
         it stays meaningful when the robot yaws. Negative = the CoM is BEHIND the feet.
  base   the fore-aft span of the loaded contact points, i.e. how much support there is
         to be behind. fore/base is the useful ratio: -0.5 means half a foot behind.
  pitch  body pitch, positive nose-up, for comparison against the robot's imu_pitch.

CoM comes from body_mass x xipos rather than data.subtree_com, which caches
body_subtreemass and returned stale answers after a mass edit on 2026-08-29.

    python grid6/com_foreaft.py 1.46/250/75/32/20 1.46/250/75/32/10 --mu 0.5
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
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402
from friction_utils import set_floor_friction    # noqa: E402

SETTLE, WINDOW, FS, NBIN = 2.0, 13.0, 200.0, 24

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



def com_of(model, data):
    m = model.body_mass[1:, None]
    return (model.body_xipos[1:] * m).sum(0) / m.sum() if False else \
           (data.xipos[1:] * m).sum(0) / m.sum()


def run(freq, phi, leg, hip, off, mu, kappa=0.0):
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _protocol()
    gc.STAND_HIP_DEG = 0.0               # the kappa PID calibrates its neutral at hips-0
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = REST_LEAN
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)

    gc.T_HOLD = 1e9
    t0 = None
    fore, base, pitch, ph, nsup, pos, tt_ = [], [], [], [], [], [], []
    rear = []
    nxt = 0.0
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
        tw = data.time - t0 - gc.T_TRANSITION - SETTLE
        if tw < 0:
            continue
        if tw > WINDOW:
            break
        if data.time < nxt:
            continue
        nxt = data.time + 1.0 / FS

        # force-weighted contact points of whatever feet are loaded
        pts, wts = [], []
        for ci in range(data.ncon):
            c = data.contact[ci]
            if c.geom1 not in foot_geom and c.geom2 not in foot_geom:
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, f)
            w = abs(float(f[0]))
            if w <= 1e-6:
                continue
            pts.append(c.pos[:2].copy())
            wts.append(w)
        if not pts:
            continue
        pts = np.array(pts)
        wts = np.array(wts)
        cop = (pts * wts[:, None]).sum(0) / wts.sum()

        R = data.xmat[root].reshape(3, 3)
        fv = R[:2, 1]                                  # body +y = front, projected on xy
        nf = float(np.linalg.norm(fv))
        if nf < 1e-9:
            continue
        fhat = fv / nf
        com = com_of(model, data)
        fore.append(float(np.dot(com[:2] - cop, fhat)) * 1000.0)
        proj = pts @ fhat
        base.append(float(proj.max() - proj.min()) * 1000.0)
        # Tipping backwards is decided by the REARMOST loaded contact point, not by the
        # force-weighted centroid: while the CoM is still ahead of the heel there is a
        # restoring moment. This is the margin a longer foot would buy directly.
        rear.append(float(np.dot(com[:2], fhat) - proj.min()) * 1000.0)
        pitch.append(math.degrees(-math.asin(max(-1.0, min(1.0, R[2, 0])))))
        nsup.append(len(pts))
        ph.append((2 * math.pi * freq * tw) % (2 * math.pi))
        pos.append(data.xpos[root][:2].copy())
        tt_.append(data.time)
    gc.T_HOLD = 5.0
    if fell is not None or len(fore) < 100:
        return None
    q = np.array(pos)
    return dict(fore=np.array(fore), rear=np.array(rear),
                base=np.array(base), pitch=np.array(pitch),
                ph=np.array(ph), nsup=np.array(nsup), fell=fell,
                v_net=float(np.linalg.norm(q[-1] - q[0])) / (tt_[-1] - tt_[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gaits", nargs="+", help="freq/phi/leg/hip/off")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--profile", action="store_true", help="print the per-phase curve")
    a = ap.parse_args()
    print(f"model {gc.XML_PATH}   mu {a.mu}   kappa {a.kappa}   rest lean {REST_LEAN}\n")
    print("fore = CoM ahead of the loaded contact point, mm, along the body's own +y.")
    print("Negative means the CoM is BEHIND the feet.\n")
    print(f"{'gait':24s}{'fore mean':>10s}{'min':>8s}{'max':>8s}{'% behind':>10s}"
          f"{'base mm':>9s}{'pitch mean':>11s}{'pitch p2p':>10s}")
    keep = []
    for g in a.gaits:
        v = [float(x) for x in g.split("/")]
        r = run(*v, a.mu, a.kappa)
        if r is None:
            print(f"{g:24s}  fell / too short")
            continue
        f = r["fore"]
        print(f"{g:24s}{f.mean():10.1f}{f.min():8.1f}{f.max():8.1f}"
              f"{100 * (f < 0).mean():9.0f}%{r['base'].mean():9.1f}"
              f"{r['pitch'].mean():11.1f}{np.ptp(r['pitch']):10.1f}")
        keep.append((g, r))
    if a.profile:
        for g, r in keep:
            print(f"\n-- {g}: CoM fore-aft through one cycle (mm), 24 bins --")
            b = np.clip((r["ph"] / (2 * math.pi) * NBIN).astype(int), 0, NBIN - 1)
            for k in range(NBIN):
                m = b == k
                if not m.any():
                    continue
                v = r["fore"][m].mean()
                bar = ("#" * int(abs(v) / 2))[:40]
                side = " " * 20 if v >= 0 else " " * max(0, 20 - len(bar))
                print(f"  {360*k/NBIN:5.0f} deg {v:8.1f}  "
                      + (side + bar if v < 0 else " " * 20 + bar)
                      + ("   BEHIND" if v < 0 else ""))


if __name__ == "__main__":
    main()
