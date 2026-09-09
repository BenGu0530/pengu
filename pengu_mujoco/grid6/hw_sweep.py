"""hw_sweep.py — ff_sweep re-run for the hardware session: measured friction, FF torso only.

Copied from ff_sweep.py (which produced the shipped ffact_mu05 data and is left untouched).
Differences: mu is the drag-measured static value (0.12 ice, 0.45 floor) instead of the
grid 0.1..0.7; the kappa = 0 PID rollout is dropped (the robot runs feedforward now); each
rollout also records `straight` = net displacement / low-passed path length, because the
robot's c1 ice takes walked in arcs (straightness 0.18-0.43 on half of them) and no
campaign metric sees that. Ben 2026-09-08: the leg model is a 354 deg/s HARD CAP only
(LEG_TAU = 0, no one-pole); speed is the whole-body COM (mass-weighted xipos), not the
root body; the torso clamp is the firmware's 25 deg; clear_ok flags clearance >= 10 mm.
Output goes to results/grid6_hw/, never grid6_report/.

    python grid6/hw_sweep.py count
    python grid6/hw_sweep.py --mu 0.12 --shard i --of n
    python grid6/hw_sweep.py --mu 0.12 --merge

Original ff_sweep header follows.

ff_sweep.py — score gaits under the torso the robot actually has.

Every map so far was scored with a torso loop that reads true state instantly at 1 kHz.
That robot does not exist. On this one the torso joint reaches its extreme 56 ms after the
hip axis reaches its own, by which time the axis is already returning in 76-90% of events,
and the kappa PID therefore pushes the lower body the way it is already going: the same
gait rolls 21 deg peak-to-peak with the torso held and 67 with the loop closed. A gait
picked under the ideal torso is picked in the wrong world.

So each cell is rolled out three ways, all of them carrying the 56 ms:

  HELD   torso commanded to home, no controller. This is kappa = 1 and it is also the
         measurement that supplies the feedforward: the hip-axis roll here, fitted at the
         gait frequency, is exactly what the torso has to cancel.
  FF     torso_deg = A0 * sin(phase + phi0 + 180 + lead), locked to the same phase the legs
         use. No measurement in the loop, so the sensor delay cannot enter; the servo's own
         lag is cancelled by leading the phase. `lead` is the one number that is not
         predictable from the fit -- on the robot the best value sat 43 deg past naive, in
         the model 47.5 and 65 for two different gaits -- so a few are tried and the best
         kept, which is what a calibration session does anyway.
  PID    the kappa = 0 loop as flashed, for the comparison.

Ranking is on the FF rollout. Recorded per rollout: net speed, per-cycle foot clearance,
roll phase drift, torso world roll (the kappa = 0 acceptance number), hip-axis roll, and
where the CoM sits fore-and-aft of the loaded feet -- the axis this robot falls about, which
no campaign has ever scored on.

The grid is centred on what walks on the robot: 1.39 / 240 / 80 / 16 / 25, and hip_off 0,
10 and 50 are dropped at Ben's instruction. Nothing is excluded for being too fast: the leg
servos are modelled instead, so a cell above the 354 deg/s ceiling is rolled out as the
clipped, delayed gait it actually becomes rather than pretended out of existence. That is
where the speed is -- inside the envelope the best cell reached 0.128 m/s, and ignoring the
ceiling entirely reached 0.394.

    python grid6/ff_sweep.py count
    python grid6/ff_sweep.py [--mu 0.5] [--shard i --of n]
    python grid6/ff_sweep.py --merge
"""
import argparse
import csv
import cmath
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
os.environ.setdefault("PENGU_MODEL", "hardware_c1")
os.environ.setdefault("CONFIG", "c1")

import mujoco                                    # noqa: E402
import gait_config as gc                         # noqa: E402
import gait_sweep as gs                          # noqa: E402
from torso_control import TorsoKappaPID          # noqa: E402
from friction_utils import set_floor_friction    # noqa: E402

# ---------------------------------------------------------------- protocol, stated
REST_LEAN = 5.0                 # = grid6_sweep.REST_LEAN_DEG
SETTLE, WINDOW, FS = 2.0, 13.0, 200.0
SERVO_LAG = 0.056               # measured: corr(J[k], goal[k-2]) = 0.984 at 28 ms/sample
LEADS = (30.0, 50.0, 70.0)      # deg past the naive cancelling phase
# The leg servos are modelled rather than fenced off. Cells above the ceiling used to be
# excluded from the grid, which hid the whole fast half of the space; instead the command
# is slew-limited and passed through a one-pole lag, so a cell above the ceiling is
# simulated as what it actually becomes. Calibrated against two hardware points on opposite
# sides of the ceiling, from the executed-vs-commanded harmonic fits:
#   1.46 Hz, peak demand 343 deg/s : robot ratio 0.991 lag 26 ms, model 0.965 / 28.3
#   1.95 Hz, peak demand 766       : robot ratio 0.534 lag 90 ms, model 0.554 / 89.5
# Raising LEG_RATE is how a voltage or motor change gets tested -- the map need not be
# rebuilt, only re-scored.
LEG_RATE = 354.0                # deg/s, twelve measurements 2026-08-30, air and ground
LEG_TAU = 0.0                   # Ben 2026-09-08: hard velocity cap only, no one-pole
TORSO_CLAMP_DEG = 25.0          # firmware TORSO_CLAMP_DEG (pengu_tune_wifi); c6 as flashed 08-29 was 45
CLEAR_MIN_MM = 10.0             # clear_ok flag threshold; the clear column stays for re-thresholding
CEILING = 1e9                   # no cell is excluded any more

# ---------------------------------------------------------------- the grid
FREQ = [round(1.20 + 0.02 * k, 2) for k in range(26)]      # 1.20 .. 1.70
PHI = list(range(200, 310, 10))                            # 200 .. 300
LEG = list(range(70, 135, 5))                              # 70 .. 130
HIP = [12, 16, 20, 24, 28, 32]
OFF = [20, 25, 30, 35, 40]                                 # 0/10/50 dropped
OUT = os.path.join(ROOT, "results", "grid6_hw")
COLS = (["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
         "A0", "phi0", "best_lead"]
        + [f"{m}_{w}" for w in ("held", "ff")
           for m in ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms",
                     "axisrms", "fore", "rearp5", "sat")])


def cells():
    out = []
    for f in FREQ:
        for a in LEG:
            if math.pi * f * a > CEILING:
                continue
            for h in HIP:
                if 2 * math.pi * f * h > CEILING:
                    continue
                for p in PHI:
                    for o in OFF:
                        out.append((f, float(p), float(a), float(h), float(o)))
    return out


def fit(t, y, f):
    """amplitude, phase and residual of y at f, in the controller's phase reference."""
    y = np.asarray(y) - np.mean(y)
    w = 2 * math.pi * f * np.asarray(t)
    M = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return (math.hypot(c[1], c[2]), math.degrees(math.atan2(c[2], c[1])),
            float(np.std(y - M @ c)))


def rollout(freq, phi, leg, hip, off, mu, mode, A=0.0, ph=0.0, kappa=0.0):
    """One bout. mode is 'held', 'ff' or 'pid'; all three carry the servo lag."""
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    gc.RAMP_HIP_OFFSET = True
    gs.STAGED_START = True
    gc.STAND_HIP_DEG = 0.0                       # the PID calibrates its neutral at hips-0
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0, ctrl_limit_deg=TORSO_CLAMP_DEG)
    gc.STAND_HIP_DEG = REST_LEAN
    set_floor_friction(model, mu)
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = gs.make_ids(model)
    legs = [act[n] for n in ("crank1-L", "crank1-R", "hip-L", "hip-R")]
    slew = math.radians(LEG_RATE) * model.opt.timestep
    a_lp = model.opt.timestep / (model.opt.timestep + LEG_TAU)
    held_cmd = None
    lagged = None
    zg = {s: [g for g, sd in foot_geom.items() if sd == s] for s in ("L", "R")}

    buf = []

    def ctrl(d, t, alpha=1.0):
        if mode == "pid":
            u = pid(d, t, alpha)
        elif mode == "held":
            u = 0.0
        else:
            w = 2 * math.pi * freq * (t - gc.T_HOLD - gc.T_TRANSITION)
            u = alpha * math.radians(A) * math.sin(w + math.radians(ph))
            u = max(-pid.limit, min(pid.limit, u))
        buf.append((t, u))
        while len(buf) > 1 and buf[1][0] <= t - SERVO_LAG:
            buf.pop(0)
        return buf[0][1]
    gc.TORSO_CONTROLLER = ctrl

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    T, AX, RO, Z, LOAD, FORE, REAR, POS, SAT = [], [], [], {"L": [], "R": []}, \
        {"L": [], "R": []}, [], [], [], []
    fell = None
    while True:
        if t0 is None:
            tt = data.time
            if (tt >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or tt >= gs.QUIET_MAX_T:
                t0 = tt
                gc.T_HOLD = tt
        gc.apply_ctrl(data, act, data.time)
        cur = np.array([data.ctrl[i] for i in legs])
        if held_cmd is None:
            held_cmd = cur.copy()
            lagged = cur.copy()
        held_cmd += np.clip(cur - held_cmd, -slew, slew)
        lagged += a_lp * (held_cmd - lagged)
        for i, j in enumerate(legs):
            data.ctrl[j] = lagged[i]
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
        h = pid.hinge(data)
        T.append(data.time)
        AX.append(math.degrees(pid.axis_roll(data, h)))
        RO.append(math.degrees(pid.torso_roll(data, h)))
        SAT.append(abs(data.ctrl[act["torso"]]) > pid.limit - 1e-9)
        loaded = {"L": False, "R": False}
        pts, wts = [], []
        for ci in range(data.ncon):
            c = data.contact[ci]
            hit = [g for g in (c.geom1, c.geom2) if g in foot_geom]
            if not hit:
                continue
            loaded[foot_geom[hit[0]]] = True
            fv = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, fv)
            if abs(float(fv[0])) > 1e-6:
                pts.append(c.pos[:2].copy())
                wts.append(abs(float(fv[0])))
        for s in ("L", "R"):
            Z[s].append(min(data.geom_xpos[g][2] for g in zg[s]))
            LOAD[s].append(loaded[s])
        com = (data.xipos[1:] * model.body_mass[1:, None]).sum(0) \
            / model.body_mass[1:].sum()
        POS.append(com[:2].copy())              # whole-body COM, not the root body
        if pts:
            pts = np.array(pts)
            wts = np.array(wts)
            cop = (pts * wts[:, None]).sum(0) / wts.sum()
            R = data.xmat[root].reshape(3, 3)
            fvv = R[:2, 1]
            nf = float(np.linalg.norm(fvv))
            if nf > 1e-9:
                fh = fvv / nf
                FORE.append(float(np.dot(com[:2] - cop, fh)) * 1000.0)
                REAR.append(float(np.dot(com[:2], fh) - (pts @ fh).min()) * 1000.0)
    gc.T_HOLD = 5.0
    if fell is not None or len(T) < 100:
        return dict(fell=fell if fell is not None else 0.0)

    t = np.array(T)
    ax = np.array(AX)
    out = dict(fell=None, rollrms=float(np.std(RO)), axisrms=float(np.std(ax)),
               sat=100.0 * float(np.mean(SAT)),
               fore=float(np.mean(FORE)) if FORE else float("nan"),
               rearp5=float(np.percentile(REAR, 5)) if REAR else float("nan"))
    q = np.array(POS)
    out["v_net"] = float(np.linalg.norm(q[-1] - q[0])) / (t[-1] - t[0])
    # straightness: net displacement over the path length of the root, the path taken on
    # the trajectory low-passed over one gait period so the waddle does not count
    k = max(1, int(round(FS / freq)))
    ker = np.ones(k) / k
    if len(q) > 2 * k:
        ql = np.column_stack([np.convolve(q[:, i], ker, "valid") for i in range(2)])
        path = float(np.sum(np.hypot(np.diff(ql[:, 0]), np.diff(ql[:, 1]))))
        out["straight"] = float(np.linalg.norm(ql[-1] - ql[0]) / path) if path > 1e-6 else float("nan")
    else:
        out["straight"] = float("nan")
    # per-foot per-cycle clearance apex, the minimum over cycles
    mins = []
    for s in ("L", "R"):
        z = np.array(Z[s])
        ld = np.array(LOAD[s])
        base = float(z[ld].mean()) if ld.any() else float(z.min())
        clr = (z - base) * 1000.0
        apex = [float(clr[(t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)].max())
                for k in range(int((t[-1] - t[0]) * freq))
                if ((t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)).sum() > 5]
        mins.append(min(apex) if apex else float("nan"))
    out["clear"] = min(mins)
    out["clear_ok"] = int(out["clear"] >= CLEAR_MIN_MM) if np.isfinite(out["clear"]) else 0
    # roll phase drift, per cycle
    y = np.array(RO) - np.mean(RO)
    psis = []
    for k in range(int((t[-1] - t[0]) * freq)):
        m = (t >= t[0] + k / freq) & (t < t[0] + (k + 1) / freq)
        if m.sum() > 5:
            psis.append(fit(t[m], y[m], freq)[1])
    d = [abs((psis[j] - psis[j - 1] + 180) % 360 - 180) for j in range(1, len(psis))]
    out["drift"] = float(np.mean(d)) if d else float("nan")
    out["_fit"] = fit(t - gc.T_HOLD - gc.T_TRANSITION, ax, freq)
    return out


def blank():
    return dict(fell=float("nan"), v_net=float("nan"), straight=float("nan"), clear=float("nan"),
                clear_ok=0,
                drift=float("nan"), rollrms=float("nan"), axisrms=float("nan"),
                fore=float("nan"), rearp5=float("nan"), sat=float("nan"))


def score(cell, mu):
    f, phi, leg, hip, off = cell
    held = rollout(f, phi, leg, hip, off, mu, "held")
    if held.get("fell") is not None:
        return None                       # cannot even stand the gait with a passive torso
    A0, p0, _ = held["_fit"]
    best, best_lead = None, float("nan")
    for lead in LEADS:
        r = rollout(f, phi, leg, hip, off, mu, "ff", A=A0, ph=p0 + 180.0 + lead)
        if r.get("fell") is not None:
            continue
        if best is None or r["rollrms"] < best["rollrms"]:
            best, best_lead = r, lead
    row = list(cell) + [mu, round(A0, 3), round(p0 % 360, 1), best_lead]
    for r in (held, best if best else blank()):
        for k in ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms",
                  "axisrms", "fore", "rearp5", "sat"):
            v = r.get(k, float("nan"))
            row.append("" if v is None else (round(v, 4) if isinstance(v, float) else v))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("--mu", type=float, default=0.12)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"hwact_mu{int(round(a.mu * 100)):03d}"
    cl = cells()

    if a.cmd == "count":
        full = len(FREQ) * len(PHI) * len(LEG) * len(HIP) * len(OFF)
        print(f"grid {len(FREQ)}x{len(PHI)}x{len(LEG)}x{len(HIP)}x{len(OFF)} = {full:,}")
        print(f"inside the {CEILING:.0f} deg/s envelope: {len(cl):,} cells")
        print(f"rollouts: {len(cl)} x (1 held + {len(LEADS)} ff) "
              f"= {len(cl) * (1 + len(LEADS)):,}")
        return

    if a.merge:
        rows = []
        for i in range(64):
            p = os.path.join(OUT, f"{tag}.{i}.csv")
            if os.path.exists(p):
                with open(p) as fh:
                    rd = csv.reader(fh)
                    next(rd, None)
                    rows += [r for r in rd if r]
        if not rows:
            raise SystemExit("no shard output")
        i_v = COLS.index("v_net_ff")
        rows.sort(key=lambda r: -(float(r[i_v]) if r[i_v] not in ("", "nan") else -1))
        with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(rows)
        print(f"{len(rows)} cells -> {OUT}/{tag}.csv")
        return

    todo = [c for i, c in enumerate(cl) if i % a.of == a.shard]
    p = os.path.join(OUT, f"{tag}.{a.shard}.csv")
    done = set()
    if os.path.exists(p):
        with open(p) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            done = {tuple(round(float(x), 4) for x in r[:5]) for r in rd if r}
    fh = open(p, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(COLS)
    for c in todo:
        if tuple(round(x, 4) for x in c) in done:
            continue
        row = score(c, a.mu)
        if row is not None:
            w.writerow(row)
            fh.flush()
    fh.close()
    print(f"shard {a.shard}/{a.of}: {len(todo)} cells -> {p}")


if __name__ == "__main__":
    main()
