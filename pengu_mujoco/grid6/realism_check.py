"""realism_check.py — re-score GRID-5 cells with the two hardware layers the map lacks.

The GRID-5 maps drive ideal actuators and an ideal torso loop. The robot has neither:
the leg servos cannot exceed 354 deg/s (twelve bench points, 2026-08-30) and the torso
command reaches the joint 56 ms late (pengu-A/-B/-10). hw_sweep.rollout (ff_sweep's copy)
models both, so this script only chooses which layers are on and which model/config the
rollout runs on, then reports the same cell under

    ideal   no cap, no lag             (the map's assumption, under ff_sweep's protocol)
    lag     torso 56 ms only
    act     354 deg/s hard cap on the four leg actuators only
    both    the robot

Speed is the whole-body COM's net displacement over the window; each row also carries
straightness, the per-cycle foot clearance and a clear_ok flag (>= 10 mm), so nothing has
to be re-simulated after the download. Torso clamp is the one flashed for that build:
c1 25 deg, c6 45 deg (pengu_champ, 2026-08-29).

Config selection follows grid5_sweep: base model pengu1_31, COM slid to the config's
target, kappa from the config table, torso in PID mode.

    CONFIG=c6 python grid6/realism_check.py --cells 1.67/340/95/24/20 --mu 0.1 0.12
    CONFIG=c6 python grid6/realism_check.py --from-map --top 500 --variants both \
        --shard 0 --of 8

--from-map takes every passing (cell, mu) row of the config's GRID-5 map (or the top N
per mu by net_fwd_mean), runs the chosen variants, and writes one row per (cell, mu,
variant). Sharded runs write results/grid6_report/realism_<config>.<shard>.csv; --merge
joins them.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "grid5"))

CONFIGS = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
           "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31),
           "c7": (0.0, 1.10), "c8": (0.0, 1.40), "c9": (2.0, 1.10), "c10": (2.0, 1.40)}
CONFIG = os.environ.get("CONFIG", "c6").lower()
KAPPA, COM_TARGET = CONFIGS[CONFIG]
os.environ["PENGU_MODEL"] = "1.31"           # same base as grid5_sweep, every config

import mujoco                                 # noqa: E402
import hw_sweep as fsw                        # noqa: E402  (ff_sweep copy: hard cap, COM speed, clamp)
import gait_config as gc                      # noqa: E402

# (torso delay s, leg velocity cap deg/s, leg one-pole s) -- hard cap only, per Ben 2026-09-08
VARIANTS = {"ideal": (0.0, 1e9, 0.0), "lag": (0.056, 1e9, 0.0),
            "act": (0.0, 354.0, 0.0), "both": (0.056, 354.0, 0.0)}
KEYS = ("fell", "v_net", "straight", "clear", "clear_ok", "drift", "rollrms", "axisrms",
        "sat", "fore", "rearp5")
# torso clamp as flashed for that build: c1 firmware 25 deg; c6 (pengu_champ, 2026-08-29) 45
TORSO_CLAMP = {"c1": 25.0, "c6": 45.0}.get(CONFIG, 25.0)
OUT = os.path.join(ROOT, "results", "grid6_report")
MAP = os.path.join(ROOT, "results", "gait_sweep",
                   f"sweep_grid5_{CONFIG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")

_slide = {}


# grid5_sweep.com_ratio_of / apply_com_variant, copied rather than imported: grid5_sweep
# asserts on grid5's gait_sweep at import and ff_sweep runs on grid6's.
def com_ratio_of(model):
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easyaxis")
    return float(d.subtree_com[1][2]) / float(d.xpos[aid][2])


def apply_com_variant(model, target):
    """Slide easytorso's inertial COM along world-up until the neutral-stand COM ratio
    hits `target`. Masses/geometry untouched."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    up = d.xmat[tid].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    ip0 = model.body_ipos[tid].copy()

    def ratio_at(s):
        model.body_ipos[tid] = ip0 + s * up
        return com_ratio_of(model)

    lo, hi = -0.30, 0.30
    assert ratio_at(lo) < target < ratio_at(hi), (target, ratio_at(lo), ratio_at(hi))
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if ratio_at(mid) < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    got = ratio_at(s)
    assert abs(got - target) < 1e-3, (got, target)
    return s, got


class _Mujoco:
    """ff_sweep.rollout builds its model from `mujoco.MjModel.from_xml_path`; route that
    through grid5's COM slide so the cell runs on the config's model, not the base."""

    def __getattr__(self, k):
        return getattr(mujoco, k)

    class MjModel:
        @staticmethod
        def from_xml_path(path):
            m = mujoco.MjModel.from_xml_path(path)
            s, got = apply_com_variant(m, COM_TARGET)
            _slide.setdefault("mm", s * 1000.0)
            _slide.setdefault("got", got)
            return m


def run_variant(cell, mu, variant, mode="pid"):
    lag, rate, tau = VARIANTS[variant]
    fsw.SERVO_LAG, fsw.LEG_RATE, fsw.LEG_TAU = lag, rate, tau
    fsw.TORSO_CLAMP_DEG = TORSO_CLAMP
    f, phi, leg, hip, off = cell
    t0 = time.time()
    r = fsw.rollout(f, phi, leg, hip, off, mu, mode, kappa=KAPPA)
    r["secs"] = time.time() - t0
    return r


def fmt(r):
    if r.get("fell") is not None:
        return f"FELL at {r['fell']:.1f}s"
    return (f"v_net {r['v_net']:.4f}  straight {r['straight']:.2f}  axis {r['axisrms']:5.2f}"
            f"  torso {r['rollrms']:5.2f}  clear {r['clear']:5.1f}mm{'' if r['clear_ok'] else '!'}"
            f"  drift {r['drift']:5.1f}  sat {r['sat']:4.1f}%")


def row_of(cell, mu, variant, r):
    row = list(cell) + [mu, variant]
    for k in KEYS:
        v = r.get(k, float("nan"))
        row.append("" if v is None else (round(v, 4) if isinstance(v, float) else v))
    return row


def cells_from_map(top, mus):
    import pandas as pd
    cols = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu", "pass_rate",
            "net_fwd_mean"]
    path = MAP if os.path.exists(MAP) else MAP + ".gz"
    df = pd.read_csv(path, usecols=cols)
    df = df[df.pass_rate >= 1.0]
    if mus:
        df = df[df.mu.round(3).isin([round(m, 3) for m in mus])]
    if top:
        df = df.sort_values("net_fwd_mean", ascending=False).groupby("mu").head(top)
    df = df.sort_values(["mu", "net_fwd_mean"], ascending=[True, False])
    return [((float(a), float(b), float(c), float(d), float(e)), float(m), float(v))
            for a, b, c, d, e, m, v in
            df[["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
                "net_fwd_mean"]].itertuples(index=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=[], help="freq/phi/leg/hip/off ...")
    ap.add_argument("--mu", nargs="*", type=float, default=[])
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--mode", default="pid")
    ap.add_argument("--from-map", action="store_true")
    ap.add_argument("--cells-file", default="",
                    help="csv with freq,hip_phi,leg_amp,hip_amp,hip_off (one cell per row); "
                         "runs every cell at every --mu, ignoring the map")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"realism_{CONFIG}"
    hdr = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu", "variant"] + list(KEYS)

    if a.merge:
        rows = []
        for i in range(256):
            p = os.path.join(OUT, f"{tag}.{i}.csv")
            if os.path.exists(p):
                with open(p) as fh:
                    rd = csv.reader(fh)
                    next(rd, None)
                    rows += [r for r in rd if r]
        with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr)
            w.writerows(rows)
        print(f"merged {len(rows)} rows -> {OUT}/{tag}.csv")
        return

    fsw.mujoco = _Mujoco()

    if a.cells_file:
        if not a.mu:
            raise SystemExit("--cells-file needs --mu")
        todo = []
        with open(a.cells_file) as fh:
            for r in csv.DictReader(fh):
                cell = tuple(float(r[k]) for k in
                             ("freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"))
                for mu in a.mu:
                    todo.append((cell, float(mu), float("nan")))
        a.from_map = True

    if a.from_map:
        if not a.cells_file:
            todo = cells_from_map(a.top, a.mu)
        todo = [x for i, x in enumerate(todo) if i % a.of == a.shard]
        path = os.path.join(OUT, f"{tag}.{a.shard}.csv")
        done = set()
        if os.path.exists(path):
            with open(path) as fh:
                rd = csv.reader(fh)
                next(rd, None)
                done = {tuple(r[:7]) for r in rd if r}
        fh = open(path, "a", newline="")
        w = csv.writer(fh)
        if not done:
            w.writerow(hdr)
        print(f"{CONFIG} kappa={KAPPA} com={COM_TARGET}  shard {a.shard}/{a.of}: "
              f"{len(todo)} (cell,mu) x {a.variants}, {len(done)} already done")
        t0 = time.time()
        n = 0
        for cell, mu, vmap in todo:
            for variant in a.variants:
                key = tuple(str(x) for x in row_of(cell, mu, variant, {})[:7])
                if key in done:
                    continue
                r = run_variant(cell, mu, variant, a.mode)
                w.writerow(row_of(cell, mu, variant, r))
                fh.flush()
                n += 1
                if n % 20 == 0:
                    print(f"  {n} rollouts, {(time.time()-t0)/n:.1f} s each", flush=True)
        fh.close()
        print(f"shard {a.shard}: {n} rollouts in {(time.time()-t0)/60:.1f} min -> {path}")
        return

    if not a.cells or not a.mu:
        raise SystemExit("give --cells and --mu, or --from-map")
    print(f"{CONFIG}: kappa={KAPPA} com_target={COM_TARGET} torso={a.mode} clamp={TORSO_CLAMP:.0f}"
          f"   layers: torso lag 56 ms, legs 354 deg/s hard cap\n")
    rows = []
    for spec in a.cells:
        cell = tuple(float(x) for x in spec.split("/"))
        for mu in a.mu:
            print(f"{spec}  mu={mu}")
            for variant in a.variants:
                r = run_variant(cell, mu, variant, a.mode)
                rows.append(row_of(cell, mu, variant, r))
                print(f"   {variant:6s} {fmt(r)}   ({r['secs']:.0f}s)", flush=True)
            print()
    if _slide:
        print(f"model: base 1.31 slid {_slide['mm']:+.2f} mm -> com ratio {_slide['got']:.4f}")
    p = os.path.join(OUT, f"{tag}_cells.csv")
    with open(p, "a", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(hdr)
        w.writerows(rows)
    print(f"appended {len(rows)} rows -> {p}")


if __name__ == "__main__":
    main()
