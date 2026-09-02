"""foreaft_scan.py — every stage-2 passer, scored on where the CoM sits fore-and-aft.

The campaign never recorded this axis: gait_sweep logs the CoM against the stance foot
laterally only (`lx = com[0] - cxy[0]`, "forward travel is +y"). On the robot the falls
are backwards, and in the model the CoM of the flashed preset sits behind the loaded feet
64% of the time, reaching 73 mm back, with the excursion peaking exactly at the hip's
swing apex.

Two knobs were already ruled out by hand at 1.46/./75/./20: hip_amp trades the excursion
against speed one for one (32 -> 0.1041 m/s at -8.6 mm; 16 -> 0.0343 at -0.6; 0 -> the
robot does not walk at all), and all 36 hip_phi values are on the same trade-off curve --
none reaches -3 mm with v_net above 0.08, and the deepest excursion barely moves (-62.7 to
-73.5 mm across every phase).

So the question is whether anything ELSE in the passing set escapes it. This runs the whole
stage-2 mu=0.5 passing list and records both numbers from one simulation each.

    python grid6/foreaft_scan.py [--mu 0.5] [--shard i --of n]
    python grid6/foreaft_scan.py --merge
"""
import argparse, csv, math, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
os.environ.setdefault("PENGU_MODEL", "hardware_c1")
os.environ.setdefault("CONFIG", "c1")

STAGE2 = os.path.join(ROOT, "results", "gait_sweep",
                      "sweep_grid6_c1_c1f_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
OUT = os.path.join(ROOT, "results", "grid6_report")
COLS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
        "v_net", "fore_mean", "fore_min", "pct_behind", "base_mm", "pitch_p2p"]


def cells(mu):
    out = []
    with open(STAGE2) as fh:
        rd = csv.reader(fh); next(rd)
        for r in rd:
            if len(r) > 8 and abs(float(r[5]) - mu) < 1e-9 and float(r[6]) > 0.5:
                out.append(tuple(float(x) for x in r[:5]))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"foreaft_mu{int(round(a.mu*10)):02d}"

    if a.merge:
        rows = []
        for i in range(64):
            p = os.path.join(OUT, f"{tag}.{i}.csv")
            if os.path.exists(p):
                with open(p) as fh:
                    rd = csv.reader(fh); next(rd, None)
                    rows += [r for r in rd if r]
        if not rows:
            raise SystemExit("no shard output")
        rows.sort(key=lambda r: -float(r[6]))
        with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(COLS); w.writerows(rows)
        print(f"{len(rows)} cells -> {OUT}/{tag}.csv")
        return

    import com_foreaft as cf
    todo = [c for i, c in enumerate(cells(a.mu)) if i % a.of == a.shard]
    p = os.path.join(OUT, f"{tag}.{a.shard}.csv")
    done = set()
    if os.path.exists(p):
        with open(p) as fh:
            rd = csv.reader(fh); next(rd, None)
            done = {tuple(round(float(x), 4) for x in r[:5]) for r in rd if r}
    fh = open(p, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(COLS)
    for c in todo:
        if tuple(round(x, 4) for x in c) in done:
            continue
        r = cf.run(*c, a.mu)
        if r is None:
            continue
        f = r["fore"]
        w.writerow(list(c) + [a.mu, round(r["v_net"], 4), round(float(f.mean()), 1),
                              round(float(f.min()), 1), round(100 * float((f < 0).mean()), 1),
                              round(float(r["base"].mean()), 1),
                              round(float(np.ptp(r["pitch"])), 1)])
        fh.flush()
    fh.close()
    print(f"shard {a.shard}/{a.of}: {len(todo)} assigned -> {p}")


if __name__ == "__main__":
    main()
