"""rerank.py — re-rank the stage-2 passers by what the pass gate ignores.

The gate asks only whether the robot stayed up, faced where it went, and covered ground.
A gait that scuffs its feet along and lurches at some frequency of its own satisfies all
three, and because scuffing is fast it can top the ranking: the mu=0.5 winner of the first
pass, 1.37/350/75/16/50, has a per-cycle minimum clearance of -5.1 mm (a step where the
foot never rose above its loaded height) and a roll phase that wanders 25.5 deg per cycle
against 2-7 for its neighbours.

So every passing (cell, mu) from stage 2 is re-simulated with the two missing measurements
recorded -- clearance and phase lock, definitions in gait_quality.py -- and written out
with the sweep's own net_fwd alongside. No thresholds are applied here. The output is a
table; where to put the cut is a judgement about what counts as walking, and belongs to
whoever reads it.

    python grid6/rerank.py [--shard i --of n]     -> results/grid6_report/rerank_raw.csv
    python grid6/rerank.py --merge                -> results/grid6_report/c1_ranked.md
"""
import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
os.environ.setdefault("PENGU_MODEL", "hardware_c1")
os.environ.setdefault("CONFIG", "c1")

OUT = os.path.join(ROOT, "results", "grid6_report")
STAGE2 = os.path.join(ROOT, "results", "gait_sweep",
                      "sweep_grid6_c1_c1f_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
COLS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu", "net_fwd",
        "clear_L", "clear_R", "clear_min", "air_L", "air_R",
        "roll_amp", "roll_pur", "roll_drift", "pitch_pur", "v_net"]


def passers():
    out = []
    with open(STAGE2) as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if len(r) > 8 and float(r[6]) > 0.5:
                out.append((float(r[0]), float(r[1]), float(r[2]), float(r[3]),
                            float(r[4]), float(r[5]), float(r[8])))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.merge:
        rows = []
        for i in range(64):
            p = os.path.join(OUT, f"rerank_raw.{i}.csv")
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                rd = csv.reader(fh)
                next(rd, None)
                rows += [r for r in rd if r]
        if not rows:
            raise SystemExit("no shard output found")
        with open(os.path.join(OUT, "rerank_raw.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(sorted(rows, key=lambda r: (float(r[5]), -float(r[6]))))
        print(f"merged {len(rows)} rows -> {OUT}/rerank_raw.csv")
        return

    import gait_quality as gq
    todo = [c for i, c in enumerate(passers()) if i % a.of == a.shard]
    p = os.path.join(OUT, f"rerank_raw.{a.shard}.csv")
    done = set()
    if os.path.exists(p):
        with open(p) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            done = {tuple(round(float(x), 4) for x in r[:6]) for r in rd if r}
    fh = open(p, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(COLS)
    for f, phi, leg, hip, off, mu, net in todo:
        key = tuple(round(x, 4) for x in (f, phi, leg, hip, off, mu))
        if key in done:
            continue
        r = gq.run(f, phi, leg, hip, off, mu)
        if r is None:
            continue
        w.writerow([f, phi, leg, hip, off, mu, round(net, 4),
                    round(r["clear_L"], 1), round(r["clear_R"], 1),
                    round(min(r["clear_L_min"], r["clear_R_min"]), 1),
                    round(r["air_L"], 0), round(r["air_R"], 0),
                    round(r["roll_amp"], 2), round(r["roll_pur"], 0),
                    round(r["roll_drift"], 1), round(r["pitch_pur"], 0),
                    round(r["v_net"], 4)])
        fh.flush()
    fh.close()
    print(f"shard {a.shard}/{a.of}: {len(todo)} assigned -> {p}")


if __name__ == "__main__":
    main()
