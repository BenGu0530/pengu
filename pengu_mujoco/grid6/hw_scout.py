"""hw_scout.py — coarse scout over the FULL GRID-5 axis ranges with the hardware model and
the hardware layers, to find where the cap-surviving region of a config actually is
before spending PSC time on a fine grid there.

GRID-5's champions for every config sit at 1.9-2.0 Hz and leg_amp 145-165 -- two to three
times over the 354 deg/s crank ceiling -- so they say nothing about where the executed
optimum lies. This runs hw_sweep.score (HELD + 3 FF [+ PID for kappa=2], hard cap, 56 ms,
COM speed, straightness, clearance) on a 5 x 12 x 5 x 3 x 3 = 2,700-cell grid spanning
the whole GRID-5 range:

    freq    1.2 1.4 1.6 1.8 2.0
    hip_phi 0 .. 330 step 30
    leg_amp 75 95 115 135 155
    hip_amp 16 24 32
    hip_off 20 30 40

    CONFIG=c5 python grid6/hw_scout.py --mu 0.12 --shard 0 --of 2
    CONFIG=c5 python grid6/hw_scout.py --mu 0.12 --merge

Output: results/grid6_hw/<cfg>/scout_<cfg>_mu012.csv, same columns as hw_sweep.
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.append(os.path.dirname(_HERE))

import hw_sweep as hs  # noqa: E402

FREQ = [1.2, 1.4, 1.6, 1.8, 2.0]
PHI = list(range(0, 360, 30))
LEG = [75, 95, 115, 135, 155]
HIP = [16, 24, 32]
OFF = [20, 30, 40]


def cells():
    return [(f, float(p), float(a), float(h), float(o))
            for f in FREQ for p in PHI for a in LEG for h in HIP for o in OFF]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(hs.OUT, exist_ok=True)
    tag = f"scout_{hs.CONFIG}_mu{int(round(a.mu * 100)):03d}"
    if a.merge:
        rows = []
        for i in range(64):
            p = os.path.join(hs.OUT, f"{tag}.{i}.csv")
            if os.path.exists(p):
                with open(p) as fh:
                    rd = csv.reader(fh)
                    next(rd, None)
                    rows += [r for r in rd if r]
        iv = hs.COLS.index("v_net_ff")
        rows.sort(key=lambda r: -(float(r[iv]) if r[iv] not in ("", "nan") else -1))
        with open(os.path.join(hs.OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hs.COLS)
            w.writerows(rows)
        print(f"{len(rows)} cells -> {hs.OUT}/{tag}.csv")
        return
    todo = [c for i, c in enumerate(cells()) if i % a.of == a.shard]
    p = os.path.join(hs.OUT, f"{tag}.{a.shard}.csv")
    done = set()
    if os.path.exists(p):
        with open(p) as fh:
            rd = csv.reader(fh)
            next(rd, None)
            done = {tuple(round(float(x), 4) for x in r[:5]) for r in rd if r}
    fh = open(p, "a", newline="")
    w = csv.writer(fh)
    if not done:
        w.writerow(hs.COLS)
    n = 0
    for c in todo:
        if tuple(round(x, 4) for x in c) in done:
            continue
        row = hs.score(c, a.mu)
        if row is not None:
            w.writerow(row)
            fh.flush()
        n += 1
    fh.close()
    print(f"{hs.CONFIG} shard {a.shard}/{a.of}: {n} cells scored -> {p}")


if __name__ == "__main__":
    main()
