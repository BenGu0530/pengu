"""plateau.py — rank stage-2 cells by the worst of their 1-unit neighbourhood.

The stage-2 champion at mu=0.5, 1.46/250/75/32/10, is a spike one grid point wide in two
axes at once: leg_amp 73/74/76 and freq 1.45/1.47 all read 0.065-0.072 against its 0.108,
and the spike survives every measurement window tried, so it is in the model rather than
in the readout. The campaign could not see this. Its robustness filter ran on stage 1 at
0.05 Hz and checked freq +-0.05 and hip_phi +-10; the champion came out of stage 2 at
0.01 Hz and nobody re-checked its neighbours at that resolution, and leg_amp's grid step
is 10, so +-1 was never simulated at all.

The robot cannot be aimed at a grid point. Preset 1 commanded leg_amp 75 and executed
74.3 -- a 0.9% tracking error, and 74 is off the spike. So what is wanted is not the
fastest cell but the fastest cell whose whole 1-unit neighbourhood is still walking.

Each candidate is scored by the MINIMUM v_net over itself and its neighbours in leg_amp
(+-1..3) and freq (+-0.01..0.03) -- a cell that falls anywhere in that box scores zero.
Ranking by the minimum is the whole point: ranking by the centre is what produced the
spike.

    python grid6/plateau.py [--mu 0.5] [--top 40] [--shard i --of n]
    python grid6/plateau.py --merge
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

STAGE2 = os.path.join(ROOT, "results", "gait_sweep",
                      "sweep_grid6_c1_c1f_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
OUT = os.path.join(ROOT, "results", "grid6_report")
DLEG = (0, -1, 1, -2, 2, -3, 3)
DFREQ = (0.0, -0.01, 0.01, -0.02, 0.02, -0.03, 0.03)
COLS = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
        "sweep_net", "centre_net", "worst_net", "mean_net", "n_fell",
        "worst_clear", "worst_drift", "worst_at"]


def candidates(mu, top):
    """The `top` fastest passing cells at this mu, one per (phi, leg, hip, off) family so
    the list is not forty consecutive frequencies of the same gait."""
    rows = []
    with open(STAGE2) as fh:
        rd = csv.reader(fh)
        next(rd)
        for r in rd:
            if len(r) > 8 and abs(float(r[5]) - mu) < 1e-9 and float(r[6]) > 0.5:
                rows.append(tuple(float(x) for x in r[:6]) + (float(r[8]),))
    rows.sort(key=lambda r: -r[6])
    seen, out = set(), []
    for r in rows:
        fam = r[1:5]
        if fam in seen:
            continue
        seen.add(fam)
        out.append(r)
        if len(out) >= top:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = f"plateau_mu{int(round(a.mu*10)):02d}"

    if a.merge:
        rows = []
        for i in range(64):
            p = os.path.join(OUT, f"{tag}.{i}.csv")
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                rd = csv.reader(fh)
                next(rd, None)
                rows += [r for r in rd if r]
        if not rows:
            raise SystemExit("no shard output")
        rows.sort(key=lambda r: -float(r[8]))
        with open(os.path.join(OUT, f"{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(COLS)
            w.writerows(rows)
        print(f"{'cell':26s}{'sweep':>8s}{'centre':>8s}{'WORST':>8s}{'mean':>8s}"
              f"{'fell':>6s}{'w.clear':>9s}{'w.drift':>8s}   worst at")
        for r in rows:
            print(f"{r[0]}/{r[1]}/{r[2]}/{r[3]}/{r[4]:<10s}"[:26].ljust(26)
                  + f"{float(r[6]):>8.4f}{float(r[7]):>8.4f}{float(r[8]):>8.4f}"
                  f"{float(r[9]):>8.4f}{r[10]:>6s}{float(r[11]):>9.1f}"
                  f"{float(r[12]):>8.1f}   {r[13]}")
        print(f"\nwrote {OUT}/{tag}.csv")
        print("WORST is the minimum v_net over the cell and its leg_amp +-3 / freq +-0.03\n"
              "neighbours. Rank on that column, not on sweep or centre.")
        return

    import gait_quality as gq
    todo = [c for i, c in enumerate(candidates(a.mu, a.top)) if i % a.of == a.shard]
    p = os.path.join(OUT, f"{tag}.{a.shard}.csv")
    fh = open(p, "w", newline="")
    w = csv.writer(fh)
    w.writerow(COLS)
    for freq, phi, leg, hip, off, mu, snet in todo:
        vals, fell, worst = {}, 0, None
        for dl in DLEG:
            for df in DFREQ:
                f2, l2 = round(freq + df, 2), leg + dl
                r = gq.run(f2, phi, l2, hip, off, mu)
                if r is None:
                    fell += 1
                    vals[(dl, df)] = (0.0, 0.0, 999.0)
                else:
                    vals[(dl, df)] = (r["v_net"],
                                      min(r["clear_L_min"], r["clear_R_min"]),
                                      r["roll_drift"])
        key = min(vals, key=lambda k: vals[k][0])
        v = [x[0] for x in vals.values()]
        w.writerow([freq, phi, leg, hip, off, mu, round(snet, 4),
                    round(vals[(0, 0.0)][0], 4), round(min(v), 4),
                    round(sum(v) / len(v), 4), fell,
                    round(vals[key][1], 1), round(vals[key][2], 1),
                    f"leg{key[0]:+d} freq{key[1]:+.2f}"])
        fh.flush()
    fh.close()
    print(f"shard {a.shard}/{a.of}: {len(todo)} cells x {len(DLEG)*len(DFREQ)} -> {p}")


if __name__ == "__main__":
    main()
