#!/usr/bin/env python
"""Select top gaits ranked at an ARBITRARY mu, not just mu=0.1.

grid4_report.py always ranks at mu=0.1 (m=0) and writes cN/top_gaits.csv. Finalists
chosen that way collapse at high mu, so "kappa=0 is better on grippy floors" cannot be
tested from them -- the candidates were never picked for that surface.

This writes cN/top_gaits_mu<XX>.csv using the SAME rule as grid4_report.py:
    cells with nbhd_pass >= 0.8 AND pass > 0 at that mu, sorted by net_fwd_mean, top 50.

usage: python physics/grid4_top_at_mu.py --mu 0.7 [--top 50] [cN ...]
"""
import os, sys, csv, gzip, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
GS = os.path.join(_ROOT, "results", "gait_sweep")
OUT = os.path.join(_ROOT, "results", "grid4_report")

FREQS = np.round(np.arange(1.00, 2.0001, 0.01), 2)
PHIS = np.round(np.arange(0, 350.01, 10), 1)
LEGS = [85, 95, 105, 115, 125]; HIPS = [12, 16, 20, 24, 28]; OFFS = [10, 20, 30, 40, 50]
MUS = [0.1, 0.3, 0.5, 0.7]
CONF = ["c1", "c2", "c3", "c4", "c5", "c6"]
fi = {f: i for i, f in enumerate(FREQS)}; pi = {p: i for i, p in enumerate(PHIS)}
li = {float(v): i for i, v in enumerate(LEGS)}; hi = {float(v): i for i, v in enumerate(HIPS)}
oi = {float(v): i for i, v in enumerate(OFFS)}; mi = {m: i for i, m in enumerate(MUS)}

ap = argparse.ArgumentParser()
ap.add_argument("cfgs", nargs="*")
ap.add_argument("--mu", type=float, required=True)
ap.add_argument("--top", type=int, default=50)
a = ap.parse_args()
if a.mu not in mi:
    sys.exit(f"mu must be one of {MUS}")
M = mi[a.mu]
wanted = [c for c in a.cfgs if c in CONF] or CONF


def load(cfg):
    base = os.path.join(GS, f"sweep_grid4_{cfg}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
    path = base + ".gz" if os.path.exists(base + ".gz") else (base if os.path.exists(base) else None)
    if path is None: return None
    P = np.full((4, 101, 36, 5, 5, 5), np.nan, np.float32); NF = np.full_like(P, np.nan)
    n = bad = 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for r in csv.reader(f):
            if r and r[0] == "freq": continue
            if len(r) != 12: bad += 1; continue
            try:
                idx = (mi[float(r[5])], fi[round(float(r[0]), 2)], pi[float(r[1])],
                       li[float(r[2])], hi[float(r[3])], oi[float(r[4])])
                P[idx] = float(r[6]); NF[idx] = float(r[8]); n += 1
            except (KeyError, ValueError):
                bad += 1
    if n < 1818000:
        print(f"  {cfg}: {n} rows (partial) -> skipped"); return None
    print(f"  {cfg}: {n} rows loaded")
    return P, NF


def nbhd(A):
    out = np.zeros_like(A)
    for df in (-2, -1, 0, 1, 2):
        for dp in (-1, 0, 1):
            out += np.roll(np.roll(A, df, axis=1), dp, axis=2)
    out /= 15.0
    out[:, :2] = np.nan; out[:, -2:] = np.nan
    return out


tag = f"{a.mu:.1f}".replace("0.", "0")
print(f"selecting top-{a.top} at mu={a.mu} (same rule as grid4_report: "
      f"nbhd>=0.8 AND pass>0, sorted by net_fwd_mean)\n")
for cfg in wanted:
    d = load(cfg)
    if d is None: continue
    P, NF = d
    N = nbhd(P)
    ok = np.isfinite(N[M]) & (N[M] >= 0.8) & (P[M] > 0)
    cand = np.argwhere(ok)
    rows = sorted(((NF[M][tuple(x)], N[M][tuple(x)], x) for x in cand),
                  key=lambda t: -t[0])[:a.top]
    os.makedirs(os.path.join(OUT, cfg), exist_ok=True)
    out = os.path.join(OUT, cfg, f"top_gaits_mu{tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off",
                    "mu", "net_fwd_mean", "nbhd_pass"])
        for nf_, nb_, x in rows:
            w.writerow([FREQS[x[0]], PHIS[x[1]], LEGS[x[2]], HIPS[x[3]], OFFS[x[4]],
                        a.mu, round(float(nf_), 4), round(float(nb_), 3)])
    print(f"  {cfg}: {int(ok.sum()):>7,} eligible cells -> wrote {os.path.basename(out)} "
          f"({len(rows)} rows, best net_fwd={rows[0][0]:.4f})" if rows
          else f"  {cfg}: NO eligible cells at mu={a.mu}")
