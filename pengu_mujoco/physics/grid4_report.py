#!/usr/bin/env python
"""GRID-4 living report: eats every COMPLETE sweep_grid4_c*.csv(.gz) it finds and
regenerates the full analysis package into results/grid4_report/.

Outputs
  REPORT.md                     summary tables (integrity, per-mu stats, freq-edge, overlap)
  heatmap_cN.png                per config: 4 mu panels, neighborhood-mean pass, best slice
  volume_vs_mu.png              robust-region volume (nbhd>=0.8) vs mu, one line per config
  passfrac_vs_mu.png            raw pass>0 share vs mu
  speed_vs_mu.png               best net_fwd among passers vs mu
  overlap_mu01.png              Jaccard overlap of pass>0 gait cells at mu=0.1
  top_gaits_cN.csv              top-50 robust+fast cells per config (topup / demo input)

Rerun any time; configs appear as their data lands. usage:
  python physics/grid4_report.py
"""
import os, sys, gzip, csv, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "results", "grid4_report"); os.makedirs(OUT, exist_ok=True)
GS = os.path.join(_ROOT, "results", "gait_sweep")

FREQS = np.round(np.arange(1.00, 2.0001, 0.01), 2)
PHIS = np.round(np.arange(0, 350.01, 10), 1)
LEGS = [85, 95, 105, 115, 125]; HIPS = [12, 16, 20, 24, 28]; OFFS = [10, 20, 30, 40, 50]
MUS = [0.1, 0.3, 0.5, 0.7]
CONF = {"c1": ("κ=0", 1.05), "c2": ("κ=0", 1.20), "c3": ("κ=0", 1.31),
        "c4": ("κ=2", 1.05), "c5": ("κ=2", 1.20), "c6": ("κ=2", 1.31)}
fi = {f: i for i, f in enumerate(FREQS)}; pi = {p: i for i, p in enumerate(PHIS)}
li = {float(v): i for i, v in enumerate(LEGS)}; hi = {float(v): i for i, v in enumerate(HIPS)}
oi = {float(v): i for i, v in enumerate(OFFS)}; mi = {m: i for i, m in enumerate(MUS)}


def load(cfg):
    base = os.path.join(GS, f"sweep_grid4_{cfg}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
    path = base + ".gz" if os.path.exists(base + ".gz") else (base if os.path.exists(base) else None)
    if path is None:
        return None
    P = np.full((4, 101, 36, 5, 5, 5), np.nan, np.float32)
    NF = np.full_like(P, np.nan)
    n = bad = 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for r in csv.reader(f):
            if r and r[0] == "freq":
                continue
            if len(r) != 12:
                bad += 1; continue
            try:
                idx = (mi[float(r[5])], fi[round(float(r[0]), 2)], pi[float(r[1])],
                       li[float(r[2])], hi[float(r[3])], oi[float(r[4])])
                P[idx] = float(r[6]); NF[idx] = float(r[8]); n += 1
            except (KeyError, ValueError):
                bad += 1
    if n < 1818000:                                     # partial -> skip (report wants apples)
        print(f"  {cfg}: {n} rows (partial) -> skipped")
        return None
    print(f"  {cfg}: {n} rows, {bad} malformed -> loaded")
    return P, NF


def nbhd(A):                                            # 15-cell mean, freq±2 phi±1(wrap)
    out = np.zeros_like(A)
    for df in (-2, -1, 0, 1, 2):
        for dp in (-1, 0, 1):
            out += np.roll(np.roll(A, df, axis=1), dp, axis=2)
    out /= 15.0
    out[:, :2] = np.nan; out[:, -2:] = np.nan
    return out


print("loading configs:")
data = {c: d for c, d in ((c, load(c)) for c in CONF) if d is not None}
if not data:
    sys.exit("no complete configs found")
N = {c: nbhd(P) for c, (P, NF) in data.items()}

md = io.StringIO()
md.write("# GRID-4 report\n\nconfigs included: " + ", ".join(
    f"{c} ({CONF[c][0]}, COM {CONF[c][1]})" for c in data) + "\n")

# ---- per-config per-mu table + top gaits csv ----
md.write("\n## Per-config, per-mu\n\n| config | mu | pass>0 % | mean pass | nbhd>=0.8 | best net_fwd |\n|---|---|---|---|---|---|\n")
for c, (P, NF) in data.items():
    for m, mu in enumerate(MUS):
        p = P[m]; v = N[c][m]
        passers = p > 0
        best = np.nanmax(np.where(passers, NF[m], np.nan))
        md.write(f"| {c} | {mu} | {100*np.nanmean(passers):.1f} | {np.nanmean(p):.4f} | "
                 f"{int(np.nansum(v >= 0.8))} | {best:.3f} |\n")
    # top-50 robust+fast at mu=0.1 -> csv
    m = 0
    ok = np.isfinite(N[c][m]) & (N[c][m] >= 0.8) & (data[c][0][m] > 0)
    cand = np.argwhere(ok)
    rows = sorted(((data[c][1][m][tuple(x)], N[c][m][tuple(x)], x) for x in cand),
                  key=lambda t: -t[0])[:50]
    with open(os.path.join(OUT, f"top_gaits_{c}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off",
                                       "mu", "net_fwd_mean", "nbhd_pass"])
        for nf_, nb_, x in rows:
            w.writerow([FREQS[x[0]], PHIS[x[1]], LEGS[x[2]], HIPS[x[3]], OFFS[x[4]],
                        0.1, round(float(nf_), 4), round(float(nb_), 3)])

# ---- cross-config curves ----
for fname, title, fn in [
    ("volume_vs_mu.png", "robust-region volume (cells with 15-cell mean pass ≥ 0.8)",
     lambda c, m: int(np.nansum(N[c][m] >= 0.8))),
    ("passfrac_vs_mu.png", "share of gait cells with pass > 0",
     lambda c, m: 100 * np.nanmean(data[c][0][m] > 0)),
    ("speed_vs_mu.png", "best net_fwd among passers [m/s]",
     lambda c, m: float(np.nanmax(np.where(data[c][0][m] > 0, data[c][1][m], np.nan)))),
]:
    plt.figure(figsize=(7, 4.5))
    for c in data:
        ys = [fn(c, m) for m in range(4)]
        plt.plot(MUS, ys, "o-", label=f"{c} ({CONF[c][0]}, COM {CONF[c][1]})")
    plt.xlabel("floor friction $\\mu$"); plt.title(title)
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    if "volume" in fname: plt.yscale("symlog", linthresh=10)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()

# ---- freq-edge pressure ----
md.write("\n## freq-edge pressure (share of nbhd>=0.8 cells at freq >= 1.9, mu=0.1)\n\n")
md.write("| config | share |\n|---|---|\n")
for c in data:
    v = N[c][0]
    hot = np.isfinite(v) & (v >= 0.8)
    if hot.sum():
        edge = hot[fi[1.9]:].sum() / hot.sum()
        md.write(f"| {c} | {100*edge:.1f}% |\n")

# ---- overlap matrix at mu=0.1 (pass>0 gait cells, Jaccard) ----
sets = {c: set(map(tuple, np.argwhere(data[c][0][0] > 0))) for c in data}
cs = list(data)
Mx = np.zeros((len(cs), len(cs)))
for a in range(len(cs)):
    for b in range(len(cs)):
        A, B = sets[cs[a]], sets[cs[b]]
        Mx[a, b] = len(A & B) / max(1, len(A | B))
plt.figure(figsize=(5.5, 4.5))
plt.imshow(Mx, vmin=0, vmax=1, cmap="magma")
plt.xticks(range(len(cs)), cs); plt.yticks(range(len(cs)), cs)
for a in range(len(cs)):
    for b in range(len(cs)):
        plt.text(b, a, f"{Mx[a,b]:.2f}", ha="center", va="center",
                 color="w" if Mx[a, b] < 0.6 else "k", fontsize=9)
plt.title("Jaccard overlap of passing gait cells @ $\\mu$=0.1")
plt.colorbar(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "overlap_mu01.png"), dpi=130); plt.close()
md.write("\n## overlap\n\nsee overlap_mu01.png — Jaccard of pass>0 cell sets at mu=0.1.\n")

# ---- per-config heatmap wall ----
for c in data:
    v = N[c][0]
    best = np.unravel_index(np.nanargmax(np.nan_to_num(v, nan=-1)), v.shape)
    a, b, o = best[2], best[3], best[4]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), sharey=True)
    for m, (ax, mu) in enumerate(zip(axes, MUS)):
        im = ax.imshow(N[c][m][:, :, a, b, o].T, aspect="auto", origin="lower",
                       extent=[1.0, 2.0, 0, 350], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"$\\mu$={mu}"); ax.set_xlabel("freq [Hz]")
    axes[0].set_ylabel("hip_phi [deg]")
    fig.suptitle(f"{c} ({CONF[c][0]}, COM {CONF[c][1]}) — nbhd-mean pass, "
                 f"slice leg={LEGS[a]} hip={HIPS[b]} off={OFFS[o]}")
    fig.colorbar(im, ax=axes, shrink=0.85)
    plt.savefig(os.path.join(OUT, f"heatmap_{c}.png"), dpi=130, bbox_inches="tight")
    plt.close()

open(os.path.join(OUT, "REPORT.md"), "w").write(md.getvalue())
print(f"\nwrote {OUT}/REPORT.md + figures + top_gaits_*.csv for: {', '.join(data)}")
