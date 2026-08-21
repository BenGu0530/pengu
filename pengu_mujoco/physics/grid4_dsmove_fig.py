#!/usr/bin/env python
"""ds_move_frac violin plot at an arbitrary mu, using that mu's OWN selection.

Matches the style of cross/ds_move_mu01.png (written by grid4_finalists.py) but reads
cN/finalists_mu<XX>.csv, i.e. gaits selected at that mu, so the comparison is
selection-matched rather than "ice gaits on a grippy floor".

ds_move_frac = share of travel that happens while BOTH feet are loaded.
High = shuffling along; low = travel happens in single support, i.e. real stepping.

usage: python physics/grid4_dsmove_fig.py --mu 0.7 [--compare 0.1]
"""
import os, sys, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "results", "grid4_report")
CROSS = os.path.join(OUT, "cross"); os.makedirs(CROSS, exist_ok=True)
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
KCOL = {0.0: "tab:blue", 2.0: "tab:red"}

ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, required=True)
ap.add_argument("--compare", type=float, default=None,
                help="also draw this mu's own selection as a left panel")
a = ap.parse_args()


def tag(m):
    return f"{m:.1f}".replace("0.", "0")


def series_for(mu):
    """(labels, data, colors) of ds_move_frac for gaits SELECTED at mu, run at mu."""
    fn = "finalists.csv" if abs(mu - 0.1) < 1e-9 else f"finalists_mu{tag(mu)}.csv"
    labels, data, cols, ns = [], [], [], []
    for cfg, (kappa, com) in CONF.items():
        p = os.path.join(OUT, cfg, fn)
        if not os.path.exists(p):
            continue
        v = []
        for r in csv.DictReader(open(p)):
            try:
                if abs(float(r["mu"]) - mu) > 1e-9 or not int(float(r["survived"])):
                    continue
                d = float(r["ds_move_frac"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(d):
                v.append(d)
        if v:
            labels.append(f"{cfg}\nκ={kappa:g} COM{com}")
            data.append(v); cols.append(KCOL[kappa]); ns.append(len(v))
    return labels, data, cols, ns


def draw(ax, mu, labels, data, cols, ns):
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for b, c in zip(parts["bodies"], cols):
        b.set_facecolor(c); b.set_alpha(0.55)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"):
        if k in parts:
            parts[k].set_color("black"); parts[k].set_linewidth(1.0)
    for i, (v, n) in enumerate(zip(data, ns), start=1):
        ax.text(i, 1.02, f"n={n}\nmed {np.median(v):.3f}", ha="center", va="bottom",
                fontsize=7, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.03, 1.0)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"finalists selected at $\\mu$={mu} and run at $\\mu$={mu}",
                 fontsize=10, pad=30)


L, D, C, N = series_for(a.mu)
if not D:
    sys.exit(f"no finalists data for mu={a.mu}")

if a.compare is None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    draw(ax, a.mu, L, D, C, N)
    ax.set_ylabel("ds_move_frac  (travel while both feet loaded)")
    axes = [ax]
else:
    L2, D2, C2, N2 = series_for(a.compare)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), sharey=True)
    draw(axes[0], a.compare, L2, D2, C2, N2)
    draw(axes[1], a.mu, L, D, C, N)
    axes[0].set_ylabel("ds_move_frac  (travel while both feet loaded)")

for ax in axes:
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(0.015, 0.52, "above: more travel shuffling on two feet",
            transform=ax.transAxes, fontsize=7, color="gray")
    ax.text(0.015, 0.44, "below: travel in single support (stepping)",
            transform=ax.transAxes, fontsize=7, color="gray", va="top")

plt.suptitle("shuffle vs stepping — blue κ=0, red κ=2 (survivors only)", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(CROSS, f"ds_move_mu{tag(a.mu)}.png")
plt.savefig(out, dpi=130); plt.close()
print("wrote cross/" + os.path.basename(out))
for lab, v, n in zip(L, D, N):
    print(f"  {lab.replace(chr(10),' '):<20} n={n:<3} median={np.median(v):.4f} "
          f"mean={np.mean(v):.4f} min={min(v):.4f} max={max(v):.4f}")
