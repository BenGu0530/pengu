#!/usr/bin/env python
"""Support-phase figures: single-support duty and double-support travel vs mu.

  cross/support_vs_mu.png    single_frac and ds_move_frac per config, all mu
  cross/support_mu07.png     the mu=0.7 slice Ben asked for, with survivor counts

ds_move_frac = share of travel that happens while BOTH feet are down (shuffling).
single_frac  = share of time in single support.
Only rows with survived=1 are averaged; configs with no survivors are drawn as
gaps and annotated, because a mean over fallen trials is meaningless.
"""
import os, sys, csv
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(_ROOT, "results", "grid4_report")
CROSS = os.path.join(OUT, "cross"); os.makedirs(CROSS, exist_ok=True)
LAB = {"c1": "c1 (κ=0, COM 1.05)", "c3": "c3 (κ=0, COM 1.31)",
       "c4": "c4 (κ=2, COM 1.05)", "c5": "c5 (κ=2, COM 1.20)",
       "c6": "c6 (κ=2, COM 1.31)"}
COL = {"c1": "#1f77b4", "c3": "#2ca02c", "c4": "#ff7f0e", "c5": "#d62728", "c6": "#9467bd"}
MUS = [0.1, 0.3, 0.5, 0.7]


def num(r, k):
    try: return float(r[k])
    except (TypeError, ValueError): return float("nan")


stats = {}
for c in LAB:
    p = os.path.join(OUT, c, "finalists.csv")
    if not os.path.exists(p): continue
    rows = list(csv.DictReader(open(p)))
    stats[c] = {}
    for m in MUS:
        sel = [r for r in rows if abs(num(r, "mu") - m) < 1e-9]
        surv = [r for r in sel if int(num(r, "survived"))]
        good = [r for r in surv if num(r, "single_frac") == num(r, "single_frac")]
        prog = [r for r in surv if num(r, "net_fwd") > 0.05]
        stats[c][m] = dict(
            n=len(sel), n_surv=len(surv), n_prog=len(prog),
            single=float(np.mean([num(r, "single_frac") for r in good])) if good else np.nan,
            ds=float(np.mean([num(r, "ds_move_frac") for r in good])) if good else np.nan,
            clear=float(np.mean([np.nanmean([num(r, "clear_L"), num(r, "clear_R")])
                                 for r in good])) if good else np.nan,
            speed=float(np.mean([num(r, "net_fwd") for r in surv])) if surv else np.nan)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
for c, s in stats.items():
    ax[0].plot(MUS, [s[m]["single"] for m in MUS], "-o", color=COL[c], label=LAB[c], ms=5)
    ax[1].plot(MUS, [s[m]["ds"] for m in MUS], "-o", color=COL[c], label=LAB[c], ms=5)
ax[0].set_ylabel("single_frac  (share of time on ONE foot)")
ax[0].set_title("single-support duty — higher = more step-like")
ax[1].set_ylabel("ds_move_frac  (share of travel on BOTH feet)")
ax[1].set_title("double-support travel — higher = more shuffling")
for a in ax:
    a.set_xlabel("floor friction μ"); a.set_xticks(MUS); a.grid(alpha=0.3)
    a.legend(fontsize=7)
    a.annotate("gaps = no surviving finalists", xy=(0.02, 0.02),
               xycoords="axes fraction", fontsize=7, color="gray")
plt.tight_layout(); plt.savefig(os.path.join(CROSS, "support_vs_mu.png"), dpi=130); plt.close()
print("wrote cross/support_vs_mu.png")

# ---- the mu=0.7 slice ----
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
cs = list(stats)
x = np.arange(len(cs))
sing = [stats[c][0.7]["single"] for c in cs]
ds = [stats[c][0.7]["ds"] for c in cs]
b1 = ax[0].bar(x, [0 if np.isnan(v) else v for v in sing],
               color=[COL[c] for c in cs])
b2 = ax[1].bar(x, [0 if np.isnan(v) else v for v in ds],
               color=[COL[c] for c in cs])
for i, c in enumerate(cs):
    st = stats[c][0.7]
    note = f"{st['n_surv']}/{st['n']} survived\n{st['n_prog']}/{st['n']} progressed"
    for a, vals in ((ax[0], sing), (ax[1], ds)):
        if np.isnan(vals[i]):
            a.text(i, 0.02, "NO\nSURVIVORS", ha="center", va="bottom",
                   fontsize=8, color="crimson", fontweight="bold")
        else:
            a.text(i, vals[i] + 0.02, note, ha="center", va="bottom", fontsize=7)
ax[0].set_ylabel("single_frac"); ax[0].set_title("μ=0.7  single-support duty")
ax[1].set_ylabel("ds_move_frac"); ax[1].set_title("μ=0.7  double-support travel")
for a in ax:
    a.set_xticks(x); a.set_xticklabels([LAB[c] for c in cs], fontsize=7, rotation=12)
    a.grid(axis="y", alpha=0.3); a.set_ylim(0, 0.85)
plt.suptitle("μ=0.7: only c1 (κ=0, COM 1.05) has surviving finalists — and none of them "
             "make net_fwd > 0.05 m/s", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(CROSS, "support_mu07.png"), dpi=130); plt.close()
print("wrote cross/support_mu07.png")

print(f"\n{'config':<20}{'mu':>5}{'surv':>7}{'progress':>10}{'single':>9}{'ds_move':>9}"
      f"{'clear':>9}{'speed':>9}")
for c in cs:
    for m in MUS:
        s = stats[c][m]
        f = lambda v: "-" if v != v else f"{v:.3f}"
        print(f"{LAB[c]:<20}{m:>5}{s['n_surv']:>4}/{s['n']:<2}{s['n_prog']:>7}/{s['n']:<2}"
              f"{f(s['single']):>9}{f(s['ds']):>9}{f(s['clear']):>9}{f(s['speed']):>9}")
