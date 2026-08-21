#!/usr/bin/env python
"""Selection-matched "diagonal" figures: every mu evaluated with its OWN top-20.

Every earlier figure ranked gaits at mu=0.1 and re-ran them elsewhere, which measures
transfer, not capability. Here each mu uses cN/finalists_mu<XX>.csv (mu=0.1 uses
finalists.csv), so each point is "the best this config can do on THAT surface".

  cross/diag_pass_vs_mu.png     pass fraction (survived AND net_fwd>0.05)
  cross/diag_speed_vs_mu.png    mean net_fwd of the passers      <- the discriminator
  cross/diag_support_vs_mu.png  single_frac and ds_move_frac
  cross/diag_matrix.png         2x3 gait x COM grid, speed vs mu

usage: python physics/grid4_diagonal_figs.py
"""
import os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, "results", "grid4_report")
CROSS = os.path.join(OUT, "cross"); os.makedirs(CROSS, exist_ok=True)
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
MUS = [0.1, 0.3, 0.5, 0.7]
COL = {1.05: "#1f77b4", 1.20: "#2ca02c", 1.31: "#d62728"}
LS = {0.0: "--", 2.0: "-"}
MK = {0.0: "s", 2.0: "o"}


def tag(m): return f"{m:.1f}".replace("0.", "0")


def num(r, k):
    try: return float(r[k])
    except (TypeError, ValueError): return float("nan")


def stats(c, mu):
    fn = "finalists.csv" if mu == 0.1 else f"finalists_mu{tag(mu)}.csv"
    p = os.path.join(OUT, c, fn)
    if not os.path.exists(p): return None
    rows = [r for r in csv.DictReader(open(p)) if abs(num(r, "mu") - mu) < 1e-9]
    if not rows: return None
    surv = [r for r in rows if int(num(r, "survived"))]
    prog = [r for r in surv if num(r, "net_fwd") > 0.05]
    f = lambda k, S: float(np.nanmean([num(r, k) for r in S])) if S else float("nan")
    return dict(pas=len(prog) / len(rows), spd=f("net_fwd", prog),
                sf=f("single_frac", surv), ds=f("ds_move_frac", surv),
                st=f("straightness", surv), slip=f("slip_ratio", surv))


D = {c: [stats(c, m) for m in MUS] for c in CONF}
D = {c: v for c, v in D.items() if all(x is not None for x in v)}
print("configs on the diagonal:", ", ".join(D))


def lab(c):
    k, com = CONF[c]
    return f"{c} (κ={k:g}, COM {com})"


def line(key, ylab, title, fname, ylim=None):
    plt.figure(figsize=(7.6, 4.8))
    for c, v in D.items():
        k, com = CONF[c]
        plt.plot(MUS, [x[key] for x in v], LS[k] + MK[k], color=COL[com],
                 label=lab(c), lw=1.9, ms=6, mfc="none" if k == 0.0 else COL[com])
    plt.xlabel("floor friction μ"); plt.ylabel(ylab); plt.title(title, fontsize=10)
    plt.xticks(MUS); plt.grid(alpha=0.3)
    if ylim: plt.ylim(*ylim)
    plt.legend(fontsize=7.5, ncol=2)
    plt.figtext(0.5, 0.005, "solid/filled = κ=2   dashed/hollow = κ=0   "
                "colour = COM ratio   (each μ uses its own top-20)",
                ha="center", fontsize=7, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(CROSS, fname), dpi=130); plt.close()
    print("wrote cross/" + fname)


line("pas", "pass fraction of top-20",
     "selection-matched pass fraction — everything walks when picked for the surface",
     "diag_pass_vs_mu.png", ylim=(0, 1.06))
line("spd", "mean net_fwd of passers [m/s]",
     "selection-matched speed — this is where the configs separate",
     "diag_speed_vs_mu.png")

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.7))
for c, v in D.items():
    k, com = CONF[c]
    ax[0].plot(MUS, [x["sf"] for x in v], LS[k] + MK[k], color=COL[com], label=lab(c),
               lw=1.8, ms=5, mfc="none" if k == 0.0 else COL[com])
    ax[1].plot(MUS, [x["ds"] for x in v], LS[k] + MK[k], color=COL[com], label=lab(c),
               lw=1.8, ms=5, mfc="none" if k == 0.0 else COL[com])
ax[0].set_ylabel("single_frac"); ax[0].set_title("single-support duty (higher = step-like)", fontsize=10)
ax[1].set_ylabel("ds_move_frac"); ax[1].set_title("double-support travel (higher = shuffling)", fontsize=10)
for a_ in ax:
    a_.set_xlabel("floor friction μ"); a_.set_xticks(MUS); a_.grid(alpha=0.3)
    a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(CROSS, "diag_support_vs_mu.png"), dpi=130)
plt.close(); print("wrote cross/diag_support_vs_mu.png")

fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex=True, sharey=True)
for i, k in enumerate([0.0, 2.0]):
    for j, com in enumerate([1.05, 1.20, 1.31]):
        c = next((x for x in CONF if CONF[x] == (k, com)), None)
        ax = axes[i, j]
        if c in D:
            ax.plot(MUS, [x["spd"] for x in D[c]], "-o", color=COL[com], lw=2, ms=6)
            for m, x in zip(MUS, D[c]):
                ax.annotate(f"{x['spd']:.3f}", (m, x["spd"]), fontsize=6.5,
                            textcoords="offset points", xytext=(0, 6), ha="center")
            ax.set_title(f"{c}  κ={k:g}  COM {com}", fontsize=9)
        else:
            ax.set_title(f"κ={k:g} COM {com} — missing", fontsize=9, color="crimson")
        ax.grid(alpha=0.3); ax.set_xticks(MUS)
        if j == 0: ax.set_ylabel(f"κ={k:g}\nnet_fwd [m/s]", fontsize=9)
        if i == 1: ax.set_xlabel("μ", fontsize=9)
plt.suptitle("gait × COM matrix — selection-matched speed of the top-20 passers", fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(CROSS, "diag_matrix.png"), dpi=130); plt.close()
print("wrote cross/diag_matrix.png")

print(f"\n{'cfg':<5}{'κ':<5}{'COM':<6}" + "".join(f"{'μ='+str(m):>9}" for m in MUS)
      + "   (speed of passers)")
for c, v in D.items():
    k, com = CONF[c]
    print(f"{c:<5}{k:<5g}{com:<6}" + "".join(f"{x['spd']:>9.4f}" for x in v))
print("\nfastest per μ:")
for i, m in enumerate(MUS):
    best = max(D, key=lambda c: D[c][i]["spd"])
    order = sorted(D, key=lambda c: -D[c][i]["spd"])
    print(f"  μ={m}: {best} ({D[best][i]['spd']:.4f})   order: {' > '.join(order)}")
