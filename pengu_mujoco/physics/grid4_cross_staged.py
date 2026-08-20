#!/usr/bin/env python
"""Cross figures comparing the ABRUPT start against the STAGED (slow) start.

Reads cN/finalists.csv (abrupt) and cN/finalists_staged.csv (staged) and writes
comparison figures to results/grid4_report/cross/:

  start_pass_vs_mu.png     pass fraction vs mu, solid = abrupt, dashed = staged
  start_speed_vs_mu.png    mean net_fwd of the passers, same convention
  start_roll_vs_mu.png     measured torso roll RMS
  start_delta.png          per-config change in pass fraction (staged - abrupt)

"pass" here uses the map's criterion minus the heading term, which is not
carried in finalists.csv:  survived AND net_fwd > 0.05.
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
MUS = [0.1, 0.3, 0.5, 0.7]
COL = {"c1": "#1f77b4", "c3": "#2ca02c", "c4": "#ff7f0e", "c5": "#d62728", "c6": "#9467bd"}


def load(path):
    if not os.path.exists(path):
        return None
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return rows


def agg(rows, mu):
    sel = [r for r in rows if abs(float(r["mu"]) - mu) < 1e-9]
    if not sel:
        return None
    n = len(sel)
    def fnum(r, k):
        v = r[k]
        try: return float(v)
        except (TypeError, ValueError): return float("nan")
    pas = [r for r in sel if int(float(r["survived"])) and fnum(r, "net_fwd") > 0.05]
    roll = [fnum(r, "torso_roll_rms_deg") for r in sel]
    roll = [x for x in roll if x == x]
    return dict(passfrac=len(pas) / n,
                speed=float(np.mean([fnum(r, "net_fwd") for r in pas])) if pas else 0.0,
                roll=float(np.mean(roll)) if roll else float("nan"))


data = {}
for c in LAB:
    A = load(os.path.join(OUT, c, "finalists.csv"))
    S = load(os.path.join(OUT, c, "finalists_staged.csv"))
    if A and S:
        data[c] = (
            [agg(A, m) for m in MUS],
            [agg(S, m) for m in MUS],
        )
if not data:
    sys.exit("no paired finalists.csv / finalists_staged.csv found")
print("configs compared:", ", ".join(data))


def line_fig(key, ylabel, title, fname, logy=False):
    plt.figure(figsize=(7.2, 4.6))
    for c, (A, S) in data.items():
        ya = [a[key] if a else np.nan for a in A]
        ys = [s[key] if s else np.nan for s in S]
        plt.plot(MUS, ya, "-o", color=COL[c], label=f"{LAB[c]} abrupt", lw=1.8, ms=5)
        plt.plot(MUS, ys, "--s", color=COL[c], label=f"{LAB[c]} staged",
                 lw=1.6, ms=5, alpha=0.85, mfc="none")
    if logy:
        plt.yscale("log")
    plt.xlabel("floor friction μ"); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(MUS); plt.grid(alpha=0.3)
    plt.legend(fontsize=6.5, ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(CROSS, fname), dpi=130); plt.close()
    print("wrote cross/" + fname)


line_fig("passfrac", "fraction of top-20 passing",
         "finalist pass fraction — abrupt (solid) vs staged (dashed)",
         "start_pass_vs_mu.png")
line_fig("speed", "mean net_fwd of passers [m/s]",
         "finalist speed — abrupt (solid) vs staged (dashed)",
         "start_speed_vs_mu.png")
line_fig("roll", "measured torso roll RMS [deg]",
         "torso roll RMS — abrupt (solid) vs staged (dashed)",
         "start_roll_vs_mu.png")

# delta bars
plt.figure(figsize=(7.6, 4.4))
w = 0.15
xs = np.arange(len(MUS))
for i, (c, (A, S)) in enumerate(data.items()):
    d = [(S[j]["passfrac"] - A[j]["passfrac"]) if (A[j] and S[j]) else 0.0
         for j in range(len(MUS))]
    plt.bar(xs + (i - len(data) / 2) * w + w / 2, d, w, color=COL[c], label=LAB[c])
plt.axhline(0, color="k", lw=0.8)
plt.xticks(xs, [f"μ={m}" for m in MUS])
plt.ylabel("Δ pass fraction  (staged − abrupt)")
plt.title("does the slow start change the result?  >0 = staged better")
plt.grid(axis="y", alpha=0.3); plt.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(CROSS, "start_delta.png"), dpi=130); plt.close()
print("wrote cross/start_delta.png")

# text table
print(f"\n{'config':<20}{'mu':>6}{'pass A':>9}{'pass S':>9}{'d':>7}"
      f"{'spd A':>8}{'spd S':>8}{'roll A':>8}{'roll S':>8}")
for c, (A, S) in data.items():
    for j, m in enumerate(MUS):
        a, s = A[j], S[j]
        if not (a and s): continue
        print(f"{LAB[c]:<20}{m:>6}{a['passfrac']:>9.2f}{s['passfrac']:>9.2f}"
              f"{s['passfrac']-a['passfrac']:>+7.2f}{a['speed']:>8.4f}{s['speed']:>8.4f}"
              f"{a['roll']:>8.2f}{s['roll']:>8.2f}")
