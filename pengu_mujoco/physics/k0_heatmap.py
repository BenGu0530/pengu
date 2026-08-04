#!/usr/bin/env python
"""Heatmaps of the complete GRID-3 k0 (Gait 1, kappa=0) landscape + best-gait readout."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1]
OUT = os.path.join(os.path.dirname(CSV), "k0_heatmaps.png")
df = pd.read_csv(CSV)
val = df[df.valid == 1].copy()
print(f"rows={len(df)}  valid={len(val)} ({len(val)/len(df)*100:.1f}%)")

best = val.sort_values("net_fwd_speed", ascending=False).iloc[0]
axc = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]
print("BEST valid gait by net_fwd_speed:")
print("  " + "  ".join(f"{a}={best[a]:g}" for a in axc))
print(f"  net_fwd={best.net_fwd_speed:.4f}  path={best.path_speed:.4f}  "
      f"straight={best.straightness:.3f}  single={best.single_frac:.3f}  "
      f"roll_rms={best.torso_roll_rms:.2f}  mu_p95={best.mu_req_p95:.3f}")


def pivot_max(a, b, metric="net_fwd_speed"):
    """max metric over all other axes for each (a,b) cell (best achievable landscape)."""
    return val.pivot_table(index=b, columns=a, values=metric, aggfunc="max")


panels = [
    ("freq", "hip_phi", "net_fwd_speed", "max net_fwd over leg/hip/hip_off"),
    ("leg_amp", "hip_amp", "net_fwd_speed", "max net_fwd over freq/hip_phi/hip_off"),
    ("freq", "hip_off", "net_fwd_speed", "max net_fwd over hip_phi/leg/hip_amp"),
    ("freq", "hip_phi", "torso_roll_rms", "min torso_roll_rms over leg/hip/hip_off"),
]
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for ax, (a, b, metric, sub) in zip(axs.flat, panels):
    agg = "min" if metric == "torso_roll_rms" else "max"
    P = val.pivot_table(index=b, columns=a, values=metric, aggfunc=agg)
    im = ax.imshow(P.values, aspect="auto", origin="lower", cmap="viridis",
                   extent=[P.columns.min(), P.columns.max(), P.index.min(), P.index.max()])
    ax.set_xlabel(a); ax.set_ylabel(b)
    ax.set_title(f"{metric}\n({sub})", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    if a in axc and b in axc:
        ax.plot(best[a], best[b], "r*", ms=16, mec="w")
fig.suptitle(f"GRID-3 k0 (Gait 1, torso upright) — complete {len(df):,} cells, "
             f"{len(val):,} valid.  red star = best net_fwd gait", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT, dpi=120, bbox_inches="tight")
print(f"# wrote {OUT}")
# emit best params for the renderer
print("BEST_PARAMS " + " ".join(f"{a}={best[a]:g}" for a in axc))
