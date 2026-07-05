"""
heatmaps.py - find robust "highlands" in the 6-DOF grid (NOT single best points).
For every pair of the 6 gait DOF, colour = MEAN path_speed over all VALID cells
(averaging over the other 4 dims). Averaging rewards broad plateaus and washes
out lone razor spikes -- so bright, wide regions = robust good regimes to fine-
sweep next. Also prints per-freq robustness so we don't crown a fragile freq.

Run from pengu_mujoco/:
  python physics/heatmaps.py
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_HERE, "..", "results", "gait_sweep",
                   "sweep_v3_p25_freq_leg_amp_hip_amp_torso_amp_hip_phi_torso_phi.csv")
AXES = ["freq", "leg_amp", "hip_amp", "torso_amp", "hip_phi", "torso_phi"]
METRIC = "path_speed"


def main():
    df = pd.read_csv(CSV)
    dfv = df[df["valid"] == 1]
    print(f"# grid rows={len(df)}  valid={len(dfv)} ({100*len(dfv)/len(df):.0f}%)")

    # robustness by freq: mean/max/valid-fraction (don't trust a fragile spike)
    print("\n=== per-freq robustness (valid cells only) ===")
    print(f"{'freq':>5}{'n_valid':>9}{'valid%':>8}{'mean_spd':>10}{'p90_spd':>9}{'max_spd':>9}")
    for fr, g in df.groupby("freq"):
        gv = g[g["valid"] == 1]
        if len(gv) == 0:
            continue
        print(f"{fr:>5.2f}{len(gv):>9}{100*len(gv)/len(g):>7.0f}%{gv[METRIC].mean():>10.3f}"
              f"{gv[METRIC].quantile(0.9):>9.3f}{gv[METRIC].max():>9.3f}")

    pairs = list(itertools.combinations(AXES, 2))    # 15
    fig, axs = plt.subplots(3, 5, figsize=(24, 13))
    for ax, (a, b) in zip(axs.ravel(), pairs):
        pt = dfv.pivot_table(index=b, columns=a, values=METRIC, aggfunc="mean")
        im = ax.pcolormesh(pt.columns.values, pt.index.values, pt.values, cmap="viridis", shading="nearest")
        fig.colorbar(im, ax=ax, label=f"mean {METRIC}")
        ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{a} x {b}", fontsize=10, loc="left")
    fig.suptitle("Robust highlands: mean path_speed over VALID cells (avg over other 4 DOF)  "
                 "-- wide bright = robust regime to fine-sweep", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(_HERE, "..", "results", "gait_sweep", "heatmaps_pathspeed.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\n# wrote {out}")

    # top robust highlands: coarse-bin the 6D space, rank bins by mean path_speed
    print("\n=== top robust regimes (mean path_speed of valid cells in each freq x hip_phi bin) ===")
    hb = dfv.groupby(["freq", "hip_phi"])[METRIC].agg(["mean", "count"]).reset_index()
    hb = hb[hb["count"] >= 5].sort_values("mean", ascending=False).head(8)
    for _, r in hb.iterrows():
        print(f"  freq={r['freq']:.2f}  hip_phi={r['hip_phi']:.0f}  mean_spd={r['mean']:.3f}  n={int(r['count'])}")


if __name__ == "__main__":
    main()
