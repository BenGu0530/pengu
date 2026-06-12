# 2D basin slices around the CMA-ES optima (clean-speed landscape / robustness).
# Holds a base gait fixed, varies two axes, colors by CLEAN forward speed
# (speed where survived & roll<15 & |pitch|<25, else gray). Marks the base point.
#
# Run from pengu_mujoco/:
#   MUJOCO_GL=egl is NOT needed (no render); just:
#   python slice_2d.py
"""slice_2d.py - 2D clean-speed basin slices around chosen gaits."""
import os
import csv
import time
from datetime import datetime
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_gait import make_model, evaluate, ROLL_CAP, PITCH_CAP
from gait_config import build_ids

LEGTORSO = dict(leg_amp=72.1, hip_amp=0.0, torso_amp=7.3, hip_phi=0.0, torso_phi=283.4, freq=1.51)
FULL     = dict(leg_amp=100.9, hip_amp=21.7, torso_amp=2.9, hip_phi=202.7, torso_phi=66.5, freq=1.69)

# (title, base, (ax1_name, ax1_vals), (ax2_name, ax2_vals))
SLICES = [
    ("legtorso: leg_amp x freq", LEGTORSO,
     ("leg_amp", np.linspace(40, 110, 15)), ("freq", np.round(np.arange(1.20, 1.901, 0.05), 3))),
    ("legtorso: torso_amp x torso_phase", LEGTORSO,
     ("torso_amp", np.linspace(0, 20, 15)), ("torso_phi", np.arange(0, 360, 20))),
    ("full: leg_amp x freq", FULL,
     ("leg_amp", np.linspace(60, 120, 15)), ("freq", np.round(np.arange(1.30, 2.001, 0.05), 3))),
    ("full: hip_amp x hip_phase", FULL,
     ("hip_amp", np.linspace(0, 30, 16)), ("hip_phi", np.arange(0, 360, 20))),
]


def main():
    model = make_model()
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"slices_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.ravel()
    t0 = time.perf_counter()
    nfev = 0
    for ax, (title, base, (a1, v1), (a2, v2)) in zip(axes, SLICES):
        Z = np.full((len(v1), len(v2)), np.nan)   # clean speed
        fell = np.zeros_like(Z, dtype=bool)
        nclean = 0
        for i, x1 in enumerate(v1):
            for j, x2 in enumerate(v2):
                p = dict(base); p[a1] = float(x1); p[a2] = float(x2)
                m = evaluate(model, data, aid, jadr, root_id, p)
                nfev += 1
                clean = (m["survived"] and np.isfinite(m["roll_amp"]) and m["roll_amp"] < ROLL_CAP
                         and np.isfinite(m["pitch_off"]) and abs(m["pitch_off"]) < PITCH_CAP)
                if clean:
                    Z[i, j] = m["speed"]; nclean += 1
                if not m["survived"]:
                    fell[i, j] = True
        Zm = np.ma.masked_invalid(Z)
        cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
        pc = ax.pcolormesh(v2, v1, Zm, shading="nearest", cmap=cmap)
        if fell.any():
            XX, YY = np.meshgrid(v2, v1)
            ax.plot(XX[fell], YY[fell], "r.", ms=3, alpha=0.5)
        ax.plot([base[a2]], [base[a1]], "w*", ms=16, mec="k", label="optimum")
        fig.colorbar(pc, ax=ax, label="clean speed [m/s]")
        ax.set_xlabel(a2); ax.set_ylabel(a1)
        ax.set_title(f"{title}   (clean {nclean}/{Z.size})", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        print(f"# {title}: clean {nclean}/{Z.size}  best_clean_speed={np.nanmax(Z) if np.isfinite(Z).any() else float('nan'):.3f}")

    plt.tight_layout()
    png = os.path.join(outdir, "basin_slices.png")
    plt.savefig(png, dpi=120, bbox_inches="tight")
    dt = time.perf_counter() - t0
    print(f"# DONE {nfev} evals  wall={dt:.1f}s  wrote {png}")


if __name__ == "__main__":
    main()
