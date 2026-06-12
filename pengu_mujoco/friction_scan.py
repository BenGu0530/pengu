# Friction-robustness scan: how each optimized gait degrades as the floor gets
# slippery. For each gait, sweep floor sliding-mu x frequency, color by CLEAN
# forward speed (gray = not clean / fell). Feet stay at 0.9 (fixed); floor mu is
# the variable, and min(foot,floor)=floor governs the contact while mu<0.9.
#
# Run from pengu_mujoco/:
#   python friction_scan.py
"""friction_scan.py - floor-friction x frequency robustness of chosen gaits."""
import os
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
from friction_utils import set_floor_friction, SURFACES

LEGTORSO = dict(leg_amp=72.1, hip_amp=0.0, torso_amp=7.3, hip_phi=0.0, torso_phi=283.4, freq=1.51)
FULL     = dict(leg_amp=100.9, hip_amp=21.7, torso_amp=2.9, hip_phi=202.7, torso_phi=66.5, freq=1.69)

MU = [0.06, 0.10, 0.14, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
GAITS = [("legtorso", LEGTORSO, np.round(np.arange(1.20, 1.851, 0.05), 3)),
         ("full",     FULL,     np.round(np.arange(1.40, 2.001, 0.05), 3))]


def main():
    model = make_model()
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"friction_{stamp}")
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    t0 = time.perf_counter(); nfev = 0
    for ax, (name, base, freqs) in zip(axes, GAITS):
        Z = np.full((len(MU), len(freqs)), np.nan)
        fell = np.zeros_like(Z, dtype=bool)
        print(f"# {name} (base freq={base['freq']}):")
        for i, mu in enumerate(MU):
            model.geom_friction[floor_id, 0] = mu
            best_at_basefreq = None
            for j, f in enumerate(freqs):
                p = dict(base); p["freq"] = float(f)
                m = evaluate(model, data, aid, jadr, root_id, p)
                nfev += 1
                clean = (m["survived"] and np.isfinite(m["roll_amp"]) and m["roll_amp"] < ROLL_CAP
                         and np.isfinite(m["pitch_off"]) and abs(m["pitch_off"]) < PITCH_CAP)
                if clean:
                    Z[i, j] = m["speed"]
                if not m["survived"]:
                    fell[i, j] = True
                if abs(f - base["freq"]) < 1e-6:
                    best_at_basefreq = (m["speed"], m["survived"], m["roll_amp"], m["pitch_off"])
            row_best = np.nanmax(Z[i]) if np.isfinite(Z[i]).any() else float("nan")
            bf = best_at_basefreq
            print(f"#   mu={mu:.2f}: best_clean_speed(any f)={row_best:.3f}  "
                  f"@base_freq speed={bf[0]:.3f} surv={int(bf[1])}" if bf else f"#   mu={mu:.2f}")
        Zm = np.ma.masked_invalid(Z)
        cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
        pc = ax.pcolormesh(freqs, MU, Zm, shading="nearest", cmap=cmap)
        if fell.any():
            FF, MM = np.meshgrid(freqs, MU)
            ax.plot(FF[fell], MM[fell], "r.", ms=3, alpha=0.5)
        ax.axhline(0.7, color="white", ls="--", lw=1)
        ax.plot([base["freq"]], [0.7], "w*", ms=15, mec="k")
        fig.colorbar(pc, ax=ax, label="clean speed [m/s]")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("floor sliding mu")
        ax.set_title(f"{name}: floor-mu x freq  (baseline mu=0.7 dashed; * = opt)")
    plt.tight_layout()
    png = os.path.join(outdir, "friction_robustness.png")
    plt.savefig(png, dpi=120, bbox_inches="tight")
    print(f"# surfaces: {SURFACES}")
    print(f"# DONE {nfev} evals  wall={time.perf_counter()-t0:.1f}s  wrote {png}")


if __name__ == "__main__":
    main()
