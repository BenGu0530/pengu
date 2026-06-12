# Hip<->leg PHASE-difference x frequency sweep for Pengu (exploration).
#
# Both leg (crank) and hip are ON, fixed at their isolated-best amplitudes
# (leg=110 deg, hip=12 deg, torso=0). We sweep the phase offset applied to BOTH
# hip signals (phi) against frequency, to find whether hip should lead/lag the
# leg-extension cycle.
#
# Phase convention (leg offsets zeroed here so phi reads cleanly):
#   leg_L built-in phase 0, leg_R 180 (po_a=po_b=0)
#   hip_L built-in phase 180, hip_R 0, both shifted by +phi (po_c=po_d=phi)
#   => relative phase(hip_L, leg_L) = 180 + phi
#      phi=180  -> hip_L IN-PHASE with leg_L   (real-robot style: same sin(s))
#      phi=0    -> hip_L ANTI-PHASE with leg_L (old sim default)
#
# Run from pengu_mujoco/:
#   nohup /home/ben/miniconda3/envs/mujoco/bin/python sweep_phase_freq.py \
#         > phasefreq_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""sweep_phase_freq.py - hip<->leg phase x frequency exploration sweep."""
import os
import csv
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gait_config as gc
import sweep_anchor_validation as sav
from sweep_anchor_validation import run_trial, CSV_FIELDS, SURFACE

# ===================================================================
#  SWEEP CONFIG  (edit here)
# ===================================================================
LEG_AMP   = 110.0   # deg (crank), isolated best
HIP_AMP   = 12.0    # deg, isolated best
TORSO_AMP = 0.0

PHASE = np.arange(0, 360, 15)               # deg, hip phase offset phi (24 pts)
FREQ  = np.round(np.arange(1.00, 2.201, 0.02), 3)  # Hz (61 pts)
SIM_DURATION = 20.0


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(here, "results", f"sweep_phasefreq_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    total = len(PHASE) * len(FREQ)
    print(f"# output dir: {outdir}")
    print(f"# leg={LEG_AMP} hip={HIP_AMP} torso={TORSO_AMP}  "
          f"phase 0-345 step15 ({len(PHASE)})  freq {FREQ[0]}-{FREQ[-1]} ({len(FREQ)})  total={total}")

    # zero the leg phase offsets so phi (hip offset) reads cleanly
    gc.PHASE_OFFSET_A_DEG = 0.0
    gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_E_DEG = 0.0

    Z = np.full((len(PHASE), len(FREQ)), np.nan)
    fell = np.zeros((len(PHASE), len(FREQ)), dtype=bool)
    all_rows = []
    idx = 0
    t0 = time.perf_counter()
    for ip, phi in enumerate(PHASE):
        gc.PHASE_OFFSET_C_DEG = float(phi)
        gc.PHASE_OFFSET_D_DEG = float(phi)
        for jf, f in enumerate(FREQ):
            idx += 1
            anchor = {"name": f"phi{phi:g}", "hip_amp_deg": HIP_AMP,
                      "crank_amp_deg": LEG_AMP, "torso_amp_deg": TORSO_AMP}
            r = run_trial(idx, total, anchor, float(f), SIM_DURATION, print)
            r["phase_deg"] = float(phi)
            all_rows.append(r)
            if r["survived"] and not r["error_msg"]:
                Z[ip, jf] = r["dist_fwd"]
            else:
                fell[ip, jf] = True

    # heatmap
    fig, ax = plt.subplots(figsize=(11, 6))
    Zm = np.ma.masked_invalid(Z)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
    pc = ax.pcolormesh(FREQ, PHASE, Zm, shading="nearest", cmap=cmap)
    if fell.any():
        FF, PP = np.meshgrid(FREQ, PHASE)
        ax.plot(FF[fell], PP[fell], "r.", ms=2, alpha=0.5)
    ax.axhline(180, color="white", ls="--", lw=1)
    ax.text(FREQ[0]+0.01, 184, "phi=180: hip IN-phase with leg (real-style)",
            color="white", fontsize=8, va="bottom")
    ax.text(FREQ[0]+0.01, 4, "phi=0: hip ANTI-phase", color="white", fontsize=8, va="bottom")
    fig.colorbar(pc, ax=ax, label="dist_fwd [m]")
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("hip phase offset phi [deg]")
    ax.set_title(f"hip<->leg phase x frequency  (leg={LEG_AMP:g}, hip={HIP_AMP:g}, torso=0)")
    plt.tight_layout()
    png = os.path.join(outdir, "phasefreq_heatmap.png")
    plt.savefig(png, dpi=130, bbox_inches="tight")

    # csv
    fields = CSV_FIELDS + ["phase_deg"]
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    if np.isfinite(Z).any():
        ip, jf = np.unravel_index(np.nanargmax(Z), Z.shape)
        best = (f"best dist_fwd={Z[ip,jf]:.3f} m at phi={PHASE[ip]:g} deg, "
                f"freq={FREQ[jf]:.3f} Hz  (survived {int(np.isfinite(Z).sum())}/{Z.size})")
        # per-phase best for the in/out-of-phase comparison
        rowbest = np.nanmax(np.where(np.isfinite(Z), Z, -1), axis=1)
    else:
        best = "nothing survived"
        rowbest = []
    dt = time.perf_counter() - t0
    print(f"# DONE {idx} trials  wall={dt:.1f}s")
    print(f"# {best}")
    for ip, phi in enumerate(PHASE):
        if len(rowbest):
            print(f"#   phi={phi:3g} deg : best dist over freq = {rowbest[ip]:+.3f} m")
    print(f"# wrote {png}")


if __name__ == "__main__":
    main()
