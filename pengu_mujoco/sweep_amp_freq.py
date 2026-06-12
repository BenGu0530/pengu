# Isolated 2D amplitude x frequency sweeps for Pengu (exploration, not validation).
#
# For each DOF we LOCK the other two motors to 0 and sweep (amplitude x frequency),
# so we see each DOF's own contribution and how its resonance moves with amplitude:
#   - leg-only   : crank amp varies,  hip=0, torso=0   (extends crank_only)
#   - hip-only   : hip amp varies,    crank=0, torso=0 (extends hip_only)
#   - torso-only : torso amp varies,  crank=0, hip=0
#
# Reuses run_trial() from sweep_anchor_validation.py (same controller, friction,
# fall/roll/pitch metrics) by building a one-off "anchor" dict per grid point.
#
# Run from pengu_mujoco/:
#   nohup /home/ben/miniconda3/envs/mujoco/bin/python sweep_amp_freq.py \
#         > ampfreq_$(date +%Y%m%d_%H%M%S).log 2>&1 &
"""sweep_amp_freq.py - isolated amplitude x frequency exploration sweeps."""
import os
import csv
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sweep_anchor_validation as sav
from sweep_anchor_validation import run_trial, CSV_FIELDS, SURFACE

# ===================================================================
#  SWEEP CONFIG  (edit here)
# ===================================================================
FREQ = np.round(np.arange(1.00, 2.201, 0.01), 3)   # Hz (0.01 is enough; see freq sweep)

# Each DOF: (label, key in anchor dict, amplitude values [deg])
DOFS = [
    ("leg",   "crank_amp_deg", [10, 20, 30, 40, 50, 60, 73, 80, 90, 100, 110, 120]),
    ("hip",   "hip_amp_deg",   [3, 6, 9, 12, 15, 20, 25, 30, 35, 40]),
    ("torso", "torso_amp_deg", [2, 4, 6, 9, 12, 15, 20, 25, 30]),
]

SIM_DURATION = 20.0   # matches the freq sweep (5 stand + 2 transition + 13 walk)


def _build_anchor(dof_key, amp, idx):
    a = {"name": f"{dof_key}", "hip_amp_deg": 0.0, "crank_amp_deg": 0.0, "torso_amp_deg": 0.0}
    a[dof_key] = float(amp)
    return a


def heatmap(ax, amps, freqs, Z, title, fell):
    Zm = np.ma.masked_invalid(Z)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgray")
    pc = ax.pcolormesh(freqs, amps, Zm, shading="nearest", cmap=cmap)
    # mark fallen cells with a red dot
    if fell.any():
        FF, AA = np.meshgrid(freqs, amps)
        ax.plot(FF[fell], AA[fell], "r.", ms=2, alpha=0.6)
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("amplitude [deg]")
    return pc


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(here, "results", f"sweep_ampfreq_{stamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"# output dir: {outdir}")
    print(f"# surface={SURFACE}  freq={FREQ[0]}-{FREQ[-1]} ({len(FREQ)} pts)  sim={SIM_DURATION}s")

    total = sum(len(amps) for _, _, amps in DOFS) * len(FREQ)
    print(f"# total trials = {total}")

    all_rows = []
    fig, axes = plt.subplots(len(DOFS), 1, figsize=(11, 3.2 * len(DOFS)))
    if len(DOFS) == 1:
        axes = [axes]

    idx = 0
    t0 = time.perf_counter()
    summary_lines = []
    for ax, (label, key, amps) in zip(axes, DOFS):
        amps = list(amps)
        Z = np.full((len(amps), len(FREQ)), np.nan)   # dist_fwd, survived only
        fell = np.zeros((len(amps), len(FREQ)), dtype=bool)
        for ia, amp in enumerate(amps):
            for jf, f in enumerate(FREQ):
                idx += 1
                anchor = _build_anchor(key, amp, idx)
                anchor["name"] = f"{label}_amp{amp:g}"
                r = run_trial(idx, total, anchor, float(f), SIM_DURATION, print)
                r["dof"] = label
                r["amp_deg"] = amp
                all_rows.append(r)
                if r["survived"] and not r["error_msg"]:
                    Z[ia, jf] = r["dist_fwd"]
                else:
                    fell[ia, jf] = True
        pc = heatmap(ax, amps, FREQ, Z, f"{label}-only  (other 2 DOF locked at 0)", fell)
        fig.colorbar(pc, ax=ax, label="dist_fwd [m]")
        # best clean (survived) cell
        if np.isfinite(Z).any():
            ia, jf = np.unravel_index(np.nanargmax(Z), Z.shape)
            line = (f"{label:6s}: best dist_fwd={Z[ia,jf]:.3f} m at "
                    f"amp={amps[ia]:g} deg, freq={FREQ[jf]:.3f} Hz "
                    f"(survived {int(np.isfinite(Z).sum())}/{Z.size})")
        else:
            line = f"{label:6s}: nothing survived"
        summary_lines.append(line)
        print("# " + line)

    axes[-1].set_xlabel("frequency [Hz]")
    plt.tight_layout()
    png = os.path.join(outdir, "ampfreq_heatmaps.png")
    plt.savefig(png, dpi=130, bbox_inches="tight")

    # write CSV
    fields = CSV_FIELDS + ["dof", "amp_deg"]
    with open(os.path.join(outdir, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")

    dt = time.perf_counter() - t0
    print(f"# DONE {idx} trials  wall={dt:.1f}s")
    print(f"# wrote {png}")
    for l in summary_lines:
        print("#   " + l)


if __name__ == "__main__":
    main()
