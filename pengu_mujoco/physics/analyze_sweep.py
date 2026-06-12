"""
analyze_sweep.py - Post-process an anchor/frequency sweep results.csv to extract
a ROBUST natural frequency per anchor.

Pengu's dist_fwd(freq) curve is jagged (nonlinear sensitivity), so the single-trial
argmax is noisy. This smooths each anchor's curve (rolling median over survived
trials) and reports:
  - raw argmax freq (max single-trial dist_fwd, survived)
  - smoothed-peak freq (peak of rolling-median dist_fwd) = robust natural freq
  - clean-best freq (max dist_fwd subject to roll <= ROLL_CAP, i.e. upright walking)
and writes freq_curves.png (raw + smoothed dist_fwd, fallen trials marked).

Usage:
  python physics/analyze_sweep.py [results_dir]
  (defaults to the newest results/sweep_anchor_validation_* dir)
"""
import os
import sys
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROLL_CAP = 15.0      # deg: "clean/upright" walking threshold
MIN_DIST = 0.3       # m: ignore non-walking trials for the clean metric
SMOOTH_WIN = 5       # samples (=0.025 Hz at 0.005 step) rolling median window


def _newest_results_dir():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = sorted(glob.glob(os.path.join(here, "results", "sweep_anchor_validation_*")))
    if not dirs:
        sys.exit("no sweep results dirs found")
    return dirs[-1]


def _rolling_median(y, win):
    n = len(y)
    out = np.full(n, np.nan)
    h = win // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        seg = y[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def main():
    rdir = sys.argv[1] if len(sys.argv) > 1 else _newest_results_dir()
    csv_path = os.path.join(rdir, "results.csv")
    print(f"analyzing {csv_path}")
    rows = list(csv.DictReader(open(csv_path)))
    anchors = []
    for r in rows:
        if r["anchor_name"] not in anchors:
            anchors.append(r["anchor_name"])

    fig, axes = plt.subplots(len(anchors), 1, figsize=(11, 2.6 * len(anchors)), sharex=True)
    if len(anchors) == 1:
        axes = [axes]

    print(f"\n{'anchor':18s} {'raw argmax':>12s} {'smoothed pk':>12s} {'clean(<%g°)':>12s}"
          % ROLL_CAP)
    print("-" * 60)
    for ax, name in zip(axes, anchors):
        rr = [r for r in rows if r["anchor_name"] == name]
        rr.sort(key=lambda r: float(r["freq_hz"]))
        f = np.array([float(r["freq_hz"]) for r in rr])
        surv = np.array([r["survived"] == "True" for r in rr])
        dist = np.array([float(r["dist_fwd"]) for r in rr])
        roll = np.array([float(r["torso_roll_amp_deg"]) if r["torso_roll_amp_deg"] not in ("", "nan") else np.nan for r in rr])

        d_surv = np.where(surv, dist, np.nan)
        sm = _rolling_median(d_surv, SMOOTH_WIN)

        # raw argmax over survived
        raw_f = f[np.nanargmax(d_surv)] if np.isfinite(d_surv).any() else np.nan
        # smoothed peak
        sm_f = f[np.nanargmax(sm)] if np.isfinite(sm).any() else np.nan
        # clean: survived & roll<=cap & dist>=min, max dist
        clean_mask = surv & (roll <= ROLL_CAP) & (dist >= MIN_DIST)
        if clean_mask.any():
            ci = np.where(clean_mask)[0]
            clean_f = f[ci[np.argmax(dist[ci])]]
            clean_d = dist[ci].max()
            clean_str = f"{clean_f:.3f}({clean_d:.2f}m)"
        else:
            clean_str = "none"
        print(f"{name:18s} {raw_f:7.3f}Hz   {sm_f:7.3f}Hz   {clean_str:>12s}")

        ax.plot(f, np.where(surv, dist, np.nan), '.', ms=3, color='tab:blue', label='survived dist')
        ax.plot(f, np.where(~surv, dist, np.nan), 'x', ms=4, color='tab:red', label='fell')
        ax.plot(f, sm, '-', color='black', lw=1.8, label=f'rolling median (w={SMOOTH_WIN})')
        if np.isfinite(sm_f):
            ax.axvline(sm_f, color='green', ls='--', lw=1, label=f'smoothed pk {sm_f:.3f}')
        ax.set_title(name, fontsize=10, loc='left')
        ax.set_ylabel("dist_fwd [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc='upper right')
    axes[-1].set_xlabel("frequency [Hz]")
    plt.tight_layout()
    out = os.path.join(rdir, "freq_curves.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
