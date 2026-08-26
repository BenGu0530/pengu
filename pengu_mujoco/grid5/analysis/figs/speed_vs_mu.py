#!/usr/bin/env python
"""Forward speed vs mu — two panels from the same per-(config, mu) selection:

  left   top-20 mean : mean net_fwd_mean of the 20 fastest passing cells
  right  champion    : net_fwd_mean of the single fastest passing cell

Selection is the frozen T-speed track (docs/grid5_design.md step 2 /
PLOT_GRID5.md §8): eligibility = pass (pass_rate > 0), ranked by
net_fwd_mean descending, chosen PER (config, mu) — never selected at one mu
and scored at another (trap T1). The champion is a best-of-best: one cell,
often one grid step wide (trap T8) — read it next to the robust-region
figure, not alone. Fewer than 20 passers -> the point is annotated with n;
zero passers -> a gap labelled "no passers", never a zero.

usage:
  python grid5/analysis/figs/speed_vs_mu.py                  # grid4 reference
  python grid5/analysis/figs/speed_vs_mu.py --round grid5
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import style5, load5
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))          # pengu_mujoco/
DEF_OUT = os.path.join(_ROOT, "results", "grid5_report", "style_ref")
TOP = 20                     # default; --top overrides (stamped in filename)


def speed_track(g, top):
    """Per mu: (top-20 mean, champion, n passers) from the T-speed ranking."""
    out = []
    for m in range(len(g.axes["mu"])):
        nf = g["net_fwd_mean"][m]
        vals = nf[(g["pass_rate"][m] > 0) & np.isfinite(nf)]
        if vals.size == 0:
            out.append((np.nan, np.nan, 0))
            continue
        best = np.sort(vals)[-top:]
        out.append((float(best.mean()), float(best[-1]),
                    int(min(vals.size, top))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="grid4", choices=["grid4", "grid5"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=TOP)
    ap.add_argument("--out", default=DEF_OUT)
    args = ap.parse_args()

    cfgs = args.configs or [c for c in style5.CONFIGS
                            if load5._csv_path(c, args.round)]
    grids = {}
    for c in cfgs:
        try:
            grids[c] = load5.load(c, rnd=args.round)
        except FileNotFoundError as e:
            print(f"  skip {c}: {e}")
    if not grids:
        sys.exit("no configs loadable")
    load5.compatible(grids.values())

    data = {c: speed_track(g, args.top) for c, g in grids.items()}
    mus = next(iter(grids.values())).axes["mu"]
    partial = [c for c, g in grids.items() if not g.complete]
    Ks = {g.K for g in grids.values()}
    commits = {g.commit for g in grids.values() if g.commit}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), sharey=True)
    for ax, col, ttl in zip(
            axes, (0, 1),
            (f"top-{args.top} mean (per-μ selection)",
             "champion (best-of-best)")):
        for c, g in grids.items():
            k, com = style5.CONFIGS[c]
            ys = [d[col] for d in data[c]]
            ax.plot(mus, ys, **style5.style_for(k, com),
                    label=style5.label_for(c)
                    + (" [PARTIAL]" if not g.complete else ""))
            for mu, d in zip(mus, data[c]):          # annotate thin selections
                if d[2] == 0:
                    ax.annotate("no passers", (mu, 0), fontsize=6.5,
                                color="crimson", ha="center",
                                xytext=(0, -12), textcoords="offset points")
                elif d[2] < args.top:
                    ax.annotate(f"n={d[2]}", (mu, d[col]), fontsize=6.5,
                                color="gray", xytext=(4, -10),
                                textcoords="offset points")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("floor friction μ"); ax.set_xticks(mus); ax.grid(alpha=0.3)
    axes[0].set_ylabel("net_fwd_mean [m/s]")
    style5.legend_two(axes[0],
                      coms=sorted({style5.CONFIGS[c][1] for c in grids}),
                      loc_gait="upper right", loc_com="upper left")

    note = "T-speed: eligibility pass_rate>0, ranked by net_fwd_mean, per (config, μ)"
    if partial:
        pres = {c: grids[c].present["hip_off"] for c in partial}
        note += "; PARTIAL " + " ".join(
            f"{c}(hip_off={[int(v) for v in p]})" for c, p in pres.items())
    style5.finish(
        fig, os.path.join(args.out, f"speed_vs_mu_{args.round}"
                          + (f"_top{args.top}" if args.top != TOP else "")
                          + ".png"),
        K="/".join(str(k) for k in sorted(Ks)),
        tier="pass (pass_rate > 0)",
        stat=f"left: mean of top-{args.top}; right: best-of-best (one cell)",
        note=note,
        commit=commits.pop() if len(commits) == 1 else "")


if __name__ == "__main__":
    main()
