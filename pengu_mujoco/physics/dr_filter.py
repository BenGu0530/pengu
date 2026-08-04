#!/usr/bin/env python
"""DR filter: take the forward-facing 'good' gaits from a clean sweep and score each
across a friction (mu) ladder, to filter out gaits that are only LOCALLY good (walk at
mu=0.7 but collapse when the floor gets slippery). Keeps the ones robust across mu.

'good' selection : valid & heading_align>0.5 (forward-facing), top-N by net_fwd_speed.
mu ladder        : fine near the top, where the failures actually happen.
walks-forward(mu): survived AND heading_align>0.5 AND net_fwd>NET_MIN.
robustness score : how many ladder rungs it walks forward on (mu_floor = lowest CONTIGUOUS
                   rung from 0.7 downward; a gait that falls then recovers is flagged).

usage: python physics/dr_filter.py <sweep_csv> [N] [KAPPA]
out:   <same dir>/dr_filter_<tag>.csv (+ _summary.csv, .png)
"""
import os, sys
os.environ.setdefault("PENGU_MODEL", "v3")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import mujoco, gait_config as gc, gait_sweep as gs
from torso_control import TorsoKappaPID

CSV = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
KAPPA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
LADDER = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
NET_MIN = 0.05
AX = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]

df = pd.read_csv(CSV)
good = df[(df.valid == 1) & (df.heading_align > 0.5)].sort_values(
    "net_fwd_speed", ascending=False).head(N).reset_index(drop=True)
print(f"# DR filter: {len(good)} forward-facing good gaits x {len(LADDER)} mu rungs "
      f"= {len(good)*len(LADDER)} trials  (kappa={KAPPA})")

model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
ids = gs.make_ids(model)
gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=KAPPA, measure_after=gs.SETTLE)

rows = []
for gi, g in good.iterrows():
    p = {a: float(g[a]) for a in AX}
    for mu in LADDER:
        gs.FLOOR_MU = mu; gs.CONDITION["hip_off"] = p["hip_off"]
        r = gs.run_trial(model, data, ids, {k: v for k, v in p.items() if k != "hip_off"})
        fwd = int(r["survived"] and np.isfinite(r["heading_align"])
                  and r["heading_align"] > 0.5 and r["net_fwd_speed"] > NET_MIN)
        rows.append(dict(gait=gi, mu=mu, walks_fwd=fwd, survived=r["survived"],
                         net_fwd=r["net_fwd_speed"], heading=r["heading_align"],
                         slip_ratio=r["slip_ratio"]))
    print(f"  [{gi+1}/{len(good)}] "
          + " ".join(f"{a}={p[a]:g}" for a in AX))

long = pd.DataFrame(rows)
outdir = os.path.dirname(CSV)
tag = f"k{KAPPA:g}".replace(".", "p")
long.to_csv(os.path.join(outdir, f"dr_filter_{tag}.csv"), index=False)

# per-gait robustness: contiguous forward-walking floor from mu=0.70 downward
summ = []
for gi, g in good.iterrows():
    w = long[long.gait == gi].sort_values("mu", ascending=False)
    flags = w.walks_fwd.tolist(); mus = w.mu.tolist()
    contig = 0
    for f in flags:
        if f: contig += 1
        else: break
    mu_floor = mus[contig-1] if contig > 0 else None      # lowest contiguous fwd rung
    n_fwd = int(w.walks_fwd.sum())
    recovers = int(n_fwd > contig)                        # walks fwd again below a failure
    summ.append(dict(gait=gi, **{a: g[a] for a in AX},
                     net_fwd_07=round(float(g.net_fwd_speed), 3),
                     slip_07=round(float(g.slip_ratio), 3),
                     n_rungs_fwd=n_fwd, mu_floor=mu_floor, recovers=recovers))
S = pd.DataFrame(summ).sort_values(["n_rungs_fwd", "net_fwd_07"], ascending=[False, False])
S.to_csv(os.path.join(outdir, f"dr_filter_{tag}_summary.csv"), index=False)
print("\n=== ranked by robustness (n_rungs_fwd = # of mu it still walks forward on) ===")
print(S.head(15).to_string(index=False))

# heatmap: gaits (ranked) x mu, colored by walks_fwd (green) with net_fwd shading
order = S.gait.tolist()
M = long.pivot(index="gait", columns="mu", values="net_fwd").reindex(order)[LADDER]
W = long.pivot(index="gait", columns="mu", values="walks_fwd").reindex(order)[LADDER]
fig, ax = plt.subplots(figsize=(8, max(4, 0.25*len(order)+1)))
im = ax.imshow(np.ma.masked_where(W.values == 0, M.values), aspect="auto", cmap="viridis")
ax.imshow(np.ma.masked_where(W.values == 1, np.ones_like(M.values)), aspect="auto",
          cmap="Reds", alpha=0.35, vmin=0, vmax=1)   # red = NOT walking forward
ax.set_xticks(range(len(LADDER))); ax.set_xticklabels([f"{m:g}" for m in LADDER])
ax.set_xlabel("floor mu"); ax.set_ylabel(f"gait (ranked by robustness), n={len(order)}")
ax.set_title(f"DR filter k{KAPPA:g}: forward-walking across mu (color=net_fwd, red=not fwd)")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); fig.savefig(os.path.join(outdir, f"dr_filter_{tag}.png"), dpi=120)
print(f"\n# wrote dr_filter_{tag}.csv / _summary.csv / .png")
