#!/usr/bin/env python
"""Automated friction (sim2real robustness) eval for one co-design cell.

"A" plan: take the top-N gaits of a completed sweep (found at its design mu), re-run each
DOWN the SURFACES mu-ladder, and record whether it still walks and how much the planted
foot slips. Foot mu is fixed/grippy; only the FLOOR mu varies (see friction_utils) — that
is the single physical knob, because the floor (slipperier) caps the interface friction.

A gait PASSES at a given mu iff:  valid==1  AND  net_fwd_speed > NET_MIN  AND
                                  slip_ratio <= SLIP_MAX   (Ben's 5%).
min_mu = the lowest ladder mu at which it still passes (contiguous from the top).

Reusable per cell: point it at any grid CSV and give its kappa.
  usage: python physics/friction_eval.py <grid_csv> [kappa=0] [topN=50]
  env:   NET_MIN (0.05)  SLIP_MAX (0.05)
  out:   <dir>/friction_eval_<tag>.csv        (long: gait x mu)
         <dir>/friction_eval_<tag>_summary.csv (per-gait min_mu + speeds + slip)
         <dir>/friction_eval_<tag>.png         (heatmaps: net_fwd & slip_ratio)
"""
import os, sys
os.environ.setdefault("PENGU_MODEL", "v3")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import SURFACES

CSV = sys.argv[1]
KAPPA = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 50
NET_MIN = float(os.environ.get("NET_MIN", "0.05"))
SLIP_MAX = float(os.environ.get("SLIP_MAX", "0.05"))
AX = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]
LADDER = sorted(set(SURFACES.values()), reverse=True)   # [0.7, 0.30, 0.14, 0.06]
KTAG = f"k{KAPPA:g}".replace(".", "p")
tag = f"{KTAG}_top{TOPN}"
outdir = os.path.dirname(CSV)

df = pd.read_csv(CSV)
val = df[df.valid == 1].copy()
top = val.sort_values("net_fwd_speed", ascending=False).head(TOPN).reset_index(drop=True)
print(f"# {os.path.basename(CSV)}  kappa={KAPPA}  valid={len(val)}  "
      f"eval top-{len(top)} x mu{LADDER}  ({len(top)*len(LADDER)} trials)")

model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model)
pid = TorsoKappaPID(model, kappa=KAPPA, measure_after=gs.SETTLE)
gc.TORSO_CONTROLLER = pid

rows = []
for gi, g in top.iterrows():
    p = {a: float(g[a]) for a in AX}
    for mu in LADDER:
        gs.FLOOR_MU = mu
        gs.CONDITION["hip_off"] = p["hip_off"]
        r = gs.run_trial(model, data, ids, {k: v for k, v in p.items() if k != "hip_off"})
        passed = int(r["valid"] == 1 and r["net_fwd_speed"] > NET_MIN
                     and np.isfinite(r["slip_ratio"]) and r["slip_ratio"] <= SLIP_MAX)
        rows.append(dict(gait=gi, **p, mu=mu, passed=passed,
                         net_fwd_speed=r["net_fwd_speed"], slip_ratio=r["slip_ratio"],
                         slip_dist=r["slip_dist"], valid=r["valid"],
                         mu_req_p95=r["mu_req_p95"]))
    print(f"  [{gi+1}/{len(top)}] f={p['freq']:.2f} phi={p['hip_phi']:.0f} "
          f"leg={p['leg_amp']:.0f} hip={p['hip_amp']:.0f} off={p['hip_off']:.0f}", flush=True)
gc.TORSO_CONTROLLER = None

long = pd.DataFrame(rows)
long_path = os.path.join(outdir, f"friction_eval_{tag}.csv")
long.to_csv(long_path, index=False)

# per-gait summary: min_mu = lowest ladder mu passing contiguously from the top
def min_mu(sub):
    s = sub.sort_values("mu", ascending=False)
    ok = float("nan")
    for _, rr in s.iterrows():
        if rr["passed"]:
            ok = rr["mu"]
        else:
            break
    return ok

summ = []
for gi, sub in long.groupby("gait"):
    g = top.loc[gi]
    summ.append(dict(gait=gi, **{a: float(g[a]) for a in AX},
                     net_fwd_mu07=float(sub[sub.mu == 0.7].net_fwd_speed.iloc[0]),
                     slip_mu07=float(sub[sub.mu == 0.7].slip_ratio.iloc[0]),
                     min_mu=min_mu(sub),
                     n_mu_pass=int(sub.passed.sum())))
summ = pd.DataFrame(summ).sort_values(["min_mu", "net_fwd_mu07"],
                                      ascending=[True, False])
summ_path = os.path.join(outdir, f"friction_eval_{tag}_summary.csv")
summ.to_csv(summ_path, index=False)

# heatmaps: rows = gaits (ordered by min_mu then speed), cols = mu ladder
order = summ.gait.tolist()
def grid(metric):
    M = long.pivot(index="gait", columns="mu", values=metric).reindex(order)
    return M[LADDER]                                  # columns high->low mu
fig, axs = plt.subplots(1, 2, figsize=(11, max(4, 0.18 * len(order) + 2)))
for ax, (metric, cmap, ttl) in zip(axs, [("net_fwd_speed", "viridis", "net_fwd_speed [m/s]"),
                                         ("slip_ratio", "inferno_r", "slip_ratio (clipped to 5; 0.05=pass)")]):
    M = grid(metric).values.astype(float)
    if metric == "slip_ratio":
        M = np.clip(M, 0.0, 5.0)                       # cap runaway slip; NaN stays NaN
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap)  # NaN (fell) masked
    ax.set_xticks(range(len(LADDER))); ax.set_xticklabels([f"{m:g}" for m in LADDER])
    ax.set_xlabel("floor mu"); ax.set_ylabel(f"gait rank (by min_mu), n={len(order)}")
    ax.set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle(f"Friction eval  {tag}  (foot fixed-grippy, floor varied)  "
             f"pass: valid & net>{NET_MIN} & slip<={SLIP_MAX}", fontweight="bold", fontsize=10)
fig.tight_layout()
png = os.path.join(outdir, f"friction_eval_{tag}.png")
fig.savefig(png, dpi=120, bbox_inches="tight")

# report
n_walk_ice = int((summ.min_mu <= 0.06).sum())
print(f"# wrote {os.path.basename(long_path)}, {os.path.basename(summ_path)}, {os.path.basename(png)}")
print(f"# min_mu distribution over top-{len(top)}:")
print(summ.min_mu.value_counts(dropna=False).sort_index().to_string())
print(f"# gaits passing at every ladder mu (incl. 0.06): {n_walk_ice}")
print("# most-robust few (low min_mu, then fast):")
print(summ.head(8)[["gait", "freq", "hip_phi", "leg_amp", "hip_amp", "hip_off",
                    "net_fwd_mu07", "slip_mu07", "min_mu"]].to_string(index=False))
