#!/usr/bin/env python
"""
MATCHED-SPEED torso-mode comparison on penguV3 — the decisive paper experiment.

Why this and not killtest_torso_modes.py: the kill test proved you CANNOT build the
upright/over_swing conditions by flipping the torso on a fixed over-stance gait — the
gait then doesn't walk, so mu_req becomes meaningless (a stationary robot has tiny Ft).
The valid comparison (same design as the v2 friction_study.py) is: for EACH torso mode,
independently re-optimize the leg drive so all three modes walk at the SAME target speed,
THEN compare friction demand. This removes the "walk-fast vs lean-hard" confound.

Modes (torso treatment fixed; leg_amp/hip_amp/freq optimized to hit V_TARGET):
  upright     : torso_amp=0                      (PD-ish hold, no swing)  -- the "human on dry ground" control
  over_stance : torso_amp=TORSO, torso_phi=0     (penguin: torso over stance foot)
  over_swing  : torso_amp=TORSO, torso_phi=180   (nonplantigrade: torso over swing foot)

Measurement reuses gait_sweep.run_trial => mu_req_p95 on the SAME stance-gate (Fn>4N)
as the 3.97M sweep and the kill test, so all three data sources are directly comparable.

min_mu_to_walk uses a CLEAN-WALK definition (survived AND net_fwd>thresh AND
single_frac>0.6), taken as the lowest mu contiguous from the top of the ladder — this
rejects the low-mu "skidding forward" artifact that fooled the naive threshold
(killtest showed net_fwd going non-monotonic / single_frac collapsing at low mu).

Runs on v3 (hip_off=30 forward pitch). No matched CMA on v2 needed: the existing
results/friction_study/penguin_configs.csv already covers v2 (cross-model reference).
"""
import os, sys, csv

os.environ["PENGU_MODEL"] = "v3"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import cma
import gait_config as gc
import gait_sweep as gs

assert gc.XML_PATH.endswith("penguV3/scene.xml"), gc.XML_PATH

V_TARGET = 0.08        # m/s matched speed (same as v2 friction_study, for cross-model compare)
W_SPEED = 4.0
TORSO = 20.0           # torso roll amplitude for the two swing modes (= marked gait value)
REF_MU = 0.7           # optimize + report friction demand on the sweep floor (comparable)
MU_LADDER = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]
WALK_TIME = gs.SIM_DURATION - gs.SETTLE
NETV_OK = 0.15 / WALK_TIME
SINGLE_OK = 0.6
BOUNDS = {"leg_amp": (10.0, 120.0), "hip_amp": (0.0, 30.0), "freq": (1.0, 2.0)}
FREE = ["leg_amp", "hip_amp", "freq"]
# usage: matched_speed_v3.py [maxfev] [seeds]   e.g. 400 1,2,3,4,5
# mu_req_p95 is a MEASURED quantity at the matched-speed optimum, not the objective —
# it is non-monotone in budget and noisy across CMA seeds, so the paper margin needs
# mean±std across seeds (single-seed 1.10x vs 1.20x is indistinguishable from noise).
MAXFEV = int(sys.argv[1]) if len(sys.argv) > 1 else 140
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]

MODES = {
    "upright":     dict(torso_amp=0.0,   torso_phi=0.0),
    "over_stance": dict(torso_amp=TORSO, torso_phi=0.0),
    "over_swing":  dict(torso_amp=TORSO, torso_phi=180.0),
}
# seed near the penguin band (matched speed collapses freq; seed does not bias the mode compare).
# hip_phi fixed at 210 = the penguin winning-family value (gait B); it is the leg<->hip phase
# that produces a true forward step, held constant so the ONLY categorical is the torso mode.
SEED = dict(leg_amp=60.0, hip_amp=14.0, freq=1.30, hip_phi=210.0)

gs.CONDITION["hip_off"] = 30.0
model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model)


def measure(p, mu):
    gs.FLOOR_MU = mu
    return gs.run_trial(model, data, ids, p)


def clean_walk(r):
    return bool(r["survived"]) and r["net_fwd_speed"] > NETV_OK and r["single_frac"] > SINGLE_OK


def objective(x, mo):
    p = dict(SEED); p.update(mo)
    for n, v in zip(FREE, [BOUNDS[k][0] + xi * (BOUNDS[k][1] - BOUNDS[k][0])
                           for k, xi in zip(FREE, np.clip(x, 0, 1))]):
        p[n] = float(v)
    r = measure(p, REF_MU)
    if not r["survived"]:
        return 2.0 - 0.3 * max(0.0, r["net_fwd_speed"])          # big cost, guide toward moving
    J = -W_SPEED * abs(r["net_fwd_speed"] - V_TARGET)             # matched speed
    J -= 0.5 * max(0.0, SINGLE_OK - r["single_frac"])            # prefer clean alternation
    return -J                                                    # cma minimizes


def optimize(mo, cma_seed):
    x0 = np.array([(SEED[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in FREE])
    es = cma.CMAEvolutionStrategy(x0, 0.30,
                                  {"bounds": [0, 1], "maxfevals": MAXFEV, "verb_disp": 0,
                                   "seed": cma_seed})
    best = {"J": 1e9, "p": None}
    while not es.stop():
        xs = es.ask()
        js = [objective(x, mo) for x in xs]
        es.tell(xs, js)
        for x, j in zip(xs, js):
            if j < best["J"]:
                p = dict(SEED); p.update(mo)
                for n, v in zip(FREE, [BOUNDS[k][0] + xi * (BOUNDS[k][1] - BOUNDS[k][0])
                                       for k, xi in zip(FREE, np.clip(x, 0, 1))]):
                    p[n] = float(v)
                best = {"J": j, "p": p}
    return best["p"]


def min_mu_contiguous(p):
    """lowest mu, contiguous from the top of the ladder, that still cleanly walks."""
    lo = None
    for mu in MU_LADDER:                     # descending
        if clean_walk(measure(p, mu)):
            lo = mu
        else:
            break
    return lo


OUTDIR = os.path.join(_ROOT, "results", "friction_study")
os.makedirs(OUTDIR, exist_ok=True)

all_rows = []
print(f"MATCHED-SPEED v3 | V_TARGET={V_TARGET} | REF_MU={REF_MU} | maxfev={MAXFEV} | "
      f"seeds={SEEDS} | clean=survived&net_fwd>{NETV_OK:.4f}&single>{SINGLE_OK}")
for cma_seed in SEEDS:
    rows = []
    print(f"--- seed {cma_seed} ---")
    for mode, mo in MODES.items():
        p = optimize(mo, cma_seed)
        r = measure(p, REF_MU)
        min_mu = min_mu_contiguous(p)
        row = dict(seed=cma_seed, mode=mode,
                   opt_leg=round(p["leg_amp"], 1), opt_hip=round(p["hip_amp"], 1),
                   opt_freq=round(p["freq"], 3), torso_amp=p["torso_amp"], torso_phi=p["torso_phi"],
                   speed=round(r["net_fwd_speed"], 4), speed_err=round(r["net_fwd_speed"] - V_TARGET, 4),
                   single=round(r["single_frac"], 3), straight=round(r["straightness"], 3),
                   mu_req_p95=round(r["mu_req_p95"], 3), min_mu_to_walk=min_mu)
        rows.append(row); all_rows.append(row)
        print(f"  {mode:12s} opt(leg={row['opt_leg']:5.1f} hip={row['opt_hip']:4.1f} f={row['opt_freq']:.3f}) "
              f"speed={row['speed']:.3f}(err{row['speed_err']:+.3f}) single={row['single']:.3f} "
              f"mu_req@{REF_MU}={row['mu_req_p95']:.3f} min_mu={min_mu}")
    out = os.path.join(OUTDIR, f"matched_speed_v3_mf{MAXFEV}_s{cma_seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  wrote {out}")


def _mean_std(vals):
    m = sum(vals) / len(vals)
    s = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, s


# ---- cross-seed aggregation ------------------------------------------------------
agg_rows = []
print(f"\n=== v3 MATCHED-SPEED @ {V_TARGET} m/s | maxfev={MAXFEV} | n_seeds={len(SEEDS)} ===")
print(f"{'mode':12s} {'mu_req mean±std':>18s} {'speed_err mean':>14s} {'min_mu (per seed)':>20s}")
for mode in MODES:
    sub = [r for r in all_rows if r["mode"] == mode]
    mu_m, mu_s = _mean_std([r["mu_req_p95"] for r in sub])
    se_m, _ = _mean_std([r["speed_err"] for r in sub])
    minmus = "|".join(str(r["min_mu_to_walk"]) for r in sub)
    agg_rows.append(dict(mode=mode, n_seeds=len(sub),
                         mu_req_mean=round(mu_m, 4), mu_req_std=round(mu_s, 4),
                         speed_err_mean=round(se_m, 4), min_mu_per_seed=minmus))
    print(f"{mode:12s} {mu_m:>10.3f} ±{mu_s:.3f} {se_m:>14.4f} {minmus:>20s}")

AGG = os.path.join(OUTDIR, f"matched_speed_v3_mf{MAXFEV}_agg.csv")
with open(AGG, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys())); w.writeheader(); w.writerows(agg_rows)

# ---- seed-robustness verdict: is over_stance < upright beyond optimizer noise? ----
# GATE on matched speed: an underspeeding config has fake-low mu_req (small Ft), so a
# seed only counts if BOTH modes actually hit V_TARGET (the smoke test showed a 20-eval
# upright at speed 0.021 faking mu_req 0.419 and reversing the verdict).
SPEED_TOL = 0.01
gaps = []
print(f"\n=== cross-seed verdict: over_stance vs upright (|speed_err|<={SPEED_TOL}) ===")
for cma_seed in SEEDS:
    up = next(r for r in all_rows if r["seed"] == cma_seed and r["mode"] == "upright")
    ov = next(r for r in all_rows if r["seed"] == cma_seed and r["mode"] == "over_stance")
    ok = abs(up["speed_err"]) <= SPEED_TOL and abs(ov["speed_err"]) <= SPEED_TOL
    if not ok:
        print(f"  seed {cma_seed}: EXCLUDED (underspeed: upright err {up['speed_err']:+.3f}, "
              f"over_stance err {ov['speed_err']:+.3f}) -> mu_req not comparable")
        continue
    gaps.append(up["mu_req_p95"] / ov["mu_req_p95"])
    print(f"  seed {cma_seed}: upright {up['mu_req_p95']:.3f} vs over_stance {ov['mu_req_p95']:.3f} "
          f"-> gap {gaps[-1]:.2f}x {'OK' if gaps[-1] > 1 else 'REVERSED'}")
if gaps:
    g_m, g_s = _mean_std(gaps)
    wins = sum(1 for g in gaps if g > 1)
    print(f"  gap mean±std = {g_m:.2f}x ± {g_s:.2f} | over_stance wins {wins}/{len(gaps)} "
          f"matched seeds ({len(SEEDS) - len(gaps)} excluded)")
else:
    print("  no seed had both modes at matched speed — increase maxfev.")
print(f"\nwrote {AGG}")
print("PAPER CLAIM holds iff over_stance mu_req < upright mu_req at matched speed, "
      "robust across seeds.")
print("real surfaces: mocap 0.7 | acrylic 0.30 | uhmw 0.14 | ptfe/ice 0.06")
