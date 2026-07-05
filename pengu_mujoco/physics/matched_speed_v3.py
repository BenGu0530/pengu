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
MAXFEV = int(sys.argv[1]) if len(sys.argv) > 1 else 140

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


def optimize(mo):
    x0 = np.array([(SEED[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in FREE])
    es = cma.CMAEvolutionStrategy(x0, 0.30,
                                  {"bounds": [0, 1], "maxfevals": MAXFEV, "verb_disp": 0, "seed": 1})
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


rows = []
print(f"MATCHED-SPEED v3 | V_TARGET={V_TARGET} | REF_MU={REF_MU} | maxfev={MAXFEV} | "
      f"clean=survived&net_fwd>{NETV_OK:.4f}&single>{SINGLE_OK}")
for mode, mo in MODES.items():
    p = optimize(mo)
    r = measure(p, REF_MU)
    min_mu = min_mu_contiguous(p)
    row = dict(mode=mode,
               opt_leg=round(p["leg_amp"], 1), opt_hip=round(p["hip_amp"], 1),
               opt_freq=round(p["freq"], 3), torso_amp=p["torso_amp"], torso_phi=p["torso_phi"],
               speed=round(r["net_fwd_speed"], 4), speed_err=round(r["net_fwd_speed"] - V_TARGET, 4),
               single=round(r["single_frac"], 3), straight=round(r["straightness"], 3),
               mu_req_p95=round(r["mu_req_p95"], 3), min_mu_to_walk=min_mu)
    rows.append(row)
    print(f"  {mode:12s} opt(leg={row['opt_leg']:5.1f} hip={row['opt_hip']:4.1f} f={row['opt_freq']:.3f}) "
          f"speed={row['speed']:.3f}(err{row['speed_err']:+.3f}) single={row['single']:.3f} "
          f"mu_req@{REF_MU}={row['mu_req_p95']:.3f} min_mu={min_mu}")

OUT = os.path.join(_ROOT, "results", "friction_study", "matched_speed_v3.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print("\n=== v3 MATCHED-SPEED @ %.2f m/s ===" % V_TARGET)
print(f"{'mode':12s} {'mu_req_p95':>10s} {'min_mu':>7s} {'speed':>7s} {'single':>7s}")
for r in rows:
    print(f"{r['mode']:12s} {r['mu_req_p95']:>10.3f} {str(r['min_mu_to_walk']):>7s} "
          f"{r['speed']:>7.3f} {r['single']:>7.3f}")
print(f"\nwrote {OUT}")
print("PAPER CLAIM holds iff over_stance mu_req < upright mu_req at matched speed.")
print("real surfaces: mocap 0.7 | acrylic 0.30 | uhmw 0.14 | ptfe/ice 0.06")
