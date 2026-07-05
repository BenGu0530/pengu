#!/usr/bin/env python
"""
KILL TEST — torso strategy x friction, on penguV3 with the TWO marked gaits.

Purpose (per Fable review): the paper claim is "torso-over-stance (penguin) walks
on lower friction than torso-held-upright". The existing results/friction_study/
penguin_configs.csv shows this on v2 + a CMA-optimized seed. This script re-tests it
CLEANLY on v3, using the actual registered gaits A (1.59 Hz) and B (1.27 Hz penguin),
reusing the *same* measurement core (gait_sweep.run_trial) as the 3.97M-cell sweep, so
mu_req_p95 is on the SAME stance-gate (Fn>4N) and directly comparable to SWEEP_ANALYSIS.

For each gait x {upright, over_stance, over_swing} we sweep a floor-mu ladder and record
survival / net forward progress / single-support / friction demand. This answers:
  (a) do all three torso modes even walk on v3? (over_swing confound check)
  (b) how much does the torso MODE move mu_req vs the within-family 0.47-0.65 spread?
  (c) does any (gait,mode) get below the real slippery surfaces (acrylic .30, uhmw .14)?

NO CMA, NO matched-speed re-optimization: we hold each gait fixed and only flip the
torso treatment, so the comparison isolates the torso strategy (not a re-tuned gait).
Read-only w.r.t. committed behavior; writes only killtest_v3_AB.csv.
"""
import os, sys, csv

os.environ["PENGU_MODEL"] = "v3"  # MUST be set before gait_config import (XML_PATH resolved at load)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)   # gait_sweep
sys.path.insert(0, _ROOT)   # gait_config, friction_utils

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs

assert gc.XML_PATH.endswith("penguV3/scene.xml"), f"expected v3, got {gc.XML_PATH}"

# ---- the two registered gaits (SWEEP_ANALYSIS.md sec 7) --------------------------
GAITS = {
    "A_f1.59": dict(freq=1.59, hip_phi=180.0, leg_amp=110.0, hip_amp=20.0, torso_amp=20.0),
    "B_f1.27": dict(freq=1.27, hip_phi=210.0, leg_amp=115.0, hip_amp=22.0, torso_amp=20.0),
}
# torso treatment: how each mode overrides (torso_amp, torso_phi) on top of the base gait
MODES = {
    "upright":     dict(torso_amp=0.0,  torso_phi=0.0),    # PD hold, no swing
    "over_stance": dict(torso_amp=None, torso_phi=0.0),    # gait's own torso_amp, in-phase (= the marked gait)
    "over_swing":  dict(torso_amp=None, torso_phi=180.0),  # antiphase (torso over swing foot)
}
MU_LADDER = [1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]  # matches friction_study
REF_MU_DEMAND = 0.7   # mu_req_p95 reported at this floor (== sweep floor, comparable to SWEEP_ANALYSIS)
WALK_TIME = gs.SIM_DURATION - gs.SETTLE  # 24 - 11 = 13 s measurement window
DIST_OK = 0.15        # m net forward to count as "still walking" (friction_study convention)
NETV_OK = DIST_OK / WALK_TIME  # ~0.0115 m/s

gs.CONDITION["hip_off"] = 30.0  # v3 forward-pitch posture (all marked gaits found under this)

model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model)

OUT = os.path.join(_ROOT, "results", "friction_study", "killtest_v3_AB.csv")
rows = []

def run_one(gait, base, mode, mo, mu):
    p = dict(base)
    p["torso_amp"] = base["torso_amp"] if mo["torso_amp"] is None else mo["torso_amp"]
    p["torso_phi"] = mo["torso_phi"]
    gs.FLOOR_MU = mu                      # <-- run_trial re-applies this global as its 1st line
    r = gs.run_trial(model, data, ids, p)
    walks = bool(r["survived"]) and (r["net_fwd_speed"] > NETV_OK)
    return dict(gait=gait, mode=mode, mu=mu,
                survived=int(bool(r["survived"])), walks=int(walks),
                net_fwd=round(r["net_fwd_speed"], 4), single=round(r["single_frac"], 3),
                straight=round(r["straightness"], 3), mu_req_p95=round(r["mu_req_p95"], 3),
                torso_amp=p["torso_amp"], torso_phi=p["torso_phi"])

print(f"model={gs.XML.split('/')[-2]}/{gs.XML.split('/')[-1]}  walk_window={WALK_TIME:.0f}s  "
      f"walk_thresh net_fwd>{NETV_OK:.4f} m/s (={DIST_OK}m)")
for gait, base in GAITS.items():
    for mode, mo in MODES.items():
        for mu in MU_LADDER:
            rows.append(run_one(gait, base, mode, mo, mu))
        # progress line as each (gait,mode) finishes its ladder
        sub = [x for x in rows if x["gait"] == gait and x["mode"] == mode]
        min_mu = min([x["mu"] for x in sub if x["walks"]], default=None)
        demand = next((x["mu_req_p95"] for x in sub if x["mu"] == REF_MU_DEMAND), float("nan"))
        print(f"  {gait:8s} {mode:12s}  min_mu_to_walk={str(min_mu):5s}  "
              f"mu_req_p95@{REF_MU_DEMAND}={demand:.3f}  "
              f"net_fwd@0.7={next(x['net_fwd'] for x in sub if x['mu']==0.7):.3f}  "
              f"single@0.7={next(x['single'] for x in sub if x['mu']==0.7):.3f}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\n=== SUMMARY: min_mu_to_walk | mu_req_p95@0.7 (demand) ===")
print(f"{'gait':9s} {'upright':>22s} {'over_stance':>22s} {'over_swing':>22s}")
for gait in GAITS:
    cells = []
    for mode in MODES:
        sub = [x for x in rows if x["gait"] == gait and x["mode"] == mode]
        min_mu = min([x["mu"] for x in sub if x["walks"]], default=None)
        demand = next((x["mu_req_p95"] for x in sub if x["mu"] == REF_MU_DEMAND), float("nan"))
        cells.append(f"minmu={str(min_mu):>4s} req={demand:.3f}")
    print(f"{gait:9s} " + " ".join(f"{c:>22s}" for c in cells))
print(f"\nwrote {OUT} ({len(rows)} rows)")
print("real slippery surfaces for reference: acrylic=0.30  uhmw_pe=0.14  ptfe_ice=0.06")
