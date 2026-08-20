#!/usr/bin/env python
"""Rich finalist evaluation for gaits selected at an arbitrary mu.

Same evaluation as grid4_finalists.py (top-N re-run at every mu, nominal conditions,
full 21-metric set) but reading cN/top_gaits_mu<XX>.csv from grid4_top_at_mu.py and
writing cN/finalists_mu<XX>.csv.

usage: python physics/grid4_finalists_at.py --mu 0.7 [--top 20] [cN ...]
"""
import os, sys, csv, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import mujoco
import gait_config as gc, gait_sweep as gs
from torso_control import TorsoKappaPID
import grid4_sweep as g4

OUT = os.path.join(_ROOT, "results", "grid4_report")
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
MUS = [0.1, 0.3, 0.5, 0.7]
FIELDS = ["rank", "freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
          "net_fwd", "slip_ratio", "ds_move_frac", "ss_move_frac", "single_frac",
          "clear_L", "clear_R", "n_steps", "cadence", "straightness",
          "torso_roll_rms_deg", "torso_sat_frac", "survived", "valid"]

ap = argparse.ArgumentParser()
ap.add_argument("cfgs", nargs="*")
ap.add_argument("--mu", type=float, required=True)
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()
tag = f"{a.mu:.1f}".replace("0.", "0")
wanted = [c for c in a.cfgs if c in CONF] or list(CONF)


def eval_config(cfg):
    top_csv = os.path.join(OUT, cfg, f"top_gaits_mu{tag}.csv")
    if not os.path.exists(top_csv):
        return None
    gaits = list(csv.DictReader(open(top_csv)))[:a.top]
    if not gaits:
        return None
    kappa, com = CONF[cfg]
    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = g4.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid
    print(f"{cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm  "
          f"finalists={len(gaits)} (selected at mu={a.mu})", flush=True)
    rows = []
    for rank, gd in enumerate(gaits, 1):
        p = dict(freq=float(gd["freq"]), hip_phi=float(gd["hip_phi"]),
                 leg_amp=float(gd["leg_amp"]), hip_amp=float(gd["hip_amp"]))
        hip_off = float(gd["hip_off"])
        for mu in MUS:
            gs.FLOOR_MU = mu; gs.POSE_JITTER = None
            gs.CONDITION["hip_off"] = hip_off
            pid.reset()
            r = gs.run_trial(model, data, ids, dict(p))
            rows.append(dict(rank=rank, freq=p["freq"], hip_phi=p["hip_phi"],
                             leg_amp=p["leg_amp"], hip_amp=p["hip_amp"], hip_off=hip_off,
                             mu=mu, net_fwd=r["net_fwd_speed"], slip_ratio=r["slip_ratio"],
                             ds_move_frac=r["ds_move_frac"], ss_move_frac=r["ss_move_frac"],
                             single_frac=r["single_frac"], clear_L=r["clear_L"],
                             clear_R=r["clear_R"], n_steps=r["n_steps"],
                             cadence=r["cadence"], straightness=r["straightness"],
                             torso_roll_rms_deg=round(pid.roll_rms(), 3),
                             torso_sat_frac=round(pid.saturation_frac(), 3),
                             survived=r["survived"], valid=r["valid"]))
        print(f"  rank {rank}/{len(gaits)} done", flush=True)
    with open(os.path.join(OUT, cfg, f"finalists_mu{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    gc.TORSO_CONTROLLER = None
    return rows


for cfg in wanted:
    eval_config(cfg)
print(f"\ndone -> cN/finalists_mu{tag}.csv")
