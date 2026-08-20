#!/usr/bin/env python
"""Re-run the finalist evaluation with a STAGED (slow) start, for comparison.

Same top-20 gaits, same 4 mu levels, same nominal conditions and the same
metric set as grid4_finalists.py -- the only change is how the walk begins:

  original (abrupt)   hold 5s -> blend 4s (hip_off applied as a STEP) -> walk
  staged (here)       hold 5s -> ramp hip_off 4s -> settle 6s -> blend 4s -> walk

Implemented by monkeypatching gait_config.apply_ctrl and extending T_HOLD to
5+ramp+settle, so gait_sweep.run_trial's measurement window (SETTLE = T_HOLD +
T_TRANSITION + 2) shifts automatically and the measured walking duration stays
identical to the original (13 s).

-> results/grid4_report/cN/finalists_staged.csv

usage: python physics/grid4_finalists_staged.py [--ramp 4] [--settle 6] [cN ...]
"""
import os, sys, csv, math, argparse
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np, mujoco
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
ap.add_argument("--ramp", type=float, default=4.0)
ap.add_argument("--settle", type=float, default=6.0)
ap.add_argument("--top", type=int, default=20)
a = ap.parse_args()
wanted = [c for c in a.cfgs if c in CONF] or [c for c in CONF]

STAND_T = 5.0
T1, T2 = STAND_T, STAND_T + a.ramp                 # ramp window
T3 = T2 + a.settle                                 # offset settled; blend starts here
gc.T_HOLD = T3                                     # run_trial's blend begins at T3
gs.SETTLE = gc.T_HOLD + gc.T_TRANSITION + 2.0
gs.SIM_DURATION = gs.SETTLE + 13.0                 # same 13 s measured walk as original

_full_off = [0.0]        # set per-trial before run_trial


def _smoothstep(t, lo, hi):
    if hi <= lo: return 1.0
    x = min(1.0, max(0.0, (t - lo) / (hi - lo)))
    return x * x * (3 - 2 * x)


def staged_apply_ctrl(data, act_ids, t):
    """apply_ctrl with hip_off ramped in during the extended hold."""
    stand = math.radians(gc.STAND_HIP_DEG)
    t_walk = gc.T_HOLD + gc.T_TRANSITION
    off = _full_off[0]
    if t < T1:
        frac = 0.0
    elif t < T2:
        frac = _smoothstep(t, T1, T2)
    else:
        frac = 1.0
    gc.WALK_HIP_OFFSET_DEG = off * frac

    if t < gc.T_HOLD:
        # stand / ramp / settle: no oscillation, torso commanded to 0 and the PID
        # is held reset so its integrator does not wind up (matches the original
        # hold branch, which never calls TORSO_CONTROLLER).
        alpha = 0.0
        hip_L, hip_R, crank_L, crank_R, _ = gc.compute_gait(0.0, alpha)
        if gc.TORSO_CONTROLLER is not None and hasattr(gc.TORSO_CONTROLLER, "reset"):
            gc.TORSO_CONTROLLER.reset()
        torso = 0.0
    else:
        alpha = _smoothstep(t, gc.T_HOLD, t_walk) if t < t_walk else 1.0
        phase = 2 * math.pi * gc.WALK_FREQ * (t - gc.T_HOLD)
        hip_L, hip_R, crank_L, crank_R, torso = gc.compute_gait(phase, alpha)
        if gc.TORSO_CONTROLLER is not None:
            torso = gc.TORSO_CONTROLLER(data, t, alpha)

    data.ctrl[act_ids["hip-L"]] = stand * (1.0 - alpha) + hip_L
    data.ctrl[act_ids["hip-R"]] = stand * (1.0 - alpha) + hip_R
    data.ctrl[act_ids["crank1-L"]] = crank_L
    data.ctrl[act_ids["crank1-R"]] = crank_R
    data.ctrl[act_ids["torso"]] = torso


gc.apply_ctrl = staged_apply_ctrl


def eval_config(cfg):
    top_csv = os.path.join(OUT, cfg, "top_gaits.csv")
    if not os.path.exists(top_csv):
        return None
    gaits = list(csv.DictReader(open(top_csv)))[:a.top]
    kappa, com = CONF[cfg]
    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = g4.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid
    print(f"{cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm  "
          f"finalists={len(gaits)}  T_HOLD={gc.T_HOLD} SETTLE={gs.SETTLE} "
          f"DUR={gs.SIM_DURATION}", flush=True)
    rows = []
    for rank, gd in enumerate(gaits, 1):
        p = dict(freq=float(gd["freq"]), hip_phi=float(gd["hip_phi"]),
                 leg_amp=float(gd["leg_amp"]), hip_amp=float(gd["hip_amp"]))
        hip_off = float(gd["hip_off"])
        for mu in MUS:
            gs.FLOOR_MU = mu; gs.POSE_JITTER = None
            gs.CONDITION["hip_off"] = hip_off
            _full_off[0] = hip_off
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
    with open(os.path.join(OUT, cfg, "finalists_staged.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    gc.TORSO_CONTROLLER = None
    return rows


for cfg in wanted:
    eval_config(cfg)
print("\ndone -> cN/finalists_staged.csv")
