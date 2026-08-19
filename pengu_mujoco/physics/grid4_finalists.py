#!/usr/bin/env python
"""GRID-4 stage-3: finalist rich evaluation + roll-to-speed + per-config demos.

For every config that has results/grid4_report/cN/top_gaits.csv (from grid4_report.py):
  - re-run the top TOP_N gaits at every mu level, NOMINAL conditions (exact mu, no jitter),
    harvesting the full metric set incl. measured torso roll RMS [deg] and PID saturation
    -> results/grid4_report/cN/finalists.csv
  - render the #1 gait at mu=0.1 -> results/grid4_report/cN/demo_mu01.mp4 (side+back)
Cross-config figures -> results/grid4_report/cross/:
  - roll_to_speed.png : x = measured torso roll RMS, y = net_fwd, mu=0.1 finalists
  - ds_move_mu01.png  : ds_move_frac distribution per config (shuffle vs stepping)

usage: python physics/grid4_finalists.py [--no-demo]
"""
import os, sys, csv
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
from friction_utils import set_floor_friction
import grid4_sweep as g4

OUT = os.path.join(_ROOT, "results", "grid4_report")
CONF = {"c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
        "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31)}
MUS = [0.1, 0.3, 0.5, 0.7]
TOP_N = int(os.environ.get("TOP_N", "20"))
DEMO = "--no-demo" not in sys.argv
FIELDS = ["rank", "freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu",
          "net_fwd", "slip_ratio", "ds_move_frac", "ss_move_frac", "single_frac",
          "clear_L", "clear_R", "n_steps", "cadence", "straightness",
          "torso_roll_rms_deg", "torso_sat_frac", "survived", "valid"]


def eval_config(cfg):
    top_csv = os.path.join(OUT, cfg, "top_gaits.csv")
    if not os.path.exists(top_csv):
        return None
    gaits = list(csv.DictReader(open(top_csv)))[:TOP_N]
    kappa, com = CONF[cfg]
    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = g4.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid
    print(f"{cfg}: kappa={kappa} com={got:.4f} slide={slide*1000:+.1f}mm  "
          f"finalists={len(gaits)}")
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
    with open(os.path.join(OUT, cfg, "finalists.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

    if DEMO:
        gd = gaits[0]
        p = dict(freq=float(gd["freq"]), hip_phi=float(gd["hip_phi"]),
                 leg_amp=float(gd["leg_amp"]), hip_amp=float(gd["hip_amp"]))
        import imageio.v2 as imageio
        set_floor_friction(model, 0.1)
        gs.CONDITION["hip_off"] = float(gd["hip_off"])
        gs._set_gait(dict(p)); pid.reset()
        act, jadr = gc.build_ids(model)
        gc.set_initial_pose(model, data, act, jadr)
        cams = []
        for dist, elev, az in [(1.4, -10, 0), (1.1, -12, -90)]:
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance, cam.elevation, cam.azimuth = dist, elev, az
            cams.append(cam)
        frames = []
        with mujoco.Renderer(model, height=480, width=640) as ren:
            nxt = 0.0
            while data.time < gs.SIM_DURATION:
                gc.apply_ctrl(data, act, data.time)
                mujoco.mj_step(model, data)
                if data.time >= nxt:
                    nxt += 1.0 / 30
                    pair = []
                    for cam in cams:
                        cam.lookat[:] = data.xpos[ids[3]]
                        ren.update_scene(data, cam)
                        pair.append(ren.render().copy())
                    frames.append(np.hstack(pair))
        out = os.path.join(OUT, cfg, "demo_mu01.mp4")
        imageio.mimsave(out, frames, fps=30, macro_block_size=1)
        print(f"  demo -> {out}")
    gc.TORSO_CONTROLLER = None
    return rows


all_rows = {}
for cfg in CONF:
    r = eval_config(cfg)
    if r:
        all_rows[cfg] = r

# ---- cross figures ----
os.makedirs(os.path.join(OUT, "cross"), exist_ok=True)
KCOL = {0.0: "tab:blue", 2.0: "tab:red"}
plt.figure(figsize=(7, 5))
for cfg, rows in all_rows.items():
    kappa, com = CONF[cfg]
    xs = [r["torso_roll_rms_deg"] for r in rows if r["mu"] == 0.1 and r["survived"]]
    ys = [r["net_fwd"] for r in rows if r["mu"] == 0.1 and r["survived"]]
    plt.scatter(xs, ys, s=28, alpha=0.75, color=KCOL[kappa],
                marker={1.05: "o", 1.20: "s", 1.31: "^"}[com],
                label=f"{cfg} (κ={kappa:g}, COM {com})")
plt.xlabel("measured torso roll RMS [deg]"); plt.ylabel("net forward speed [m/s]")
plt.title("finalists @ $\\mu$=0.1 — torso roll vs speed")
plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cross", "roll_to_speed.png"), dpi=130)
plt.close()

plt.figure(figsize=(7, 4.5))
labels, series = [], []
for cfg, rows in all_rows.items():
    v = [r["ds_move_frac"] for r in rows if r["mu"] == 0.1 and r["survived"]
         and np.isfinite(r["ds_move_frac"])]
    if v:
        labels.append(cfg); series.append(v)
plt.violinplot(series, showmedians=True)
plt.xticks(range(1, len(labels) + 1), labels)
plt.ylabel("ds_move_frac (travel while both feet loaded)")
plt.title("finalists @ $\\mu$=0.1 — shuffle vs stepping")
plt.grid(alpha=0.3, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "cross", "ds_move_mu01.png"), dpi=130)
plt.close()
print("cross figures written")
