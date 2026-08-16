#!/usr/bin/env python
"""Top up a K=1 GRID-4 map to K=TARGET_K on SELECTED rows, without re-running trial r=0.

Trials are seeded per (cell, mu, repeat): a K=1 sweep ran exactly repeat r=0, so this
script runs r=1..TARGET_K-1 for each selected (cell,mu) and merges with the stored
r=0 aggregates -- numerically identical (to CSV rounding, ~1e-4) to having run K=TARGET_K
from the start.

usage:
  CONFIG=c1 python physics/topup_k.py <base_k1_csv> <select_csv|->  [out_csv]
    select_csv: rows to upgrade, needs the 6 axis columns (header ok). '-' = ALL rows.
    out: default <base>.topupK<TARGET_K>.csv -- contains ONLY the upgraded rows
         (full 12-col schema, K=TARGET_K aggregates). Analysis: topup rows override base.
env: TARGET_K (default 5); CONFIG/GRID4_SMOKE must match the base sweep.
Resume: safe -- rows already in the out csv are skipped on relaunch. Shardable with
N_SHARDS/SHARD_ID over the selection.
"""
import os, sys, csv
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
import grid4_sweep as g4

TARGET_K = int(os.environ.get("TARGET_K", "5"))
BASE_K = 1                                   # merge math below assumes the base ran r=0 only

def main():
    base_csv = sys.argv[1]
    sel_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else base_csv + f".topupK{TARGET_K}.csv"
    fields = g4.AXNAMES + g4.DR_FIELDS

    # canonical cell ordering -> cell index (must match grid4_sweep.cells())
    combos = list(g4.cells())
    cell_idx = {tuple(round(v, 4) for v in c): i for i, c in enumerate(combos)}
    mu_idx = {round(float(m), 4): j for j, m in enumerate(g4.MUS)}

    base = {}
    with open(base_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = tuple(round(float(row[n]), 4) for n in g4.AXNAMES)
            except (ValueError, TypeError):
                continue
            base[key] = row
    print(f"# base rows: {len(base)}")

    if sel_arg == "-":
        selected = list(base.keys())
    else:
        selected = []
        with open(sel_arg) as f:
            for row in csv.DictReader(f):
                selected.append(tuple(round(float(row[n]), 4) for n in g4.AXNAMES))
    done = gs._load_done(out_csv, g4.AXNAMES) if os.path.exists(out_csv) else set()
    n_shards = int(os.environ.get("N_SHARDS", "1")); shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = g4.apply_com_variant(model, g4.COM_TARGET)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=g4.KAPPA, measure_after=gs.SETTLE)
    print(f"# TOPUP {g4.CONFIG} (kappa={g4.KAPPA} com={got:.4f}) K {BASE_K}->{TARGET_K}  "
          f"selected={len(selected)} done={len(done)} shard={shard_id}/{n_shards}")

    new_file = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        w.writeheader(); f.flush()
    for k_i, key in enumerate(selected):
        if k_i % n_shards != shard_id: continue
        if key in done or key not in base: continue
        b = base[key]
        cell = key[:5]; mu0 = key[5]
        i = cell_idx[cell]; mi = mu_idx[mu0]
        # base (r=0) aggregates -> counts/values
        passes = int(round(float(b["pass_rate"]) * BASE_K))
        surv = [int(round(float(b["surv_rate"])))]
        netf = [float(b["net_fwd_mean"])]
        slp = [] if b["slip_mean"] in ("nan", "") else [float(b["slip_mean"])]
        hd = [] if b["head_mean"] in ("nan", "") else [float(b["head_mean"])]
        p = dict(zip(g4.AXNAMES[:5], cell)); hip_off = p.pop("hip_off")
        for r in range(BASE_K, TARGET_K):
            rng = np.random.default_rng((i * len(g4.MUS) + mi) * 100 + r)
            gs.FLOOR_MU = float(mu0) * float(rng.uniform(1 - g4.MU_JIT, 1 + g4.MU_JIT))
            gs.POSE_JITTER = {"yaw": np.radians(rng.uniform(-g4.YAW_DEG, g4.YAW_DEG)),
                              "pitch": np.radians(rng.uniform(-g4.PITCH_DEG, g4.PITCH_DEG)),
                              "lat": float(rng.uniform(-g4.LAT_M, g4.LAT_M))}
            gs.CONDITION["hip_off"] = hip_off
            rr = gs.run_trial(model, data, ids, dict(p))
            sv = int(rr["survived"]); nf = rr["net_fwd_speed"]
            he = rr["heading_align"]; sl = rr["slip_ratio"]
            surv.append(sv); netf.append(nf)
            if np.isfinite(sl): slp.append(sl)
            if np.isfinite(he): hd.append(he)
            ok = (sv and np.isfinite(he) and he > g4.HEAD_MIN and nf > g4.NET_MIN)
            passes += int(ok)
        row = {n: v for n, v in zip(g4.AXNAMES, key)}
        row["pass_rate"] = round(passes / TARGET_K, 3)
        row["surv_rate"] = round(float(np.mean(surv)), 3)
        row["net_fwd_mean"] = round(float(np.mean(netf)), 4)
        row["net_fwd_min"] = round(float(np.min(netf)), 4)
        row["slip_mean"] = round(float(np.mean(slp)), 4) if slp else float("nan")
        row["head_mean"] = round(float(np.mean(hd)), 4) if hd else float("nan")
        w.writerow(row); f.flush()
    f.close()
    gs.POSE_JITTER = None
    print("# topup done")


if __name__ == "__main__":
    main()
