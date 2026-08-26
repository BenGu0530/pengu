#!/usr/bin/env python
"""Top up a K=1 GRID-5 map to K=TARGET_K on SELECTED rows, without re-running trial r=0.

Duplicated from physics/topup_k.py (GRID-4, untouched backup) and adapted: grid5
protocol switches (staged start, rest lean, ramped hip_off, extended metrics) are
inherited by importing grid5_sweep; the merge covers the extended columns too.

Trials are seeded per (cell, mu, repeat): the K=1 map ran exactly repeat r=0, so this
runs r=1..TARGET_K-1 per selected (cell,mu) and merges with the stored r=0 aggregates
-- numerically identical (to CSV rounding) to a native K=TARGET_K sweep.

usage:
  CONFIG=c1 python topup_k.py <base_k1_csv> <select_csv|->  [out_csv]
    select_csv: rows to upgrade, needs the 6 axis columns (header ok). '-' = ALL rows.
    out: default <base>.topupK<TARGET_K>.csv -- ONLY the upgraded rows (full grid5
         schema, K=TARGET_K aggregates). Analysis: topup rows override base.
env: TARGET_K (default 5); CONFIG/GRID5_SMOKE must match the base sweep.
Resume: safe -- rows already in the out csv are skipped. Shardable with N_SHARDS/SHARD_ID.
"""
import os, sys, csv, json
os.environ["PENGU_MODEL"] = "1.31"
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                      # grid5 copies FIRST

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID
import grid5_sweep as g5                       # sets the grid5 protocol switches

assert gs.EXTENDED_METRICS and gs.STAGED_START and gc.RAMP_HIP_OFFSET

TARGET_K = int(os.environ.get("TARGET_K", "5"))
BASE_K = 1                                   # merge math below assumes the base ran r=0 only
EXT_NUM = [k for k in g5.EXT_AGG if k != "fall_phase"]


def check_base_manifest(base_csv):
    mpath = base_csv.replace(".csv", ".manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"manifest missing for base map: {mpath}")
    man = json.load(open(mpath))
    want = dict(protocol="grid5-v1", config=g5.CONFIG,
                mujoco_version=mujoco.__version__)
    for k, v in want.items():
        if man.get(k) != v:
            raise SystemExit(f"MANIFEST MISMATCH on {k!r}: base has {man.get(k)!r}, "
                             f"this process has {v!r} — refusing")
    slip_want = dict(cone_eps=gs.SLIP_CONE_EPS, v0=gs.SLIP_V0, c=gs.SLIP_C)
    if man.get("slip") != slip_want:
        raise SystemExit(f"MANIFEST MISMATCH on 'slip': base has {man.get('slip')!r}, "
                         f"this process has {slip_want!r} — refusing")


def _f(row, k):
    v = row.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_tally(s):
    out = {}
    for part in (s or "").split("|"):
        if ":" in part:
            k, v = part.split(":")
            out[k] = int(v)
    return out


def main():
    base_csv = sys.argv[1]
    sel_arg = sys.argv[2] if len(sys.argv) > 2 else "-"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else base_csv + f".topupK{TARGET_K}.csv"
    check_base_manifest(base_csv)
    fields = g5.AXNAMES + g5.DR_FIELDS + g5.EXT_AGG

    combos = list(g5.cells())
    cell_idx = {tuple(round(v, 4) for v in c): i for i, c in enumerate(combos)}
    mu_idx = {round(float(m), 4): j for j, m in enumerate(g5.MUS)}

    base = {}
    with open(base_csv) as f:
        for row in csv.DictReader(f):
            try:
                key = tuple(round(float(row[n]), 4) for n in g5.AXNAMES)
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
                selected.append(tuple(round(float(row[n]), 4) for n in g5.AXNAMES))
    done = gs._load_done(out_csv, g5.AXNAMES) if os.path.exists(out_csv) else set()
    n_shards = int(os.environ.get("N_SHARDS", "1")); shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    # calibrate at the hips-0 design neutral (identical to grid5_sweep.main)
    _lean = gc.STAND_HIP_DEG; gc.STAND_HIP_DEG = 0.0
    slide, got = g5.apply_com_variant(model, g5.COM_TARGET)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=g5.KAPPA, measure_after=0.0)
    gc.STAND_HIP_DEG = _lean
    print(f"# GRID5 TOPUP {g5.CONFIG} (kappa={g5.KAPPA} com={got:.4f}) K {BASE_K}->{TARGET_K}  "
          f"selected={len(selected)} done={len(done)} shard={shard_id}/{n_shards}")
    print(f"# start: staged rest_lean={gc.STAND_HIP_DEG}deg ramp_off={gc.RAMP_HIP_OFFSET}  "
          f"slip: eps={gs.SLIP_CONE_EPS} v0={gs.SLIP_V0} c={gs.SLIP_C}")

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
        # base (r=0) aggregates -> counts/values (K=1: aggregates ARE the r=0 trial)
        passes = int(round(float(b["pass_rate"]) * BASE_K))
        surv = [int(round(float(b["surv_rate"])))]
        netf = [float(b["net_fwd_mean"])]
        slp = [] if b["slip_mean"] in ("nan", "") else [float(b["slip_mean"])]
        hd = [] if b["head_mean"] in ("nan", "") else [float(b["head_mean"])]
        ext = {k: [] for k in EXT_NUM}
        for k in EXT_NUM:
            v = _f(b, k)
            if np.isfinite(v):
                ext[k].append(v)
        phase_tally = _parse_tally(b.get("fall_phase", ""))
        p = dict(zip(g5.AXNAMES[:5], cell)); hip_off = p.pop("hip_off")
        for r in range(BASE_K, TARGET_K):
            rng = np.random.default_rng((i * len(g5.MUS) + mi) * 100 + r)
            gs.FLOOR_MU = float(mu0) * float(rng.uniform(1 - g5.MU_JIT, 1 + g5.MU_JIT))
            gs.POSE_JITTER = {"yaw": np.radians(rng.uniform(-g5.YAW_DEG, g5.YAW_DEG)),
                              "pitch": np.radians(rng.uniform(-g5.PITCH_DEG, g5.PITCH_DEG)),
                              "lat": float(rng.uniform(-g5.LAT_M, g5.LAT_M))}
            gs.CONDITION["hip_off"] = hip_off
            rr = gs.run_trial(model, data, ids, dict(p))
            sv = int(rr["survived"]); nf = rr["net_fwd_speed"]
            he = rr["heading_align"]; sl = rr["slip_ratio"]
            surv.append(sv); netf.append(nf)
            if np.isfinite(sl): slp.append(sl)
            if np.isfinite(he): hd.append(he)
            ok = (sv and np.isfinite(he) and he > g5.HEAD_MIN and nf > g5.NET_MIN)
            passes += int(ok)
            for k in EXT_NUM:
                v = rr[k]
                if isinstance(v, (int, float)) and np.isfinite(v):
                    ext[k].append(v)
            if rr["fall_phase"]:
                phase_tally[rr["fall_phase"]] = phase_tally.get(rr["fall_phase"], 0) + 1
        row = {n: v for n, v in zip(g5.AXNAMES, key)}
        row["pass_rate"] = round(passes / TARGET_K, 3)
        row["surv_rate"] = round(float(np.mean(surv)), 3)
        row["net_fwd_mean"] = round(float(np.mean(netf)), 4)
        row["net_fwd_min"] = round(float(np.min(netf)), 4)
        row["slip_mean"] = round(float(np.mean(slp)), 4) if slp else float("nan")
        row["head_mean"] = round(float(np.mean(hd)), 4) if hd else float("nan")
        for k in EXT_NUM:
            row[k] = round(float(np.mean(ext[k])), 4) if ext[k] else float("nan")
        row["fall_phase"] = "|".join(f"{k}:{v}" for k, v in sorted(phase_tally.items()))
        w.writerow(row); f.flush()
    f.close()
    gs.POSE_JITTER = None
    print("# topup done")


if __name__ == "__main__":
    main()
