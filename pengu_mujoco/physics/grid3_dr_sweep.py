#!/usr/bin/env python
"""GRID-3 DR re-sweep of k0 (Gait 1, kappa=0) with in-the-loop domain randomization.

Same 454,500-cell grid as grid3_kappa_sweep, but each cell is scored over K randomized
repeats instead of one deterministic mu=0.7 trial. Per repeat we sample:
  - floor friction  mu  ~ U(0.45, 0.90)
  - torso mass       f  ~ U(0.85, 1.15)  (scales easytorso mass + inertia, +-15%)
  - initial pose     yaw ~ U(-5,5) deg, pitch ~ U(-3,3) deg, lateral ~ U(-1,1) cm
Randomization is seeded per (cell-index, repeat) so a resumed cell reproduces exactly.

A repeat "passes" = survived AND forward-facing (heading_align>0.5) AND net_fwd>0.05 AND
slip_ratio<=0.15. Per cell we log pass_rate + net_fwd mean/worst + slip/heading means, so
robustness (not single-mu speed) drives which cells win. This addresses the finding that a
single-mu-point verdict is a non-monotonic coin-flip (see dr_filter.py / CLAUDE.md).

usage:
  python physics/grid3_dr_sweep.py count
  python physics/grid3_dr_sweep.py initcsv
  N_SHARDS=12 SHARD_ID=3 python physics/grid3_dr_sweep.py
env: GRID3_SMOKE=1 -> tiny grid;  DR_K -> repeats per cell (default 5);
     KAPPA=2 -> Gait 2 re-sweep (default 0 = Gait 1); tag/filename follow KAPPA (k0dr/k2dr);
     PENGU_MODEL=1.20|1.31 -> COM-ladder model (default v3 == the 1.05 rung); the COM tag
     is baked into the filename (grid3_com120_k0dr etc.; v3 filenames unchanged).
"""
import os, sys, csv
os.environ.setdefault("PENGU_MODEL", "v3")
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID

# penguV3 == the 1.05 COM rung; 1.20/1.31 are the hardened COM-ladder exports
assert gc.XML_PATH.endswith(("penguV3/scene.xml", "pengu1_20/scene.xml",
                             "pengu1_31/scene.xml")), gc.XML_PATH

K = int(os.environ.get("DR_K", "5"))
MU_LO, MU_HI = 0.45, 0.90
MASS_JIT = 0.15                       # +-15% on easytorso mass/inertia
YAW_DEG, PITCH_DEG, LAT_M = 5.0, 3.0, 0.01
NET_MIN, HEAD_MIN, SLIP_OK = 0.05, 0.5, 0.15
SMOKE = os.environ.get("GRID3_SMOKE", "") == "1"
KAPPA = float(os.environ.get("KAPPA", "0"))            # 0 = Gait 1 (k0dr); 2 = Gait 2 (k2dr)
_KTAG = ("%g" % KAPPA).replace("-", "m").replace(".", "p")   # 0->"0", 2->"2", 0.5->"0p5"
_PM = os.environ.get("PENGU_MODEL", "v3")
_CTAG = "" if _PM == "v3" else "com" + _PM.replace(".", "") + "_"   # 1.20 -> "com120_"

if SMOKE:
    FREQS    = np.round(np.arange(1.60, 1.8001, 0.10), 3)
    HIP_PHIS = np.array([250.0, 260.0])
    LEG_AMPS = np.array([115.0]); HIP_AMPS = np.array([28.0]); HIP_OFFS = np.array([10.0])
else:
    FREQS    = np.round(np.arange(1.00, 2.0001, 0.01), 3)   # 101
    HIP_PHIS = np.round(np.arange(0.0, 350.01, 10.0), 1)    # 36
    LEG_AMPS = np.array([85.0, 95.0, 105.0, 115.0, 125.0])  # 5
    HIP_AMPS = np.array([12.0, 16.0, 20.0, 24.0, 28.0])     # 5
    HIP_OFFS = np.array([10.0, 20.0, 30.0, 40.0, 50.0])     # 5

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]
DR_FIELDS = ["pass_rate", "surv_rate", "net_fwd_mean", "net_fwd_min",
             "slip_mean", "head_mean"]
TAG = f"grid3_{_CTAG}k{_KTAG}dr" + ("_smoke" if SMOKE else "")


def cells():
    for ho in HIP_OFFS:
        for f in FREQS:
            for hp in HIP_PHIS:
                for la in LEG_AMPS:
                    for ha in HIP_AMPS:
                        yield (float(f), float(hp), float(la), float(ha), float(ho))


def main():
    combos = list(cells())
    outdir = os.path.join(_ROOT, "results", "gait_sweep"); os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"sweep_v3_{TAG}_{'_'.join(AXNAMES)}.csv")
    fields = AXNAMES + DR_FIELDS

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(f"cells={len(combos)}  K={K}  trials={len(combos)*K}  "
              f"csv={os.path.basename(csv_path)}"); return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        print(f"# initcsv {csv_path}  cells={len(combos)}  K={K}"); return

    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=KAPPA, measure_after=gs.SETTLE)
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    nom_m = float(model.body_mass[tid]); nom_I = model.body_inertia[tid].copy()

    done = gs._load_done(csv_path, AXNAMES)
    print(f"# GRID3-DR {_CTAG}k{_KTAG} (model={_PM} kappa={KAPPA})  cells={len(combos)}  done={len(done)}  K={K}  "
          f"shard={shard_id}/{n_shards}  mu~U({MU_LO},{MU_HI}) mass+-{MASS_JIT}")
    f = open(csv_path, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    n_mine = 0
    for i, combo in enumerate(combos):
        if i % n_shards != shard_id:
            continue
        n_mine += 1
        key = tuple(round(v, 4) for v in combo)
        if key in done:
            continue
        p = dict(zip(AXNAMES, combo))
        hip_off = p.pop("hip_off")
        surv = []; netf = []; slp = []; hd = []; passes = 0
        for r in range(K):
            rng = np.random.default_rng(i * 100 + r)          # deterministic per (cell,rep)
            mu = float(rng.uniform(MU_LO, MU_HI))
            fac = float(rng.uniform(1 - MASS_JIT, 1 + MASS_JIT))
            gs.FLOOR_MU = mu
            model.body_mass[tid] = nom_m * fac
            model.body_inertia[tid] = nom_I * fac
            gs.POSE_JITTER = {"yaw": np.radians(rng.uniform(-YAW_DEG, YAW_DEG)),
                              "pitch": np.radians(rng.uniform(-PITCH_DEG, PITCH_DEG)),
                              "lat": float(rng.uniform(-LAT_M, LAT_M))}
            gs.CONDITION["hip_off"] = hip_off
            rr = gs.run_trial(model, data, ids, dict(p))
            sv = int(rr["survived"]); nf = rr["net_fwd_speed"]
            he = rr["heading_align"]; sl = rr["slip_ratio"]
            surv.append(sv); netf.append(nf)
            if np.isfinite(sl): slp.append(sl)
            if np.isfinite(he): hd.append(he)
            ok = (sv and np.isfinite(he) and he > HEAD_MIN
                  and nf > NET_MIN and np.isfinite(sl) and sl <= SLIP_OK)
            passes += int(ok)
        model.body_mass[tid] = nom_m; model.body_inertia[tid] = nom_I     # restore nominal
        row = {n: round(v, 4) for n, v in zip(AXNAMES, combo)}
        row["pass_rate"] = round(passes / K, 3)
        row["surv_rate"] = round(float(np.mean(surv)), 3)
        row["net_fwd_mean"] = round(float(np.mean(netf)), 4)
        row["net_fwd_min"] = round(float(np.min(netf)), 4)
        row["slip_mean"] = round(float(np.mean(slp)), 4) if slp else float("nan")
        row["head_mean"] = round(float(np.mean(hd)), 4) if hd else float("nan")
        w.writerow(row); f.flush()
        if n_mine % 25 == 0:
            print(f"  [shard{shard_id} {i+1}/{len(combos)}] "
                  + " ".join(f"{n}={row[n]}" for n in AXNAMES)
                  + f" | pass={row['pass_rate']} netfwd_mean={row['net_fwd_mean']}")
    f.close()
    gs.POSE_JITTER = None
    if n_shards > 1:
        open(f"{csv_path}.shard{shard_id}of{n_shards}.done", "w").close()
        if all(os.path.exists(f"{csv_path}.shard{s}of{n_shards}.done")
               for s in range(n_shards)):
            open(csv_path + ".done", "w").close()
            print(f"# ALL {n_shards} shards complete -> {os.path.basename(csv_path)}.done")
    else:
        open(csv_path + ".done", "w").close()


if __name__ == "__main__":
    main()
