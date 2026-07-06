#!/usr/bin/env python
"""
GRID-2 Stage A: walkability sweep for the min_mu landscape study (docs/minmu_grid_design.md).

penguV3, penguin mass, floor mu=0.7, one run_trial per cell. Differences vs the fine3c
sweep this is derived from:
  - freq extended to 2.00 (upright's low-friction gaits live at 1.7-2.0; fine3c's 1.5
    cap never gave the torso_amp=0 slice its space)
  - torso_amp includes 0 -> "upright" is a slice of the SAME grid, classified post-hoc
    by measured torso_stance_corr in Stage B (phi labels lie across gait families)
  - torso_phi COLLAPSES to a single value when torso_amp==0 (phase of a zero-amplitude
    sine is meaningless -> 17x fewer wasted cells than a full product)
  - hip_off (forward-pitch posture) is a swept dimension, not a constant

Reuses gait_sweep.run_trial (identical metrics/stance gates) and clones its harness
semantics exactly: initcsv-once, append-per-cell CSV, resume by axis-tuple, modulo
sharding via N_SHARDS/SHARD_ID, per-shard + master .done sentinels.

usage:
  python physics/grid2_sweep.py count        # print cell count and exit
  python physics/grid2_sweep.py initcsv      # write header once (BEFORE sharded workers)
  N_SHARDS=16 SHARD_ID=3 python physics/grid2_sweep.py   # one worker
env: GRID2_SMOKE=1 -> tiny decimated grid for an end-to-end pipe check.
"""
import os, sys, csv

os.environ.setdefault("PENGU_MODEL", "v3")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs

assert gc.XML_PATH.endswith("penguV3/scene.xml"), gc.XML_PATH

SMOKE = os.environ.get("GRID2_SMOKE", "") == "1"

if SMOKE:
    FREQS      = np.round(np.arange(1.20, 1.3001, 0.05), 3)          # 3
    HIP_PHIS   = np.array([180.0, 210.0])                            # 2
    LEG_AMPS   = np.array([115.0])                                   # 1
    HIP_AMPS   = np.array([22.0])                                    # 1
    TORSO_AMPS = np.array([0.0, 20.0])                               # 2
    TORSO_PHIS = np.array([0.0, 180.0])                              # 2 (->1 when amp=0)
    HIP_OFFS   = np.array([30.0])                                    # 1
else:
    FREQS      = np.round(np.arange(1.00, 2.0001, 0.01), 3)          # 101
    HIP_PHIS   = np.round(np.arange(0.0, 350.01, 10.0), 1)           # 36
    LEG_AMPS   = np.array([95.0, 105.0, 115.0])                      # 3
    HIP_AMPS   = np.array([16.0, 20.0, 24.0])                        # 3
    TORSO_AMPS = np.array([0.0, 10.0, 20.0])                         # 3
    TORSO_PHIS = np.round(np.arange(0.0, 315.01, 45.0), 1)           # 8 (->1 when amp=0)
    HIP_OFFS   = np.array([10.0, 20.0, 30.0, 40.0])                  # 4

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "torso_amp", "torso_phi", "hip_off"]
TAG = "grid2smoke" if SMOKE else "grid2"


def cells():
    """deterministic enumeration; torso_phi collapses to 0.0 when torso_amp == 0."""
    for ho in HIP_OFFS:
        for ta in TORSO_AMPS:
            tphis = [0.0] if ta == 0.0 else TORSO_PHIS
            for tp in tphis:
                for f in FREQS:
                    for hp in HIP_PHIS:
                        for la in LEG_AMPS:
                            for ha in HIP_AMPS:
                                yield (float(f), float(hp), float(la), float(ha),
                                       float(ta), float(tp), float(ho))


def main():
    combos = list(cells())
    outdir = os.path.join(_ROOT, "results", "gait_sweep")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"sweep_v3_{TAG}_{'_'.join(AXNAMES)}.csv")
    fields = AXNAMES + gs.METRIC_FIELDS

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(f"cells={len(combos)}  csv={os.path.basename(csv_path)}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        print(f"# initcsv {csv_path}  total cells={len(combos)}")
        return

    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)

    done = gs._load_done(csv_path, AXNAMES)
    new_file = not os.path.exists(csv_path)
    print(f"# GRID2 stage A  cells={len(combos)}  done={len(done)}  "
          f"shard={shard_id}/{n_shards}  mu=0.7")
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file and shard_id == 0:
        w.writeheader(); f.flush()
    gs.FLOOR_MU = 0.7
    n_mine = 0
    for i, combo in enumerate(combos):
        if i % n_shards != shard_id:
            continue
        n_mine += 1
        key = tuple(round(v, 4) for v in combo)
        if key in done:
            continue
        p = dict(zip(AXNAMES, combo))
        gs.CONDITION["hip_off"] = p.pop("hip_off")   # hip_off rides CONDITION, not p
        r = gs.run_trial(model, data, ids, p)
        row = {n: round(v, 4) for n, v in zip(AXNAMES, combo)}
        row.update(r)
        w.writerow(row); f.flush()
        if n_mine % 50 == 0:
            print(f"  [shard{shard_id} {i + 1}/{len(combos)}] " +
                  " ".join(f"{n}={row[n]}" for n in AXNAMES) +
                  f" | valid={r['valid']} netfwd={r['net_fwd_speed']} mu={r['mu_req_p95']}")
    f.close()
    if n_shards > 1:
        open(f"{csv_path}.shard{shard_id}of{n_shards}.done", "w").close()
        if all(os.path.exists(f"{csv_path}.shard{s}of{n_shards}.done")
               for s in range(n_shards)):
            open(csv_path + ".done", "w").close()
            print(f"# ALL {n_shards} shards complete -> {os.path.basename(csv_path)}.done")
    else:
        open(csv_path + ".done", "w").close()
        print(f"# wrote {csv_path}  (all {len(combos)} cells complete)")


if __name__ == "__main__":
    main()
