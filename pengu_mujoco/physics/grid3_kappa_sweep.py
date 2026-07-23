#!/usr/bin/env python
"""
GRID-3: gait sweep for ONE design/control cell of the co-design table.

Fixed for this run (the selected table cell + control mode):
  - model penguV3, COM ratio 1.108 (current), foot gap current, floor mu=0.7
  - torso control = TorsoKappaPID with KAPPA (default 0.0 = Gait 1, world-upright torso)
Swept (find the best gait in this cell):
  - freq, hip_phi, leg_amp, hip_amp, hip_off

vs GRID-2: the open-loop torso axes (torso_amp, torso_phi) are GONE — the torso is now
the reactive PID at fixed kappa. Two extra readouts per cell measure whether the control
mode actually did its job: torso_roll_rms (Gait-1 benchmark: ~0 = torso held upright) and
torso_sat_frac (fraction of the walk window the +-4.1 N.m motor was clamped).

Reuses gait_sweep.run_trial (same metrics/stance gates) + its harness semantics
(initcsv-once, append-per-cell, resume by axis-tuple, modulo sharding, sentinels).

usage:
  KAPPA=0 python physics/grid3_kappa_sweep.py count
  KAPPA=0 python physics/grid3_kappa_sweep.py initcsv
  KAPPA=0 N_SHARDS=16 SHARD_ID=3 python physics/grid3_kappa_sweep.py
env: GRID3_SMOKE=1 -> tiny grid.  KAPPA -> the fixed follow gain (default 0.0).
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
from torso_control import TorsoKappaPID

assert gc.XML_PATH.endswith("penguV3/scene.xml"), gc.XML_PATH

KAPPA = float(os.environ.get("KAPPA", "0.0"))
SMOKE = os.environ.get("GRID3_SMOKE", "") == "1"

if SMOKE:
    FREQS    = np.round(np.arange(1.60, 1.8001, 0.10), 3)   # 3
    HIP_PHIS = np.array([180.0, 210.0])                     # 2
    LEG_AMPS = np.array([105.0])                            # 1
    HIP_AMPS = np.array([20.0])                             # 1
    HIP_OFFS = np.array([20.0, 30.0])                       # 2
else:
    FREQS    = np.round(np.arange(1.00, 2.0001, 0.01), 3)   # 101
    HIP_PHIS = np.round(np.arange(0.0, 350.01, 10.0), 1)    # 36
    LEG_AMPS = np.array([85.0, 95.0, 105.0, 115.0, 125.0])  # 5
    HIP_AMPS = np.array([12.0, 16.0, 20.0, 24.0, 28.0])     # 5
    HIP_OFFS = np.array([10.0, 20.0, 30.0, 40.0, 50.0])     # 5

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"]
# torso benchmark readouts + (opt-out) whole-robot CoM vs stance-foot contact point.
# COM_STANCE=1 (default) adds the 4 CoM columns for the Gait-2 cells; pass COM_STANCE=0 to
# reproduce the original 22-column schema (needed to finish k0, whose rows predate this).
TRACK_CS = os.environ.get("COM_STANCE", "1") == "1"
gs.TRACK_COM_STANCE = TRACK_CS
EXTRA = ["torso_roll_rms", "torso_sat_frac"] + (gs.COM_STANCE_FIELDS if TRACK_CS else [])
KTAG = f"k{KAPPA:g}".replace(".", "p")
TAG = f"grid3_{KTAG}" + ("_smoke" if SMOKE else "")


def cells():
    for ho in HIP_OFFS:
        for f in FREQS:
            for hp in HIP_PHIS:
                for la in LEG_AMPS:
                    for ha in HIP_AMPS:
                        yield (float(f), float(hp), float(la), float(ha), float(ho))


def main():
    combos = list(cells())
    outdir = os.path.join(_ROOT, "results", "gait_sweep")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"sweep_v3_{TAG}_{'_'.join(AXNAMES)}.csv")
    fields = AXNAMES + gs.METRIC_FIELDS + EXTRA

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(f"kappa={KAPPA} cells={len(combos)}  csv={os.path.basename(csv_path)}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        print(f"# initcsv {csv_path}  cells={len(combos)}  kappa={KAPPA}")
        return

    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    # install the fixed-kappa reactive torso; measure roll only after the settle window
    pid = TorsoKappaPID(model, kappa=KAPPA, measure_after=gs.SETTLE)
    gc.TORSO_CONTROLLER = pid

    done = gs._load_done(csv_path, AXNAMES)
    new_file = not os.path.exists(csv_path)
    print(f"# GRID3 kappa={KAPPA}  cells={len(combos)}  done={len(done)}  "
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
        gs.CONDITION["hip_off"] = p.pop("hip_off")
        r = gs.run_trial(model, data, ids, p)     # run_trial resets data -> pid auto-resets
        row = {n: round(v, 4) for n, v in zip(AXNAMES, combo)}
        row.update(r)
        row["torso_roll_rms"] = round(pid.roll_rms(), 3)
        row["torso_sat_frac"] = round(pid.saturation_frac(), 3)
        w.writerow(row); f.flush()
        if n_mine % 50 == 0:
            print(f"  [shard{shard_id} {i + 1}/{len(combos)}] " +
                  " ".join(f"{n}={row[n]}" for n in AXNAMES) +
                  f" | valid={r['valid']} netfwd={r['net_fwd_speed']} "
                  f"roll_rms={row['torso_roll_rms']} sat={row['torso_sat_frac']}")
    f.close()
    if n_shards > 1:
        open(f"{csv_path}.shard{shard_id}of{n_shards}.done", "w").close()
        if all(os.path.exists(f"{csv_path}.shard{s}of{n_shards}.done")
               for s in range(n_shards)):
            open(csv_path + ".done", "w").close()
            print(f"# ALL {n_shards} shards complete -> {os.path.basename(csv_path)}.done")
    else:
        open(csv_path + ".done", "w").close()
        print(f"# wrote {csv_path}")


if __name__ == "__main__":
    main()
