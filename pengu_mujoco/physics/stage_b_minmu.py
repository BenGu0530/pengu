#!/usr/bin/env python
"""
GRID-2 Stage B: min_mu ladder + kinematic readouts on the Stage-A clean walkers.

For each clean walker (survived & net_fwd>0.0115 & single_frac>0.6 at mu=0.7 in Stage A):
  - instrumented rollout @ mu=0.7: torso_stance_corr (penguin-posture signature),
    foot_roll_amp, foot_pitch_amp (plan's roll-vs-pitch single-peak readout)
  - mu ladder 0.5 -> 0.06: min_mu_to_walk = lowest rung with a CLEAN walk, taken
    CONTIGUOUS from 0.7 down (rejects the low-mu skidding artifact)

Readout layer for the paper's gait x mass matrix (friction as a measurement, not the
subject). min_mu may diverge from Stage A's mu_req_p95 (peak demand != slip robustness).

Sharding: works on the SHARED clean-walker list (grid2_cleanwalkers.csv, committed) so
Linux (shards 0-11) and Mac (12-15) partition the SAME ordered list by index % 16.

usage:
  python physics/stage_b_minmu.py initlist    # build grid2_cleanwalkers.csv from merged Stage A
  python physics/stage_b_minmu.py initcsv      # output header once
  N_SHARDS=16 SHARD_ID=3 python physics/stage_b_minmu.py
env: STAGEB_SMOKE=1 -> only the first 24 clean walkers (pipe check).
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
from friction_utils import set_floor_friction

assert gc.XML_PATH.endswith("penguV3/scene.xml"), gc.XML_PATH

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "torso_amp", "torso_phi", "hip_off"]
MU_LADDER = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]   # 0.7 already clean (selection)
WALK_TIME = gs.SIM_DURATION - gs.SETTLE
NETV_OK = 0.15 / WALK_TIME
SINGLE_OK = 0.6
F_LOAD = 4.0

OUTDIR = os.path.join(_ROOT, "results", "gait_sweep")
MERGED = os.path.join(OUTDIR, "grid2_merged.csv")
LIST = os.path.join(OUTDIR, "grid2_cleanwalkers.csv")
OUTCSV = os.path.join(OUTDIR, "grid2_stageB_minmu.csv")
OUTFIELDS = AXNAMES + ["a_netfwd", "a_single", "a_mureq", "min_mu_to_walk",
                       "torso_corr", "foot_roll_amp", "foot_pitch_amp"]


def build_list():
    """filter merged Stage A -> ordered clean-walker list (params + Stage-A refs)."""
    n = 0
    with open(MERGED) as fi, open(LIST, "w", newline="") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo)
        header = next(r)
        w.writerow(AXNAMES + ["a_netfwd", "a_single", "a_mureq"])
        for row in r:
            # cols: 0-6 params, 7 survived, 10 net_fwd, 13 single_frac, 21 mu_req
            if row[7] == "1" and float(row[10]) > NETV_OK and float(row[13]) > SINGLE_OK:
                w.writerow(row[0:7] + [row[10], row[13], row[21]])
                n += 1
    print(f"# clean walkers -> {LIST}  n={n}")


def load_list():
    rows = []
    with open(LIST) as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append(d)
    return rows


def _p(d):
    return {k: float(d[k]) for k in AXNAMES}


def clean(r):
    return bool(r["survived"]) and r["net_fwd_speed"] > NETV_OK and r["single_frac"] > SINGLE_OK


def measure(model, data, ids, p, mu):
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = p["hip_off"]
    pp = {k: v for k, v in p.items() if k != "hip_off"}
    return gs.run_trial(model, data, ids, pp)


def instrumented(model, data, ids, p, mu=0.7):
    """one rollout: torso_stance_corr + foot roll/pitch peak-to-peak amplitude (deg)."""
    set_floor_friction(model, mu)
    gs.CONDITION["hip_off"] = p["hip_off"]
    pp = {k: v for k, v in p.items() if k != "hip_off"}
    gs._set_gait(pp)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = ids
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    fb = {s: b for b, s in foot_bid.items()}
    rfoot = fb["R"]
    agree, rolls, pitches = [], [], []
    while data.time < gs.SIM_DURATION:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.time < gs.SETTLE:
            continue
        R = data.xmat[rfoot].reshape(3, 3)     # foot body-to-world rotation
        rolls.append(np.degrees(np.arctan2(R[2, 0], R[2, 2])))    # about fwd y (lateral tilt)
        pitches.append(np.degrees(np.arctan2(R[2, 1], R[2, 2])))  # about lateral x (toe up/down)
        fn = {"L": 0.0, "R": 0.0}
        for ci in range(data.ncon):
            c = data.contact[ci]
            side = foot_geom.get(c.geom1) or foot_geom.get(c.geom2)
            if side is None or (c.geom1 != floor_id and c.geom2 != floor_id):
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, f6)
            fn[side] += abs(f6[0])
        if (fn["L"] > F_LOAD) != (fn["R"] > F_LOAD):
            s = "L" if fn["L"] > F_LOAD else "R"
            mid = (data.xipos[fb["L"]][0] + data.xipos[fb["R"]][0]) / 2
            sf = np.sign(data.xipos[fb[s]][0] - mid)
            st = np.sign(data.xipos[tb][0] - mid)
            if sf != 0 and st != 0:
                agree.append(1.0 if sf == st else -1.0)

    def amp(a):
        # atan2 wraps at +-180: a foot crossing the branch cut faked amp=180 (2% of cells).
        # unwrap the series before taking peak-to-peak.
        if not a:
            return float("nan")
        d = np.degrees(np.unwrap(np.radians(np.asarray(a))))
        return round((d.max() - d.min()) / 2.0, 2)
    corr = round(float(np.mean(agree)), 3) if agree else float("nan")
    return corr, amp(rolls), amp(pitches)


def min_mu_contiguous(model, data, ids, p):
    lo = 0.7
    for mu in MU_LADDER:
        if clean(measure(model, data, ids, p, mu)):
            lo = mu
        else:
            break
    return lo


def worker():
    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))
    rows = load_list()
    if os.environ.get("STAGEB_SMOKE") == "1":
        rows = rows[:24]
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)

    done = set()
    if os.path.exists(OUTCSV):
        with open(OUTCSV) as f:
            for d in csv.DictReader(f):
                try:
                    done.add(tuple(round(float(d[n]), 4) for n in AXNAMES))
                except (KeyError, ValueError):
                    pass
    new = not os.path.exists(OUTCSV)
    fo = open(OUTCSV, "a", newline="")
    w = csv.DictWriter(fo, fieldnames=OUTFIELDS)
    if new and shard_id == 0:
        w.writeheader(); fo.flush()
    print(f"# StageB cleanwalkers={len(rows)} done={len(done)} shard={shard_id}/{n_shards}")
    n_mine = 0
    for i, d in enumerate(rows):
        if i % n_shards != shard_id:
            continue
        n_mine += 1
        p = _p(d)
        key = tuple(round(p[n], 4) for n in AXNAMES)
        if key in done:
            continue
        corr, roll, pitch = instrumented(model, data, ids, p)
        min_mu = min_mu_contiguous(model, data, ids, p)
        row = {n: round(p[n], 4) for n in AXNAMES}
        row.update(a_netfwd=d["a_netfwd"], a_single=d["a_single"], a_mureq=d["a_mureq"],
                   min_mu_to_walk=min_mu, torso_corr=corr,
                   foot_roll_amp=roll, foot_pitch_amp=pitch)
        w.writerow(row); fo.flush()
        if n_mine % 50 == 0:
            print(f"  [shard{shard_id} {i+1}/{len(rows)}] min_mu={min_mu} "
                  f"corr={corr} roll={roll} pitch={pitch}")
    fo.close()
    if n_shards > 1:
        open(f"{OUTCSV}.shard{shard_id}of{n_shards}.done", "w").close()
        print(f"# shard {shard_id} done ({n_mine} cells)")
    else:
        open(OUTCSV + ".done", "w").close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "initlist":
        build_list(); return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(OUTCSV):
            with open(OUTCSV, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=OUTFIELDS).writeheader()
        print(f"# initcsv {OUTCSV}"); return
    worker()


if __name__ == "__main__":
    main()
