#!/usr/bin/env python
"""GRID-4: the gait x friction co-design sweep (paper Sec. III, docs/grid4_guide.md).

Base model = hardened models/pengu1_31 (native 2.2724 kg). The COM-ratio ladder
{1.05, 1.20, 1.31} is built at load time by sliding easytorso's inertial COM along
the counterweight axis (world-up at neutral stand) with total/per-link mass, geometry,
actuation and contact untouched -- bisection hits the target ratio exactly.

6 configs (paper tab:configs):
  c1 kappa=0 COM 1.05 | c2 kappa=0 1.20 | c3 kappa=0 1.31
  c4 kappa=2 COM 1.05 | c5 kappa=2 1.20 | c6 kappa=2 1.31

Axes: freq(101) x hip_phi(36) x leg_amp(5) x hip_amp(5) x hip_off(5) x mu{0.1,0.3,0.5,0.7}
    = 1,818,000 rows/config. Per (cell,mu): K=5 trials, mu jittered relative +-5%,
pose jitter yaw +-5deg / pitch +-3deg / lateral +-1cm, NO mass jitter, seeded per
(cell, mu, repeat) -> machine-independent resume by 6-axis tuple.

Pass = survived AND heading_align>0.5 AND net_fwd>0.05. Slip is RECORDED, not gated.

usage:
  CONFIG=c1 python physics/grid4_sweep.py count|initcsv
  CONFIG=c1 N_SHARDS=10 SHARD_ID=3 python physics/grid4_sweep.py
env: CONFIG=c1..c6; GRID4_SMOKE=1 -> tiny grid; DR_K -> repeats (default 5).
"""
import os, sys, csv
os.environ["PENGU_MODEL"] = "1.31"          # base = models/pengu1_31 for every config
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID

assert gc.XML_PATH.endswith("pengu1_31/scene.xml"), gc.XML_PATH

CONFIGS = {  # config -> (kappa, target com_ratio)
    "c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
    "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31),
}
CONFIG = os.environ.get("CONFIG", "c1").lower()
assert CONFIG in CONFIGS, f"CONFIG={CONFIG!r} (want c1..c6)"
KAPPA, COM_TARGET = CONFIGS[CONFIG]

K = int(os.environ.get("DR_K", "5"))
MU_JIT = 0.05                          # relative: mu * U(1-MU_JIT, 1+MU_JIT)
YAW_DEG, PITCH_DEG, LAT_M = 5.0, 3.0, 0.01
NET_MIN, HEAD_MIN = 0.05, 0.5          # slip is recorded, NOT a pass gate
SMOKE = os.environ.get("GRID4_SMOKE", "") == "1"

if SMOKE:
    FREQS    = np.round(np.arange(1.60, 1.8001, 0.10), 3)
    HIP_PHIS = np.array([250.0, 260.0])
    LEG_AMPS = np.array([115.0]); HIP_AMPS = np.array([28.0]); HIP_OFFS = np.array([10.0])
    MUS      = np.array([0.3, 0.7])
else:
    FREQS    = np.round(np.arange(1.00, 2.0001, 0.01), 3)   # 101
    HIP_PHIS = np.round(np.arange(0.0, 350.01, 10.0), 1)    # 36
    LEG_AMPS = np.array([85.0, 95.0, 105.0, 115.0, 125.0])  # 5
    HIP_AMPS = np.array([12.0, 16.0, 20.0, 24.0, 28.0])     # 5
    HIP_OFFS = np.array([10.0, 20.0, 30.0, 40.0, 50.0])     # 5
    MUS      = np.array([0.1, 0.3, 0.5, 0.7])               # 4

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu"]
DR_FIELDS = ["pass_rate", "surv_rate", "net_fwd_mean", "net_fwd_min",
             "slip_mean", "head_mean"]
TAG = f"grid4_{CONFIG}" + ("_smoke" if SMOKE else "")


def com_ratio_of(model):
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easyaxis")
    return float(d.subtree_com[1][2]) / float(d.xpos[aid][2])


def apply_com_variant(model, target):
    """Slide easytorso's inertial COM along world-up (the counterweight axis) until
    the neutral-stand COM ratio hits `target`. Masses/geometry untouched."""
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    d = mujoco.MjData(model)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, d, act, jadr)
    mujoco.mj_forward(model, d)
    up = d.xmat[tid].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    ip0 = model.body_ipos[tid].copy()

    def ratio_at(s):
        model.body_ipos[tid] = ip0 + s * up
        return com_ratio_of(model)

    lo, hi = -0.30, 0.30
    assert ratio_at(lo) < target < ratio_at(hi), (target, ratio_at(lo), ratio_at(hi))
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if ratio_at(mid) < target:
            lo = mid
        else:
            hi = mid
    s = 0.5 * (lo + hi)
    got = ratio_at(s)
    assert abs(got - target) < 1e-3, (got, target)
    return s, got


def cells():
    for ho in HIP_OFFS:
        for f in FREQS:
            for hp in HIP_PHIS:
                for la in LEG_AMPS:
                    for ha in HIP_AMPS:
                        yield (float(f), float(hp), float(la), float(ha), float(ho))


def main():
    combos = list(cells())
    n_rows = len(combos) * len(MUS)
    outdir = os.path.join(_ROOT, "results", "gait_sweep"); os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"sweep_{TAG}_{'_'.join(AXNAMES)}.csv")
    fields = AXNAMES + DR_FIELDS

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(f"config={CONFIG} kappa={KAPPA} com={COM_TARGET}  cells={len(combos)}  "
              f"mus={list(MUS)}  rows={n_rows}  K={K}  trials={n_rows*K}  "
              f"csv={os.path.basename(csv_path)}"); return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        print(f"# initcsv {csv_path}  rows={n_rows}  K={K}"); return

    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    slide, got = apply_com_variant(model, COM_TARGET)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=KAPPA, measure_after=gs.SETTLE)

    done = gs._load_done(csv_path, AXNAMES)
    print(f"# GRID4 {CONFIG} (kappa={KAPPA} com={got:.4f} slide={slide*1000:+.2f}mm "
          f"mass={model.body_mass.sum():.4f}kg)  cells={len(combos)} mus={list(MUS)} "
          f"done={len(done)}/{n_rows}  K={K}  shard={shard_id}/{n_shards}")
    f = open(csv_path, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    n_mine = 0
    row = None      # progress print below must survive resume (all-done cells skip the mu loop)
    for i, combo in enumerate(combos):
        if i % n_shards != shard_id:
            continue
        n_mine += 1
        p0 = dict(zip(AXNAMES[:5], combo))
        hip_off = p0.pop("hip_off")
        for mi, mu0 in enumerate(MUS):
            key = tuple(round(v, 4) for v in combo) + (round(float(mu0), 4),)
            if key in done:
                continue
            surv = []; netf = []; slp = []; hd = []; passes = 0
            for r in range(K):
                rng = np.random.default_rng((i * len(MUS) + mi) * 100 + r)
                gs.FLOOR_MU = float(mu0) * float(rng.uniform(1 - MU_JIT, 1 + MU_JIT))
                gs.POSE_JITTER = {"yaw": np.radians(rng.uniform(-YAW_DEG, YAW_DEG)),
                                  "pitch": np.radians(rng.uniform(-PITCH_DEG, PITCH_DEG)),
                                  "lat": float(rng.uniform(-LAT_M, LAT_M))}
                gs.CONDITION["hip_off"] = hip_off
                rr = gs.run_trial(model, data, ids, dict(p0))
                sv = int(rr["survived"]); nf = rr["net_fwd_speed"]
                he = rr["heading_align"]; sl = rr["slip_ratio"]
                surv.append(sv); netf.append(nf)
                if np.isfinite(sl): slp.append(sl)
                if np.isfinite(he): hd.append(he)
                ok = (sv and np.isfinite(he) and he > HEAD_MIN and nf > NET_MIN)
                passes += int(ok)
            row = {n: round(v, 4) for n, v in zip(AXNAMES[:5], combo)}
            row["mu"] = round(float(mu0), 4)
            row["pass_rate"] = round(passes / K, 3)
            row["surv_rate"] = round(float(np.mean(surv)), 3)
            row["net_fwd_mean"] = round(float(np.mean(netf)), 4)
            row["net_fwd_min"] = round(float(np.min(netf)), 4)
            row["slip_mean"] = round(float(np.mean(slp)), 4) if slp else float("nan")
            row["head_mean"] = round(float(np.mean(hd)), 4) if hd else float("nan")
            w.writerow(row); f.flush()
        if n_mine % 25 == 0 and row is not None:
            print(f"  [shard{shard_id} cell {i+1}/{len(combos)}] "
                  + " ".join(f"{n}={row[n]}" for n in AXNAMES)
                  + f" | pass={row['pass_rate']} netfwd={row['net_fwd_mean']}")
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
