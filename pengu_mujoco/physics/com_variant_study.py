#!/usr/bin/env python
"""
COM-VARIANT x TORSO-MODE study on penguV3 — the plan's mass-distribution axis (issue #6),
runnable NOW without Onshape exports.

Per the summer plan, the manipulated variables are torso phasing x mass distribution;
friction is the READOUT (min mu to walk), not the subject. This harness:

1. Builds COM variants IN MEMORY: transfers mass from low bodies (axis/thigh-motors/feet)
   into `easytorso`, scaling each body's inertia with its mass ratio, total mass constant.
   Bisection hits any target COM fraction up to ~57% of standing height
   (baseline penguin = 36.7%; human target 54-57%). Model files untouched.
2. Calibrates the STAND pose per variant (grid over gc.STAND_HIP_DEG): a top-heavy variant
   falls during T_HOLD with the penguin stand pose — probe showed >=47% cannot stand at
   STAND_HIP=0. Humans and penguins stand differently; this is a physical re-calibration.
3. For each (variant x torso mode x seed): CMA-optimizes leg_amp/hip_amp/freq (+hip_off,
   which top-heavy variants need to re-balance posture during walking) to MATCHED SPEED,
   then measures the min-mu ladder with the CLEAN-WALK definition (contiguous from top,
   single_frac gate — rejects low-mu skidding).

Probe result feeding this design: gait B (penguin winner) fails outright at COM 42% —
gaits do NOT transfer across mass distributions, so per-config re-optimization (as the
plan specifies) is mandatory, and is what this script does.

usage: com_variant_study.py [maxfev] [seeds] [fracs]
   e.g. com_variant_study.py 400 1,2,3 base,0.42,0.47,0.52,0.57
outputs: results/friction_study/com_variant_mf{maxfev}_s{seed}_{fractag}.csv
         results/friction_study/com_variant_mf{maxfev}_agg.csv (rewritten from all present)
"""
import os, sys, csv, glob

os.environ["PENGU_MODEL"] = "v3"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE); sys.path.insert(0, _ROOT)

import numpy as np
import mujoco
import cma
import gait_config as gc
import gait_sweep as gs

assert gc.XML_PATH.endswith("penguV3/scene.xml"), gc.XML_PATH

V_TARGET = 0.08
W_SPEED = 4.0
TORSO_AMP = 20.0
REF_MU = 0.7
MU_LADDER = [0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]  # anchored at REF_MU: a gait
# optimized at 0.7 may legitimately fail at 1.0 (sticky), which would fake min_mu=None
WALK_TIME = gs.SIM_DURATION - gs.SETTLE
NETV_OK = 0.15 / WALK_TIME
SINGLE_OK = 0.6
SPEED_TOL = 0.01
BOUNDS = {"leg_amp": (10.0, 120.0), "hip_amp": (0.0, 30.0), "freq": (1.0, 2.0),
          "hip_off": (0.0, 45.0)}
FREE = ["leg_amp", "hip_amp", "freq", "hip_off"]   # hip_off free: posture re-balance per variant
SEED0 = dict(leg_amp=60.0, hip_amp=14.0, freq=1.30, hip_off=30.0, hip_phi=210.0)

MAXFEV = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
FRACS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["base", "0.42", "0.47", "0.52", "0.57"]

MODES = {
    "upright":     dict(torso_amp=0.0,       torso_phi=0.0),
    "over_stance": dict(torso_amp=TORSO_AMP, torso_phi=0.0),
    "over_swing":  dict(torso_amp=TORSO_AMP, torso_phi=180.0),
}

TORSO_BODY = "easytorso"
LOW_BODIES = ["easyaxis", "rightthighmotor", "leftthighmotor",
              "right_foot0080", "right_foot0080___fillet13"]

OUTDIR = os.path.join(_ROOT, "results", "friction_study")
os.makedirs(OUTDIR, exist_ok=True)


# ---------- variant construction --------------------------------------------------
def com_frac_of(m):
    d = mujoco.MjData(m)
    act, jadr = gc.build_ids(m)
    gc.set_initial_pose(m, d, act, jadr)
    mujoco.mj_forward(m, d)
    H = max(d.geom_xpos[g][2] + m.geom_rbound[g]
            for g in range(m.ngeom) if m.geom_bodyid[g] != 0)
    return d.subtree_com[1][2] / H


def make_variant(target):
    """target None -> baseline; else bisect mass-transfer alpha to hit COM fraction."""
    m = mujoco.MjModel.from_xml_path(gs.XML)
    if target is None:
        return m, com_frac_of(m), 0.0
    bt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    bl = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in LOW_BODIES]
    m0t = m.body_mass[bt]; m0l = m.body_mass[bl].copy()
    I0t = m.body_inertia[bt].copy(); I0l = [m.body_inertia[b].copy() for b in bl]

    def apply(a):
        mt = m0t + a * m0l.sum()
        m.body_mass[bt] = mt; m.body_inertia[bt] = I0t * (mt / m0t)
        for i, b in enumerate(bl):
            mb = (1 - a) * m0l[i]
            m.body_mass[b] = mb; m.body_inertia[b] = I0l[i] * max(mb / m0l[i], 1e-3)

    lo, hi = 0.0, 0.85
    for _ in range(24):
        mid = (lo + hi) / 2
        apply(mid)
        if com_frac_of(m) < target:
            lo = mid
        else:
            hi = mid
    apply(hi)
    return m, com_frac_of(m), hi


# ---------- stand calibration ------------------------------------------------------
def calibrate_stand(m):
    """grid gc.STAND_HIP_DEG; keep the value that survives a T_HOLD+1s hold.
    Returns chosen stand hip (deg) or None if nothing stands."""
    act, jadr = gc.build_ids(m)
    best = None
    for sh in [0., -5., 5., -10., 10., -15., 15., -20., 20., -25., 25.]:
        gc.STAND_HIP_DEG = sh
        d = mujoco.MjData(m)
        gc.set_initial_pose(m, d, act, jadr)
        ok = True
        while d.time < gc.T_HOLD + 1.0:
            gc.apply_ctrl(d, act, d.time)
            mujoco.mj_step(m, d)
            if d.qpos[2] < 0.08:
                ok = False; break
        if ok:
            best = sh; break        # first (smallest-|.|) stable value wins
    return best


# ---------- measurement ------------------------------------------------------------
def measure(m, d, ids, p, mu):
    gs.FLOOR_MU = mu
    return gs.run_trial(m, d, ids, p)


def clean_walk(r):
    return bool(r["survived"]) and r["net_fwd_speed"] > NETV_OK and r["single_frac"] > SINGLE_OK


def min_mu_contiguous(m, d, ids, p):
    lo = None
    for mu in MU_LADDER:
        if clean_walk(measure(m, d, ids, p, mu)):
            lo = mu
        else:
            break
    return lo


def build_p(x, mo):
    p = dict(SEED0); p.update(mo)
    for n, v in zip(FREE, [BOUNDS[k][0] + xi * (BOUNDS[k][1] - BOUNDS[k][0])
                           for k, xi in zip(FREE, np.clip(x, 0, 1))]):
        p[n] = float(v)
    return p


def optimize(m, d, ids, mo, cma_seed):
    def obj(x):
        p = build_p(x, mo)
        gs.CONDITION["hip_off"] = p["hip_off"]
        r = measure(m, d, ids, p, REF_MU)
        if not r["survived"]:
            return 2.0 - 0.3 * max(0.0, r["net_fwd_speed"])
        J = -W_SPEED * abs(r["net_fwd_speed"] - V_TARGET)
        J -= 0.5 * max(0.0, SINGLE_OK - r["single_frac"])
        return -J
    x0 = np.array([(SEED0[k] - BOUNDS[k][0]) / (BOUNDS[k][1] - BOUNDS[k][0]) for k in FREE])
    es = cma.CMAEvolutionStrategy(x0, 0.30, {"bounds": [0, 1], "maxfevals": MAXFEV,
                                             "verb_disp": 0, "seed": cma_seed})
    best = {"J": 1e9, "p": None}
    while not es.stop():
        xs = es.ask()
        js = [obj(x) for x in xs]
        es.tell(xs, js)
        for x, j in zip(xs, js):
            if j < best["J"]:
                best = {"J": j, "p": build_p(x, mo)}
    return best["p"]


# ---------- main -------------------------------------------------------------------
for frac_tag in FRACS:
    target = None if frac_tag == "base" else float(frac_tag)
    m, frac, alpha = make_variant(target)
    stand_hip = calibrate_stand(m)
    tag = "base" if target is None else f"{frac:.2f}"
    print(f"=== variant {tag}: COM {frac:.1%}, transfer a={alpha:.2f}, "
          f"stand_hip={stand_hip} ===")
    if stand_hip is None:
        print("    CANNOT STAND at any calibrated pose — recorded as infeasible")
    for cma_seed in SEEDS:
        rows = []
        for mode, mo in MODES.items():
            if stand_hip is None:
                rows.append(dict(frac_tag=tag, com_frac=round(frac, 4), seed=cma_seed,
                                 mode=mode, stand_hip=None, feasible=0))
                continue
            gc.STAND_HIP_DEG = stand_hip
            d = mujoco.MjData(m); ids = gs.make_ids(m)
            p = optimize(m, d, ids, mo, cma_seed)
            gs.CONDITION["hip_off"] = p["hip_off"]
            r = measure(m, d, ids, p, REF_MU)
            min_mu = min_mu_contiguous(m, d, ids, p)
            rows.append(dict(frac_tag=tag, com_frac=round(frac, 4), seed=cma_seed,
                             mode=mode, stand_hip=stand_hip, feasible=1,
                             opt_leg=round(p["leg_amp"], 1), opt_hip=round(p["hip_amp"], 1),
                             opt_freq=round(p["freq"], 3), opt_hip_off=round(p["hip_off"], 1),
                             speed=round(r["net_fwd_speed"], 4),
                             speed_err=round(r["net_fwd_speed"] - V_TARGET, 4),
                             single=round(r["single_frac"], 3),
                             mu_req_p95=round(r["mu_req_p95"], 3), min_mu_to_walk=min_mu))
            rr = rows[-1]
            print(f"  s{cma_seed} {mode:12s} f={rr['opt_freq']:.3f} hip_off={rr['opt_hip_off']:4.1f} "
                  f"speed={rr['speed']:.3f}(err{rr['speed_err']:+.3f}) single={rr['single']:.3f} "
                  f"min_mu={rr['min_mu_to_walk']} mu_req={rr['mu_req_p95']:.3f}")
        out = os.path.join(OUTDIR, f"com_variant_mf{MAXFEV}_s{cma_seed}_{tag}.csv")
        keys = max(rows, key=len).keys()
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(keys)); w.writeheader(); w.writerows(rows)

# ---------- aggregate everything present at this maxfev ----------------------------
allrows = []
for path in sorted(glob.glob(os.path.join(OUTDIR, f"com_variant_mf{MAXFEV}_s*_*.csv"))):
    with open(path) as f:
        allrows += [r for r in csv.DictReader(f)]
if allrows:
    print(f"\n=== AGG over {len(allrows)} rows (all seeds found on disk, mf{MAXFEV}) ===")
    print(f"{'variant':>8s} {'mode':12s} {'n':>2s} {'min_mu per seed':>18s} {'speed ok':>8s}")
    agg = []
    tags = sorted({r["frac_tag"] for r in allrows})
    for tagv in tags:
        for mode in MODES:
            sub = [r for r in allrows if r["frac_tag"] == tagv and r["mode"] == mode
                   and r.get("feasible") == "1"]
            if not sub:
                agg.append(dict(frac_tag=tagv, mode=mode, n=0, note="infeasible/no data"))
                print(f"{tagv:>8s} {mode:12s}  0 {'—':>18s}")
                continue
            minmus = "|".join(str(r["min_mu_to_walk"]) for r in sub)
            n_ok = sum(1 for r in sub if abs(float(r["speed_err"])) <= SPEED_TOL)
            agg.append(dict(frac_tag=tagv, mode=mode, n=len(sub),
                            min_mu_per_seed=minmus, n_matched_speed=n_ok))
            print(f"{tagv:>8s} {mode:12s} {len(sub):>2d} {minmus:>18s} {n_ok:>5d}/{len(sub)}")
    AGG = os.path.join(OUTDIR, f"com_variant_mf{MAXFEV}_agg.csv")
    FIELDS = ["frac_tag", "mode", "n", "min_mu_per_seed", "n_matched_speed", "note"]
    with open(AGG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        w.writeheader(); w.writerows(agg)
    print(f"wrote {AGG}")
