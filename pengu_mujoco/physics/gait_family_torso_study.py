#!/usr/bin/env python
"""
GAIT-FAMILY x TORSO-MODE study on penguV3, current (penguin) mass distribution.

Question: is the over_stance min_mu advantage specific to the fine3c winning family
(hip_phi ~210) or does it generalize across gait families? fine3c found two other
regimes: family A (hip_phi 110, "fast but scrappy") and the secondary lobe (hip_phi 270).

Method per family (hip_phi held fixed; the family is the categorical):
1. PHASE CALIBRATION: hip_phi shifts stance timing, so torso_phi=0 is NOT guaranteed to
   mean "torso over stance foot" outside the 210 family (v2 friction_study calibrated
   PHI_STANCE=150 for the same reason). We sweep torso_phi 0..315 in 45deg steps with an
   instrumented rollout measuring torso_stance_corr (sign agreement between torso lateral
   offset and loaded-foot lateral offset, single-support steps only, after settle) and
   pick phi_stance = argmax corr, phi_swing = argmin corr.
2. For each mode in {upright(torso_amp=0), over_stance(phi_stance), over_swing(phi_swing)}:
   CMA-optimize leg_amp/hip_amp/freq/hip_off to MATCHED SPEED V=0.08 on mu=0.7, then
   measure the min-mu ladder (clean-walk: survived & net_fwd & single_frac>0.6,
   contiguous from REF_MU downward).

usage: gait_family_torso_study.py [maxfev] [seeds] [families]
  e.g. gait_family_torso_study.py 400 1,2,3 210,110,270
outputs: results/friction_study/gait_family_mf{maxfev}_s{seed}_phi{family}.csv
         + gait_family_mf{maxfev}_agg.csv (rebuilt from everything on disk)
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
MU_LADDER = [0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.06]  # anchored at REF_MU
WALK_TIME = gs.SIM_DURATION - gs.SETTLE
NETV_OK = 0.15 / WALK_TIME
SINGLE_OK = 0.6
SPEED_TOL = 0.01
BOUNDS = {"leg_amp": (10.0, 120.0), "hip_amp": (0.0, 30.0), "freq": (1.0, 2.0),
          "hip_off": (0.0, 45.0)}
FREE = ["leg_amp", "hip_amp", "freq", "hip_off"]
F_LOAD = 4.0   # same stance gate as run_trial

MAXFEV = int(sys.argv[1]) if len(sys.argv) > 1 else 400
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
FAMILIES = [float(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3
                               else ["210", "110", "270"])]

# family seeds from fine3c (SWEEP_ANALYSIS sec 6): representative walking cells
FAMILY_SEED = {
    210.0: dict(leg_amp=115.0, hip_amp=22.0, freq=1.27),
    110.0: dict(leg_amp=110.0, hip_amp=22.0, freq=1.21),   # family A interloper
    270.0: dict(leg_amp=110.0, hip_amp=20.0, freq=1.30),   # secondary lobe
}

gs.CONDITION["hip_off"] = 30.0
model = mujoco.MjModel.from_xml_path(gs.XML)
data = mujoco.MjData(model)
ids = gs.make_ids(model)
OUTDIR = os.path.join(_ROOT, "results", "friction_study")
os.makedirs(OUTDIR, exist_ok=True)


def measure(p, mu):
    gs.FLOOR_MU = mu
    gs.CONDITION["hip_off"] = p.get("hip_off", 30.0)
    return gs.run_trial(model, data, ids, p)


def clean_walk(r):
    return bool(r["survived"]) and r["net_fwd_speed"] > NETV_OK and r["single_frac"] > SINGLE_OK


def min_mu_contiguous(p):
    lo = None
    for mu in MU_LADDER:
        if clean_walk(measure(p, mu)):
            lo = mu
        else:
            break
    return lo


# ---------- instrumented rollout: torso_stance_corr --------------------------------
def torso_stance_corr(p, mu=REF_MU):
    """sign agreement between torso lateral offset and the loaded foot's lateral offset,
    over single-support timesteps after settle. +1 = torso over stance foot."""
    from friction_utils import set_floor_friction
    set_floor_friction(model, mu)
    gs.CONDITION["hip_off"] = p.get("hip_off", 30.0)
    gs._set_gait(p)
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    floor_id, foot_geom, foot_bid, root = ids
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    fb = {s: b for b, s in foot_bid.items()}
    agree = []
    while data.time < gs.SIM_DURATION:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.time < gs.SETTLE:
            continue
        # per-foot normal force via contacts with floor
        fn = {"L": 0.0, "R": 0.0}
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            side = foot_geom.get(g1) or foot_geom.get(g2)
            if side is None or (g1 != floor_id and g2 != floor_id):
                continue
            f6 = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, f6)
            fn[side] += abs(f6[0])
        single = (fn["L"] > F_LOAD) != (fn["R"] > F_LOAD)
        if not single:
            continue
        side = "L" if fn["L"] > F_LOAD else "R"
        mid_x = (data.xipos[fb["L"]][0] + data.xipos[fb["R"]][0]) / 2
        s_foot = np.sign(data.xipos[fb[side]][0] - mid_x)
        s_torso = np.sign(data.xipos[tb][0] - mid_x)
        if s_foot != 0 and s_torso != 0:
            agree.append(float(s_foot == s_torso) * 2 - 1)
    return float(np.mean(agree)) if agree else float("nan"), len(agree)


def calibrate_phases(hip_phi):
    """sweep torso_phi, measure realized corr; return (phi_stance, phi_swing, table)."""
    base = dict(FAMILY_SEED[hip_phi], hip_phi=hip_phi, torso_amp=TORSO_AMP, hip_off=30.0)
    table = []
    for phi in range(0, 360, 45):
        p = dict(base, torso_phi=float(phi))
        corr, n = torso_stance_corr(p)
        table.append((float(phi), corr, n))
    ok = [(phi, c) for phi, c, n in table if np.isfinite(c) and n > 50]
    if not ok:
        return None, None, table
    phi_st = max(ok, key=lambda t: t[1])[0]
    phi_sw = min(ok, key=lambda t: t[1])[0]
    return phi_st, phi_sw, table


# ---------- matched-speed optimization ----------------------------------------------
def build_p(x, fixed):
    p = dict(fixed)
    for n, v in zip(FREE, [BOUNDS[k][0] + xi * (BOUNDS[k][1] - BOUNDS[k][0])
                           for k, xi in zip(FREE, np.clip(x, 0, 1))]):
        p[n] = float(v)
    return p


def optimize(fixed, seed_gait, cma_seed):
    def obj(x):
        r = measure(build_p(x, fixed), REF_MU)
        if not r["survived"]:
            return 2.0 - 0.3 * max(0.0, r["net_fwd_speed"])
        J = -W_SPEED * abs(r["net_fwd_speed"] - V_TARGET)
        J -= 0.5 * max(0.0, SINGLE_OK - r["single_frac"])
        return -J
    x0 = np.array([(seed_gait.get(k, 30.0) - BOUNDS[k][0]) /
                   (BOUNDS[k][1] - BOUNDS[k][0]) for k in FREE])
    es = cma.CMAEvolutionStrategy(np.clip(x0, 0, 1), 0.30,
                                  {"bounds": [0, 1], "maxfevals": MAXFEV,
                                   "verb_disp": 0, "seed": cma_seed})
    best = {"J": 1e9, "p": None}
    while not es.stop():
        xs = es.ask()
        js = [obj(x) for x in xs]
        es.tell(xs, js)
        for x, j in zip(xs, js):
            if j < best["J"]:
                best = {"J": j, "p": build_p(x, fixed)}
    return best["p"]


# ---------- main --------------------------------------------------------------------
for fam in FAMILIES:
    phi_st, phi_sw, table = calibrate_phases(fam)
    ftag = f"phi{int(fam)}"
    print(f"=== family hip_phi={fam:.0f}: calibration "
          f"{[(int(a), None if not np.isfinite(b) else round(b, 2)) for a, b, _ in table]}")
    print(f"    -> phi_stance={phi_st} phi_swing={phi_sw}")
    if phi_st is None:
        print("    family produces no single-support walking at calibration params — skip")
        continue
    modes = {
        "upright":     dict(torso_amp=0.0,       torso_phi=0.0),
        "over_stance": dict(torso_amp=TORSO_AMP, torso_phi=phi_st),
        "over_swing":  dict(torso_amp=TORSO_AMP, torso_phi=phi_sw),
    }
    for cma_seed in SEEDS:
        rows = []
        for mode, mo in modes.items():
            fixed = dict(mo, hip_phi=fam)
            p = optimize(fixed, FAMILY_SEED[fam], cma_seed)
            r = measure(p, REF_MU)
            corr, _ = torso_stance_corr(p)
            min_mu = min_mu_contiguous(p)
            rows.append(dict(family=int(fam), seed=cma_seed, mode=mode,
                             torso_phi=mo["torso_phi"], torso_amp=mo["torso_amp"],
                             opt_leg=round(p["leg_amp"], 1), opt_hip=round(p["hip_amp"], 1),
                             opt_freq=round(p["freq"], 3), opt_hip_off=round(p["hip_off"], 1),
                             speed=round(r["net_fwd_speed"], 4),
                             speed_err=round(r["net_fwd_speed"] - V_TARGET, 4),
                             single=round(r["single_frac"], 3),
                             corr=round(corr, 3) if np.isfinite(corr) else "",
                             mu_req_p95=round(r["mu_req_p95"], 3),
                             min_mu_to_walk=min_mu))
            rr = rows[-1]
            print(f"  s{cma_seed} {mode:12s} phi={rr['torso_phi']:5.1f} f={rr['opt_freq']:.3f} "
                  f"speed={rr['speed']:.3f}(err{rr['speed_err']:+.3f}) single={rr['single']:.3f} "
                  f"corr={rr['corr']} min_mu={rr['min_mu_to_walk']} mu_req={rr['mu_req_p95']:.3f}")
        out = os.path.join(OUTDIR, f"gait_family_mf{MAXFEV}_s{cma_seed}_{ftag}.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"  wrote {out}")

# ---------- aggregate everything on disk at this maxfev ------------------------------
allrows = []
for path in sorted(glob.glob(os.path.join(OUTDIR, f"gait_family_mf{MAXFEV}_s*_phi*.csv"))):
    with open(path) as f:
        allrows += list(csv.DictReader(f))
if allrows:
    fams = sorted({int(r["family"]) for r in allrows})
    print(f"\n=== AGG mf{MAXFEV}: min_mu per seed (speed-matched rows only) ===")
    print(f"{'family':>7s} {'mode':12s} {'n':>2s} {'min_mu per seed':>20s} {'corr mean':>10s}")
    agg = []
    for fam in fams:
        for mode in ["upright", "over_stance", "over_swing"]:
            sub = [r for r in allrows if int(r["family"]) == fam and r["mode"] == mode
                   and abs(float(r["speed_err"])) <= SPEED_TOL]
            minmus = "|".join(str(r["min_mu_to_walk"]) for r in sub) if sub else "-"
            corrs = [float(r["corr"]) for r in sub if r["corr"] not in ("", "nan")]
            cmean = round(float(np.mean(corrs)), 3) if corrs else ""
            agg.append(dict(family=fam, mode=mode, n_matched=len(sub),
                            min_mu_per_seed=minmus, corr_mean=cmean))
            print(f"{fam:>7d} {mode:12s} {len(sub):>2d} {minmus:>20s} {str(cmean):>10s}")
    AGG = os.path.join(OUTDIR, f"gait_family_mf{MAXFEV}_agg.csv")
    with open(AGG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    print(f"wrote {AGG}")
