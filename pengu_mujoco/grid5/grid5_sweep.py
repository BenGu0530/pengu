#!/usr/bin/env python
"""GRID-5: the gait x friction co-design sweep, round 2 (docs/grid5_design.md).

Self-contained under grid5/ — duplicated from the frozen GRID-4 pipeline (427b701)
and edited here only; physics/ and the root modules are the untouched GRID-4 backup.

Differences from GRID-4 (physics/grid4_sweep.py, protocol a22f80b):
  MODELS   10 configs: kappa {0,2} x COM {1.05,1.10,1.20,1.31,1.40}. Same single-base
           construction (hardened models/pengu1_31, 2.2724 kg; easytorso inertial COM
           slid at load time, mass-conserving). Baked reference models exist at
           models/pengu1_10 and models/pengu1_40 (verified vs the slide to <1e-10 m).
  AXES     freq 1.21-2.00 (GRID-4 top-20 had no champion below 1.21 — a physics
           rationale, cadence bounds speed, robust to the protocol change);
           hip_phi FULL 360 deg (the v1 trim of {150..190} was dropped 2026-08-26:
           its "no passers" evidence came from the jittered + step-start GRID-4
           protocol, which provably killed whole parameter strata);
           leg_amp 75-165 @10 (v2.3: 12.6-48.7 mm of the 50 mm slider stroke)
           and hip_amp 12-32 @4;
           hip_off 0..50 (high-COM champions sat on the off=10 edge; no 60+, Ben).
  START    staged: quiescence-gated hold (max|qvel|<0.3, 2-10 s) with a 5 deg stand
           rest lean, then hip_off RAMPS in with the transition alpha (firmware READY
           behavior). Fixes the GRID-4 hip_off step input that killed high-COM cells.
  METRICS  EXT_FIELDS appended: fall timing/phase, dual-criterion slip, cone/GRF,
           lateral drift/velocity, positive-work COT, torso-IMU orientation.
  MANIFEST manifest.json written at initcsv; consumers must refuse on mismatch.
  DR       NONE in the map (grid5-v2, Ben 2026-08-26): every trial is DETERMINISTIC —
           exact nominal mu, no pose jitter, no RNG, K=1. "Don't contaminate the
           sweep." DR/robustness testing happens ONLY at the post-map champion stage
           (designed after the map completes).

usage (run from grid5/ so the local module copies win):
  CONFIG=c1 python grid5_sweep.py count|initcsv
  CONFIG=c1 N_SHARDS=10 SHARD_ID=3 python grid5_sweep.py
env: CONFIG=c1..c10; GRID5_SMOKE=1 -> tiny grid. K is fixed at 1 (deterministic
trials — repeats would be identical).
"""
import os, sys, csv, json, hashlib, subprocess
os.environ["PENGU_MODEL"] = "1.31"          # base = models/pengu1_31 for every config
_HERE = os.path.dirname(os.path.abspath(__file__)); _ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                   # grid5 copies FIRST (never the grid4 tree)

import numpy as np
import mujoco
import gait_config as gc
import gait_sweep as gs
from torso_control import TorsoKappaPID

assert "grid5" in os.path.abspath(gs.__file__), gs.__file__
assert "grid5" in os.path.abspath(gc.__file__), gc.__file__
assert gc.XML_PATH.endswith("pengu1_31/scene.xml"), gc.XML_PATH

# ---- GRID-5 protocol switches (see gait_sweep.py / gait_config.py in this folder) ----
gs.EXTENDED_METRICS = True
gs.STAGED_START = True
gc.RAMP_HIP_OFFSET = True
REST_LEAN_DEG = 5.0                        # stand rest lean (hardware READY analogue)
gc.STAND_HIP_DEG = REST_LEAN_DEG

CONFIGS = {  # config -> (kappa, target com_ratio)
    "c1": (0.0, 1.05), "c2": (0.0, 1.20), "c3": (0.0, 1.31),
    "c4": (2.0, 1.05), "c5": (2.0, 1.20), "c6": (2.0, 1.31),
    "c7": (0.0, 1.10), "c8": (0.0, 1.40),
    "c9": (2.0, 1.10), "c10": (2.0, 1.40),
}
CONFIG = os.environ.get("CONFIG", "c1").lower()
assert CONFIG in CONFIGS, f"CONFIG={CONFIG!r} (want c1..c10)"
KAPPA, COM_TARGET = CONFIGS[CONFIG]

K = 1                                  # deterministic map: repeats would be identical
NET_MIN, HEAD_MIN = 0.05, 0.5          # slip is recorded, NOT a pass gate
SMOKE = os.environ.get("GRID5_SMOKE", "") == "1"

if SMOKE:
    FREQS    = np.round(np.arange(1.60, 1.8001, 0.10), 3)
    HIP_PHIS = np.array([250.0, 260.0])
    LEG_AMPS = np.array([115.0]); HIP_AMPS = np.array([28.0]); HIP_OFFS = np.array([10.0])
    MUS      = np.array([0.3, 0.7])
else:
    FREQS    = np.round(np.arange(1.21, 2.0001, 0.01), 3)          # 80
    HIP_PHIS = np.round(np.arange(0.0, 350.01, 10.0), 1)          # 36 (full circle;
    #   the {150..190} trim was dropped 2026-08-26 — its GRID-4 evidence is start-
    #   protocol-contaminated; grid5-v2 measures the full circle cleanly)
    LEG_AMPS = np.round(np.arange(75.0, 165.01, 10.0), 1)          # 10 (v2.3: 75-165;
    #   low end 75 = 12.6 mm stroke, high end 165 = 48.7 mm = 97% of the 50 mm rail;
    #   65 was dropped 2026-08-26 with hip 8 to save ~22% map time)
    HIP_AMPS = np.array([12.0, 16.0, 20.0, 24.0, 28.0, 32.0])      # 6 (v2.3: 12-32)
    HIP_OFFS = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])       # 6
    MUS      = np.array([0.1, 0.3, 0.5, 0.7])                      # 4

AXNAMES = ["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off", "mu"]
DR_FIELDS = ["pass_rate", "surv_rate", "net_fwd_mean", "net_fwd_min",
             "slip_mean", "head_mean"]
# extended aggregates (nan-mean over the K repeats; fall_phase = "phase:count|..." tally)
EXT_AGG = ["t_start", "t_fall", "fall_phase", "slip_dist2", "roll_dist", "slip_frac",
           "cone_util_p50", "cone_util_p95", "fn_peak", "fn_mean",
           "lat_disp", "lat_vel_rms", "e_pos", "cot_net", "cot_path", "power_mean",
           "imu_roll_mean", "imu_roll_rms", "imu_pitch_rms"]
TAG = f"grid5_{CONFIG}" + ("_smoke" if SMOKE else "")


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


def _nanmean(vals):
    a = [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    return round(float(np.mean(a)), 4) if a else float("nan")


def write_manifest(outdir, csv_path, n_rows):
    """Per-artifact manifest (lessons_2026-08-25 §5h): every option that changes the
    meaning of a row, written at production time. Consumers refuse on mismatch."""
    xml_abs = os.path.normpath(os.path.join(_ROOT, gc.XML_PATH))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    man = dict(
        protocol="grid5-v2",
        config=CONFIG, kappa=KAPPA, com_target=COM_TARGET,
        base_model=os.path.relpath(xml_abs, _ROOT),
        base_model_md5=hashlib.md5(open(os.path.join(os.path.dirname(xml_abs),
                                                     "robot.xml"), "rb").read()).hexdigest(),
        mujoco_version=mujoco.__version__,
        repo_commit=commit,
        axes={n: [float(v) for v in vals] for n, vals in
              [("freq", FREQS), ("hip_phi", HIP_PHIS), ("leg_amp", LEG_AMPS),
               ("hip_amp", HIP_AMPS), ("hip_off", HIP_OFFS), ("mu", MUS)]},
        rows=n_rows, K=K,
        dr=dict(map="deterministic: exact nominal mu, no pose jitter, no RNG, K=1 "
                    "(grid5-v2, Ben 2026-08-26 — champion-stage DR designed post-map)"),
        start=dict(staged=True, quiet_qvel=gs.QUIET_QVEL,
                   quiet_min_t=gs.QUIET_MIN_T, quiet_max_t=gs.QUIET_MAX_T,
                   rest_lean_deg=REST_LEAN_DEG, ramp_hip_offset=True,
                   transition_s=gc.T_TRANSITION, settle_s=2.0, measure_s=13.0),
        slip=dict(cone_eps=gs.SLIP_CONE_EPS, v0=gs.SLIP_V0, c=gs.SLIP_C),
        gates=dict(net_min=NET_MIN, head_min=HEAD_MIN,
                   note="pass gate unchanged from GRID-4; tiers recomputed post-hoc"),
        schema=AXNAMES + DR_FIELDS + EXT_AGG,
        execution_layer=dict(slew=None, cmd_fc=None, torso_clamp_deg=45.0,
                             note="GRID-4-identical execution layer (Ben, 2026-08-26)"),
    )
    mpath = os.path.join(outdir, os.path.basename(csv_path).replace(".csv", ".manifest.json"))
    with open(mpath, "w") as f:
        json.dump(man, f, indent=1)
    return mpath


def check_manifest(outdir, csv_path):
    """Refuse to append rows under a manifest that does not match this process."""
    mpath = os.path.join(outdir, os.path.basename(csv_path).replace(".csv", ".manifest.json"))
    if not os.path.exists(mpath):
        raise SystemExit(f"manifest missing: {mpath} — run initcsv first")
    man = json.load(open(mpath))
    want = dict(protocol="grid5-v2", config=CONFIG, K=K,
                mujoco_version=mujoco.__version__)
    for k, v in want.items():
        if man.get(k) != v:
            raise SystemExit(f"MANIFEST MISMATCH on {k!r}: artifact has {man.get(k)!r}, "
                             f"this process has {v!r} — refusing to write")
    slip_want = dict(cone_eps=gs.SLIP_CONE_EPS, v0=gs.SLIP_V0, c=gs.SLIP_C)
    if man.get("slip") != slip_want:
        raise SystemExit(f"MANIFEST MISMATCH on 'slip': artifact has {man.get('slip')!r}, "
                         f"this process has {slip_want!r} — refusing to write")
    return man


def main():
    combos = list(cells())
    n_rows = len(combos) * len(MUS)
    outdir = os.path.join(_ROOT, "results", "gait_sweep"); os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"sweep_{TAG}_{'_'.join(AXNAMES)}.csv")
    fields = AXNAMES + DR_FIELDS + EXT_AGG

    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(f"config={CONFIG} kappa={KAPPA} com={COM_TARGET}  cells={len(combos)}  "
              f"mus={list(MUS)}  rows={n_rows}  K={K}  trials={n_rows*K}  "
              f"csv={os.path.basename(csv_path)}"); return
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        mpath = write_manifest(outdir, csv_path, n_rows)
        print(f"# initcsv {csv_path}  rows={n_rows}  K={K}")
        print(f"# manifest {mpath}"); return

    check_manifest(outdir, csv_path)
    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))

    model = mujoco.MjModel.from_xml_path(gs.XML); data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    # COM ratio and the kappa-PID are DEFINED at the hips-0 design neutral (same as
    # GRID-4 and the baked models/pengu1_10, 1_40); the rest lean is a start-protocol
    # pose only. Calibrate with lean off, then restore it for the trials.
    _lean = gc.STAND_HIP_DEG; gc.STAND_HIP_DEG = 0.0
    slide, got = apply_com_variant(model, COM_TARGET)
    gc.TORSO_CONTROLLER = TorsoKappaPID(model, kappa=KAPPA, measure_after=0.0)
    gc.STAND_HIP_DEG = _lean

    done = gs._load_done(csv_path, AXNAMES)
    # runtime probe (lessons §5i): print what will actually be simulated
    print(f"# GRID5 {CONFIG} (kappa={KAPPA} com={got:.4f} slide={slide*1000:+.2f}mm "
          f"mass={model.body_mass.sum():.4f}kg)  cells={len(combos)} mus={list(MUS)} "
          f"done={len(done)}/{n_rows}  K={K}  shard={shard_id}/{n_shards}")
    print(f"# start: staged quiet<{gs.QUIET_QVEL} rest_lean={gc.STAND_HIP_DEG}deg "
          f"ramp_off={gc.RAMP_HIP_OFFSET}  slip: eps={gs.SLIP_CONE_EPS} v0={gs.SLIP_V0} "
          f"c={gs.SLIP_C}  ext={gs.EXTENDED_METRICS}  DR=NONE(deterministic map)")
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
            ext = {k: [] for k in EXT_AGG if k != "fall_phase"}
            phase_tally = {}
            for r in range(K):
                # grid5-v2: PURE deterministic trial — exact nominal mu, no jitter,
                # no RNG. DR enters only at the post-map champion stage.
                gs.FLOOR_MU = float(mu0)
                gs.POSE_JITTER = None
                gs.CONDITION["hip_off"] = hip_off
                rr = gs.run_trial(model, data, ids, dict(p0))
                sv = int(rr["survived"]); nf = rr["net_fwd_speed"]
                he = rr["heading_align"]; sl = rr["slip_ratio"]
                surv.append(sv); netf.append(nf)
                if np.isfinite(sl): slp.append(sl)
                if np.isfinite(he): hd.append(he)
                ok = (sv and np.isfinite(he) and he > HEAD_MIN and nf > NET_MIN)
                passes += int(ok)
                for k in ext:
                    ext[k].append(rr[k])
                if rr["fall_phase"]:
                    phase_tally[rr["fall_phase"]] = phase_tally.get(rr["fall_phase"], 0) + 1
            row = {n: round(v, 4) for n, v in zip(AXNAMES[:5], combo)}
            row["mu"] = round(float(mu0), 4)
            row["pass_rate"] = round(passes / K, 3)
            row["surv_rate"] = round(float(np.mean(surv)), 3)
            row["net_fwd_mean"] = round(float(np.mean(netf)), 4)
            row["net_fwd_min"] = round(float(np.min(netf)), 4)
            row["slip_mean"] = round(float(np.mean(slp)), 4) if slp else float("nan")
            row["head_mean"] = round(float(np.mean(hd)), 4) if hd else float("nan")
            for k in ext:
                row[k] = _nanmean(ext[k])
            row["fall_phase"] = "|".join(f"{k}:{v}" for k, v in sorted(phase_tally.items()))
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
