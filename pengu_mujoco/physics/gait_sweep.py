"""
gait_sweep.py - plain, from-scratch, axis-driven gait sweep. NO optimization,
NO scalar objective. We sweep a grid and MEASURE-and-REPORT each cell; "natural /
best" is judged by eye from rendered gifs, supported by the gait-quality metrics.

Design:
  - BASE        : the fixed parameter set (the kept gait).
  - CONDITION   : torso mode (upright / over_stance / over_swing) applied on top.
  - AXES        : list of (param, values) to sweep -> Cartesian product of cells.
                  Resolution is per-axis (freq down to 0.01 -- this is a nonlinear
                  system, narrow regimes matter).
  - Automation  : incremental CSV (append per cell) + RESUME (skip done cells) +
                  progress log, so a long high-dim sweep can be interrupted/resumed.

Footfall detection is de-bounced (Schmitt trigger on normal force + minimum
swing-clearance gate) so stride/cadence/clearance are not corrupted by contact
chatter.

Run from pengu_mujoco/:
  python physics/gait_sweep.py                       # run the configured sweep
  MUJOCO_GL=egl python physics/gait_sweep.py viz <freq> [leg_amp]   # render a cell
"""
import os
import sys
import csv
import math
import itertools
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_config as gc
from gait_config import build_ids, set_initial_pose
from friction_utils import set_floor_friction

_HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(_HERE, "..", gc.XML_PATH)   # follows PENGU_MODEL (v2/v3)
MODEL = gc._MODEL                              # tag outputs so v2/v3 don't collide
ROOT_BODY = "leftthighmotor"
FOOT_BODIES = {"right_foot0080": "R", "right_foot0080___fillet13": "L"}

# ===================================================================
#  EXPERIMENT SETUP  (review / edit here)
# ===================================================================
# BASE = the kept parameter set (the f=1.8/leg=90 gait Ben liked, freq lowered to
# the penguin 1.27 Hz as the nominal; the freq AXIS below overrides it per cell).
BASE = dict(leg_amp=90.0, hip_amp=12.0, hip_phi=250.0,
            torso_amp=0.0, torso_phi=0.0, freq=2.05)   # freq fixed in the clean band
# upright torso (no swing) + a FIXED 25deg forward pitch via a symmetric hip
# offset (hip_off=30 -> ~24.6deg; legs hold it via PD, eased in over the
# transition, NOT a teleport). Pitch is fixed, not a swept parameter.
CONDITION = dict(name="p25", torso_amp=0.0, torso_phi=0.0, hip_off=30.0)
SWEEP_TAG = "fine3c"     # issue #2: LOW-FREQ band, TIER C = fine phase(10deg) + dense amps, sharded parallel
FLOOR_MU = 0.7

# --- opt-in: whole-robot CoM vs stance-foot contact point (Gait-2 benchmark) ----------
# Off by default so every existing tool's run_trial output is bit-identical. When a caller
# sets gs.TRACK_COM_STANCE=True it must ALSO append COM_STANCE_FIELDS to its CSV fieldnames
# (csv.DictWriter would otherwise raise on the extra keys). During single support (exactly
# one foot loaded) we log the whole-robot CoM (subtree_com[0], world frame) against the
# force-weighted stance-foot contact point. Forward travel is +y, so x is the lateral axis.
TRACK_COM_STANCE = False
COM_STANCE_FIELDS = ["com_lat_mean", "com_lat_rms", "com_stance_dist", "com_ss_n"]
gc.T_HOLD = 5.0          # wait longer to settle upright before pitching
gc.T_TRANSITION = 4.0    # MODERATE pitch-in ramp (not a sudden change)

# AXES: each (param, values). Cartesian product = the cells. Fine resolution.
# Default: fine frequency sweep (0.01 Hz) over the full band + the 1.27+-0.5 band
# is contained in it. Add more axes (e.g. ("torso_phi", np.arange(0,360,5))) for
# high-dim sweeps -- the automation handles product + resume.
# 6-DOF JOINT grid (no DOF locked): leg/hip/torso amplitudes + freq + 2 phase
# offsets. freq 0.05 as requested; phases at 45deg coarse (0.1rad would be ~5yr).
# ~2.7M cells. CMA-ES (cma_search.py) + local refine cover fine phase resolution.
# FOCUSED FINE sweep of the robust highland found via heatmaps (hip_phi~270,
# freq 1.5-2.0, high leg/hip amps). Fine on the two dominant axes (freq 0.01,
# hip_phi 5deg); coarse on the amps/torso the highland already pins. ~22k cells.
# issue #1 fine2: freq still 0.01, but phase opened WIDE to include +-90 (hip_phi
# 30-330 covers 90 and 270 clusters + below the 250 edge fine1 pinned; torso_phi
# now includes 90/270). ~27k cells.
# fine3c (TIER C): penguin-friendly LOW-FREQ band 1.00-1.50, FINE on every DOF that
# is feasible -- both phases at 10deg and the three amplitudes densified. Full phase
# coverage (do NOT prejudge from the high-freq winner). ~3.97M cells; run SHARDED
# across many workers (N_SHARDS env) -- 24 cores => ~20h wall.
AXES = [
    ("freq",      np.round(np.arange(1.00, 1.5001, 0.01), 3)),          # 51  (0.01 fine)
    ("hip_phi",   np.round(np.arange(0.0, 350.01, 10.0), 1)),           # 36  (10deg full phase)
    ("leg_amp",   np.round(np.array([95.,100.,105.,110.,115.]), 1)),    # 5   (dense)
    ("hip_amp",   np.round(np.array([16.,18.,20.,22.]), 1)),            # 4   (dense)
    ("torso_amp", np.round(np.array([12.,16.,20.]), 1)),               # 3   (dense)
    ("torso_phi", np.round(np.arange(0.0, 350.01, 10.0), 1)),           # 36  (10deg full phase)
]

SIM_DURATION = 24.0
SETTLE = gc.T_HOLD + gc.T_TRANSITION + 2.0   # 5+4+2 = 11s; measure the steady walk after

# footfall de-bounce
F_HI = 4.0          # N: enter STANCE above this normal force
F_LO = 1.0          # N: enter SWING below this
CLEAR_MIN = 0.003   # m: foot just has to truly clear the ground (>0) to count a step
TD_REFRACTORY = 0.25  # s: min time between two same-foot touchdowns (issue #4: reject
                      #    contact-force chatter that logs the same foot 2-3x in a burst)

METRIC_FIELDS = ["survived", "valid", "path_speed", "net_fwd_speed", "straightness",
                 "path", "single_frac", "stride_L", "stride_R", "stride_sym",
                 "clear_L", "clear_R", "cadence", "n_steps", "mu_req_p95"]


def _set_gait(p):
    gc.set_crank_amp(p["leg_amp"]); gc.set_hip_amp(p["hip_amp"])
    gc.set_torso_amp(p.get("torso_amp", CONDITION["torso_amp"]))
    gc.WALK_HIP_OFFSET_DEG = CONDITION.get("hip_off", 0.0); gc.WALK_HIP_LEAN_DEG = 0.0
    gc.set_walk_freq(p["freq"])
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = p["hip_phi"]; gc.PHASE_OFFSET_D_DEG = p["hip_phi"]
    gc.PHASE_OFFSET_E_DEG = p.get("torso_phi", CONDITION["torso_phi"])


def make_ids(model):
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    foot_bid = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n): s for n, s in FOOT_BODIES.items()}
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_geom = {g: foot_bid[model.geom_bodyid[g]] for g in range(model.ngeom)
                 if model.geom_bodyid[g] in foot_bid and model.geom_contype[g]}
    return floor_id, foot_geom, foot_bid, root


def run_trial(model, data, ids, p):
    """Run one cell; measure de-bounced gait-quality metrics. Pure measurement."""
    floor_id, foot_geom, foot_bid, root = ids
    set_floor_friction(model, FLOOR_MU)
    _set_gait(p)
    act, jadr = build_ids(model)
    set_initial_pose(model, data, act, jadr)
    f6 = np.zeros(6)

    # per-foot footfall state machine (Schmitt + clearance gate)
    state = {"L": "stance", "R": "stance"}      # stance / swing
    td_pos = {"L": None, "R": None}             # last counted touchdown (x,y)
    stance_z = {"L": None, "R": None}           # foot z while loaded (ground ref)
    swing_peak_z = {"L": -1e9, "R": -1e9}
    strides = {"L": [], "R": []}; clears = {"L": [], "R": []}
    n_steps = {"L": 0, "R": 0}
    td_time = {"L": -1e9, "R": -1e9}            # last counted touchdown time (refractory)
    mu_req = []
    cs_lat = []; cs_dist = []                    # CoM-over-stance (opt-in, single support)
    n_single = 0; n_double = 0; n_air = 0; n_meas = 0   # support-pattern counters
    pos_ws = None; last = data.xpos[root][:2].copy(); fell = False

    while data.time < SIM_DURATION:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            fell = True; break
        last = data.xpos[root][:2].copy()
        if data.time < SETTLE:
            continue
        if pos_ws is None:
            pos_ws = data.xpos[root][:2].copy()
        # per-foot normal/tangential force
        Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}
        cpos = {"L": np.zeros(2), "R": np.zeros(2)}  # force-weighted contact xy (opt-in)
        for c in range(data.ncon):
            ct = data.contact[c]
            ft = foot_geom.get(ct.geom2) if ct.geom1 == floor_id else (
                 foot_geom.get(ct.geom1) if ct.geom2 == floor_id else None)
            if ft:
                mujoco.mj_contactForce(model, data, c, f6)
                Fn[ft] += abs(f6[0]); Ft[ft] += math.hypot(f6[1], f6[2])
                if TRACK_COM_STANCE:
                    cpos[ft] += abs(f6[0]) * ct.pos[:2]
        for s in ("L", "R"):
            if Fn[s] > F_HI:                       # issue #5: stance-gated (real load only)
                mu_req.append(Ft[s] / Fn[s])
        # support pattern: single (one foot down) is the conventional walking signature
        nc = sum(1 for s in ("L", "R") if Fn[s] > F_HI)
        # CoM-over-stance benchmark: only meaningful in single support (one contact point)
        if TRACK_COM_STANCE and nc == 1:
            s = "L" if Fn["L"] > F_HI else "R"
            if Fn[s] > 1e-6:
                cxy = cpos[s] / Fn[s]              # ground contact point of the stance foot
                com = data.subtree_com[0]          # whole-robot CoM, world frame
                lx = float(com[0] - cxy[0])        # lateral (forward travel is +y)
                ly = float(com[1] - cxy[1])        # fore-aft
                cs_lat.append(lx); cs_dist.append(math.hypot(lx, ly))
        n_meas += 1
        if nc == 1: n_single += 1
        elif nc == 2: n_double += 1
        else: n_air += 1
        # footfall state machine per foot
        for bid, s in foot_bid.items():
            fz = float(data.xpos[bid][2])
            if state[s] == "stance":
                stance_z[s] = fz if stance_z[s] is None else min(stance_z[s], fz)
                if Fn[s] < F_LO:                       # lift off -> swing
                    state[s] = "swing"; swing_peak_z[s] = fz
            else:                                       # swing
                swing_peak_z[s] = max(swing_peak_z[s], fz)
                if Fn[s] > F_HI:                        # touchdown candidate
                    lift = swing_peak_z[s] - (stance_z[s] if stance_z[s] is not None else fz)
                    # issue #4: real step = cleared ground AND not a chatter re-tap
                    if lift > CLEAR_MIN and (data.time - td_time[s]) > TD_REFRACTORY:
                        pos = data.xpos[bid][:2].copy()
                        if td_pos[s] is not None:       # 2D step length (path, robust to turning)
                            strides[s].append(float(math.hypot(pos[0] - td_pos[s][0], pos[1] - td_pos[s][1])))
                        td_pos[s] = pos; clears[s].append(lift); n_steps[s] += 1
                        td_time[s] = data.time
                    state[s] = "stance"; stance_z[s] = fz

    survived = not fell
    wt = max(1e-6, (SIM_DURATION if survived else data.time) - SETTLE)
    sL = float(np.mean(strides["L"])) if strides["L"] else float("nan")
    sR = float(np.mean(strides["R"])) if strides["R"] else float("nan")
    sym = abs(sL - sR) if (np.isfinite(sL) and np.isfinite(sR)) else float("nan")
    # criterion 1: distance via SUM of per-step 2D step lengths (robust to turning)
    path = float(sum(strides["L"]) + sum(strides["R"]))
    path_speed = path / wt
    # issue #3: distinguish real forward travel from curling/getting-stuck-in-place.
    # straightness = net displacement / path length (1=straight line, ~0=looping in place);
    # net_fwd_speed = net forward (y) progress / time (low if it curls/stalls despite high path).
    net_disp = float(math.hypot(last[0] - pos_ws[0], last[1] - pos_ws[1])) if pos_ws is not None else 0.0
    net_fwd = float(last[1] - pos_ws[1]) if pos_ws is not None else 0.0
    straightness = net_disp / path if path > 1e-6 else float("nan")
    net_fwd_speed = net_fwd / wt
    cL = float(np.mean(clears["L"])) if clears["L"] else float("nan")
    cR = float(np.mean(clears["R"])) if clears["R"] else float("nan")
    single_frac = n_single / max(1, n_meas)
    # criterion 2: conventional alternating gait = both feet truly clear the ground
    # AND there is real single-support (stance foot down while the other swings)
    valid = int(survived and n_steps["L"] >= 2 and n_steps["R"] >= 2
                and np.isfinite(cL) and cL > 0 and np.isfinite(cR) and cR > 0
                and single_frac >= 0.3)
    out = dict(survived=int(survived), valid=valid,
               path_speed=round(path_speed, 4), net_fwd_speed=round(net_fwd_speed, 4),
               straightness=round(straightness, 3) if np.isfinite(straightness) else float("nan"),
               path=round(path, 4),
               stride_L=round(sL, 4), stride_R=round(sR, 4), stride_sym=round(sym, 4),
               clear_L=round(cL, 4), clear_R=round(cR, 4),
               single_frac=round(single_frac, 3),
               cadence=round((n_steps["L"] + n_steps["R"]) / wt, 3),
               n_steps=n_steps["L"] + n_steps["R"],
               mu_req_p95=round(float(np.percentile(mu_req, 95)) if mu_req else float("nan"), 3))
    if TRACK_COM_STANCE:
        lat = np.asarray(cs_lat); dist = np.asarray(cs_dist)
        out["com_lat_mean"] = round(float(np.mean(np.abs(lat))), 5) if lat.size else float("nan")
        out["com_lat_rms"] = round(float(np.sqrt(np.mean(lat * lat))), 5) if lat.size else float("nan")
        out["com_stance_dist"] = round(float(np.mean(dist)), 5) if dist.size else float("nan")
        out["com_ss_n"] = int(lat.size)
    return out


def _outpaths():
    outdir = os.path.join(_HERE, "..", "results", "gait_sweep")
    os.makedirs(outdir, exist_ok=True)
    axsig = "_".join(a[0] for a in AXES)
    tag = f"_{SWEEP_TAG}" if SWEEP_TAG else ""
    return outdir, os.path.join(outdir, f"sweep_{MODEL}_{CONDITION['name']}{tag}_{axsig}.csv")


def _load_done(csv_path, axnames):
    """For resume: set of already-computed axis-value tuples."""
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    done.add(tuple(round(float(row[n]), 4) for n in axnames))
                except (KeyError, ValueError):
                    pass
    return done


def sweep():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    ids = make_ids(model)
    axnames = [a[0] for a in AXES]
    combos = list(itertools.product(*[a[1] for a in AXES]))
    outdir, csv_path = _outpaths()
    # Parallel sharding: launch N_SHARDS workers, each owns cells where
    # global_index % N_SHARDS == SHARD_ID (disjoint -> no duplicate work).
    n_shards = int(os.environ.get("N_SHARDS", "1"))
    shard_id = int(os.environ.get("SHARD_ID", "0"))
    fields = axnames + METRIC_FIELDS
    if len(sys.argv) > 1 and sys.argv[1] == "initcsv":
        # one-shot: create the CSV with header so worker shards never race the header
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f0:
                csv.DictWriter(f0, fieldnames=fields).writeheader()
        with open(os.path.join(outdir, ".active_csv"), "w") as af:
            af.write(os.path.basename(csv_path))
        print(f"# initcsv {csv_path}  total cells={len(combos)}  shards={n_shards}")
        return
    with open(os.path.join(outdir, ".active_csv"), "w") as af:   # issue #7: tell run_grid.sh which sweep is active
        af.write(os.path.basename(csv_path))
    done = _load_done(csv_path, axnames)
    new_file = not os.path.exists(csv_path)
    print(f"# EXPERIMENT condition={CONDITION['name']}  BASE={BASE}")
    print(f"# AXES={[(n, len(v)) for n, v in AXES]}  total cells={len(combos)}  "
          f"already done={len(done)}  shard={shard_id}/{n_shards}")
    f = open(csv_path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new_file and shard_id == 0:       # single-worker fallback; sharded runs use initcsv
        w.writeheader(); f.flush()
    n_mine = 0
    for i, combo in enumerate(combos):
        if i % n_shards != shard_id:     # not this shard's cell
            continue
        n_mine += 1
        key = tuple(round(float(v), 4) for v in combo)
        if key in done:
            continue
        p = dict(BASE)
        for n, v in zip(axnames, combo):
            p[n] = float(v)
        r = run_trial(model, data, ids, p)
        row = {n: round(float(v), 4) for n, v in zip(axnames, combo)}
        row.update(r)
        w.writerow(row); f.flush()
        if n_mine % 50 == 0:
            print(f"  [shard{shard_id} {i+1}/{len(combos)}] " +
                  " ".join(f"{n}={row[n]}" for n in axnames) +
                  f" | valid={r['valid']} pathspd={r['path_speed']} mu={r['mu_req_p95']}")
    f.close()
    if n_shards > 1:
        open(f"{csv_path}.shard{shard_id}of{n_shards}.done", "w").close()
        if all(os.path.exists(f"{csv_path}.shard{s}of{n_shards}.done") for s in range(n_shards)):
            open(csv_path + ".done", "w").close()   # last shard to finish -> master sentinel
            print(f"# ALL {n_shards} shards complete -> {os.path.basename(csv_path)}.done")
            _plots(csv_path, axnames, outdir)
        else:
            print(f"# shard {shard_id} done ({n_mine} cells); waiting on other shards")
    else:
        open(csv_path + ".done", "w").close()   # issue #7: sentinel so the watchdog stops relaunching
        print(f"# wrote {csv_path}  (all {len(combos)} cells complete)")
        _plots(csv_path, axnames, outdir)


def _plots(csv_path, axnames, outdir):
    rows = list(csv.DictReader(open(csv_path)))
    def col(k):
        return np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
    tag = f"{MODEL}_{CONDITION['name']}_{'_'.join(axnames)}"
    if len(axnames) == 1:
        x = col(axnames[0]); order = np.argsort(x)
        keys = ["path_speed", "single_frac", "clear_L", "cadence", "mu_req_p95"]
        fig, axs = plt.subplots(len(keys), 1, figsize=(11, 2.2 * len(keys)), sharex=True)
        surv = col("survived")
        for ax, k in zip(axs, keys):
            y = col(k)
            ax.plot(x[order], y[order], "-", lw=0.8)
            ax.scatter(x[order], y[order], c=["g" if s > 0.5 else "r" for s in surv[order]], s=8)
            ax.set_ylabel(k); ax.grid(alpha=0.3)
        axs[-1].set_xlabel(axnames[0])
        fig.suptitle(f"gait sweep {tag}  (green=survived)", fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"sweep_{tag}.png"), dpi=130, bbox_inches="tight")
    print(f"# wrote plot for {tag}")


def viz(freq, leg_amp=None):
    import imageio.v2 as imageio
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    ids = make_ids(model); root = ids[3]
    p = dict(BASE); p["freq"] = freq
    if leg_amp is not None:
        p["leg_amp"] = leg_amp
    set_floor_friction(model, FLOOR_MU)
    _set_gait(p)
    aid, jadr = build_ids(model)
    set_initial_pose(model, data, aid, jadr)
    R = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera(); cam.azimuth = 135; cam.elevation = -12; cam.distance = 1.8
    render_every = max(1, int(round(0.02 / model.opt.timestep)))
    frames = []; k = 0
    while data.time < SIM_DURATION:
        gc.apply_ctrl(data, aid, data.time); mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            break
        if data.time >= gc.T_HOLD and k % render_every == 0:
            cam.lookat[:] = data.xpos[root]
            R.update_scene(data, camera=cam); frames.append(R.render().copy())
        k += 1
    outdir = os.path.join(_HERE, "..", "results", "gait_sweep")
    os.makedirs(outdir, exist_ok=True)
    tag = f"{MODEL}_{CONDITION['name']}_f{freq:.2f}_leg{p['leg_amp']:.0f}"
    imageio.mimsave(os.path.join(outdir, f"viz_{tag}.gif"), frames[::3], fps=18)
    imageio.mimsave(os.path.join(outdir, f"viz_{tag}.mp4"), frames, fps=25, quality=8)
    print(f"# wrote viz_{tag}.gif/.mp4  ({len(frames)} frames)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        viz(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        sweep()
