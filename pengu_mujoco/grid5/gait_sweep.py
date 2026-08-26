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

# --- opt-in: initial-pose jitter for domain randomization -----------------------------
# None = the fixed nominal stand pose (bit-identical default). A dict with keys
# {"yaw","pitch"} in radians and "lat" in metres perturbs the initial base attitude/
# position per trial (the DR sweep samples it fresh each repeat).
POSE_JITTER = None
gc.T_HOLD = 5.0          # wait longer to settle upright before pitching
gc.T_TRANSITION = 4.0    # MODERATE pitch-in ramp (not a sudden change)

# --- opt-in: GRID-5 measurement layer -------------------------------------------------
# Off by default: run_trial output is bit-identical to GRID-4. When a caller sets
# gs.EXTENDED_METRICS=True it must ALSO append EXT_FIELDS to its CSV fieldnames.
# All extended quantities are passive measurements of the same trajectory.
EXTENDED_METRICS = False
EXT_FIELDS = ["t_start", "t_fall", "fall_phase", "slip_dist2", "roll_dist", "slip_frac",
              "cone_util_p50", "cone_util_p95", "fn_peak", "fn_mean",
              "lat_disp", "lat_vel_rms", "e_pos", "cot_net", "cot_path", "power_mean",
              "imu_roll_mean", "imu_roll_rms", "imu_pitch_rms"]
# slip DUAL criterion: a contact is SLIPPING iff BOTH hold
#   cone:      |Ft| >= (1-SLIP_CONE_EPS) * mu_trial * Fn      (penetration/patch-immune)
#   kinematic: |v_tan| >= SLIP_C * |omega_foot| * r_patch + SLIP_V0
# r_patch = measured half-spread of that foot's contacts this step (0 for the usual
# single-contact case), so pure patch-edge rolling falls inside the deadband.
# Constants frozen by grid5/slip_calib_probe.py.
SLIP_CONE_EPS = 0.05
SLIP_V0 = 0.005          # [m/s] provisional until calibrated
SLIP_C = 1.0             # provisional until calibrated

# --- opt-in: GRID-5 staged start ------------------------------------------------------
# Off by default (legacy fixed T_HOLD=5 schedule, bit-identical). When True the hold
# lasts until the robot is quiescent (max|qvel| < QUIET_QVEL, between QUIET_MIN_T and
# QUIET_MAX_T after reset), then transition/settle/measure run t0-relative with the
# same durations (4s transition + 2s settle + 13s measure window). Callers normally
# combine this with gc.RAMP_HIP_OFFSET=True and a stand rest lean (gc.STAND_HIP_DEG=5).
STAGED_START = False
QUIET_QVEL = 0.3
QUIET_MIN_T, QUIET_MAX_T = 2.0, 10.0

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
                 "clear_L", "clear_R", "cadence", "n_steps", "mu_req_p95",
                 # slip = the planted (loaded) foot sliding on the ground. Appended at the
                 # END so the position of every earlier column is unchanged. slip_ratio =
                 # total sliding distance / total stride distance (avg fraction of each
                 # step spent sliding); a gait "slips" if slip_ratio > 0.05 (Ben's 5%).
                 "slip_dist", "slip_ratio",
                 # heading_align = base-front axis (body +y) . net-travel direction, in
                 # [-1,1]. +1 = robot FACES the way it travels (forward gait); ~-1 = it
                 # moves backward-facing (moonwalk). net_fwd alone is facing-blind, so this
                 # separates real forward gaits from retrograde ones. Filter: >0.5 = forward.
                 "heading_align"]


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
    """Run one cell; measure de-bounced gait-quality metrics. Pure measurement.

    GRID-5 extensions (both opt-in; defaults reproduce GRID-4 bit-identically):
      STAGED_START     — quiescence-gated hold, then transition/settle/measure run
                         t0-relative with the same durations (4+2+13 s).
      EXTENDED_METRICS — appends EXT_FIELDS: slip dual criterion, cone/GRF stats,
                         lateral drift/velocity, positive-work COT, torso-IMU
                         orientation, and fall timing/phase.
    """
    floor_id, foot_geom, foot_bid, root = ids
    set_floor_friction(model, FLOOR_MU)
    _set_gait(p)
    act, jadr = build_ids(model)
    set_initial_pose(model, data, act, jadr)
    if EXTENDED_METRICS:
        # torso-IMU mount calibration on the UN-jittered neutral pose: the recorded
        # orientation is relative to this attitude (an IMU mounted level at neutral).
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
        R0 = data.xmat[tid].reshape(3, 3).copy()
        m_g = float(model.body_mass.sum()) * 9.81
        mu_trial = FLOOR_MU
    if POSE_JITTER is not None:                    # DR: perturb the initial base pose
        j = POSE_JITTER
        data.qpos[0] += j.get("lat", 0.0)
        pp = math.radians(gc.INIT_PITCH_DEG) + j.get("pitch", 0.0)
        yy = j.get("yaw", 0.0)
        qpx = np.array([math.cos(pp / 2), math.sin(pp / 2), 0.0, 0.0])
        qy = np.array([math.cos(yy / 2), 0.0, 0.0, math.sin(yy / 2)])
        qj = np.zeros(4); mujoco.mju_mulQuat(qj, qy, qpx)
        data.qpos[3:7] = qj
        mujoco.mj_forward(model, data)
    f6 = np.zeros(6); vf6 = np.zeros(6)
    DT = float(model.opt.timestep)          # physics step == one measurement sample
    slip_dist_tot = 0.0                     # integral of stance-foot slide distance [m]
    face_sum = np.zeros(2)                   # sum of base-front (body +y) xy over the walk

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
    # displacement-weighted support split (Ben, 2026-08-17): how much of the base's travel
    # happens while BOTH feet are loaded (shuffle signature) vs single support.
    move_tot = 0.0; ds_move = 0.0; ss_move = 0.0; meas_prev = None
    pos_ws = None; last = data.xpos[root][:2].copy(); fell = False

    # --- start-protocol scheduling (legacy: fixed windows, bit-identical) -------------
    t_hold_orig = gc.T_HOLD
    if STAGED_START:
        gc.T_HOLD = 1e9                     # hold branch until quiescent
        t0 = None
        settle_t = float("inf"); end_t = float("inf")
    else:
        t0 = t_hold_orig
        settle_t = SETTLE; end_t = SIM_DURATION

    # --- extended accumulators --------------------------------------------------------
    if EXTENDED_METRICS:
        t_fall = float("nan")
        slip2 = 0.0; roll_dist = 0.0
        slip_samp = 0; loaded_samp = 0
        cone_utils = []; fn_samples = []
        e_pos = 0.0
        face0 = None
        lat_sq = 0.0; lat_n = 0
        imu_sum = 0.0; imu_sq = 0.0; pit_sq = 0.0; imu_n = 0
        prev_cent = {"L": None, "R": None}

    while data.time < end_t:
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if data.xpos[root][2] < 0.05:
            fell = True
            if EXTENDED_METRICS:
                t_fall = data.time
            break
        if STAGED_START and t0 is None:
            tq = data.time
            if (tq >= QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < QUIET_QVEL) \
                    or tq >= QUIET_MAX_T:
                t0 = tq; gc.T_HOLD = t0
                settle_t = t0 + gc.T_TRANSITION + 2.0
                end_t = settle_t + 13.0
        last = data.xpos[root][:2].copy()
        if data.time < settle_t:
            continue
        if pos_ws is None:
            pos_ws = data.xpos[root][:2].copy()
        face_sum += data.xmat[root].reshape(3, 3)[:2, 1]   # base +y (front) axis, xy
        if EXTENDED_METRICS:
            # torso-IMU orientation relative to the neutral mount: z-x-y intrinsic
            # (yaw about world z, pitch about lateral x, roll about forward y).
            # roll here equals the gravity-vector (pitch-immune) readout analytically.
            Rr = data.xmat[tid].reshape(3, 3) @ R0.T
            i_roll = math.atan2(-float(Rr[2, 0]), float(Rr[2, 2]))
            i_pitch = math.asin(max(-1.0, min(1.0, float(Rr[2, 1]))))
            imu_sum += i_roll; imu_sq += i_roll * i_roll
            pit_sq += i_pitch * i_pitch; imu_n += 1
            # positive mechanical actuator work (servos do not regenerate)
            pw = float(np.sum(np.maximum(data.qfrc_actuator * data.qvel, 0.0)))
            e_pos += pw * DT
        # per-foot normal/tangential force
        Fn = {"L": 0.0, "R": 0.0}; Ft = {"L": 0.0, "R": 0.0}
        cpos = {"L": np.zeros(2), "R": np.zeros(2)}  # force-weighted contact xy (opt-in)
        slipnum = {"L": 0.0, "R": 0.0}               # force-weighted slip-speed numerator
        if EXTENDED_METRICS:
            con_pts = {"L": [], "R": []}             # (fn, |Ft|, |v_tan|, pos_xy)
            omega_foot = {"L": 0.0, "R": 0.0}
        for c in range(data.ncon):
            ct = data.contact[c]
            fg = ct.geom2 if ct.geom1 == floor_id else (
                 ct.geom1 if ct.geom2 == floor_id else -1)
            ft = foot_geom.get(fg)
            if ft:
                mujoco.mj_contactForce(model, data, c, f6)
                fn = abs(f6[0])
                Fn[ft] += fn; Ft[ft] += math.hypot(f6[1], f6[2])
                if TRACK_COM_STANCE:
                    cpos[ft] += fn * ct.pos[:2]
                # slip speed = tangential velocity of the foot AT the contact point
                # (floor is static). Pure rolling -> ~0 here even as the foot body moves.
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, fg, vf6, 0)
                v_pt = vf6[3:6] + np.cross(vf6[0:3], ct.pos - data.geom_xpos[fg])
                n = ct.frame[0:3]
                v_tan = v_pt - np.dot(v_pt, n) * n
                slipnum[ft] += fn * float(np.linalg.norm(v_tan))
                if EXTENDED_METRICS:
                    con_pts[ft].append((fn, math.hypot(f6[1], f6[2]),
                                        float(np.linalg.norm(v_tan)), ct.pos[:2].copy()))
                    omega_foot[ft] = float(np.linalg.norm(vf6[0:3]))
        for s in ("L", "R"):
            if Fn[s] > F_HI:                       # issue #5: stance-gated (real load only)
                mu_req.append(Ft[s] / Fn[s])
                slip_dist_tot += (slipnum[s] / Fn[s]) * DT   # force-weighted slide, integrated
        if EXTENDED_METRICS:
            for s in ("L", "R"):
                pts = con_pts[s]
                if Fn[s] > F_HI and pts:
                    loaded_samp += 1
                    fn_samples.append(Fn[s])
                    r_patch = 0.0                  # measured patch half-spread this step
                    if len(pts) > 1:
                        for i2 in range(len(pts)):
                            for j2 in range(i2 + 1, len(pts)):
                                dxy = pts[i2][3] - pts[j2][3]
                                r_patch = max(r_patch, 0.5 * math.hypot(dxy[0], dxy[1]))
                    v_eps = SLIP_C * omega_foot[s] * r_patch + SLIP_V0
                    fn_slip = 0.0; num2 = 0.0
                    cent = np.zeros(2); fn_tot = 0.0
                    for fn_i, ftm_i, vt_i, pxy in pts:
                        if fn_i > 1e-9:
                            cone_utils.append(ftm_i / (mu_trial * fn_i))
                        slipping = (ftm_i >= (1.0 - SLIP_CONE_EPS) * mu_trial * fn_i
                                    and vt_i >= v_eps)
                        if slipping:
                            fn_slip += fn_i; num2 += fn_i * vt_i
                        cent += fn_i * pxy; fn_tot += fn_i
                    slip2 += (num2 / Fn[s]) * DT   # dual-criterion slide, integrated
                    foot_slipping = fn_slip > 0.5 * fn_tot
                    if foot_slipping:
                        slip_samp += 1
                    cxy = cent / fn_tot if fn_tot > 1e-9 else None
                    if (not foot_slipping) and cxy is not None and prev_cent[s] is not None:
                        # sticking stance: contact-centroid travel = rolling distance
                        roll_dist += math.hypot(cxy[0] - prev_cent[s][0],
                                                cxy[1] - prev_cent[s][1])
                    prev_cent[s] = cxy
                else:
                    prev_cent[s] = None
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
        # displacement-weighted split: attribute this step's base travel to the support state
        cur_xy = data.xpos[root][:2]
        if meas_prev is not None:
            d_step = float(math.hypot(cur_xy[0] - meas_prev[0], cur_xy[1] - meas_prev[1]))
            move_tot += d_step
            if nc == 2: ds_move += d_step
            elif nc == 1: ss_move += d_step
            if EXTENDED_METRICS:
                fvec = data.xmat[root].reshape(3, 3)[:2, 1]
                nf2 = float(np.linalg.norm(fvec))
                if nf2 > 1e-9:
                    fhat = fvec / nf2
                    vx = (cur_xy[0] - meas_prev[0]) / DT
                    vy = (cur_xy[1] - meas_prev[1]) / DT
                    lat_v = -vx * fhat[1] + vy * fhat[0]   # heading-frame lateral velocity
                    lat_sq += lat_v * lat_v; lat_n += 1
        if EXTENDED_METRICS and face0 is None:
            fvec = data.xmat[root].reshape(3, 3)[:2, 1]
            nf2 = float(np.linalg.norm(fvec))
            if nf2 > 1e-9:
                face0 = (fvec / nf2).copy()        # initial facing (lat_disp reference)
        meas_prev = cur_xy.copy()
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
    wt = max(1e-6, ((end_t if survived else data.time) - settle_t)
             if STAGED_START else
             ((SIM_DURATION if survived else data.time) - SETTLE))
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
               mu_req_p95=round(float(np.percentile(mu_req, 95)) if mu_req else float("nan"), 3),
               slip_dist=round(slip_dist_tot, 5),
               slip_ratio=round(slip_dist_tot / path, 5) if path > 1e-6 else float("nan"),
               # displacement-weighted support split (NOT in any sweep CSV schema; returned
               # for evaluation/topup/demo tools). ds_move_frac low = travel happens in
               # single support (conventional stepping); high = double-support shuffle.
               ds_move_frac=round(ds_move / move_tot, 4) if move_tot > 1e-6 else float("nan"),
               ss_move_frac=round(ss_move / move_tot, 4) if move_tot > 1e-6 else float("nan"))
    # heading: does the base front axis point the way the robot actually travelled?
    disp_vec = (last - pos_ws) if pos_ws is not None else np.zeros(2)
    nf, nd = np.linalg.norm(face_sum), np.linalg.norm(disp_vec)
    out["heading_align"] = (round(float(np.dot(face_sum / nf, disp_vec / nd)), 4)
                            if nf > 1e-9 and nd > 1e-9 else float("nan"))
    if TRACK_COM_STANCE:
        lat = np.asarray(cs_lat); dist = np.asarray(cs_dist)
        out["com_lat_mean"] = round(float(np.mean(np.abs(lat))), 5) if lat.size else float("nan")
        out["com_lat_rms"] = round(float(np.sqrt(np.mean(lat * lat))), 5) if lat.size else float("nan")
        out["com_stance_dist"] = round(float(np.mean(dist)), 5) if dist.size else float("nan")
        out["com_ss_n"] = int(lat.size)
    if EXTENDED_METRICS:
        nan = float("nan")
        out["t_start"] = round(float(t0), 3) if t0 is not None else nan
        out["t_fall"] = round(float(t_fall), 3) if fell else nan
        if fell:
            tf = float(t_fall)
            if t0 is None or tf < t0:                     ph = "hold"
            elif tf < t0 + gc.T_TRANSITION:               ph = "trans"
            elif tf < settle_t:                           ph = "settle"
            else:                                         ph = "walk"
        else:
            ph = ""
        out["fall_phase"] = ph
        out["slip_dist2"] = round(slip2, 5)
        out["roll_dist"] = round(roll_dist, 5)
        out["slip_frac"] = round(slip_samp / loaded_samp, 4) if loaded_samp else nan
        cu = np.asarray(cone_utils); fa = np.asarray(fn_samples)
        out["cone_util_p50"] = round(float(np.percentile(cu, 50)), 4) if cu.size else nan
        out["cone_util_p95"] = round(float(np.percentile(cu, 95)), 4) if cu.size else nan
        out["fn_peak"] = round(float(fa.max()) / m_g, 4) if fa.size else nan    # [BW]
        out["fn_mean"] = round(float(fa.mean()) / m_g, 4) if fa.size else nan   # [BW]
        if pos_ws is not None and face0 is not None:
            dvec = last - pos_ws
            out["lat_disp"] = round(abs(float(-dvec[0] * face0[1] + dvec[1] * face0[0])), 4)
        else:
            out["lat_disp"] = nan
        out["lat_vel_rms"] = round(math.sqrt(lat_sq / lat_n), 4) if lat_n else nan
        out["e_pos"] = round(e_pos, 4)
        net_abs = abs(net_fwd)
        out["cot_net"] = round(e_pos / (m_g * net_abs), 3) if net_abs > 0.02 else nan
        out["cot_path"] = round(e_pos / (m_g * path), 3) if path > 0.02 else nan
        out["power_mean"] = round(e_pos / wt, 4)
        out["imu_roll_mean"] = round(math.degrees(imu_sum / imu_n), 3) if imu_n else nan
        out["imu_roll_rms"] = round(math.degrees(math.sqrt(imu_sq / imu_n)), 3) if imu_n else nan
        out["imu_pitch_rms"] = round(math.degrees(math.sqrt(pit_sq / imu_n)), 3) if imu_n else nan
    if STAGED_START:
        gc.T_HOLD = t_hold_orig                 # do not leak the per-trial hold override
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
