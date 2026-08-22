"""Direct-joint-control RL environment for the GRID-4 RL phase (ice arm).

Scratch training on the COM-1.31 body: mu ~ U(0.1, 0.4) per episode (NOT
observable), vx_cmd = 0.47 m/s. 0.47 is the c6 K=5 net_fwd_mean ceiling at
mu=0.1 (topupK5: 0.4689); the kappa=0 sweep ceiling there is 0.164, so the
commanded speed itself creates the pressure toward torso use -- the reward
carries no torso preference in either direction.

Contract (frozen after Gate 0; see docs/rl_e2_ice_memo.md):
- action: 5-dim [-1,1] @ 50 Hz -> position targets over the full ctrlrange,
  canonical actuator order gait_config.ACTUATORS; first-order filter ALPHA
- obs: 36-dim proprioception, no clock/phase, no mu, no world pose
- reward: task terms + regularizers + declared stepping prior (swing/scrub);
  no term touches the stepping MECHANISM (torso weight-shift vs leg extension)
- DR tier-1 on for training; eval_mode reproduces the frozen sweep protocol
  conditions (mu +-5% jitter, pose jitter only, no noise/delay)

Unlike rl/pengu_env.py there is no CPG: no oscillator, no phase, no latching.
"""
import collections
import math
import os
import sys

os.environ["PENGU_MODEL"] = "1.31"
os.environ.setdefault("CONFIG", "c6")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (_PKG, os.path.join(_PKG, "physics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

import gait_config as gc                    # noqa: E402
import friction_utils as fu                 # noqa: E402
from grid4_sweep import apply_com_variant   # noqa: E402

assert gc.XML_PATH.endswith("pengu1_31/scene.xml"), gc.XML_PATH
assert mujoco.__version__.startswith("3.8"), mujoco.__version__

XML = os.path.join(_PKG, gc.XML_PATH)

COM_TARGET = 1.31
TOTAL_MASS = 2.2724       # kg, native pengu1_31 (grid4 base)
COM_SLIDE = 0.00873       # m, expected apply_com_variant slide on the native body
VX_CMD = 0.47             # m/s
ALPHA = 0.2               # action filter (capability knob, freezes after Gate 0)
STALL_TORQUE = 4.1        # N*m, XM430 forcerange

# Reward version log (analysis in docs/rl_e2_ice_memo.md):
# v1: progress 4*max(0,vx), fall -5. Outcome: lunge local optimum -- the dash
#     harvests progress+track+swing from step 1 (~70/ep) while survival is
#     priced at ~0, so "what it gives up" costs the optimizer nothing.
# v2: progress 1*max(0,vx), fall -10. Sustained terms (tracking kernel, swing)
#     become the main income; per-episode ladder now stand(0) < dash(~+4)
#     < step-in-place(~+100) < walk(~+750): steepest ascent = live longer.
# v3: r2 + hf (high-frequency action penalty). The e2x2 cells all converged to
#     ~5 Hz aerial mincing (42% airborne, commanded slew past the XM430
#     no-load speed 56-81% of the time): the alpha=0.2 filter attenuates but
#     the policy repays with 2.7x pre-filter amplitude. hf prices exactly the
#     content the filter rejects: ||a - act_filt||^2. Calibration 2026-08-22
#     (per-step resid^2, mu=0.1): a1p1-final 0.81, a2p0-final 0.30, c6 teacher
#     0.13 -> w=0.5 taxes c6 7% of its positive reward, a1p1 54%, a2p0 16%.
# v3b: hf residual scaled to band-independent units (x ctrl_half / a1-ref
#     halves). Round-1 r3 priced in normalized units, so the a2 band diluted
#     the crank tax ~10x and bought high frequency back. a1 pricing unchanged.
# v3c: hf priced on the EXECUTED signal (act_filt vs a second alpha cascade)
#     instead of the commanded residual. r3b taxed exploration NOISE at
#     -0.94/step in the a2 band (scale^2 x sigma^2), making death (-10 ~= 11
#     steps of tax) cheaper than living -> learned suicide at 250k (ep_len
#     18, fall 1.00). The servo tracks act_filt, so executed HF is the
#     physically meaningful quantity; white-noise tax drops ~9x and cannot be
#     dodged by pre-filter amplitude inflation (tax is on what executes).
#     Priced on HIPS+TORSO ONLY: per-dim audit (2026-08-22) shows the cheat
#     policies' HF lives in torso/hips (a1p1 torso 0.045) while c6's lives in
#     the cranks (0.231/dim, hips/torso 0.003/0.010) — and the crank's honest
#     speed limit is an open hardware question (servo model/gearing pending
#     from Ben), so cranks are unpriced until that is settled. w recalibrated
#     0.5 -> 6.0 on the 3-dim quantity: 5Hz-cheat tax 50% of its positive
#     reward (round-1 deterrent level), a2p0-cheat 25%, c6 9%, noise
#     0.085/step (suicide breakeven ~118 steps: no death gradient).
#     OUTCOME: stand-deadlock, 2/2 cells 0/5 at every mu. Executed pricing
#     exempts noise (the alpha filter absorbs it first) and taxes coherent
#     oscillation, with the cascade residual peaking on the 1-3 Hz walking
#     band: "stand deterministically + let sigma collect swing" wins.
# v3d: back to the commanded residual (the round-1 form that killed 5 Hz and
#     still walked), restricted to hips+torso (cranks unpriced pending
#     hardware facts), w=1.0. Calibration (HT commanded resid^2, mu=0.1):
#     5Hz-cheat 78% of pos reward, r2-a2p0-cheat 29%, r3-a2p0-walker 13%,
#     c6 4%, noise -0.236/step (round-1's empirically survivable level).
#     Executed-accel variant rejected offline: broadband noise accel (0.033)
#     exceeds the 5 Hz cheat's (0.025) -> any deterrent w suicides.
REWARD_VERSION = "r3d"
HF_IDX = [i for i in range(len(gc.ACTUATORS)) if i not in
          (gc.ACTUATORS.index("crank1-R"), gc.ACTUATORS.index("crank1-L"))]
# Frozen r2 reward weights. Every one is overridable per-run via
# Grid4RLEnv(rw={...}) / train_grid4.py --rw key=val, which also retags the run,
# so a tuning arm can never be pooled with a frozen-recipe run. Absent an
# override the numbers below are used and the frozen path is unchanged.
RW_DEFAULT = {
    "track":    0.8,      # tracking kernel gain
    "sigma2":   0.02,     # kernel variance: exp(-(vx-cmd)^2 / sigma2), sigma ~ 0.14
    "progress": 1.0,      # W_PROGRESS, forward driver (v1 used 4.0)
    "back":     2.0,      # backward penalty gain
    "energy":   0.0005,   # torso EXCLUDED
    "swing":    1.0,      # stepping prior, clipped at swing_cap
    "swing_cap": 0.6,
    "scrub":    0.8,      # stance slip
    "smooth":   0.01,     # action rate
    "fall":     10.0,     # magnitude; applied as -fall (v2 raised this from 5.0)
    "hf":       1.0,      # commanded-HF residual, hips+torso only (r3d; see log)
}
W_SMOOTH_DEFAULT = RW_DEFAULT["smooth"]
W_PROGRESS = 1.0
FALL_PENALTY = 10.0

# Action-mapping version (capability knob, Gate 0 period, logged in memo):
# a0: full ctrlrange on all 5 actuators. Outcome: with policy sigma ~0.9 the
#     crank +-180deg mapping turns exploration noise into huge joint excursions
#     -- even "stand still" dies in <1 s under noise, so the dash stays optimal
#     regardless of reward pricing (observed on r1 and r2).
# a1: cranks narrowed toward the measured effective region (the
#     train_penguin_crank_fix finding, [-1.9,-1.1]). Probe 2026-08-20: crank
#     held anywhere in [-1.9,-1.0] stands; stance-angle scan over 8 jitter
#     seeds picked -1.2 as the most topple-robust neutral (7/8 vs 6/8 at
#     -1.5). Band [-1.8,-0.6]: neutral action = settle stance = -1.2.
#     Hips/torso untouched.
ACTION_VERSION = "a1"
CRANK_MID = -1.2
CRANK_HALF = 0.6

ACT_NAMES = list(gc.ACTUATORS)            # ["hip-L","hip-R","crank1-R","torso","crank1-L"]
TORSO_IDX = ACT_NAMES.index("torso")
LEG_IDX = [i for i in range(len(ACT_NAMES)) if i != TORSO_IDX]
CRANK_IDX = [ACT_NAMES.index("crank1-R"), ACT_NAMES.index("crank1-L")]

_WORLD_Z = np.array([0.0, 0.0, 1.0])
_WORLD_FWD = np.array([0.0, 1.0, 0.0])    # spawn faces world +y (sweep convention)
_AXES6 = [np.array(v, float) for v in
          ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _quat_about(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


class Grid4RLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, control_hz=50.0, episode_s=10.0, vx_cmd=VX_CMD,
                 mu_lo=0.1, mu_hi=0.4, mu_fixed=None, mu_jitter=0.05,
                 seed=None, eval_mode=False,
                 filter_alpha=ALPHA, action_delay=1,
                 obs_noise=True, init_jitter=True,
                 rand_gains=False, push=False, w_smooth=None, rw=None,
                 crank_band=None):
        # crank_band=(mid, half): override the a1 crank action band. a2 probe
        # (reward audit 2026-08-21): (0.0, 1.9) — symmetric band covering both
        # the c6 designed gait's command domain [0,+1.83] (inexpressible under
        # a1) and the a1 negative band; settle stance follows the band mid.
        super().__init__()
        self.rw = dict(RW_DEFAULT)
        if rw:
            unknown = set(rw) - set(RW_DEFAULT)
            if unknown:
                raise ValueError(f"unknown reward weight(s): {sorted(unknown)}; "
                                 f"valid: {sorted(RW_DEFAULT)}")
            self.rw.update({k: float(v) for k, v in rw.items()})
            if self.rw["sigma2"] <= 0:
                raise ValueError("sigma2 must be > 0 (it is the kernel variance, "
                                 "exp(-(vx-cmd)^2/sigma2)); to switch the kernel "
                                 "off use track=0 instead")
        if w_smooth is not None:          # back-compat with --no-smooth
            self.rw["smooth"] = float(w_smooth)
        self.w_smooth = self.rw["smooth"]
        self.eval_mode = bool(eval_mode)
        if self.eval_mode:                 # frozen protocol: pose jitter only
            action_delay, obs_noise, init_jitter = 0, False, False
            rand_gains, push = False, False

        self.model = mujoco.MjModel.from_xml_path(XML)
        s, got = apply_com_variant(self.model, COM_TARGET)
        m_tot = float(self.model.body_mass.sum())
        assert abs(m_tot - TOTAL_MASS) < 2e-3, f"total mass {m_tot}"
        assert abs(s - COM_SLIDE) < 1.5e-3, f"COM slide {s * 1e3:.2f} mm (expect ~8.73)"
        self.com_slide, self.com_ratio = float(s), float(got)
        self.data = mujoco.MjData(self.model)

        self.dt = float(self.model.opt.timestep)
        self.decim = max(1, int(round((1.0 / control_hz) / self.dt)))
        self.control_dt = self.decim * self.dt
        self.max_steps = int(episode_s * control_hz)
        self.vx_cmd = float(vx_cmd)
        self.mu_lo, self.mu_hi = float(mu_lo), float(mu_hi)
        self.mu_fixed, self.mu_jitter = mu_fixed, float(mu_jitter)
        self.alpha = float(filter_alpha)
        self.delay = int(action_delay)
        self.obs_noise = bool(obs_noise)
        self.init_jitter = bool(init_jitter)
        self.rand_gains = bool(rand_gains)
        self.push = bool(push)

        self._act_ids, self._jnt_adr = gc.build_ids(self.model)
        self.aid = np.array([self._act_ids[n] for n in ACT_NAMES])
        cr = self.model.actuator_ctrlrange[self.aid]
        self.ctrl_mid = cr.mean(axis=1)
        self.ctrl_half = (cr[:, 1] - cr[:, 0]) / 2.0
        self.crank_mid, self.crank_half = (
            (float(crank_band[0]), float(crank_band[1])) if crank_band
            else (CRANK_MID, CRANK_HALF))                 # a1 default
        self.ctrl_mid[CRANK_IDX] = self.crank_mid
        self.ctrl_half[CRANK_IDX] = self.crank_half
        # r3b: hf penalty is priced in a band-independent frame. Round-1 (r3)
        # priced ||a - act_filt||^2 in NORMALIZED units, so the a2 crank band
        # (half 1.9 vs a1's 0.6) paid ~1/10 the tax for the same physical
        # motion and bought the high frequency back (session memo, Round 1
        # mid-run finding). Scale residuals by ctrl_half relative to the a1
        # reference halves: a1 cells price exactly as in round 1.
        href = (cr[:, 1] - cr[:, 0]) / 2.0
        href[CRANK_IDX] = CRANK_HALF
        self.hf_scale = self.ctrl_half / href

        jids = self.model.actuator_trnid[self.aid, 0]
        self.jqadr = self.model.jnt_qposadr[jids]
        self.jvadr = self.model.jnt_dofadr[jids]

        sj = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
              for n in ("slider-L", "slider-R")]
        assert min(sj) >= 0, "slider joints not found"
        self.sqadr = self.model.jnt_qposadr[sj]
        self.svadr = self.model.jnt_dofadr[sj]
        srange = self.model.jnt_range[sj]
        self.slider_mid = srange.mean(axis=1)
        self.slider_half = np.maximum((srange[:, 1] - srange[:, 0]) / 2.0, 1e-4)

        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_bids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080"): "R",
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13"): "L",
        }

        self._gain0 = self.model.actuator_gainprm[self.aid].copy()
        self._bias0 = self.model.actuator_biasprm[self.aid].copy()
        self._damp0 = self.model.dof_damping.copy()

        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (36,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.mu = None

    def set_vx_cmd(self, v):
        """Curriculum hook (trainer-side, via VecEnv.env_method)."""
        self.vx_cmd = float(v)

    # ---------------------------------------------------------------- frames
    def _calibrate_frames(self):
        """com_wiper self-calibration: pick the root local axis best aligned with
        world +y at the neutral stand (easytorso local +y points world-DOWN, so
        never trust raw body axes)."""
        R0 = self.data.xmat[self.root].reshape(3, 3)
        self.fwd_local = max(_AXES6, key=lambda v: float((R0 @ v) @ _WORLD_FWD))
        f = R0 @ self.fwd_local
        fh = np.array([f[0], f[1], 0.0])
        n = np.linalg.norm(fh)
        self.fh0 = fh / n if n > 1e-9 else _WORLD_FWD.copy()
        self.left0 = np.cross(_WORLD_Z, self.fh0)
        self.up_local = R0.T @ _WORLD_Z
        Rt = self.data.xmat[self.torso_bid].reshape(3, 3)
        self.torso_up_local = Rt.T @ _WORLD_Z

    def _tilt(self):
        """(roll, pitch) of the root body: tilt of its calibrated up-axis about the
        spawn heading (pitch) / spawn left (roll) axes."""
        R = self.data.xmat[self.root].reshape(3, 3)
        u = R @ self.up_local
        roll = math.atan2(float(u @ self.left0), max(float(u @ _WORLD_Z), 1e-9))
        pitch = math.atan2(float(u @ self.fh0), max(float(u @ _WORLD_Z), 1e-9))
        return roll, pitch

    def torso_roll(self):
        """World-frame roll of the torso body about the spawn heading axis (rad)."""
        Rt = self.data.xmat[self.torso_bid].reshape(3, 3)
        u = Rt @ self.torso_up_local
        return math.atan2(float(u @ self.left0), max(float(u @ _WORLD_Z), 1e-9))

    def root_roll(self):
        """World-frame roll of the root (hip-axis) body about spawn heading (rad)."""
        r, _ = self._tilt()
        return r

    def _foot_contacts(self):
        out = {b: False for b in self.foot_bids}
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            if self.floor_gid in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.floor_gid else c.geom1
                b = self.model.geom_bodyid[other]
                if b in out:
                    out[b] = True
        return out

    # ---------------------------------------------------------------- reset
    def _settle(self):
        """Hold the stand targets until rocking decays (staged-start analog for
        RL episodes; the sweep protocol settles ~11 s before walking). Returns
        False if the robot topples during the hold -> caller resamples jitter."""
        d = self.data
        R_pre = d.xmat[self.root].reshape(3, 3)
        u_local = R_pre.T @ _WORLD_Z
        n_min = int(0.3 / self.dt)
        n_max = int(1.0 / self.dt)
        for i in range(n_max):
            mujoco.mj_step(self.model, d)
            if not np.isfinite(d.qpos).all() or d.xpos[self.root][2] < 0.08:
                return False
            if i >= n_min and float(np.max(np.abs(d.qvel))) < 0.3:
                break
        u = d.xmat[self.root].reshape(3, 3) @ u_local
        return float(u @ _WORLD_Z) > math.cos(math.radians(30))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng

        if self.mu_fixed is not None:
            mu = float(self.mu_fixed) * (1.0 + rng.uniform(-self.mu_jitter, self.mu_jitter))
        else:
            mu = float(rng.uniform(self.mu_lo, self.mu_hi))
        self.mu = mu
        fu.set_floor_friction(self.model, mu)

        self.model.actuator_gainprm[self.aid] = self._gain0
        self.model.actuator_biasprm[self.aid] = self._bias0
        self.model.dof_damping[:] = self._damp0
        if self.rand_gains:                             # tier-2 DR (off by default)
            f = rng.uniform(0.9, 1.1)
            self.model.actuator_gainprm[self.aid, 0] *= f
            self.model.actuator_biasprm[self.aid, 1] *= f
            self.model.dof_damping[:] *= rng.uniform(0.9, 1.1)

        d = self.data
        for _attempt in range(8):
            mujoco.mj_resetData(self.model, self.data)
            gc.set_initial_pose(self.model, self.data, self._act_ids, self._jnt_adr)

            # sweep-protocol pose jitter (kept in eval too): yaw +-5deg,
            # pitch +-3deg, lateral +-1cm
            yaw = math.radians(rng.uniform(-5, 5))
            pitch = math.radians(rng.uniform(-3, 3))
            qj = _quat_mul(_quat_about(_WORLD_Z, yaw), _quat_about([1, 0, 0], pitch))
            d.qpos[3:7] = _quat_mul(qj, d.qpos[3:7].copy())
            d.qpos[0] += rng.uniform(-0.01, 0.01)

            if self.init_jitter:                        # tier-1 DR (training only)
                d.qpos[self.jqadr] += rng.uniform(-0.05, 0.05, size=len(self.jqadr))

            mujoco.mj_forward(self.model, d)
            ctrl0 = d.ctrl[self.aid].copy()
            if not np.any(ctrl0):
                ctrl0 = d.qpos[self.jqadr].copy()
            ctrl0[CRANK_IDX] = self.crank_mid           # settle at working stance
            d.ctrl[self.aid] = ctrl0
            if self._settle():                          # quiet stand reached
                break
            # else: this jitter draw topples on its own -> resample

        if self.init_jitter:
            d.qvel[:] += rng.normal(0.0, 0.02, size=d.qvel.shape)
        mujoco.mj_forward(self.model, d)
        self._calibrate_frames()
        ctrl0 = d.ctrl[self.aid].copy()
        a0 = np.clip((ctrl0 - self.ctrl_mid) / self.ctrl_half, -1.0, 1.0)
        self.act_filt = a0.astype(np.float64)
        self.act_filt2 = a0.astype(np.float64)    # r3c: cascade for executed-HF pricing
        self.last_action = a0.astype(np.float32)
        self._queue = collections.deque([ctrl0.copy()] * self.delay, maxlen=self.delay + 1)

        self.step_i = 0
        self._prev_xy = d.xpos[self.root][:2].copy()
        self._push_next = rng.uniform(2.0, 4.0) if self.push else np.inf
        self._push_left = 0
        self._push_force = np.zeros(3)
        self._ep = {k: 0.0 for k in
                    ("r_track", "r_progress", "r_back", "r_energy", "r_swing",
                     "r_scrub", "r_smooth", "r_hf", "r_fall", "vx")}
        self._torso_roll_sq = 0.0
        self._torso_roll_sum = 0.0   # DC component: separates a steady lean from a waddle
        self._torso_roll_prev = self.torso_roll()
        self._torso_rate_sq = 0.0    # RMS(d roll/dt): a steady lean has ~0, a waddle does not
        # stride bookkeeping: touchdown-to-touchdown distance per foot. A robot
        # walking straight takes equal strides left and right; a turning one does
        # not. Measures the LEGS, so it stays clear of the torso variable.
        self._con_prev = {b: True for b in self.foot_bids}
        self._td_xy = {b: None for b in self.foot_bids}
        self._stride = {b: [] for b in self.foot_bids}
        self._single = 0
        self._sub = 0
        # left-right alternation stats: running sums of hip-L/hip-R angles
        self._hip = np.zeros(6)          # n, sx, sy, sxx, syy, sxy
        return self._obs(), {}

    # ---------------------------------------------------------------- step
    def step(self, action):
        d = self.data
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.act_filt = (1.0 - self.alpha) * self.act_filt + self.alpha * a
        target = self.ctrl_mid + self.act_filt * self.ctrl_half
        if self.delay > 0:
            self._queue.append(target.copy())
            applied = self._queue.popleft()
        else:
            applied = target
        d.ctrl[self.aid] = applied

        t = self.step_i * self.control_dt
        if self.push and t >= self._push_next:
            ang = self._rng.uniform(0, 2 * math.pi)
            mag = self._rng.uniform(1.0, 4.0)           # N, small shove
            self._push_force = np.array([mag * math.cos(ang), mag * math.sin(ang), 0.0])
            self._push_left = int(0.1 / self.dt)
            self._push_next = t + self._rng.uniform(2.0, 4.0)

        energy = 0.0
        scrub = 0.0
        n_contact = 0
        swing_fwd = 0.0
        fh2 = self.fh0[:2]
        root_xy = d.xpos[self.root][:2]
        rel_fwd = {b: float((d.xpos[b][:2] - root_xy) @ fh2) for b in self.foot_bids}
        foot_abs = {b: d.xpos[b][:2].copy() for b in self.foot_bids}

        for _ in range(self.decim):
            if self._push_left > 0:
                d.xfrc_applied[self.root, :3] = self._push_force
                self._push_left -= 1
            elif self.push:
                d.xfrc_applied[self.root, :3] = 0.0
            mujoco.mj_step(self.model, d)
            f = d.actuator_force[self.aid]
            v = d.actuator_velocity[self.aid]
            energy += float(np.abs(f[LEG_IDX] * v[LEG_IDX]).sum())   # torso EXCLUDED
            con = self._foot_contacts()
            if sum(con.values()) == 1:
                self._single += 1
            self._sub += 1
            root_xy = d.xpos[self.root][:2]
            for b in self.foot_bids:
                xy = d.xpos[b][:2]
                rf = float((xy - root_xy) @ fh2)
                if con[b]:
                    scrub += float(np.linalg.norm(xy - foot_abs[b])) / self.dt
                    n_contact += 1
                else:
                    swing_fwd += rf - rel_fwd[b]
                if con[b] and not self._con_prev[b]:          # touchdown edge
                    if self._td_xy[b] is not None:
                        self._stride[b].append(
                            float(np.linalg.norm(xy - self._td_xy[b])))
                    self._td_xy[b] = xy.copy()
                self._con_prev[b] = con[b]
                foot_abs[b] = xy.copy()
                rel_fwd[b] = rf
        energy /= self.decim
        scrub = scrub / max(1, n_contact)
        swing_rate = swing_fwd / self.control_dt

        xy = d.xpos[self.root][:2].copy()
        vx = float((xy - self._prev_xy) @ fh2) / self.control_dt
        self._prev_xy = xy

        w = self.rw
        r_track = w["track"] * math.exp(-((vx - self.vx_cmd) ** 2) / w["sigma2"])
        r_progress = w["progress"] * max(0.0, vx)
        r_back = w["back"] * min(0.0, vx)
        r_energy = -w["energy"] * energy
        r_swing = w["swing"] * float(np.clip(swing_rate, 0.0, w["swing_cap"]))
        r_scrub = -w["scrub"] * scrub
        r_smooth = -self.w_smooth * float(np.sum((a - self.last_action.astype(np.float64)) ** 2))
        # r3d: commanded residual (round-1 form), hips+torso only; hf_scale
        # is 1 on those dims in both bands (same ctrlrange), kept for safety.
        _hf_resid = ((a - self.act_filt) * self.hf_scale)[HF_IDX]
        r_hf = -w["hf"] * float(_hf_resid @ _hf_resid)
        reward = r_track + r_progress + r_back + r_energy + r_swing + r_scrub + r_smooth + r_hf

        roll, pitch_t = self._tilt()
        z = float(d.xpos[self.root][2])
        fell = (z < 0.08 or abs(roll) > math.radians(60)
                or abs(pitch_t) > math.radians(60)
                or not np.isfinite(d.qpos).all())
        r_fall = -self.rw["fall"] if fell else 0.0
        reward += r_fall

        self.last_action = a.astype(np.float32)
        self.step_i += 1
        _tr = self.torso_roll()
        self._torso_roll_sq += _tr ** 2
        self._torso_roll_sum += _tr
        self._torso_rate_sq += ((_tr - self._torso_roll_prev) / self.control_dt) ** 2
        self._torso_roll_prev = _tr
        hl = float(d.qpos[self.jqadr[0]])
        hr = float(d.qpos[self.jqadr[1]])
        self._hip += (1.0, hl, hr, hl * hl, hr * hr, hl * hr)
        for k, val in (("r_track", r_track), ("r_progress", r_progress),
                       ("r_back", r_back), ("r_energy", r_energy),
                       ("r_swing", r_swing), ("r_scrub", r_scrub),
                       ("r_smooth", r_smooth), ("r_hf", r_hf),
                       ("r_fall", r_fall), ("vx", vx)):
            self._ep[k] += val

        terminated = bool(fell)
        truncated = self.step_i >= self.max_steps
        info = {}
        if terminated or truncated:
            n = max(1, self.step_i)
            info["ep"] = {
                **{k: v / n for k, v in self._ep.items()},
                "len": self.step_i,
                "mu": self.mu,
                "torso_roll_rms_deg": math.degrees(math.sqrt(self._torso_roll_sq / n)),
                # RMS^2 = mean^2 + var, so RMS alone cannot tell a +-30 deg waddle
                # from a steady 30 deg lean. Log the mean so they separate:
                # |mean| << RMS -> waddling; |mean| ~ RMS -> leaning and holding.
                "torso_roll_mean_deg": math.degrees(self._torso_roll_sum / n),
                # RMS of d(roll)/dt. diff(RMS) cannot separate a lean from a
                # waddle (both are flat), but RMS(diff) can: a held lean has
                # roll rate ~0 while a waddle does not.
                "torso_roll_rate_rms_dps":
                    math.degrees(math.sqrt(self._torso_rate_sq / n)),
                **self.stride_symmetry(),
                "single_frac": self._single / max(1, self._sub),
                "fell": float(fell),
                **self.hip_alternation(),
            }
        return self._obs(), float(reward), terminated, truncated, info

    def stride_symmetry(self):
        """Touchdown-to-touchdown distance per foot, and their asymmetry.

        Equal left and right strides = walking straight; a persistent
        difference = curving. asym = (L-R)/(L+R) in [-1,1], 0 = symmetric.
        Leg-only measure, so it does not touch the torso variable.
        """
        out = {}
        vals = {}
        for b, s_ in self._stride.items():
            side = self.foot_bids[b]
            vals[side] = float(np.mean(s_)) if s_ else float("nan")
            out[f"stride_{side}_m"] = vals[side]
            out[f"n_stride_{side}"] = len(s_)
        L, R = vals.get("L", float("nan")), vals.get("R", float("nan"))
        if L == L and R == R and (L + R) > 1e-9:
            out["stride_asym"] = float((L - R) / (L + R))
        else:
            out["stride_asym"] = float("nan")
        return out

    def hip_alternation(self):
        """Walk indicator without rendering: RMS of the (hip_L - hip_R) swing
        difference in deg (stepping amplitude; lunge/synchronous push ~ 0) and
        the hip_L/hip_R correlation (alternating gait -> ~ -1, in-phase -> +1)."""
        hn, sx, sy, sxx, syy, sxy = self._hip
        if hn < 10:
            return {"hip_diff_rms_deg": 0.0, "hip_corr": 0.0}
        mx, my = sx / hn, sy / hn
        vx_ = max(0.0, sxx / hn - mx * mx)
        vy_ = max(0.0, syy / hn - my * my)
        cov = sxy / hn - mx * my
        diff_var = max(0.0, vx_ + vy_ - 2.0 * cov)
        corr = cov / math.sqrt(vx_ * vy_) if vx_ > 1e-10 and vy_ > 1e-10 else 0.0
        return {"hip_diff_rms_deg": math.degrees(math.sqrt(diff_var)),
                "hip_corr": float(np.clip(corr, -1.0, 1.0))}

    # ---------------------------------------------------------------- obs
    def _obs(self):
        d = self.data
        R = d.xmat[self.root].reshape(3, 3)
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        angvel = d.qvel[3:6] / 5.0
        vbody = (R.T @ d.qvel[0:3]) / 1.0
        jpos = d.qpos[self.jqadr] / self.ctrl_half
        jvel = d.qvel[self.jvadr] / 10.0
        tau = d.actuator_force[self.aid] / STALL_TORQUE
        spos = (d.qpos[self.sqadr] - self.slider_mid) / self.slider_half
        svel = d.qvel[self.svadr] / 0.5
        con = self._foot_contacts()
        by_side = {s: float(v) for b, v in con.items() for s in [self.foot_bids[b]]}
        contacts = np.array([by_side["L"], by_side["R"]])

        obs = np.concatenate([
            proj_g, angvel, vbody, jpos, jvel, tau, spos, svel, contacts,
            self.last_action, [self.vx_cmd],
        ]).astype(np.float32)

        if self.obs_noise:
            noise = np.zeros(36)
            noise[0:3] = self._rng.normal(0, 0.02, 3)      # gravity dir
            noise[3:6] = self._rng.normal(0, 0.01, 3)      # angvel (scaled /5)
            noise[6:9] = self._rng.normal(0, 0.02, 3)      # linvel
            noise[9:14] = self._rng.normal(0, 0.01, 5)     # jpos (scaled)
            noise[14:19] = self._rng.normal(0, 0.01, 5)    # jvel (scaled /10)
            noise[19:24] = self._rng.normal(0, 0.025, 5)   # torque (scaled /4.1)
            noise[24:26] = self._rng.normal(0, 0.02, 2)    # slider pos
            noise[26:28] = self._rng.normal(0, 0.02, 2)    # slider vel
            obs = (obs + noise.astype(np.float32))
        return obs
