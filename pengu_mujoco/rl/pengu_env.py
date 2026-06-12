"""
pengu_env.py - CPG-RL Gymnasium environment for the Pengu prismatic model.

Scheme A (Bellegarda & Ijspeert style): the policy modulates a small set of
Central-Pattern-Generator parameters at the control rate; an internal phase
oscillator turns them into joint targets every sim step. Low-dim, smooth,
sim-to-real friendly, and built directly on the prismatic RL model
(penguV2/scene_rl.xml) so there is no closed-loop / equality constraint.

Action (6, in [-1,1], low-pass smoothed, mapped to):
  0 freq        [0.8, 2.2] Hz
  1 leg_ext_amp [0.0, 0.05] m   (foot extension stroke; both legs, antiphase)
  2 hip_amp     [0.0, 0.45] rad
  3 torso_amp   [0.0, 0.50] rad
  4 hip_phase   [0, 2pi]        (hip vs leg-extension)
  5 torso_phase [0, 2pi]        (torso vs leg-extension)

Observation (28): base-frame projected gravity (3), base ang vel (3),
  base lin vel body-frame (3), joint pos (5), joint vel (5), CPG (sin,cos) (2),
  last action (6), vx command (1).

Reward (per control step): forward-velocity tracking (exp) + alive
  - upright(roll,pitch) - lateral drift - energy - action smoothness.
Termination: fall (base z < 0.08) or tilt > 60 deg.
"""
import os
import math
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

_HERE = os.path.dirname(os.path.abspath(__file__))

# Per-model config. "prismatic" = open-chain slider RL model; "crank" = original
# closed-loop crank-slider (CPU-only; leg extension commanded via crank disk angle).
_CFG = {
    "prismatic": dict(
        xml="scene_rl.xml",
        acts=["hip-L", "hip-R", "slider-L", "slider-R", "torso"],
        obs_joints=["hip-L", "hip-R", "slider-L", "slider-R", "torso"],
        leg_acts=("slider-L", "slider-R"),
        leg_mid=0.035, leg_rng=0.012,   # m  -> [0.023, 0.047]
    ),
    "crank": dict(
        xml="scene.xml",
        acts=["hip-L", "hip-R", "crank1-L", "crank1-R", "torso"],
        obs_joints=["hip-L", "hip-R", "crank1-L", "crank2_R", "torso"],
        leg_acts=("crank1-L", "crank1-R"),
        leg_mid=1.0, leg_rng=0.6,        # rad -> [0.4, 1.6]
    ),
}

# Action is modulation AROUND a nominal walking gait (so PPO starts walking and
# refines, instead of starting at standstill and getting stuck in a shuffle).
# a in [-1,1] maps to MID + a*RNG.  Order: freq, leg, hip_amp, torso_amp, hip_ph, torso_ph
# freq cap REMOVED (range [0.8,2.0]) -- penguins are short-legged high-cadence
# walkers; speed comes from stride FREQUENCY, not stride length, so we let the
# policy use high cadence and instead reward the hip swinging the foot FORWARD.
_MID = np.array([1.4, 0.0, 0.18, 0.16, math.pi, math.pi])
_RNG = np.array([0.6, 0.0, 0.12, 0.12, math.pi, math.pi])

# --- PENGUIN-PRIOR CPG (Phase 2: bio-imitation) -----------------------------
# Real king-penguin gait signature (Willener 2015/2016; Griffin & Kram 2000):
#   stride frequency ~1.27 Hz (== our crank natural freq), frontal/lateral
#   "waddle" roll amplitude ~8 deg, sagittal lean amplitude ~2 deg, upright.
# The energy-optimal RL policy converged to a high-cadence (~2 Hz) tiny shuffle
# that LOOKS mechanical. To teach the slow pendular waddle we (a) re-center the
# CPG nominal on penguin values and NARROW the action range (freq pinned near
# 1.27 Hz so it can't sprint up to 2 Hz), and (b) add a torso-level imitation
# reward that tracks the penguin cadence + 8 deg lateral rock + small lean.
PENGUIN_FREQ = 1.27       # Hz, stride frequency
PENGUIN_ROLL_DEG = 8.0    # deg, frontal waddle (roll) amplitude
PENGUIN_LEAN_DEG = 2.0    # deg, sagittal lean amplitude
_BIO_MID = np.array([PENGUIN_FREQ, 0.0, 0.10, 0.20, math.pi, math.pi])
_BIO_RNG = np.array([0.15, 0.0, 0.10, 0.10, math.pi, math.pi])  # freq -> [1.12,1.42]

HIP_CENTER = -0.25      # rad, nominal stand hip
INIT_Z = 0.20
INIT_PITCH = math.radians(-30.0)


class PenguCPGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, control_hz=50.0, episode_s=10.0, vx_cmd=0.15,
                 domain_rand=True, seed=None, model_kind="prismatic",
                 bio_imitate=False, propulsion=True):
        super().__init__()
        cfg = _CFG[model_kind]
        self.kind = model_kind
        self.bio_imitate = bio_imitate
        # bio-reward variant: propulsion=False -> v3 (clean waddle, cleanest lean,
        # rocks ~in place); propulsion=True -> v4 (adds r_swing: swing foot reaches
        # FORWARD so the rock turns into a step; shifts lateral KE toward forward).
        self.propulsion = propulsion
        ACTS = cfg["acts"]
        OBS_JOINTS = cfg["obs_joints"]
        self.leg_acts = cfg["leg_acts"]
        mid, rng = (_BIO_MID, _BIO_RNG) if bio_imitate else (_MID, _RNG)
        self.A_MID = mid.copy(); self.A_MID[1] = cfg["leg_mid"]
        self.A_RNG = rng.copy(); self.A_RNG[1] = cfg["leg_rng"]
        # one-stride sliding window (in control steps) for body roll/pitch
        # amplitude estimation, used by the bio-imitation reward.
        self._amp_win = max(4, int(round(control_hz / PENGUIN_FREQ)))
        self._roll_hist = []
        self._pitch_hist = []
        self.model = mujoco.MjModel.from_xml_path(os.path.join(_HERE, "..", "penguV2", cfg["xml"]))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.decim = max(1, int(round((1.0 / control_hz) / self.dt)))
        self.control_dt = self.decim * self.dt
        self.max_steps = int(episode_s * control_hz)
        self.vx_cmd = vx_cmd
        self.cmd_lo = vx_cmd
        self.cmd_hi = vx_cmd
        self.domain_rand = domain_rand

        self.obs_joints = OBS_JOINTS
        self.aid = {n: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ACTS}
        self.jadr = {n: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in (set(OBS_JOINTS) | {"hip-L", "hip-R"})}
        self.jvadr = {n: self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in OBS_JOINTS}
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot_bids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080"): "R",
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13"): "L",
        }

        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        obs_dim = 3 + 3 + 3 + 5 + 5 + 2 + 6 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self._rng = np.random.default_rng(seed)
        self.init_pitch = INIT_PITCH      # spawn pitch (rad); -30deg = ~upright
        self.phase = 0.0
        self.cpg = np.zeros(6)
        self.last_action = np.zeros(6, np.float32)
        self.R0_up = np.array([0.0, 0.0, 1.0])
        self.step_i = 0
        self._prev_xy = np.zeros(2)

    # ---- helpers ----
    def _map_action(self, a):
        a = np.clip(a, -1, 1)
        return self.A_MID + a * self.A_RNG

    def _set_targets(self):
        f, leg, hip_a, tor_a, hip_p, tor_p = self.cpg
        ph = self.phase
        legL_act, legR_act = self.leg_acts
        if self.kind == "prismatic":
            # slider extension in meters, range [-leg, 0]
            legL = -0.5 * leg * (1.0 + math.sin(ph))
            legR = -0.5 * leg * (1.0 + math.sin(ph + math.pi))
        else:
            # crank disk angle in radians, range [0, leg]
            legL = 0.5 * leg * (1.0 + math.sin(ph))
            legR = 0.5 * leg * (1.0 + math.sin(ph + math.pi))
        hL = HIP_CENTER + hip_a * math.sin(ph + hip_p)
        hR = HIP_CENTER + hip_a * math.sin(ph + hip_p + math.pi)
        tor = tor_a * math.sin(ph + tor_p)
        d = self.data
        d.ctrl[self.aid["hip-L"]] = hL
        d.ctrl[self.aid["hip-R"]] = hR
        d.ctrl[self.aid[legL_act]] = legL
        d.ctrl[self.aid[legR_act]] = legR
        d.ctrl[self.aid["torso"]] = tor

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

    def _base_frame(self):
        R = self.data.xmat[self.root].reshape(3, 3)
        return R

    def _roll_pitch(self):
        R = self._base_frame()
        up = R @ self.R0_up
        roll = math.atan2(up[0], up[2])
        pitch = math.atan2(up[1], up[2])
        return roll, pitch

    def _obs(self, vbody):
        d = self.data
        R = self._base_frame()
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        ang_vel = d.qvel[3:6].copy()  # free joint angular (body) velocity
        jpos = np.array([d.qpos[self.jadr[n]] for n in self.obs_joints])
        jvel = np.array([d.qvel[self.jvadr[n]] for n in self.obs_joints])
        cpg = np.array([math.sin(self.phase), math.cos(self.phase)])
        return np.concatenate([proj_g, ang_vel, vbody, jpos, jvel, cpg,
                               self.last_action, [self.vx_cmd]]).astype(np.float32)

    def set_cmd_range(self, lo, hi):
        """Curriculum hook: widen/shift the per-episode forward-speed command."""
        self.cmd_lo = float(lo)
        self.cmd_hi = float(hi)

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.vx_cmd = float(self._rng.uniform(self.cmd_lo, self.cmd_hi))
        mujoco.mj_resetData(self.model, self.data)
        d = self.data
        d.qpos[2] = INIT_Z
        pr = self.init_pitch + (self._rng.uniform(-0.05, 0.05) if self.domain_rand else 0.0)
        d.qpos[3] = math.cos(pr / 2); d.qpos[4] = math.sin(pr / 2)
        d.qpos[self.jadr["hip-L"]] = HIP_CENTER
        d.qpos[self.jadr["hip-R"]] = HIP_CENTER
        if self.domain_rand:
            self.model.geom_friction[self.floor_gid, 0] = self._rng.uniform(0.3, 0.9)
        mujoco.mj_forward(self.model, self.data)
        R0 = self._base_frame()
        self.R0_up = R0.T @ np.array([0.0, 0.0, 1.0])
        self.phase = float(self._rng.uniform(0, 2 * math.pi))
        self.cpg = self._map_action(np.zeros(6))
        self._target = self.cpg.copy()
        self._last_hc = math.floor(self.phase / math.pi)  # leg-switch counter
        self.last_action = np.zeros(6, np.float32)
        self.step_i = 0
        self._prev_xy = self.data.xpos[self.root][:2].copy()
        self._roll_hist = []
        self._pitch_hist = []
        return self._obs(np.zeros(3)), {}

    def step(self, action):
        action = np.asarray(action, np.float32)
        target = self._map_action(action)
        self._target = target
        # AMPLITUDES (leg, hip, torso) adapt every control step (per-step balance,
        # smoothed). RHYTHM params (freq idx0, hip-phase idx4, torso-phase idx5)
        # are LATCHED -- only updated at a leg-switch boundary (phase crosses k*pi)
        # so the cadence/phasing stays constant within each half-stride.
        amp = [1, 2, 3]
        self.cpg[amp] = 0.8 * self.cpg[amp] + 0.2 * target[amp]

        xy0 = self.data.xpos[self.root][:2].copy()
        energy = 0.0
        scrub = 0.0; n_contact = 0; n_single = 0      # gait-quality accumulators
        swing_fwd = 0.0; vert = 0.0
        foot_abs = {b: self.data.xpos[b][:2].copy() for b in self.foot_bids}
        relY = {b: float(self.data.xpos[b][1] - self.data.xpos[self.root][1]) for b in self.foot_bids}
        rootz = float(self.data.xpos[self.root][2])
        for _ in range(self.decim):
            self.phase += 2 * math.pi * self.cpg[0] * self.dt
            hc = math.floor(self.phase / math.pi)
            if hc != self._last_hc:           # crossed a leg-switch boundary
                self._last_hc = hc
                self.cpg[0] = target[0]       # latch frequency
                self.cpg[4] = target[4]       # latch hip phase
                self.cpg[5] = target[5]       # latch torso phase
            self._set_targets()
            mujoco.mj_step(self.model, self.data)
            energy += float(np.abs(self.data.actuator_force * self.data.actuator_velocity).sum())
            con = self._foot_contacts()
            if sum(con.values()) == 1:        # clean single-support (one foot bears load)
                n_single += 1
            bodyy = float(self.data.xpos[self.root][1])
            for b in self.foot_bids:
                ax = self.data.xpos[b][:2]
                ry = float(self.data.xpos[b][1] - bodyy)
                if con[b]:                    # stance foot should NOT slide / scrub
                    scrub += float(np.linalg.norm(ax - foot_abs[b])) / self.dt
                    n_contact += 1
                else:                         # swing foot should protract FORWARD (rel. body)
                    swing_fwd += (ry - relY[b])
                foot_abs[b] = ax.copy(); relY[b] = ry
            rz = float(self.data.xpos[self.root][2]); vert += abs(rz - rootz); rootz = rz
        energy /= self.decim
        scrub = scrub / max(1, n_contact)     # mean horizontal slip speed of stance feet
        single_frac = n_single / self.decim
        swing_rate = swing_fwd / self.control_dt   # forward protraction of swing feet (m/s)
        bob_rate = vert / self.control_dt          # vertical bobbing of the body (m/s)

        d = self.data
        xy1 = d.xpos[self.root][:2].copy()
        v_global = np.array([(xy1[0] - xy0[0]) / self.control_dt,
                             (xy1[1] - xy0[1]) / self.control_dt, 0.0])
        R = self._base_frame()
        vbody = R.T @ v_global
        vx = v_global[1]      # forward = world +y
        vy = v_global[0]      # lateral = world +x
        roll, pitch = self._roll_pitch()

        # tracking (sharp exp) + STRONG forward-progress. NOTE: we penalize ROLL
        # (lateral tip = unstable) but NOT moderate forward pitch -- walking needs
        # a forward lean (penguins lean more at speed), so penalizing pitch fought
        # locomotion and produced a "stand upright, don't move" policy. Only large
        # pitch (>~34 deg) is penalized.
        # WADDLE-TARGETED reward (phase-locked weight-transfer gait):
        #  - strong forward progress, penalize backward
        #  - single_support: reward clean alternating one-foot stance (weight transfer)
        #  - scrub: penalize stance foot sliding / "still extending after landing"
        #  - roll allowed (the waddle lateral rock); only excess tilt penalized
        # update one-stride roll/pitch window (for bio amplitude estimate)
        self._roll_hist.append(roll); self._pitch_hist.append(pitch)
        if len(self._roll_hist) > self._amp_win:
            self._roll_hist.pop(0); self._pitch_hist.pop(0)
        roll_amp_deg = math.degrees(0.5 * (max(self._roll_hist) - min(self._roll_hist)))
        pitch_amp_deg = math.degrees(0.5 * (max(self._pitch_hist) - min(self._pitch_hist)))
        f_cpg = float(self.cpg[0])

        r_energy = -0.0005 * energy
        r_smooth = -0.01 * float(np.sum((action - self.last_action) ** 2))
        if self.bio_imitate:
            # PENGUIN TEACHING reward: pull the gait onto the real-penguin signature
            # (cadence 1.27 Hz, 8 deg lateral rock, small lean) while RL keeps it
            # balanced and gently moving forward. Forward-progress pressure is LOW
            # on purpose -- chasing speed is what pushed the baseline up to 2 Hz.
            # Penguin imitation reward. Two variants (see self.propulsion):
            #   v3 (propulsion=False): cadence + 8deg rock + small-lean + gentle
            #     forward pressure. Clean slow waddle, cleanest lean, but rocks
            #     nearly IN PLACE (lateral KE ~76% vs penguin 30%, net speed ~0.04).
            #   v4 (propulsion=True): v3 + r_swing, rewarding the swing foot
            #     protracting FORWARD relative to the body so it lands ahead and the
            #     body vaults over the planted stance foot -> turns the rock into a
            #     STEP (lateral KE ~76%->47%) WITHOUT a velocity command (v2's
            #     velocity push only made it fall). Single-term diff vs v3.
            r_cadence = 1.0 * math.exp(-((f_cpg - PENGUIN_FREQ) / 0.15) ** 2)
            r_rock = 1.2 * math.exp(-((roll_amp_deg - PENGUIN_ROLL_DEG) / 3.0) ** 2)
            r_lean = -0.05 * max(0.0, pitch_amp_deg - 2.0 * PENGUIN_LEAN_DEG) ** 2
            r_track = 0.5 * math.exp(-((vx - self.vx_cmd) ** 2) / 0.02)
            r_progress = 1.0 * max(0.0, vx)
            r_back = 2.0 * min(0.0, vx)
            # v4 only: reward swing foot reaching FWD -> real steps (0 in v3)
            r_swing = 1.5 * float(np.clip(swing_rate, 0.0, 0.6)) if self.propulsion else 0.0
            r_single = 0.3 * single_frac
            r_scrub = -0.8 * scrub
            r_bob = -0.15 * bob_rate
            reward = (r_cadence + r_rock + r_lean + r_track + r_progress + r_back
                      + r_swing + r_single + r_scrub + r_bob + r_energy + r_smooth)
        else:
            r_track = 0.8 * math.exp(-((vx - self.vx_cmd) ** 2) / 0.02)
            r_progress = 4.0 * max(0.0, vx)
            r_back = 2.0 * min(0.0, vx)                       # <0 when moving backward
            r_single = 0.3 * single_frac                       # clean weight transfer
            r_scrub = -0.8 * scrub                             # no stance-foot slip
            r_swing = 1.0 * float(np.clip(swing_rate, 0.0, 0.6))  # hip protracts foot FORWARD -> real steps
            r_bob = -0.15 * bob_rate                           # discourage vertical bobbing/stomp
            r_roll = -0.3 * max(0.0, abs(roll) - 0.5) ** 2     # allow waddle up to ~29 deg
            r_pitch = -0.3 * max(0.0, abs(pitch) - 0.6) ** 2   # allow lean up to ~34 deg
            reward = (r_track + r_progress + r_back + r_single + r_scrub + r_swing + r_bob
                      + r_roll + r_pitch + r_energy + r_smooth)

        self.last_action = action
        self.step_i += 1
        z = d.xpos[self.root][2]
        fell = (z < 0.08) or (abs(roll) > math.radians(60)) or (abs(pitch) > math.radians(60)) \
            or not np.isfinite(d.qpos).all()
        terminated = bool(fell)
        truncated = self.step_i >= self.max_steps
        if fell:
            reward -= 5.0
        info = {"vx": vx, "roll": roll, "pitch": pitch, "energy": energy,
                "f_cpg": f_cpg, "roll_amp_deg": roll_amp_deg, "pitch_amp_deg": pitch_amp_deg}
        return self._obs(vbody), float(reward), terminated, truncated, info


def make_env(**kw):
    def _f():
        return PenguCPGEnv(**kw)
    return _f
