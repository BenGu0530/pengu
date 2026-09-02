"""
Reactive torso roll control — the kappa follow-gain law.

WHY: `easytorso` is a child of `easyaxis`, so the torso joint angle is defined RELATIVE
to the hip axis. With the torso motor at a fixed angle the torso rides the axis and rolls
WITH it. So torso strategy is naturally one scalar — how much of the axis roll the torso
is asked to follow:

    target torso world roll  =  kappa * (hip-axis world roll)

    kappa = 0   -> torso held at absolute 0 roll in the world (counter-rotates the axis)
    kappa = 1   -> torso rides the axis (what the open-loop torso_amp=0 slice did)
    kappa > 1   -> torso leans further into the roll, toward the stance leg
    kappa < 0   -> torso leans away from the stance leg (control condition)

SIGN CONVENTION (measured, not assumed — this is a trap). The torso hinge is
`axis="0 0 1"` inside a body with `quat="0.5 0.5 -0.5 0.5"`, so a static sweep of the
joint gives d(torso world roll)/d(joint) = **-1.0**, i.e.

    T = A - J        (torso world roll = hip-axis roll  MINUS  joint angle)

Hence the joint needed for the target is J = (1 - kappa)*A, and a positive roll error
must be corrected by DECREASING J. Tracked by an outer-loop PID on the torso joint:
    cmd = (1-kappa)*axis_roll            # feedforward: joint needed for the target
        - (Kp*e + Ki*int(e) + Kd*de/dt)  # correction, e = target_roll - torso_roll
clamped to the torso ctrlrange. The torso motor is a position servo with a hard
+-4.1 N.m (XM430 stall) forcerange, so at high COM ratios this loop can SATURATE and
fail to hold the target — that is a measurement, not a bug. `roll_rms()` reports it.

ROLL DEFINITION — measured ABOUT THE HINGE AXIS, not in the world frame.
A world-frame lateral-tilt measure (atan2(v_x, hypot(v_y,v_z))) was tried first and is
WRONG: it agrees with the joint 1:1 only while the hinge axis stays near-horizontal and
aligned with the heading. Measured, it held in open-loop walking (corr 0.9998) but broke
badly once the robot leaned 11-33 deg to one side (A ~ J ~ -21 deg predicted T ~ 0, actual
T = -35.6 deg), because rotating a tilted hinge mixes roll and yaw.

The hinge can only rotate the torso about its own axis h, so that is the only error it can
null, and about h the kinematics are exact at ANY attitude:

    h        = data.xaxis[torso_joint]                  # hinge axis, world frame
    u_body   = R0^T @ z_world                           # body-fixed vector, up at neutral
    v(t)     = R(t) @ u_body                            # where it points now
    tilt(b)  = signed angle from world-up to v, about h # what the hinge can null
             = atan2( (z_perp x v_perp) . h , z_perp . v_perp )   with _perp = component
                                                                   perpendicular to h
Then torso_tilt = axis_tilt + s*J exactly (s = +-1, measured at import), because the joint
rotation IS a rotation about h. tilt = 0 means the body is as vertical as this DOF allows.
"""
import math

import mujoco
import numpy as np
from vec3 import cross3, norm3

import gait_config as gc

TORSO_BODY = "easytorso"
AXIS_BODY = "easyaxis"


_Z = np.array([0.0, 0.0, 1.0])


def _tilt_about(data, bid, u_body, h):
    """Signed tilt (rad) of the body's calibrated up-vector away from vertical,
    measured ABOUT the hinge axis h (world frame). 0 = as vertical as this DOF allows.
    Exact at any attitude, because the hinge rotation is exactly a rotation about h."""
    v = data.xmat[bid].reshape(3, 3) @ u_body
    v_p = v - np.dot(v, h) * h
    z_p = _Z - np.dot(_Z, h) * h
    nv, nz = norm3(v_p), norm3(z_p)
    if nv < 1e-9 or nz < 1e-9:          # body axis parallel to hinge -> tilt undefined
        return 0.0
    v_p /= nv
    z_p /= nz
    return math.atan2(float(np.dot(cross3(z_p, v_p), h)), float(np.dot(z_p, v_p)))


def _roll_of(data, bid, u_body):
    """DEPRECATED world-frame lateral tilt. Kept only for diagnostics; it is not
    hinge-consistent at large lean (see module docstring)."""
    v = data.xmat[bid].reshape(3, 3) @ u_body
    return math.atan2(v[0], math.hypot(v[1], v[2]))


class TorsoKappaPID:
    """Outer-loop PID realizing target_torso_roll = kappa * axis_roll.

    Self-calibrating: builds its own neutral pose internally, so the caller does not
    have to sequence anything. Stateful (integral + derivative) -> auto-resets when sim
    time goes backwards (i.e. a new trial started).
    """

    def __init__(self, model, kappa, kp=2.0, ki=0.1, kd=0.0, ctrl_limit_deg=45.0,
                 measure_after=0.0):
        self.kappa = float(kappa)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = math.radians(ctrl_limit_deg)
        self.measure_after = measure_after   # only accumulate roll_rms/sat once t >= this
        self.tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self.aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, AXIS_BODY)
        self.jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "torso")
        if min(self.tid, self.aid, self.jid) < 0:
            raise ValueError(f"missing: {TORSO_BODY}={self.tid} {AXIS_BODY}={self.aid} "
                             f"joint torso={self.jid}")

        # calibrate 'up' on a private neutral pose (deterministic for a given model)
        d0 = mujoco.MjData(model)
        act, jadr = gc.build_ids(model)
        gc.set_initial_pose(model, d0, act, jadr)
        mujoco.mj_forward(model, d0)
        self.u_torso = d0.xmat[self.tid].reshape(3, 3).T @ _Z
        self.u_axis = d0.xmat[self.aid].reshape(3, 3).T @ _Z

        # measure the joint->tilt sign s in  torso_tilt = axis_tilt + s*J  (expect +-1)
        qa = model.jnt_qposadr[self.jid]
        t0 = _tilt_about(d0, self.tid, self.u_torso, d0.xaxis[self.jid].copy())
        d0.qpos[qa] = math.radians(10.0)
        mujoco.mj_forward(model, d0)
        t1 = _tilt_about(d0, self.tid, self.u_torso, d0.xaxis[self.jid].copy())
        self.s = math.copysign(1.0, (t1 - t0) / math.radians(10.0))

        self.reset()

    def hinge(self, data):
        return data.xaxis[self.jid].copy()

    def reset(self):
        self._int = 0.0
        self._prev_e = None
        self._prev_t = None
        self._roll_sq = 0.0
        self._roll_n = 0
        self._sat_n = 0

    # --- measurement helpers (also usable standalone as readouts) -----------------
    def torso_roll(self, data, h=None):
        return _tilt_about(data, self.tid, self.u_torso, self.hinge(data) if h is None else h)

    def axis_roll(self, data, h=None):
        return _tilt_about(data, self.aid, self.u_axis, self.hinge(data) if h is None else h)

    def roll_rms(self):
        """RMS of torso world roll [deg] over the measured window (Gait 1 benchmark)."""
        if not self._roll_n:
            return float("nan")
        return math.degrees(math.sqrt(self._roll_sq / self._roll_n))

    def saturation_frac(self):
        """fraction of control steps where the joint command hit the ctrl limit."""
        if not self._roll_n:
            return float("nan")
        return self._sat_n / self._roll_n

    def start_measuring(self):
        self._roll_sq = 0.0
        self._roll_n = 0
        self._sat_n = 0

    # --- the controller ----------------------------------------------------------
    def __call__(self, data, t, alpha=1.0):
        if self._prev_t is not None and t < self._prev_t:
            self.reset()                      # sim time went backwards -> new trial
        dt = 0.0 if self._prev_t is None else (t - self._prev_t)
        self._prev_t = t

        h = self.hinge(data)
        a_roll = self.axis_roll(data, h)
        t_roll = self.torso_roll(data, h)
        target = self.kappa * a_roll
        e = target - t_roll

        if dt > 0.0:
            self._int += e * dt
            de = (e - self._prev_e) / dt if self._prev_e is not None else 0.0
        else:
            de = 0.0
        self._prev_e = e

        # torso_tilt = axis_tilt + s*J  (about the hinge, exact) => J = s*(kappa-1)*A
        # for the target, and a positive error e needs J to move by s*e.
        cmd = self.s * ((self.kappa - 1.0) * a_roll
                        + self.kp * e + self.ki * self._int + self.kd * de)
        cmd *= alpha                          # blend in with the gait transition
        sat = abs(cmd) > self.limit
        cmd = max(-self.limit, min(self.limit, cmd))

        if sat:                               # anti-windup
            self._int -= e * dt
        if t >= self.measure_after:           # gate stats to the measurement window
            self._roll_sq += t_roll * t_roll
            self._roll_n += 1
            self._sat_n += int(sat)
        return cmd
