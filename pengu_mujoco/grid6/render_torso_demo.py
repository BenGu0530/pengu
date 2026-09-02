"""render_torso_demo.py — the kappa PID and a phase-locked feedforward, side by side,
both carrying the servo lag the robot actually has.

Measured on the robot (pengu-A, -B, -10): the torso joint reaches its extreme 56 ms after
the hip axis reaches its own, by which time the axis is already returning in 76-90% of
events, and corr(dJ, d_axis) peaks at -0.94 at exactly -56 ms. The correction has the
right sign and the wrong timing, so it pushes the lower body the way it is already going.
With the loop open the same gait rolls 21 deg peak-to-peak; with it closed, 67.

Putting that 56 ms into the model reproduces it: torso_roll_rms goes 0.67 -> 4.57 and the
hip axis gets WORSE (5.13 -> 6.31 rms). A feedforward timed off the gait phase carries no
measurement at all, so the delay cannot enter; only the servo's own lag remains and that
is cancelled by a phase lead. Same lag, same gait: 0.59 rms, and the axis stays at 5.01.

Both robots are stepped in lockstep so the two halves of the frame are the same instant.

    GAIT=1.39/240/80/16/30 LAG=0.056 FF=1.0,7.55,194 MU=0.5 \
        python grid6/render_torso_demo.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PENGU_MODEL", "hardware_c1")

import imageio.v2 as imageio                    # noqa: E402
import mujoco                                   # noqa: E402
import gait_config as gc                        # noqa: E402
import gait_sweep as gs                         # noqa: E402
from torso_control import TorsoKappaPID         # noqa: E402
from friction_utils import set_floor_friction   # noqa: E402

MU = float(os.environ.get("MU", "0.5"))
LAG = float(os.environ.get("LAG", "0.056"))
WINDOW = float(os.environ.get("WINDOW", "12.0"))
SETTLE = 2.0
FPS = 30
OUT = "results/grid6_probes"
GAIT = tuple(float(x) for x in os.environ.get("GAIT", "1.39/240/80/16/30").split("/"))
FF = tuple(float(x) for x in os.environ.get("FF", "1.0,7.55,194").split(","))
CAM = (1.3, -10, -60)


class Arm:
    """One robot, one torso strategy, its own servo-lag queue."""

    def __init__(self, mode):
        self.mode = mode
        self.model = mujoco.MjModel.from_xml_path(gs.XML)
        self.data = mujoco.MjData(self.model)
        self.pid = TorsoKappaPID(self.model, kappa=0.0, measure_after=0.0)
        self.buf = []
        self.roll = []
        self.axis = []
        self.t0 = None
        self.fell = None
        self.root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = CAM

    def ctrl(self, d, t, alpha=1.0):
        if self.mode == "pid":
            u = self.pid(d, t, alpha)
        elif self.mode == "frozen":
            u = 0.0
        else:
            g, A, p = FF
            ph = 2 * math.pi * GAIT[0] * (t - gc.T_HOLD - gc.T_TRANSITION)
            u = alpha * g * math.radians(A) * math.sin(ph + math.radians(p))
        if LAG > 0:
            self.buf.append((t, u))
            while len(self.buf) > 1 and self.buf[1][0] <= t - LAG:
                self.buf.pop(0)
            return self.buf[0][1]
        return u


def main():
    freq, phi, leg, hip, off = GAIT
    arms = [Arm("pid"), Arm("ff")]
    label = {"pid": "kappa=0 PID", "ff": "feedforward", "frozen": "torso held"}

    # one shared gait configuration; each arm supplies only its own torso strategy
    set_floor_friction(arms[0].model, MU)
    for a in arms:
        set_floor_friction(a.model, MU)
    gs.FLOOR_MU = MU
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    ids = []
    for a in arms:
        act, jadr = gc.build_ids(a.model)
        gc.set_initial_pose(a.model, a.data, act, jadr)
        ids.append(act)

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    frames = []
    rens = [mujoco.Renderer(a.model, height=480, width=640) for a in arms]
    while True:
        if t0 is None:
            t = arms[0].data.time
            if (t >= gs.QUIET_MIN_T
                    and max(float(np.max(np.abs(a.data.qvel))) for a in arms) < gs.QUIET_QVEL) \
                    or t >= gs.QUIET_MAX_T:
                t0 = t
                gc.T_HOLD = t
        for a, act in zip(arms, ids):
            gc.TORSO_CONTROLLER = a.ctrl
            gc.apply_ctrl(a.data, act, a.data.time)
            mujoco.mj_step(a.model, a.data)
            if a.data.xpos[a.root][2] < 0.05 and a.fell is None:
                a.fell = a.data.time - (t0 or 0.0)
        if t0 is None:
            continue
        tw = arms[0].data.time - t0 - gc.T_TRANSITION - SETTLE
        if tw > WINDOW:
            break
        if tw >= 0:
            for a in arms:
                h = a.pid.hinge(a.data)
                a.roll.append(math.degrees(a.pid.torso_roll(a.data, h)))
                a.axis.append(math.degrees(a.pid.axis_roll(a.data, h)))
        if arms[0].data.time >= nxt:
            nxt += 1.0 / FPS
            pair = []
            for a, r in zip(arms, rens):
                a.cam.lookat[:] = a.data.xpos[a.root]
                r.update_scene(a.data, a.cam)
                pair.append(r.render().copy())
            frames.append(np.hstack(pair))
    for r in rens:
        r.close()
    gc.T_HOLD = 5.0

    os.makedirs(OUT, exist_ok=True)
    cell = "-".join(f"{x:g}" for x in GAIT)
    path = os.path.join(OUT, f"demo_torso_{cell}_lag{LAG*1000:.0f}ms.mp4")
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=1)
    print(f"gait {'/'.join(f'{x:g}' for x in GAIT)}   mu {MU}   servo lag {LAG*1000:.0f} ms"
          f"   feedforward gain {FF[0]:g} A {FF[1]:g} deg phi {FF[2]:g} deg")
    print(f"LEFT = {label[arms[0].mode]}   RIGHT = {label[arms[1].mode]}\n")
    for a in arms:
        print(f"  {label[a.mode]:14s} torso_roll_rms {np.std(a.roll):6.2f} deg"
              f"   axis_rms {np.std(a.axis):6.2f} deg"
              + (f"   FELL at t0+{a.fell:.1f}s" if a.fell else ""))
    print(f"\n{len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
