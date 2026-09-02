"""render_gait_demo.py — mp4 of one specific gait, under the grid5 staged protocol.

Renders exactly what walk_prediction.py measures, so the video and the numbers describe
the same rollout: same staged start (quiescence hold, rest lean, hip_off ramp), same COM
variant, same TorsoKappaPID, same mu. Two views side by side, the camera tracking the
robot.

    GAITS="k2:2.0,1.31,1.06,220,105,28,10; k0:0.0,1.31,1.06,220,105,28,10" \
    MU=0.3 python grid5/render_gait_demo.py

Writes results/grid5_probes/demo_<label>_mu<..>.mp4
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PENGU_MODEL", "hardware_c1")   # GRID-6 default: the as-built robot

import imageio.v2 as imageio                    # noqa: E402
import mujoco                                   # noqa: E402
import gait_config as gc                        # noqa: E402
import gait_sweep as gs                         # noqa: E402
import grid6_sweep as g6                        # noqa: E402
from torso_control import TorsoKappaPID         # noqa: E402

MU = float(os.environ.get("MU", "0.3"))
WINDOW = float(os.environ.get("WINDOW", "12.0"))    # seconds of walking after the settle
SETTLE = 6.0
FPS = 30
CAMS = [(1.4, -10, 0), (1.1, -12, -90)]             # (distance, elevation, azimuth)
OUT = "results/grid6_probes"

DEFAULT = ("k2:2.0,1.31,1.06,220,105,28,10; "
           "k0:0.0,1.31,1.06,220,105,28,10")


def parse():
    out = {}
    for item in os.environ.get("GAITS", DEFAULT).split(";"):
        item = item.strip()
        if not item:
            continue
        lab, nums = item.split(":")
        out[lab.strip()] = tuple(float(x) for x in nums.split(","))
    return out


def render(label, kappa, com, freq, phi, leg, hip, off):
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    _lean = gc.STAND_HIP_DEG
    gc.STAND_HIP_DEG = 0.0
    if gc._MODEL.endswith("cad"):
        pass                      # the as-built model's COM ratio is geometry, not a slide
    else:
        g6.apply_com_variant(model, com)
    pid = TorsoKappaPID(model, kappa=kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = _lean
    from friction_utils import set_floor_friction
    set_floor_friction(model, MU)
    gs.FLOOR_MU = MU
    gs.CONDITION["hip_off"] = off
    gs._set_gait(dict(freq=freq, hip_phi=phi, leg_amp=leg, hip_amp=hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)

    cams = []
    for dist, elev, az in CAMS:
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.distance, c.elevation, c.azimuth = dist, elev, az
        cams.append(c)

    gc.T_HOLD = 1e9
    t0 = None
    nxt = 0.0
    frames = []
    fell_at = None
    with mujoco.Renderer(model, height=480, width=640) as ren:
        while True:
            if t0 is None:
                t = data.time
                if (t >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                        or t >= gs.QUIET_MAX_T:
                    t0 = t
                    gc.T_HOLD = t
            gc.apply_ctrl(data, act, data.time)
            mujoco.mj_step(model, data)
            if data.xpos[root][2] < 0.05 and fell_at is None:
                fell_at = data.time - (t0 or 0.0)
            if t0 is not None and data.time > t0 + SETTLE + WINDOW:
                break
            if data.time >= nxt:
                nxt += 1.0 / FPS
                pair = []
                for c in cams:
                    c.lookat[:] = data.xpos[root]
                    ren.update_scene(data, c)
                    pair.append(ren.render().copy())
                frames.append(np.hstack(pair))
    gc.T_HOLD = 5.0

    os.makedirs(OUT, exist_ok=True)
    tag = f"{MU:.1f}".replace("0.", "0")
    path = os.path.join(OUT, f"demo_{label}_mu{tag}.mp4")
    imageio.mimsave(path, frames, fps=FPS, macro_block_size=1)
    print(f"  {label}: {len(frames)} frames -> {path}"
          f"   torso roll rms {pid.roll_rms():.1f} deg"
          f"   ctrl saturation {100 * pid.saturation_frac():.1f}%"
          + (f"   FELL at t0+{fell_at:.1f}s" if fell_at else ""))


def main():
    gaits = parse()
    print(f"=== rendering {len(gaits)} gait(s) at mu={MU}, "
          f"{SETTLE:.0f}s settle + {WINDOW:.0f}s walk ===")
    for lab, cfg in gaits.items():
        print(f"--- {lab}: kappa={cfg[0]} com={cfg[1]} "
              f"gait {cfg[2]}/{cfg[3]:.0f}/{cfg[4]:.0f}/{cfg[5]:.0f}/{cfg[6]:.0f}")
        render(lab, *cfg)


if __name__ == "__main__":
    main()
