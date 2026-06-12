# Offscreen render of a chosen Pengu gait -> GIF + key PNG frames (for analysis).
# Run from pengu_mujoco/:
#   MUJOCO_GL=egl /home/ben/miniconda3/envs/mujoco/bin/python render_gait.py
"""render_gait.py - headless render of one gait config to GIF + frames."""
import os
import sys
import math
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gait_config as gc
from gait_config import build_ids, set_initial_pose

# ===== GAIT presets (override via argv[1]); TORSO_PHI = po_e =====
PRESETS = {
    "phasewin":  dict(LEG_AMP=110.0, HIP_AMP=12.0, TORSO_AMP=0.0, HIP_PHI=90.0,  TORSO_PHI=0.0,   FREQ=1.24),
    "legtorso":  dict(LEG_AMP=72.1,  HIP_AMP=0.0,  TORSO_AMP=7.3, HIP_PHI=0.0,   TORSO_PHI=283.4, FREQ=1.51),
    "full":      dict(LEG_AMP=100.9, HIP_AMP=21.7, TORSO_AMP=2.9, HIP_PHI=202.7, TORSO_PHI=66.5,  FREQ=1.69),
}
_preset = sys.argv[1] if len(sys.argv) > 1 else "full"
_p = PRESETS[_preset]
LEG_AMP   = _p["LEG_AMP"]
HIP_AMP   = _p["HIP_AMP"]
TORSO_AMP = _p["TORSO_AMP"]
HIP_PHI   = _p["HIP_PHI"]
TORSO_PHI = _p["TORSO_PHI"]
FREQ      = _p["FREQ"]
TAG       = f"opt_{_preset}"

T_START = 5.0       # begin recording (end of stand hold)
T_END   = 13.0
FPS     = 30
W, H    = 640, 480


def main():
    gc.set_crank_amp(LEG_AMP); gc.set_hip_amp(HIP_AMP); gc.set_torso_amp(TORSO_AMP)
    gc.set_walk_freq(FREQ)
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_E_DEG = TORSO_PHI
    gc.PHASE_OFFSET_C_DEG = HIP_PHI; gc.PHASE_OFFSET_D_DEG = HIP_PHI

    model = mujoco.MjModel.from_xml_path("penguV2/scene.xml")
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)
    set_initial_pose(model, data, aid, jadr)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "leftthighmotor")

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.azimuth = 120.0
    cam.elevation = -18.0
    cam.distance = 1.4

    dt = model.opt.timestep
    render_every = max(1, round((1.0 / FPS) / dt))
    frames = []
    step = 0
    y0 = data.xpos[root][1]
    while data.time < T_END:
        gc.apply_ctrl(data, aid, data.time)
        mujoco.mj_step(model, data)
        step += 1
        if data.time >= T_START and step % render_every == 0:
            cam.lookat[:] = data.xpos[root]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    fwd = data.xpos[root][1] - y0
    outdir = "results"
    os.makedirs(outdir, exist_ok=True)
    gif = os.path.join(outdir, f"{TAG}.gif")
    imageio.mimsave(gif, frames, fps=FPS)
    # 6 evenly spaced key frames (one ~gait cycle spread)
    n = len(frames)
    idxs = np.linspace(0, n - 1, 6).astype(int)
    paths = []
    for k, i in enumerate(idxs):
        p = os.path.join(outdir, f"{TAG}_frame{k}.png")
        imageio.imwrite(p, frames[i])
        paths.append(p)
    print(f"frames={n}  fwd_dist over window={fwd:+.3f} m")
    print(f"gif: {gif}")
    for p in paths:
        print(f"frame: {p}")


if __name__ == "__main__":
    main()
