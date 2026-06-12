"""Render the bio penguin policy from several camera angles in ONE rollout.

Produces, in rl/runs/:
  walk_front.gif/.mp4   - head-on (the lateral waddle rock is most visible)
  walk_3q.gif/.mp4      - 3/4 view
  walk_orbit.gif/.mp4   - camera slowly orbits the walking robot (all angles)

Run from pengu_mujoco/:
  MUJOCO_GL=egl python rl/render_views.py [model.zip] [kind] [bio0|bio1] [vx] [seed]
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from rl.pengu_env import PenguCPGEnv

mpath = sys.argv[1] if len(sys.argv) > 1 else "rl/runs/ppo_penguin.zip"
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
bio = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
vx = float(sys.argv[4]) if len(sys.argv) > 4 else 0.05
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
W, H = 640, 480

model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, model_kind=kind, bio_imitate=bio)
env.set_cmd_range(vx, vx)
m, d = env.model, env.data
R = mujoco.Renderer(m, height=H, width=W)

# camera presets; "orbit" azimuth is overwritten per-frame below
VIEWS = {
    "walk_front": dict(azimuth=90,  elevation=-8,  distance=1.6),
    "walk_3q":    dict(azimuth=135, elevation=-15, distance=1.8),
    "walk_orbit": dict(azimuth=90,  elevation=-12, distance=1.9),
}
cams = {}
for name, p in VIEWS.items():
    c = mujoco.MjvCamera()
    c.azimuth = p["azimuth"]; c.elevation = p["elevation"]; c.distance = p["distance"]
    cams[name] = c
buffers = {name: [] for name in VIEWS}

o, _ = env.reset(seed=seed)
N = env.max_steps
for k in range(N):
    a, _ = model.predict(o, deterministic=True)
    o, r, t, tr, info = env.step(a)
    cams["walk_orbit"].azimuth = 90 + 360.0 * (k / N)   # one full revolution
    for name, c in cams.items():
        c.lookat[:] = d.xpos[env.root]
        R.update_scene(d, camera=c)
        buffers[name].append(R.render().copy())
    if t or tr:
        break

outdir = os.path.join(os.path.dirname(__file__), "runs")
for name, frames in buffers.items():
    imageio.mimsave(os.path.join(outdir, name + ".mp4"), frames, fps=25, quality=8)
    imageio.mimsave(os.path.join(outdir, name + ".gif"), frames[::3], fps=18)
    print(f"wrote rl/runs/{name}.mp4 / .gif  ({len(frames)} frames)")
