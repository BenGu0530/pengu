"""Render a trained CPG-RL policy to GIF + frames. Run from pengu_mujoco/:
   MUJOCO_GL=egl python rl/render_policy.py [model.zip]"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from rl.pengu_env import PenguCPGEnv

mpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "runs", "ppo_smoke.zip")
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
bio = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
gif_name = sys.argv[4] if len(sys.argv) > 4 else "ppo_smoke"
model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, seed=7, model_kind=kind, bio_imitate=bio)
env.set_cmd_range(0.05 if bio else 0.12, 0.05 if bio else 0.12)
o, _ = env.reset()
m, d = env.model, env.data
renderer = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); cam.azimuth = 120; cam.elevation = -18; cam.distance = 1.4
frames = []
y0 = d.xpos[env.root][1]
for k in range(env.max_steps):
    a, _ = model.predict(o, deterministic=True)
    o, r, term, trunc, info = env.step(a)
    cam.lookat[:] = d.xpos[env.root]
    renderer.update_scene(d, camera=cam)
    frames.append(renderer.render().copy())
    if term or trunc:
        break
dist = d.xpos[env.root][1] - y0
out = os.path.join(os.path.dirname(__file__), "runs")
imageio.mimsave(os.path.join(out, f"{gif_name}.gif"), frames[::2], fps=25)
for i, fi in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
    imageio.imwrite(os.path.join(out, f"{gif_name}_frame{i}.png"), frames[fi])
print(f"steps={len(frames)} fwd_dist={dist:+.3f} m  gif=rl/runs/{gif_name}.gif")
