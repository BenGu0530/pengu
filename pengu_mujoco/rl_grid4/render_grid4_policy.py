"""Render a trained grid4 RL policy to mp4 (side + 3/4 oblique dolly cams).

Usage (from pengu_mujoco/):
  python rl/render_grid4_policy.py rl/runs/grid4/e2_s0/final.zip [--mu 0.1]
      [--dur 10] [--out demo.mp4] [--fps 50]
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import mujoco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import imageio
    from stable_baselines3 import PPO
    from grid4_rl_env import Grid4RLEnv

    model = PPO.load(a.ckpt, device="cpu")
    env = Grid4RLEnv(eval_mode=True, mu_fixed=a.mu, episode_s=a.dur, seed=a.seed)
    obs, _ = env.reset(seed=a.seed)

    W, H = 640, 480
    ren = mujoco.Renderer(env.model, height=H, width=W)
    cams = []
    for az in (270.0, 315.0):
        c = mujoco.MjvCamera()
        c.azimuth, c.elevation, c.distance = az, -15.0, 0.9
        cams.append(c)

    base = os.path.dirname(os.path.abspath(a.ckpt))
    if os.path.basename(base) == "ckpts":               # run/ckpts/x.zip -> run/videos/
        base = os.path.join(os.path.dirname(base), "videos")
    os.makedirs(base, exist_ok=True)
    out = a.out or os.path.join(
        base, f"demo_mu{a.mu:g}_{os.path.splitext(os.path.basename(a.ckpt))[0]}.mp4")
    frames = []
    n_steps = int(a.dur / env.control_dt)
    per_frame = max(1, int(round(1.0 / (a.fps * env.control_dt))))
    for i in range(n_steps):
        act, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _ = env.step(act)
        if i % per_frame == 0:
            row = []
            for c in cams:
                c.lookat[:] = env.data.xpos[env.root]
                ren.update_scene(env.data, camera=c)
                row.append(ren.render())
            frames.append(np.concatenate(row, axis=1))
        if term or trunc:
            if term:  # hold the fall frame briefly so it reads in the video
                frames += [frames[-1]] * (a.fps // 2)
            break
    imageio.mimsave(out, frames, fps=a.fps, macro_block_size=1)
    print(f"wrote {out} ({len(frames)} frames, mu={env.mu:.3f})")


if __name__ == "__main__":
    main()
