"""Render the latched-CPG policy (side view) with a live top-down footstep /
trajectory inset in the bottom-right corner.

Run from pengu_mujoco/:
  MUJOCO_GL=egl python rl/render_with_trajectory.py [model.zip] [kind] [seed]
"""
import os
import sys
import numpy as np
import mujoco
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from rl.pengu_env import PenguCPGEnv

mpath = sys.argv[1] if len(sys.argv) > 1 else "rl/runs/ppo_prismatic_latched.zip"
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
bio = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
vx = float(sys.argv[5]) if len(sys.argv) > 5 else (0.05 if bio else 0.12)
az = float(sys.argv[6]) if len(sys.argv) > 6 else 180.0      # camera azimuth
el = float(sys.argv[7]) if len(sys.argv) > 7 else -10.0      # camera elevation
dist = float(sys.argv[8]) if len(sys.argv) > 8 else 1.7      # camera distance
tag = sys.argv[9] if len(sys.argv) > 9 else "latched_side"   # output filename stem
W, H = 640, 480
IW, IH = 250, 210                       # inset size (px)

model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, model_kind=kind, bio_imitate=bio)
env.set_cmd_range(vx, vx)
m, d = env.model, env.data

# ---- pass 1: roll out, store mujoco frames + body path + footfalls ----
R = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera(); cam.azimuth = az; cam.elevation = el; cam.distance = dist
o, _ = env.reset(seed=seed)
frames, path, falls = [], [], []          # falls: (frame_idx, foot, x, y)
prevc = {b: True for b in env.foot_bids}
for k in range(env.max_steps):
    a, _ = model.predict(o, deterministic=True)
    o, r, t, tr, info = env.step(a)
    path.append(d.xpos[env.root][:2].copy())
    con = env._foot_contacts()
    for b, name in env.foot_bids.items():
        if con[b] and not prevc[b]:
            falls.append((len(frames), name, float(d.xpos[b][0]), float(d.xpos[b][1])))
    prevc = con
    cam.lookat[:] = d.xpos[env.root]
    R.update_scene(d, camera=cam)
    frames.append(R.render().copy())
    if t or tr:
        break
path = np.array(path)

# ---- fixed plot limits (whole trajectory + margin), equal aspect ----
allx = np.concatenate([path[:, 0], [f[2] for f in falls] or [0]])
ally = np.concatenate([path[:, 1], [f[3] for f in falls] or [0]])
cx, cy = (allx.min() + allx.max()) / 2, (ally.min() + ally.max()) / 2
half = max(allx.max() - allx.min(), ally.max() - ally.min(), 0.2) / 2 + 0.05

fig = plt.figure(figsize=(IW / 100, IH / 100), dpi=100)
ax = fig.add_axes([0.18, 0.16, 0.80, 0.74])

def inset(i):
    ax.clear()
    ax.plot(path[:i + 1, 0], path[:i + 1, 1], "-", color="0.5", lw=1.0, label="body path")
    Lx = [f[2] for f in falls if f[0] <= i and f[1] == "L"]; Ly = [f[3] for f in falls if f[0] <= i and f[1] == "L"]
    Rx = [f[2] for f in falls if f[0] <= i and f[1] == "R"]; Ry = [f[3] for f in falls if f[0] <= i and f[1] == "R"]
    ax.plot(Lx, Ly, "o", color="tab:red", ms=4, label="L step")
    ax.plot(Rx, Ry, "s", color="tab:blue", ms=4, label="R step")
    ax.plot(path[i, 0], path[i, 1], "*", color="k", ms=10)        # current
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5, pad=1)
    ax.set_xlabel("x lat [m]", fontsize=6, labelpad=1); ax.set_ylabel("y fwd [m]", fontsize=6, labelpad=1)
    ax.set_title("footsteps (top-down)", fontsize=7, pad=2)
    ax.legend(fontsize=5, loc="upper left", framealpha=0.7, handletextpad=0.2, borderpad=0.2)
    ax.grid(alpha=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return buf

# ---- pass 2: composite inset into bottom-right ----
out = []
for i, fr in enumerate(frames):
    ins = inset(i)
    ih, iw = ins.shape[:2]
    comp = fr.copy()
    y0, x0 = H - ih - 4, W - iw - 4
    comp[y0 - 2:y0 + ih + 2, x0 - 2:x0 + iw + 2] = 0          # black border
    comp[y0:y0 + ih, x0:x0 + iw] = ins
    out.append(comp)

outdir = os.path.join(os.path.dirname(__file__), "runs")
imageio.mimsave(os.path.join(outdir, f"{tag}_traj.mp4"), out, fps=25, quality=8)
imageio.mimsave(os.path.join(outdir, f"{tag}_traj.gif"), out[::3], fps=18)
print(f"steps={len(frames)} footfalls={len(falls)}  wrote rl/runs/{tag}_traj.mp4 / .gif")
