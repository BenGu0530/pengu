"""
stepping_test.py - PER-STEP gait measurement for a learned policy.

The robot walks AND drifts/turns, so straight-line (end-start) distance is wrong.
This detects each FOOTFALL (foot touchdown on the floor), records the foot-plant
xy ("stepping point"), and measures:
  - step length  = distance between consecutive footfalls (L<->R)
  - stride length= distance between successive SAME-foot footfalls
  - cadence      = footfalls / sec
  - path length  = integral of |body velocity| (true distance walked along the path)
  - straight-line displacement (for contrast)
  - net heading change (turning)
  - speeds derived from path length and from stride*cadence
Also plots the footprint trail + body path (top-down).

Run from pengu_mujoco/:
  MUJOCO_GL not needed.  python rl/stepping_test.py [model.zip] [kind] [vx_cmd]
"""
import os
import sys
import math
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from rl.pengu_env import PenguCPGEnv

mpath = sys.argv[1] if len(sys.argv) > 1 else "rl/runs/ppo_curriculum_prismatic.zip"
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
vx_cmd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.12

model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, model_kind=kind)
env.set_cmd_range(vx_cmd, vx_cmd)
m, d = env.model, env.data
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
foot_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080"): "R",
               mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13"): "L"}


def feet_in_contact():
    out = {"L": False, "R": False}
    for k in range(d.ncon):
        c = d.contact[k]
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            b = m.geom_bodyid[other]
            if b in foot_bodies:
                out[foot_bodies[b]] = True
    return out


o, _ = env.reset(seed=0)
prev_contact = {"L": True, "R": True}
footfalls = []         # (t, foot, x, y)
path_len = 0.0
prev_xy = d.xpos[env.root][:2].copy()
start_xy = prev_xy.copy()
R0 = d.xmat[env.root].reshape(3, 3).copy()
t = 0.0
for k in range(env.max_steps):
    a, _ = model.predict(o, deterministic=True)
    o, r, term, trunc, info = env.step(a)
    t += env.control_dt
    xy = d.xpos[env.root][:2].copy()
    path_len += float(np.linalg.norm(xy - prev_xy))
    prev_xy = xy
    con = feet_in_contact()
    for f in ("L", "R"):
        if con[f] and not prev_contact[f]:    # touchdown
            fb = [bid for bid, n in foot_bodies.items() if n == f][0]
            fx, fy = d.xpos[fb][:2]
            footfalls.append((t, f, float(fx), float(fy)))
    prev_contact = con
    if term or trunc:
        break

ff = footfalls
dur = t
# step lengths (consecutive footfalls), stride (same foot)
step_lens = [math.hypot(ff[i][2] - ff[i - 1][2], ff[i][3] - ff[i - 1][3]) for i in range(1, len(ff))]
stride_lens = [math.hypot(ff[i][2] - ff[i - 2][2], ff[i][3] - ff[i - 2][3]) for i in range(2, len(ff))]
n_steps = len(ff)
cadence = n_steps / dur if dur > 0 else 0.0
straight = float(np.linalg.norm(d.xpos[env.root][:2] - start_xy))
# net heading change
Rf = d.xmat[env.root].reshape(3, 3)
rel = R0.T @ Rf
yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def stats(a):
    return (np.mean(a), np.std(a), np.min(a), np.max(a)) if a else (float("nan"),) * 4


sm = stats(step_lens); st = stats(stride_lens)
print(f"model={mpath} cmd={vx_cmd}  duration={dur:.1f}s  survived={k+1>=env.max_steps}")
print(f"footfalls={n_steps}  cadence={cadence:.2f} steps/s")
print(f"step length  : mean={sm[0]*1000:.0f} mm  std={sm[1]*1000:.0f}  range[{sm[2]*1000:.0f},{sm[3]*1000:.0f}] mm")
print(f"stride length: mean={st[0]*1000:.0f} mm  std={st[1]*1000:.0f}  range[{st[2]*1000:.0f},{st[3]*1000:.0f}] mm")
print(f"path length (true)      = {path_len:.3f} m  -> path speed   = {path_len/dur:.3f} m/s")
print(f"straight-line disp      = {straight:.3f} m  -> straight speed= {straight/dur:.3f} m/s")
print(f"net heading change(turn)= {yaw:+.1f} deg   (curvature -> straight underestimates path)")
print(f"speed via stride*cadence= {sm[0]*cadence:.3f} m/s")

# plot footprints + path
out = os.path.join(os.path.dirname(__file__), "runs")
plt.figure(figsize=(7, 7))
L = [(x, y) for (_, f, x, y) in ff if f == "L"]
Rp = [(x, y) for (_, f, x, y) in ff if f == "R"]
if L:
    L = np.array(L); plt.plot(L[:, 0], L[:, 1], "o-", color="tab:red", ms=7, label="L footfalls", alpha=.7)
if Rp:
    Rp = np.array(Rp); plt.plot(Rp[:, 0], Rp[:, 1], "s-", color="tab:blue", ms=7, label="R footfalls", alpha=.7)
plt.plot(start_xy[0], start_xy[1], "k*", ms=15, label="start")
plt.gca().set_aspect("equal"); plt.grid(alpha=.3); plt.legend()
plt.xlabel("x (lateral) [m]"); plt.ylabel("y (forward) [m]")
plt.title(f"footprint trail  (step {sm[0]*1000:.0f}mm, cadence {cadence:.1f}/s, path {path_len:.2f}m)")
png = os.path.join(out, "stepping_footprints.png")
plt.savefig(png, dpi=120, bbox_inches="tight")
print("wrote", png)
