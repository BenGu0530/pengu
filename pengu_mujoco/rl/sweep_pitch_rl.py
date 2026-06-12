"""
sweep_pitch_rl.py - sweep the spawn PITCH (net forward lean) for the learned
CPG-RL policy and measure how it walks at each. net_lean in [0,30] deg:
  0  = upright   (INIT_PITCH = -30 deg)
  30 = max forward lean (INIT_PITCH = 0 deg, the CAD neutral)
Policy was trained at net_lean=0 (upright), so this also tests robustness.

Run from pengu_mujoco/:
  python rl/sweep_pitch_rl.py [model.zip] [kind]
"""
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3 import PPO
from rl.pengu_env import PenguCPGEnv

mpath = sys.argv[1] if len(sys.argv) > 1 else "rl/runs/ppo_curriculum_prismatic.zip"
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
VX_CMD = 0.12
NET_LEAN = np.round(np.arange(0.0, 60.001, 0.5), 2)   # forward lean 0..60 (INIT_PITCH -30..+30)
SEEDS = [0, 1, 2]

model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, model_kind=kind)
env.set_cmd_range(VX_CMD, VX_CMD)

rows = []
for nl in NET_LEAN:
    env.init_pitch = math.radians(-30.0 + nl)
    surv, dists, speeds, rolls = [], [], [], []
    for s in SEEDS:
        o, _ = env.reset(seed=s)
        y0 = env.data.xpos[env.root][1]
        vlast, rs, k = [], [], 0
        while True:
            a, _ = model.predict(o, deterministic=True)
            o, r, t, tr, info = env.step(a)
            vlast.append(info["vx"]); rs.append(abs(info["roll"])); k += 1
            if t or tr:
                break
        surv.append(1.0 if k >= env.max_steps else 0.0)
        dists.append(env.data.xpos[env.root][1] - y0)
        speeds.append(float(np.mean(vlast[-50:])) if vlast else 0.0)
        rolls.append(math.degrees(float(np.mean(rs))) if rs else float("nan"))
    rows.append((nl, np.mean(surv), np.mean(dists), np.mean(speeds), np.nanmean(rolls)))
    print(f"net_lean={nl:4.1f} (INIT_PITCH={-30+nl:+5.1f})  surv={np.mean(surv):.2f}  "
          f"dist={np.mean(dists):+.2f}  speed={np.mean(speeds):.3f}  roll={np.nanmean(rolls):.1f}")

rows = np.array(rows)
out = os.path.join(os.path.dirname(__file__), "runs")
np.savetxt(os.path.join(out, "pitch_sweep_rl.csv"), rows, delimiter=",",
           header="net_lean_deg,survival,dist_fwd_m,speed_mps,roll_deg", comments="")

fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(rows[:, 0], rows[:, 1], "-o", ms=3); ax[0].set_ylabel("survival"); ax[0].grid(alpha=.3)
ax[0].axvline(0, color="g", ls="--", lw=1, label="trained pitch (upright)"); ax[0].legend()
ax[1].plot(rows[:, 0], rows[:, 3], "-o", ms=3, color="tab:orange"); ax[1].set_ylabel("speed [m/s]"); ax[1].grid(alpha=.3)
ax[1].axhline(VX_CMD, color="gray", ls=":", label=f"cmd {VX_CMD}"); ax[1].legend()
ax[2].plot(rows[:, 0], rows[:, 4], "-o", ms=3, color="tab:red"); ax[2].set_ylabel("roll [deg]")
ax[2].set_xlabel("net forward lean [deg]  (0=upright, 30=max forward)"); ax[2].grid(alpha=.3)
fig.suptitle("RL policy vs spawn pitch (net forward lean)")
plt.tight_layout()
png = os.path.join(out, "pitch_sweep_rl.png")
plt.savefig(png, dpi=120, bbox_inches="tight")
print("wrote", png)
