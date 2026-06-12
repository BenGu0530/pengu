"""
penguin_metrics.py - measure a learned policy against the REAL king-penguin
gait signature. This is the "ruler" for the bio-inspired science story: it turns
a rollout into the same accelerometer-level numbers biologists report, plus the
Griffin & Kram (2000) inverted-pendulum mechanical-energy-recovery metric.

Computed (steady-state window, transient skipped):
  - stride frequency   [Hz]   (CPG freq + cross-check via footfall cadence/2)
  - waddle roll amplitude [deg] (frontal/lateral body rock)
  - sagittal lean amplitude [deg]
  - forward path speed [m/s]
  - COM external mechanical energy recovery R [%]  (pendular exchange)
  - lateral KE fraction of total KE [%]

Reference (real king penguin @ ~1.4 km/h; Willener 2015/2016, Griffin&Kram 2000):
  stride freq 1.27 Hz | roll amp ~8 deg | lean amp ~2 deg |
  energy recovery up to ~80% | lateral KE ~30% of KE fluctuation.

Run from pengu_mujoco/:
  python rl/penguin_metrics.py [model.zip] [kind] [vx_cmd] [bio0|bio1]
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
from rl.pengu_env import (PenguCPGEnv, PENGUIN_FREQ, PENGUIN_ROLL_DEG,
                          PENGUIN_LEAN_DEG)

mpath = sys.argv[1] if len(sys.argv) > 1 else "rl/runs/ppo_curriculum_prismatic.zip"
kind = sys.argv[2] if len(sys.argv) > 2 else "prismatic"
vx_cmd = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
bio = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False

GRAV = 9.81
SKIP_S = 2.0          # drop the start-up transient before measuring

model = PPO.load(mpath, device="cpu")
env = PenguCPGEnv(domain_rand=False, model_kind=kind, bio_imitate=bio)
env.set_cmd_range(vx_cmd, vx_cmd)
m, d = env.model, env.data
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
foot_bodies = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080"): "R",
               mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_foot0080___fillet13"): "L"}

# whole-robot mass + mass-weighted COM (skip world body 0)
bmass = m.body_mass.copy()
M = float(bmass[1:].sum())


def robot_com():
    return (bmass[1:, None] * d.xipos[1:]).sum(0) / M


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
dt = env.control_dt
T, ROLL, PITCH, FCPG = [], [], [], []
COM, FFALL = [], []                # COM xyz per step; footfall times
prev_contact = {"L": False, "R": False}
t = 0.0
for k in range(env.max_steps):
    a, _ = model.predict(o, deterministic=True)
    o, r, term, trunc, info = env.step(a)
    t += dt
    T.append(t)
    ROLL.append(info["roll"]); PITCH.append(info["pitch"]); FCPG.append(info["f_cpg"])
    COM.append(robot_com().copy())
    con = feet_in_contact()
    for f in ("L", "R"):
        if con[f] and not prev_contact[f]:
            FFALL.append(t)
    prev_contact = con
    if term or trunc:
        break

T = np.array(T); ROLL = np.array(ROLL); PITCH = np.array(PITCH)
FCPG = np.array(FCPG); COM = np.array(COM)
survived = (k + 1) >= env.max_steps
sel = T >= SKIP_S                    # steady-state mask
if sel.sum() < 20:
    sel = np.ones_like(T, dtype=bool)

# --- kinematic signature (accelerometer-equivalent) ---
roll_amp = math.degrees(0.5 * (ROLL[sel].max() - ROLL[sel].min()))
pitch_amp = math.degrees(0.5 * (PITCH[sel].max() - PITCH[sel].min()))
f_cpg = float(np.mean(FCPG[sel]))
ff = np.array([x for x in FFALL if x >= SKIP_S])
cad = (len(ff) / (T[sel][-1] - T[sel][0])) if len(ff) > 1 else float("nan")
f_foot = cad / 2.0                   # 2 footfalls per stride
path = float(np.sum(np.linalg.norm(np.diff(COM[sel, :2], axis=0), axis=1)))
path_speed = path / (T[sel][-1] - T[sel][0])

# --- Griffin & Kram external mechanical energy recovery ---
v = np.gradient(COM[sel], dt, axis=0)            # COM velocity (m/s), world xyz
Ek_x = 0.5 * M * v[:, 0] ** 2                    # lateral KE
Ek_y = 0.5 * M * v[:, 1] ** 2                    # forward KE
Ek_z = 0.5 * M * v[:, 2] ** 2                    # vertical KE
Ek = Ek_x + Ek_y + Ek_z
Ep = M * GRAV * COM[sel, 2]                       # gravitational PE
Etot = Ek + Ep


def pos_work(E):                                  # sum of positive increments
    dE = np.diff(E)
    return float(dE[dE > 0].sum())


Wk, Wp, Wext = pos_work(Ek), pos_work(Ep), pos_work(Etot)
recovery = 100.0 * (Wk + Wp - Wext) / (Wk + Wp) if (Wk + Wp) > 0 else float("nan")
# lateral share of KE FLUCTUATION (de-meaned, more meaningful than raw mean)
def fluc(x):
    return float(np.mean(np.abs(x - x.mean())))
lat_frac = 100.0 * fluc(Ek_x) / (fluc(Ek_x) + fluc(Ek_y) + fluc(Ek_z) + 1e-12)

# --- report: ours vs real penguin ---
print(f"\n=== PENGUIN GAIT METRICS  model={os.path.basename(mpath)} bio={bio} ===")
print(f"survived={survived}  steady window={T[sel][0]:.1f}-{T[sel][-1]:.1f}s  M={M:.2f} kg\n")
rows = [
    ("stride frequency [Hz]",   f"{f_cpg:.2f} (foot {f_foot:.2f})", f"{PENGUIN_FREQ:.2f}"),
    ("waddle roll amp [deg]",   f"{roll_amp:.1f}",                  f"~{PENGUIN_ROLL_DEG:.0f}"),
    ("sagittal lean amp [deg]", f"{pitch_amp:.1f}",                 f"~{PENGUIN_LEAN_DEG:.0f}"),
    ("forward speed [m/s]",     f"{path_speed:.3f}",                "0.26-0.39"),
    ("energy recovery [%]",     f"{recovery:.0f}",                  "up to ~80"),
    ("lateral KE fraction [%]", f"{lat_frac:.0f}",                  "~30"),
]
print(f"{'metric':<26}{'OURS':>18}{'PENGUIN':>14}")
print("-" * 58)
for name, ours, peng in rows:
    print(f"{name:<26}{ours:>18}{peng:>14}")

# --- figure: rock + energy exchange ---
out = os.path.join(os.path.dirname(__file__), "runs")
fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(T, np.degrees(ROLL), label="roll (lateral rock)")
ax[0].plot(T, np.degrees(PITCH), label="pitch (lean)", alpha=.7)
ax[0].axhline(PENGUIN_ROLL_DEG, ls="--", c="r", lw=.8); ax[0].axhline(-PENGUIN_ROLL_DEG, ls="--", c="r", lw=.8)
ax[0].set_ylabel("deg"); ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
ax[0].set_title(f"{os.path.basename(mpath)}  f={f_cpg:.2f}Hz roll={roll_amp:.1f}deg recovery={recovery:.0f}%")
ax[1].plot(T[sel], COM[sel, 0] - COM[sel, 0].mean(), label="COM lateral x")
ax[1].plot(T[sel], COM[sel, 2] - COM[sel, 2].mean(), label="COM height z")
ax[1].set_ylabel("m"); ax[1].legend(loc="upper right"); ax[1].grid(alpha=.3)
ax[2].plot(T[sel], Ek - Ek.mean(), label="KE (de-meaned)")
ax[2].plot(T[sel], Ep - Ep.mean(), label="PE (de-meaned)")
ax[2].set_ylabel("J"); ax[2].set_xlabel("t [s]"); ax[2].legend(loc="upper right"); ax[2].grid(alpha=.3)
png = os.path.join(out, f"penguin_metrics_{'bio' if bio else 'base'}.png")
plt.tight_layout(); plt.savefig(png, dpi=120, bbox_inches="tight")
print(f"\nwrote {png}")
