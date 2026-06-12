"""
train_curriculum.py - longer CPG-RL PPO run with a vx-command curriculum.

Curriculum widens the per-episode forward-speed command as training progresses:
  p<0.15: 0.15 fixed | 0.15-0.40: [0.10,0.20] | 0.40-0.70: [0.05,0.28] | >0.70: [0.05,0.35]

Periodically evaluates (deterministic, no DR) at fixed commands and logs
survival / speed-tracking / roll to a CSV so we can see clean-gait progress.

CPU only. Run from pengu_mujoco/:
  python rl/train_curriculum.py [total_timesteps] [n_envs]
"""
import os
import sys
import csv
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from rl.pengu_env import PenguCPGEnv

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(OUT, exist_ok=True)


def _mk(seed, kind):
    def f():
        return Monitor(PenguCPGEnv(domain_rand=True, seed=seed, model_kind=kind))
    return f


def curriculum_range(p):
    # push speed now that the gait (waddle+lean) is allowed
    if p < 0.20:
        return (0.10, 0.10)
    if p < 0.50:
        return (0.08, 0.18)
    if p < 0.80:
        return (0.08, 0.25)
    return (0.05, 0.30)


def eval_policy(model, cmds=(0.10, 0.18, 0.25), n_ep=3, kind="prismatic"):
    """Deterministic eval at fixed commands. Returns aggregate dict."""
    env = PenguCPGEnv(domain_rand=False, seed=12345, model_kind=kind)
    surv, dists, terrs, rolls, speeds = [], [], [], [], []
    for c in cmds:
        env.set_cmd_range(c, c)
        for _ in range(n_ep):
            o, _ = env.reset()
            y0 = env.data.xpos[env.root][1]
            k = 0; rs = []; ros = []
            while True:
                a, _ = model.predict(o, deterministic=True)
                o, r, term, trunc, info = env.step(a)
                rs.append(info["vx"]); ros.append(abs(info["roll"]))
                k += 1
                if term or trunc:
                    break
            surv.append(1.0 if k >= env.max_steps else 0.0)
            dists.append(env.data.xpos[env.root][1] - y0)
            mv = float(np.mean(rs[-50:])) if rs else 0.0
            speeds.append(mv)
            terrs.append(abs(mv - c))
            rolls.append(math.degrees(float(np.mean(ros))) if ros else float("nan"))
    return dict(surv=float(np.mean(surv)), dist=float(np.mean(dists)),
                speed=float(np.mean(speeds)), track_err=float(np.mean(terrs)),
                roll=float(np.nanmean(rolls)))


class CurriculumEval(BaseCallback):
    def __init__(self, total, kind="prismatic", eval_every=200000):
        super().__init__()
        self.total = total
        self.kind = kind
        self.eval_every = eval_every
        self._next_eval = eval_every
        self.csv = open(os.path.join(OUT, f"curriculum_eval_{kind}.csv"), "w", newline="")
        self.w = csv.writer(self.csv)
        self.w.writerow(["timesteps", "cmd_lo", "cmd_hi", "surv", "dist", "speed", "track_err", "roll_deg"])

    def _on_rollout_start(self):
        p = self.num_timesteps / self.total
        lo, hi = curriculum_range(p)
        self.training_env.env_method("set_cmd_range", lo, hi)
        self._cur = (lo, hi)

    def _on_step(self):
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            m = eval_policy(self.model, kind=self.kind)
            lo, hi = getattr(self, "_cur", (0, 0))
            self.w.writerow([self.num_timesteps, lo, hi, round(m["surv"], 2),
                             round(m["dist"], 3), round(m["speed"], 3),
                             round(m["track_err"], 3), round(m["roll"], 1)])
            self.csv.flush()
            print(f"[eval @{self.num_timesteps}] cmd[{lo:.2f},{hi:.2f}] surv={m['surv']:.2f} "
                  f"dist={m['dist']:+.2f} speed={m['speed']:.3f} track_err={m['track_err']:.3f} roll={m['roll']:.1f}deg")
        return True

    def _on_training_end(self):
        self.csv.close()


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    kind = sys.argv[3] if len(sys.argv) > 3 else "prismatic"
    venv = SubprocVecEnv([_mk(i, kind) for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, device="cpu", verbose=0,
                n_steps=1024, batch_size=4096, n_epochs=5,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=0.005, clip_range=0.2, target_kl=0.03,
                policy_kwargs=dict(net_arch=[256, 256]))
    cb = CurriculumEval(total, kind=kind, eval_every=250000)
    print(f"# CURRICULUM TRAIN total={total} n_envs={n_envs} kind={kind} device=cpu")
    t0 = time.time()
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    dt = time.time() - t0
    model.save(os.path.join(OUT, f"ppo_curriculum_{kind}.zip"))
    fm = eval_policy(model, cmds=(0.10, 0.18, 0.25), n_ep=5, kind=kind)
    print(f"# DONE {total} steps in {dt/60:.1f} min ({total/dt:.0f} steps/s)")
    print(f"# FINAL eval: surv={fm['surv']:.2f} dist={fm['dist']:+.2f} speed={fm['speed']:.3f} "
          f"track_err={fm['track_err']:.3f} roll={fm['roll']:.1f}deg  saved ppo_curriculum.zip")
    venv.close()


if __name__ == "__main__":
    main()
