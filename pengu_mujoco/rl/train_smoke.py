"""
train_smoke.py - end-to-end smoke train of CPG-RL PPO on the Pengu prismatic env.

CPU only (AutoFloat has GPU priority). Verifies the whole pipeline runs and that
reward improves. Not a full training run.

Run from pengu_mujoco/:
  python rl/train_smoke.py [total_timesteps] [n_envs]
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from rl.pengu_env import PenguCPGEnv

torch.set_num_threads(2)


def _mk(seed):
    def f():
        return Monitor(PenguCPGEnv(domain_rand=True, seed=seed))
    return f


def evaluate(model, n_ep=5):
    env = PenguCPGEnv(domain_rand=False, seed=999)
    rets, dists, lens = [], [], []
    for _ in range(n_ep):
        o, _ = env.reset()
        y0 = env.data.xpos[env.root][1]
        ret = 0.0; k = 0
        while True:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            ret += r; k += 1
            if term or trunc:
                break
        rets.append(ret); lens.append(k)
        dists.append(env.data.xpos[env.root][1] - y0)
    return np.mean(rets), np.mean(lens), np.mean(dists)


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 80000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
    os.makedirs(outdir, exist_ok=True)

    venv = SubprocVecEnv([_mk(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, device="cpu", verbose=1,
                n_steps=512, batch_size=1024, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.0, learning_rate=3e-4, n_epochs=5,
                policy_kwargs=dict(net_arch=[128, 128]))

    print(f"# SMOKE TRAIN: total={total} n_envs={n_envs} device=cpu")
    r0, l0, d0 = evaluate(model)
    print(f"# PRE-TRAIN eval: ep_ret={r0:.1f} ep_len={l0:.0f} fwd_dist={d0:+.3f} m")
    t0 = time.time()
    model.learn(total_timesteps=total, progress_bar=False)
    dt = time.time() - t0
    r1, l1, d1 = evaluate(model)
    path = os.path.join(outdir, "ppo_smoke.zip")
    model.save(path)
    print(f"# POST-TRAIN eval: ep_ret={r1:.1f} ep_len={l1:.0f} fwd_dist={d1:+.3f} m")
    print(f"# trained {total} steps in {dt:.1f}s ({total/dt:.0f} steps/s); saved {path}")
    print(f"# DELTA ep_ret {r0:.1f} -> {r1:.1f}   ep_len {l0:.0f} -> {l1:.0f}   dist {d0:+.3f} -> {d1:+.3f}")
    venv.close()


if __name__ == "__main__":
    main()
