"""PPO trainer for the GRID-4 RL phase (direct joint control, COM-1.31 body).

Modes:
  gate0  -- qualification gate, NOT an experiment arm: mu fixed 0.7 (+-5%),
            proves the frozen setup can learn locomotion at all before any
            multi-seed arm runs. Capability knobs may be tuned here ONLY.
  e2     -- the ice arm: mu ~ U(0.1, 0.4) per episode, vx_cmd = 0.47.

Usage (from pengu_mujoco/):
  python rl/train_grid4.py --mode gate0 --seed 0
  python rl/train_grid4.py --mode e2 --seed 0 --steps 3000000 --n-envs 8
  python rl/train_grid4.py --mode gate0 --seed 0 --smoke     # 50k pipeline check

Hyperparams are the validated set: PPO MlpPolicy [256,256], lr 3e-4, gamma .99,
lambda .95, clip .2, ent .005, target_kl .03, n_steps 1024, batch 4096,
n_epochs 5, CPU only. Diagnostics every 250k steps to diag.csv: per-component
episode reward means, per-dim policy sigma (torso sigma collapse = exploration
death), torso roll RMS, ep_len (lunge-trap signature: low ep_len + high
per-step progress).
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np


def make_env(rank, seed, env_kwargs):
    def _f():
        from grid4_rl_env import Grid4RLEnv
        from stable_baselines3.common.monitor import Monitor
        return Monitor(Grid4RLEnv(seed=seed * 1000 + rank, **env_kwargs))
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate0", "e2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="50k steps, 4 envs")
    ap.add_argument("--tier2", action="store_true", help="enable tier-2 DR (gains+push)")
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--out", default=os.path.join(_HERE, "runs"))
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.n_envs = 50_000, 4

    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

    env_kwargs = dict(rand_gains=a.tier2, push=a.tier2)
    if a.mode == "gate0":
        env_kwargs.update(mu_fixed=0.7)
    else:
        env_kwargs.update(mu_lo=0.1, mu_hi=0.4)

    tag = f"{a.mode}_s{a.seed}" + ("_smoke" if a.smoke else "") + ("_t2" if a.tier2 else "")
    outdir = os.path.join(a.out, tag)
    ckptdir = os.path.join(outdir, "ckpts")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "videos"), exist_ok=True)

    venv = SubprocVecEnv([make_env(i, a.seed, env_kwargs) for i in range(a.n_envs)])

    class Diag(BaseCallback):
        EVERY = 250_000 if not a.smoke else 10_000
        COMP = ["r_track", "r_progress", "r_back", "r_energy", "r_swing",
                "r_scrub", "r_smooth", "r_fall", "vx"]

        def __init__(self):
            super().__init__()
            self._eps = []
            self._next = self.EVERY

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "ep" in info:
                    self._eps.append(info["ep"])
            if self.num_timesteps >= self._next:
                self._next += self.EVERY
                self._flush()
            return True

        def _flush(self):
            eps = self._eps
            self._eps = []
            sig = np.exp(self.model.policy.log_std.detach().cpu().numpy()).round(4)
            row = {"steps": self.num_timesteps,
                   "n_ep": len(eps),
                   "ep_len": np.mean([e["len"] for e in eps]) if eps else float("nan"),
                   "fall_rate": np.mean([e["fell"] for e in eps]) if eps else float("nan"),
                   "torso_roll_rms_deg":
                       np.mean([e["torso_roll_rms_deg"] for e in eps]) if eps else float("nan"),
                   "single_frac": np.mean([e["single_frac"] for e in eps]) if eps else float("nan")}
            for k in self.COMP:
                row[k] = np.mean([e[k] for e in eps]) if eps else float("nan")
            for i, name in enumerate(["hipL", "hipR", "crankR", "torso", "crankL"]):
                row[f"sigma_{name}"] = float(sig[i])
            path = os.path.join(outdir, "diag.csv")
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if new:
                    w.writeheader()
                w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                            for k, v in row.items()})
            print(f"[diag] {row['steps']:>9} steps  ep_len={row['ep_len']:.0f} "
                  f"fall={row['fall_rate']:.2f}  vx={row['vx']:.3f}  "
                  f"torso_rms={row['torso_roll_rms_deg']:.1f}deg  "
                  f"sigma_torso={row['sigma_torso']:.3f}", flush=True)

    callbacks = [Diag()]
    if not a.smoke:
        callbacks.append(CheckpointCallback(
            save_freq=max(250_000 // a.n_envs, 1), save_path=ckptdir, name_prefix="ckpt"))

    model = PPO("MlpPolicy", venv, device="cpu", seed=a.seed,
                n_steps=1024, batch_size=4096, n_epochs=5,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=a.ent, clip_range=0.2, target_kl=0.03,
                policy_kwargs=dict(net_arch=[256, 256]), verbose=0)
    print(f"[train_grid4] {tag}: {a.steps} steps x {a.n_envs} envs -> {outdir}", flush=True)
    model.learn(total_timesteps=a.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(ckptdir, "final"))
    venv.close()
    print(f"[train_grid4] done -> {ckptdir}/final.zip", flush=True)


if __name__ == "__main__":
    main()
