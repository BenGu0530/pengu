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
    ap.add_argument("--curriculum", action="store_true",
                    help="vx_cmd curriculum c1 (declared amendment): start 0.12, "
                         "+0.05 whenever recent mean vx >= 0.6*cmd and fall<=0.5, "
                         "cap 0.47. Eval stays fixed at 0.47.")
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

    from grid4_rl_env import REWARD_VERSION, ACTION_VERSION
    # e1 (capability knob, 2026-08-21): log_std_init 0 -> -1.0. Probe: a
    # zero-mean policy under exploration noise survives 0.7 s at sigma 0.85
    # but the robot is open-loop unstable at ANY sigma (2.7 s even at 0.15),
    # so with default sigma~1 no surviving samples exist and the dash is the
    # only harvestable strategy. Smaller init sigma lets early rollouts
    # contain multi-second survival for the value function to see.
    LOG_STD_INIT = -1.0
    EXPLORATION_VERSION = "e1"
    tag = (f"{a.mode}_{REWARD_VERSION}{ACTION_VERSION}{EXPLORATION_VERSION}"
           + ("c2" if a.curriculum else "") + f"_s{a.seed}"
           + ("_smoke" if a.smoke else "") + ("_t2" if a.tier2 else ""))
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
                   "single_frac": np.mean([e["single_frac"] for e in eps]) if eps else float("nan"),
                   "hip_diff_rms_deg":
                       np.mean([e.get("hip_diff_rms_deg", 0.0) for e in eps]) if eps else float("nan"),
                   "hip_corr": np.mean([e.get("hip_corr", 0.0) for e in eps]) if eps else float("nan")}
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
                  f"hip_diff={row['hip_diff_rms_deg']:.1f}deg "
                  f"hip_corr={row['hip_corr']:+.2f}  "
                  f"torso_rms={row['torso_roll_rms_deg']:.1f}deg  "
                  f"sigma_torso={row['sigma_torso']:.3f}", flush=True)

    class Curriculum(BaseCallback):
        """vx_cmd ramp c2: performance-gated, checked every CHECK steps over the
        last WINDOW finished episodes. Starts at CMD0; +STEP when fall<=0.5 and
        (mean ep vx >= 0.6*cmd  OR  survival is solved: mean ep_len>=400 and
        fall<=0.3). The second clause breaks the stand-still deadlock seen in
        c1: at low cmd the kernel tail pays ~0.4/step at vx=0, so standing is
        profitable and the vx gate never fires; raising cmd moves the kernel
        away from zero (receding carrot). Cap 0.47 (the experiment command)."""
        CHECK, WINDOW, CMD0, STEP, CAP = 25_000, 100, 0.12, 0.05, 0.47

        def __init__(self):
            super().__init__()
            self.cmd = self.CMD0
            self._eps = []
            self._next = self.CHECK

        def _on_training_start(self):
            self.training_env.env_method("set_vx_cmd", self.cmd)
            print(f"[curriculum] vx_cmd start {self.cmd:.2f}", flush=True)

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "ep" in info:
                    self._eps.append((info["ep"]["vx"], info["ep"]["fell"],
                                      info["ep"]["len"]))
                    if len(self._eps) > self.WINDOW:
                        self._eps.pop(0)
            if self.num_timesteps >= self._next:
                self._next += self.CHECK
                if len(self._eps) >= 30 and self.cmd < self.CAP:
                    vx = float(np.mean([e[0] for e in self._eps]))
                    fall = float(np.mean([e[1] for e in self._eps]))
                    elen = float(np.mean([e[2] for e in self._eps]))
                    speed_ok = vx >= 0.6 * self.cmd
                    survival_ok = elen >= 400 and fall <= 0.3
                    if fall <= 0.5 and (speed_ok or survival_ok):
                        self.cmd = min(self.CAP, self.cmd + self.STEP)
                        self.training_env.env_method("set_vx_cmd", self.cmd)
                        self._eps = []
                        print(f"[curriculum] {self.num_timesteps} steps -> "
                              f"vx_cmd {self.cmd:.2f} "
                              f"({'speed' if speed_ok else 'survival'} gate)",
                              flush=True)
            return True

    callbacks = [Diag()] + ([Curriculum()] if a.curriculum else [])
    if not a.smoke:
        callbacks.append(CheckpointCallback(
            save_freq=max(250_000 // a.n_envs, 1), save_path=ckptdir, name_prefix="ckpt"))

    model = PPO("MlpPolicy", venv, device="cpu", seed=a.seed,
                n_steps=1024, batch_size=4096, n_epochs=5,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=a.ent, clip_range=0.2, target_kl=0.03,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=LOG_STD_INIT),
                verbose=0)
    print(f"[train_grid4] {tag}: {a.steps} steps x {a.n_envs} envs -> {outdir}", flush=True)
    model.learn(total_timesteps=a.steps, callback=callbacks, progress_bar=False)
    model.save(os.path.join(ckptdir, "final"))
    venv.close()
    print(f"[train_grid4] done -> {ckptdir}/final.zip", flush=True)


if __name__ == "__main__":
    main()
