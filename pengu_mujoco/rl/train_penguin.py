"""
train_penguin.py - Phase 2 bio-imitation CPG-RL.

Teaches the penguin gait signature (cadence ~1.27 Hz, ~8 deg lateral rock,
small sagittal lean) by training the env in bio_imitate mode:
  - penguin-prior CPG nominal + NARROW action range (freq pinned ~1.27 Hz),
  - torso-level imitation reward (cadence + roll-amp + lean),
  - domain randomization OFF (DR was the real source of the conservative,
    high-cadence shuffle; we trade some sim2real robustness for animal fidelity),
  - LOW, flat forward-speed command (chasing speed is what pushed it to 2 Hz).

Periodically evaluates and logs the bio signature (f_cpg, roll/lean amplitude)
so we can watch the gait converge onto the real-penguin numbers.

CPU only. Run from pengu_mujoco/:
  python rl/train_penguin.py [total_timesteps] [n_envs]
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

from rl.pengu_env import (PenguCPGEnv, PENGUIN_FREQ, PENGUIN_ROLL_DEG,
                          PENGUIN_LEAN_DEG)
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
os.makedirs(OUT, exist_ok=True)
VX_CMD = 0.05            # v1 (restored): slow, penguin-like; imitation reward leads,
                         # not speed. (v2's 0.15 push only made it fall -> reverted.)
# bio-reward variant: "v4" = propulsion (swing-foot reaches forward -> steps);
# "v3" = clean waddle (no r_swing, rocks ~in place, cleanest lean). Override with
# 3rd CLI arg, e.g.  python rl/train_penguin.py 3000000 8 v3
PROPULSION = (len(sys.argv) <= 3) or (sys.argv[3].lower() not in ("v3", "0", "false"))


def _mk(seed):
    def f():
        return Monitor(PenguCPGEnv(domain_rand=False, seed=seed, bio_imitate=True,
                                   vx_cmd=VX_CMD, propulsion=PROPULSION))
    return f


def eval_bio(model, n_ep=4):
    """Deterministic eval: report the penguin signature + survival."""
    env = PenguCPGEnv(domain_rand=False, seed=999, bio_imitate=True, vx_cmd=VX_CMD,
                      propulsion=PROPULSION)
    env.set_cmd_range(VX_CMD, VX_CMD)
    surv, fs, rolls, leans, speeds = [], [], [], [], []
    for ep in range(n_ep):
        o, _ = env.reset()
        y0 = env.data.xpos[env.root][1]
        k = 0; ff, rr, ll = [], [], []
        while True:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            if k > 100:                 # skip transient
                ff.append(info["f_cpg"]); rr.append(info["roll_amp_deg"]); ll.append(info["pitch_amp_deg"])
            k += 1
            if term or trunc:
                break
        surv.append(1.0 if k >= env.max_steps else 0.0)
        fs.append(np.mean(ff) if ff else float("nan"))
        rolls.append(np.mean(rr) if rr else float("nan"))
        leans.append(np.mean(ll) if ll else float("nan"))
        speeds.append((env.data.xpos[env.root][1] - y0) / (k * env.control_dt))
    return dict(surv=float(np.mean(surv)), f=float(np.nanmean(fs)),
                roll=float(np.nanmean(rolls)), lean=float(np.nanmean(leans)),
                speed=float(np.mean(speeds)))


class BioEval(BaseCallback):
    def __init__(self, eval_every=250000):
        super().__init__()
        self.eval_every = eval_every
        self._next = eval_every
        self.csv = open(os.path.join(OUT, "penguin_eval.csv"), "w", newline="")
        self.w = csv.writer(self.csv)
        self.w.writerow(["timesteps", "surv", "f_cpg", "roll_amp_deg", "lean_amp_deg", "speed"])

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            e = eval_bio(self.model)
            self.w.writerow([self.num_timesteps, round(e["surv"], 2), round(e["f"], 2),
                             round(e["roll"], 1), round(e["lean"], 1), round(e["speed"], 3)])
            self.csv.flush()
            print(f"[eval @{self.num_timesteps}] surv={e['surv']:.2f} f={e['f']:.2f}Hz "
                  f"roll={e['roll']:.1f}deg lean={e['lean']:.1f}deg speed={e['speed']:.3f}  "
                  f"(target f={PENGUIN_FREQ} roll~{PENGUIN_ROLL_DEG} lean~{PENGUIN_LEAN_DEG})")
        return True

    def _on_training_end(self):
        self.csv.close()


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 3000000
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    venv = SubprocVecEnv([_mk(i) for i in range(n_envs)])
    model = PPO("MlpPolicy", venv, device="cpu", verbose=0,
                n_steps=1024, batch_size=4096, n_epochs=5,
                gamma=0.99, gae_lambda=0.95, learning_rate=3e-4,
                ent_coef=0.005, clip_range=0.2, target_kl=0.03,
                policy_kwargs=dict(net_arch=[256, 256]))
    # v2: also dump an intermediate checkpoint every 250k steps -- the most
    # penguin-like gait (roll ~8 deg) showed up mid-run last time and we lost it
    # because only the final (over-smoothed) policy was saved.
    variant = "v4" if PROPULSION else "v3"
    out_zip = os.path.join(OUT, f"ppo_penguin_{variant}.zip")
    ckpt = CheckpointCallback(save_freq=max(1, 250000 // n_envs),
                              save_path=OUT, name_prefix=f"penguin_ckpt_{variant}")
    cb = CallbackList([BioEval(eval_every=250000), ckpt])
    print(f"# PENGUIN BIO-IMITATION TRAIN {variant} (propulsion={PROPULSION}) "
          f"total={total} n_envs={n_envs} device=cpu  DR=OFF  vx_cmd={VX_CMD}")
    t0 = time.time()
    model.learn(total_timesteps=total, callback=cb, progress_bar=False)
    dt = time.time() - t0
    model.save(out_zip)
    e = eval_bio(model, n_ep=6)
    print(f"# DONE {total} steps in {dt/60:.1f} min ({total/dt:.0f} steps/s)")
    print(f"# FINAL: surv={e['surv']:.2f} f={e['f']:.2f}Hz roll={e['roll']:.1f}deg "
          f"lean={e['lean']:.1f}deg speed={e['speed']:.3f}  saved {os.path.basename(out_zip)}")
    venv.close()


if __name__ == "__main__":
    main()
