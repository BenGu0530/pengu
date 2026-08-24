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
    ap.add_argument("--shape", choices=["reward", "penalty"], default="reward",
                    help="penalty: shift track/progress/swing so each tops out "
                         "at 0 and the whole reward is <= 0. Standing then "
                         "bleeds instead of collecting income. This changes what "
                         "DYING is worth, so --rw fall= must clear the suicide "
                         "breakeven (floor/(1-gamma) ~ 186 at the defaults); the "
                         "launcher refuses to start otherwise.")
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
    ap.add_argument("--mu-fixed", type=float, default=None,
                    help="override mu to a fixed value +-5%% (mu-curriculum "
                         "amendment: e2 stage A runs at 0.4, the easy end of "
                         "the arm's range; stage B runs the full U(0.1,0.4))")
    ap.add_argument("--rw", nargs="+", default=None, metavar="KEY=VAL",
                    help="override frozen reward weights, e.g. --rw progress=3.0 "
                         "swing=0.3 fall=5 sigma2=0.06. Keys: track sigma2 progress "
                         "back energy swing swing_cap scrub smooth fall. The run tag "
                         "records every override so a tuning arm can never be pooled "
                         "with a frozen-recipe run.")
    ap.add_argument("--no-smooth", action="store_true",
                    help="ABLATION: set the action-rate weight to 0 (frozen r2 uses "
                         "-0.01*||a_t - a_{t-1}||^2). Run tag gets an 'ns' marker so "
                         "these can never be pooled with frozen-recipe runs.")
    ap.add_argument("--cmd0", type=float, default=None,
                    help="start the c2 curriculum at this vx_cmd instead of 0.12. "
                         "Needed to RESUME a curriculum run: --init-from restores "
                         "policy+value weights only, so without this the ramp "
                         "silently restarts from 0.12.")
    ap.add_argument("--init-from", default=None,
                    help="warm-start from a prior final.zip (policy+value weights); "
                         "tag gets _w suffix")
    ap.add_argument("--crank-band", nargs=2, type=float, default=None,
                    metavar=("MID", "HALF"),
                    help="override the a1 crank action band, e.g. 0.0 1.9 (a2 "
                         "probe: covers the c6 designed gait's command domain)")
    ap.add_argument("--name", default=None,
                    help="run dir name under --out (e.g. e2/s0/stageA); "
                         "overrides the auto version tag")
    ap.add_argument("--no-slew", action="store_true",
                    help="disable the sv1 servo slew clamp (legacy sv0 repro)")
    ap.add_argument("--cmd-fc", type=float, default=None,
                    help="sv2 command bandwidth cap: 2nd-order LPF cutoff in Hz "
                         "on the filtered action (firmware-replicable). Tag f<fc>.")
    ap.add_argument("--cmd-cap", type=float, default=0.47,
                    help="curriculum vx_cmd cap (default 0.47, the c6 ceiling)")
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
    rw = {}
    if a.rw:
        for kv in a.rw:
            k, _, v = kv.partition("=")
            if not _:
                raise SystemExit(f"--rw expects KEY=VAL, got {kv!r}")
            rw[k.strip()] = float(v)
        env_kwargs.update(rw=rw)
    if a.no_smooth:
        env_kwargs.update(w_smooth=0.0)
    if a.crank_band:
        env_kwargs.update(crank_band=tuple(a.crank_band))
    if a.shape != "reward":
        env_kwargs.update(shape=a.shape)
        # suicide preflight. With every term <= 0, an agent that cannot yet walk
        # bleeds the floor each step; if a single -fall is cheaper than the
        # discounted remaining bleed, dying is optimal and the run is wasted.
        # This has already burned two fleet rounds, so it is a hard gate.
        from grid4_rl_env import RW_DEFAULT as _RWD
        _w = dict(_RWD); _w.update(rw)
        _floor = _w["track"] + _w["progress"] * 0.47 + _w["swing"] * _w["swing_cap"]
        _g, _T = 0.99, 500
        _need = _floor * (1 - _g ** _T) / (1 - _g)
        print(f"[preflight] shape=penalty floor -{_floor:.2f}/step -> discounted "
              f"{_need:.0f}; fall={_w['fall']:g}", flush=True)
        if _w["fall"] <= _need:
            raise SystemExit(
                f"[preflight] REFUSING: fall={_w['fall']:g} <= suicide breakeven "
                f"{_need:.0f}. Dying would beat living. Set --rw fall=<bigger>.")
    if a.no_slew:
        env_kwargs.update(slew_vmax=0.0)
    if a.cmd_fc:
        env_kwargs.update(cmd_fc_hz=a.cmd_fc)
    if a.mu_fixed is not None:
        env_kwargs.update(mu_fixed=a.mu_fixed)
    elif a.mode == "gate0":
        env_kwargs.update(mu_fixed=0.7)
    else:
        env_kwargs.update(mu_lo=0.1, mu_hi=0.4)

    from grid4_rl_env import REWARD_VERSION, ACTION_VERSION, SLEW_VERSION
    # e1 (capability knob, 2026-08-21): log_std_init 0 -> -1.0. Probe: a
    # zero-mean policy under exploration noise survives 0.7 s at sigma 0.85
    # but the robot is open-loop unstable at ANY sigma (2.7 s even at 0.15),
    # so with default sigma~1 no surviving samples exist and the dash is the
    # only harvestable strategy. Smaller init sigma lets early rollouts
    # contain multi-second survival for the value function to see.
    LOG_STD_INIT = -1.0
    EXPLORATION_VERSION = "e1"
    rwtag = "".join(f"-{k}{v:g}" for k, v in sorted(rw.items())) if rw else ""
    rwtag += "" if a.shape == "reward" else f"-{a.shape}"
    tag = a.name or (
        f"{a.mode}_{REWARD_VERSION}{'ns' if a.no_smooth else ''}{rwtag}"
        f"{ACTION_VERSION}{EXPLORATION_VERSION}"
        + ("" if a.no_slew else SLEW_VERSION)
        + (f"f{a.cmd_fc:g}" if a.cmd_fc else "")
        + (f"cap{a.cmd_cap:g}" if a.cmd_cap != 0.47 else "")
        + ("c2" if a.curriculum else "") + ("_w" if a.init_from else "")
        + (f"_mu{a.mu_fixed:g}" if a.mu_fixed is not None and a.mode != "gate0" else "")
        + f"_s{a.seed}"
        + ("_smoke" if a.smoke else "") + ("_t2" if a.tier2 else ""))
    outdir = os.path.join(a.out, tag)
    ckptdir = os.path.join(outdir, "ckpts")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "videos"), exist_ok=True)

    venv = SubprocVecEnv([make_env(i, a.seed, env_kwargs) for i in range(a.n_envs)])

    class Diag(BaseCallback):
        EVERY = 250_000 if not a.smoke else 10_000
        COMP = ["r_track", "r_progress", "r_back", "r_energy", "r_swing",
                "r_scrub", "r_smooth", "r_hf", "r_straight", "r_dutybal", "r_fall", "vx"]

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
                   "torso_roll_mean_deg":
                       np.mean([e.get("torso_roll_mean_deg", float("nan")) for e in eps])
                       if eps else float("nan"),
                   "vx_cmd": float(getattr(curric, "cmd", float("nan"))),
                   "torso_roll_rate_rms_dps":
                       np.mean([e.get("torso_roll_rate_rms_dps", float("nan")) for e in eps])
                       if eps else float("nan"),
                   "stride_L_m": np.nanmean([e.get("stride_L_m", float("nan")) for e in eps])
                       if eps else float("nan"),
                   "stride_R_m": np.nanmean([e.get("stride_R_m", float("nan")) for e in eps])
                       if eps else float("nan"),
                   "stride_asym": np.nanmean([e.get("stride_asym", float("nan")) for e in eps])
                       if eps else float("nan"),
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
                  f"torso_rms={row['torso_roll_rms_deg']:.1f}deg "
                  f"roll_rate={row['torso_roll_rate_rms_dps']:.0f}dps "
                  f"stride_asym={row['stride_asym']:+.3f}  "
                  f"r_hf={row['r_hf']:+.3f}  "
                  f"r_dutybal={row['r_dutybal']:+.3f}  "
                  f"sigma_torso={row['sigma_torso']:.3f}", flush=True)

    class Curriculum(BaseCallback):
        """vx_cmd ramp c2: performance-gated, checked every CHECK steps over the
        last WINDOW finished episodes. Starts at CMD0; +STEP when fall<=0.5 and
        (mean ep vx >= 0.6*cmd  OR  survival is solved: mean ep_len>=400 and
        fall<=0.3). The second clause breaks the stand-still deadlock seen in
        c1: at low cmd the kernel tail pays ~0.4/step at vx=0, so standing is
        profitable and the vx gate never fires; raising cmd moves the kernel
        away from zero (receding carrot). Cap 0.47 (the experiment command)."""
        CHECK, WINDOW, CMD0, STEP = 25_000, 100, 0.12, 0.05
        CAP = None  # set from --cmd-cap at construction


        def __init__(self, cmd0=None, cap=0.47):
            super().__init__()
            self.CAP = float(cap)
            self.cmd = self.CMD0 if cmd0 is None else float(cmd0)
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

    curric = Curriculum(a.cmd0, cap=a.cmd_cap) if a.curriculum else None
    callbacks = [Diag()] + ([curric] if curric else [])
    if not a.smoke:
        callbacks.append(CheckpointCallback(
            save_freq=max(250_000 // a.n_envs, 1), save_path=ckptdir, name_prefix="ckpt"))

    if a.init_from:
        model = PPO.load(a.init_from, env=venv, device="cpu")
        print(f"[train_grid4] warm-start from {a.init_from}", flush=True)
    else:
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
