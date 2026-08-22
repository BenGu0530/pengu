"""Reward audit: what does the c6 designed gait earn under the frozen RL reward?

Replays the c6 champion (freq 1.77, hip_phi 270, leg_amp 105, hip_amp 28,
hip_off 10, kappa=2 TorsoKappaPID) through Grid4RLEnv's reward accounting and
prints per-component means, next to a learned policy scored the same way.

Two probe conditions, both declared:
- the designed gait is expressed through a FULL-ctrlrange action mapping with
  no filter/delay (the frozen a1 crank band [-1.8,-0.6] cannot represent its
  crank commands [0,+1.83] at all -- that expressibility gap is part of the
  report, not silently worked around);
- the learned policy is scored through its own frozen interface (a1, alpha).

Usage (from pengu_mujoco/):
  python rl_grid4/score_designed_gait.py [--policy rl_grid4/runs/e2/s2/stageB/ckpts/final.zip]
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import grid4_rl_env as ge
from grid4_rl_env import Grid4RLEnv

C6 = dict(freq=1.77, hip_phi=270.0, leg_amp=105.0, hip_amp=28.0, hip_off=10.0)
KAPPA = 2.0
T_HOLD, T_TRANSITION = 5.0, 4.0
WALK_FROM = T_HOLD + T_TRANSITION + 2.0     # sweep SETTLE convention: 11 s
DUR = 24.0
COMP = ["r_track", "r_progress", "r_back", "r_energy", "r_swing",
        "r_scrub", "r_smooth", "r_hf", "r_fall", "vx"]


def set_c6_params():
    import gait_config as gc
    gc.set_crank_amp(C6["leg_amp"])
    gc.set_hip_amp(C6["hip_amp"])
    gc.set_torso_amp(0.0)
    gc.WALK_HIP_OFFSET_DEG = C6["hip_off"]
    gc.WALK_HIP_LEAN_DEG = 0.0
    gc.set_walk_freq(C6["freq"])
    gc.PHASE_OFFSET_A_DEG = 0.0
    gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = C6["hip_phi"]
    gc.PHASE_OFFSET_D_DEG = C6["hip_phi"]
    gc.PHASE_OFFSET_E_DEG = 0.0
    gc.T_HOLD, gc.T_TRANSITION = T_HOLD, T_TRANSITION
    return gc


def snapshot(env):
    return dict(env._ep), env.step_i


def window_means(ep0, n0, ep1, n1):
    n = max(1, n1 - n0)
    return {k: (ep1[k] - ep0.get(k, 0.0)) / n for k in ep1}


def run_designed(mu, seed):
    gc = set_c6_params()
    env = Grid4RLEnv(eval_mode=True, mu_fixed=mu, episode_s=DUR, seed=seed,
                     filter_alpha=1.0, action_delay=0)
    # probe condition: full-range mapping so the designed ctrl is expressible
    cr = env.model.actuator_ctrlrange[env.aid]
    env.ctrl_mid = cr.mean(axis=1)
    env.ctrl_half = (cr[:, 1] - cr[:, 0]) / 2.0

    from torso_control import TorsoKappaPID
    gc.TORSO_CONTROLLER = TorsoKappaPID(env.model, kappa=KAPPA,
                                        measure_after=WALK_FROM)
    obs, _ = env.reset(seed=seed)
    act_ids = {n: int(a) for n, a in zip(ge.ACT_NAMES, env.aid)}

    clip_hits = 0
    walk_snap = None
    n_steps = int(DUR / env.control_dt)
    ep = {}
    # r3 hf accounting: the replay bypasses the action filter (alpha=1.0), so
    # env-side r_hf is identically 0. Score the designed action stream against
    # the same alpha=0.2 reference offline and write the sum into env._ep.
    # r3b: residuals are scaled to band-fair units; this replay normalizes by
    # the FULL ctrlrange, so scale = full-range halves / a1 reference halves.
    from grid4_rl_env import (ALPHA as _ALPHA, RW_DEFAULT as _RW,
                              CRANK_IDX as _CIDX, CRANK_HALF as _CHALF,
                              HF_IDX as _HFIDX)
    _href = env.ctrl_half.copy()      # full-range halves (overridden above)
    _href[_CIDX] = _CHALF
    _hf_scale = env.ctrl_half / _href
    _hf_filt = None
    _hf_sum = 0.0
    for i in range(n_steps):
        t = i * env.control_dt
        # designed controller writes data.ctrl; we lift it into action space
        gc.apply_ctrl(env.data, act_ids, t)
        ctrl_des = env.data.ctrl[env.aid].copy()
        a = (ctrl_des - env.ctrl_mid) / env.ctrl_half
        if np.any(np.abs(a) > 1.0):
            clip_hits += 1
        a_cl = np.clip(a, -1, 1)
        if _hf_filt is None:
            _hf_filt = a_cl.copy()
            _hf_filt2 = a_cl.copy()
        else:
            _hf_filt = (1 - _ALPHA) * _hf_filt + _ALPHA * a_cl
        _hf_filt2 = (1 - _ALPHA) * _hf_filt2 + _ALPHA * _hf_filt
        _resid = ((_hf_filt - _hf_filt2) * _hf_scale)[_HFIDX]  # r3c: executed HF, hips+torso
        _hf_sum += -_RW["hf"] * float(_resid @ _resid)
        obs, r, term, trunc, info = env.step(a_cl)
        env._ep["r_hf"] = _hf_sum
        if walk_snap is None and t >= WALK_FROM:
            walk_snap = snapshot(env)
        if term or trunc:
            ep = info.get("ep", {})
            break
    gc.TORSO_CONTROLLER = None
    end_snap = snapshot(env)
    full = {k: v for k, v in ep.items()} if ep else {}
    out = {
        "ep_len_s": (end_snap[1]) * env.control_dt,
        "fell": full.get("fell", 0.0) if full else float("nan"),
        "clip_frac": clip_hits / max(1, end_snap[1]),
        "full": {k: full.get(k, float("nan")) for k in COMP} if full else {},
        "torso_rms": full.get("torso_roll_rms_deg", float("nan")),
        "hip_corr": full.get("hip_corr", float("nan")),
    }
    if walk_snap is not None:
        out["walk"] = window_means(walk_snap[0], walk_snap[1],
                                   end_snap[0], end_snap[1])
    return out


def run_policy(ckpt, mu, seed):
    from stable_baselines3 import PPO
    model = PPO.load(ckpt, device="cpu")
    env = Grid4RLEnv(eval_mode=True, mu_fixed=mu, episode_s=DUR, seed=seed)
    obs, _ = env.reset(seed=seed)
    walk_snap = None
    ep = {}
    n_steps = int(DUR / env.control_dt)
    for i in range(n_steps):
        t = i * env.control_dt
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(act)
        if walk_snap is None and t >= 2.0:      # policy has no staging; 2 s settle
            walk_snap = snapshot(env)
        if term or trunc:
            ep = info.get("ep", {})
            break
    end_snap = snapshot(env)
    out = {
        "ep_len_s": end_snap[1] * env.control_dt,
        "fell": ep.get("fell", float("nan")) if ep else float("nan"),
        "full": {k: ep.get(k, float("nan")) for k in COMP} if ep else {},
        "torso_rms": ep.get("torso_roll_rms_deg", float("nan")),
        "hip_corr": ep.get("hip_corr", float("nan")),
    }
    if walk_snap is not None:
        out["walk"] = window_means(walk_snap[0], walk_snap[1],
                                   end_snap[0], end_snap[1])
    return out


def fmt(res, label):
    w = res.get("walk", {})
    line1 = (f"  {label:<22} ep {res['ep_len_s']:４.1f}s fell={res['fell']:.0f} "
             f"torso_rms={res['torso_rms']:.1f} hip_corr={res.get('hip_corr', float('nan')):+.2f}"
             + (f" clip={res['clip_frac']:.2f}" if "clip_frac" in res else ""))
    per = " ".join(f"{k.replace('r_','')}={w.get(k, float('nan')):+.3f}"
                   for k in COMP)
    tot = sum(w.get(k, 0.0) for k in COMP if k != "vx")
    return line1 + f"\n    walk-window per-step: {per}\n    TOTAL/step={tot:+.3f}  (x500 steps -> {tot*500:+.0f}/10s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",
                    default=os.path.join(_HERE, "runs", "e2", "s2", "stageB",
                                         "ckpts", "final.zip"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-policy", action="store_true")
    a = ap.parse_args()
    for mu in (0.1, 0.3):
        print(f"== mu={mu} ==")
        for s in range(a.seeds):
            print(fmt(run_designed(mu, seed=s), f"c6 designed (seed {s})"))
        if not a.skip_policy:
            for s in range(a.seeds):
                print(fmt(run_policy(a.policy, mu, seed=s), f"learned s2 (seed {s})"))


if __name__ == "__main__":
    main()
