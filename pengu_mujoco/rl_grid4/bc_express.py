"""BC expressibility test: clone the c6 designed gait into the RL policy net.

Question (reward audit 2026-08-21): the frozen reward prefers c6 by 2-10x at
mu=0.1, but can the policy NETWORK + a2 action band even express that gait?
This script does the whole test in one command:

  1. collect (obs, action) pairs from the scripted c6 teacher (kappa=2 PID),
     DART-style: small exploration noise on the executed action, label = the
     teacher's clean action; mu ~ U(0.1,0.4) per episode, pose jitter on;
  2. supervised-train an SB3 PPO MlpPolicy [256,256] (log_std_init -1) on MSE;
  3. save as a normal PPO zip (directly usable with train_grid4 --init-from
     for retention fine-tuning later);
  4. run the frozen eval on the clone (a2 band) and print the table.

Phase note: the teacher is time-indexed and obs has no clock, but at steady
state each joint's (pos, vel) pair determines the sinusoid phase, so a single
frame is in principle sufficient; the printed clone-vs-teacher tracking error
is the check. If it fails, frame-stacking is the declared fallback (not
implemented here).

Usage (from pengu_mujoco/):
  python rl_grid4/bc_express.py                # full run, ~30-60 min CPU
  python rl_grid4/bc_express.py --episodes 20 --epochs 5   # quick smoke
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

CRANK_BAND = (0.0, 1.9)          # a2: covers the c6 crank command domain
C6 = dict(freq=1.77, hip_phi=270.0, leg_amp=105.0, hip_amp=28.0, hip_off=10.0)
KAPPA = 2.0
T_HOLD, T_TRANSITION = 5.0, 4.0
EP_S = 24.0


def set_c6_params():
    import gait_config as gc
    gc.set_crank_amp(C6["leg_amp"]); gc.set_hip_amp(C6["hip_amp"])
    gc.set_torso_amp(0.0)
    gc.WALK_HIP_OFFSET_DEG = C6["hip_off"]; gc.WALK_HIP_LEAN_DEG = 0.0
    gc.set_walk_freq(C6["freq"])
    gc.PHASE_OFFSET_A_DEG = 0.0; gc.PHASE_OFFSET_B_DEG = 0.0
    gc.PHASE_OFFSET_C_DEG = C6["hip_phi"]; gc.PHASE_OFFSET_D_DEG = C6["hip_phi"]
    gc.PHASE_OFFSET_E_DEG = 0.0
    gc.T_HOLD, gc.T_TRANSITION = T_HOLD, T_TRANSITION
    return gc


def collect(episodes, dart, seed):
    import grid4_rl_env as ge
    from grid4_rl_env import Grid4RLEnv
    from torso_control import TorsoKappaPID
    gc = set_c6_params()
    env = Grid4RLEnv(eval_mode=True, mu_lo=0.1, mu_hi=0.4, episode_s=EP_S,
                     seed=seed, filter_alpha=1.0, action_delay=0,
                     crank_band=CRANK_BAND)
    gc.TORSO_CONTROLLER = TorsoKappaPID(env.model, kappa=KAPPA,
                                        measure_after=T_HOLD + T_TRANSITION + 2.0)
    act_ids = {n: int(a) for n, a in zip(ge.ACT_NAMES, env.aid)}
    rng = np.random.default_rng(seed)
    X, Y = [], []
    n_fell = 0
    n_steps = int(EP_S / env.control_dt)
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed * 10_000 + ep)
        for i in range(n_steps):
            t = i * env.control_dt
            gc.apply_ctrl(env.data, act_ids, t)
            ctrl_des = env.data.ctrl[env.aid].copy()
            a_teacher = np.clip((ctrl_des - env.ctrl_mid) / env.ctrl_half, -1, 1)
            X.append(obs.copy()); Y.append(a_teacher.astype(np.float32))
            a_exec = np.clip(a_teacher + rng.normal(0, dart, 5), -1, 1)
            obs, _r, term, trunc, _ = env.step(a_exec)
            if term:
                n_fell += 1
                break
            if trunc:
                break
    gc.TORSO_CONTROLLER = None
    print(f"[collect] {len(X)} pairs from {episodes} episodes ({n_fell} fell)",
          flush=True)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def train_bc(X, Y, epochs, outdir, seed):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from grid4_rl_env import Grid4RLEnv
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    venv = DummyVecEnv([lambda: Grid4RLEnv(seed=seed, crank_band=CRANK_BAND)])
    model = PPO("MlpPolicy", venv, device="cpu", seed=seed,
                n_steps=64, batch_size=64,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
                verbose=0)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)
    n = len(Xt)
    idx = np.arange(n)
    for epoch in range(epochs):
        np.random.default_rng(seed + epoch).shuffle(idx)
        tot = 0.0
        for lo in range(0, n, 1024):
            b = idx[lo:lo + 1024]
            xb, yb = Xt[b], Yt[b]
            feats = pol.extract_features(xb)
            if isinstance(feats, tuple):
                feats = feats[0]
            latent = pol.mlp_extractor.forward_actor(feats)
            pred = pol.action_net(latent)
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        print(f"[bc] epoch {epoch + 1}/{epochs}  mse {tot / n:.5f}", flush=True)
    path = os.path.join(outdir, "bc_c6.zip")
    model.save(path)
    venv.close()
    return path, tot / n


def tracking_check(ckpt, seed=0):
    """Run the clone open-loop-free (its own feedback) and report speed."""
    from stable_baselines3 import PPO
    from grid4_rl_env import Grid4RLEnv
    model = PPO.load(ckpt, device="cpu")
    out = []
    for mu in (0.1, 0.3):
        env = Grid4RLEnv(eval_mode=True, mu_fixed=mu, episode_s=EP_S, seed=seed,
                         crank_band=CRANK_BAND)
        obs, _ = env.reset(seed=seed)
        n = 0; info = {}
        for i in range(int(EP_S / env.control_dt)):
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(act)
            n += 1
            if term or trunc:
                break
        ep = info.get("ep", {})
        out.append((mu, n * env.control_dt, ep.get("fell"), ep.get("vx"),
                    ep.get("torso_roll_rms_deg"), ep.get("hip_corr")))
        print(f"[clone] mu={mu}: {n * env.control_dt:.1f}s fell={ep.get('fell')} "
              f"vx={ep.get('vx', float('nan')):.3f} "
              f"torso_rms={ep.get('torso_roll_rms_deg', float('nan')):.1f} "
              f"hip_corr={ep.get('hip_corr', float('nan')):+.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dart", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "runs", "bc_c6"))
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    X, Y = collect(a.episodes, a.dart, a.seed)
    np.savez_compressed(os.path.join(a.out, "dataset.npz"), X=X, Y=Y)
    ckpt, mse = train_bc(X, Y, a.epochs, a.out, a.seed)
    print(f"[bc] saved {ckpt} (final mse {mse:.5f})", flush=True)
    tracking_check(ckpt, seed=a.seed)
    if not a.skip_eval:
        subprocess.run([sys.executable,
                        os.path.join(_HERE, "eval_grid4_policy.py"), ckpt,
                        "--repeats", "5", "--crank-band", "0.0", "1.9"],
                       check=False)


if __name__ == "__main__":
    main()
