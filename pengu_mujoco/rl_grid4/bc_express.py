"""BC expressibility test: clone the c6 designed gait into the RL policy net.

Round 1 (rml3, 2026-08-22, docs/bc_express_result.md): single-frame BC
regressed to the conditional mean (constant action, hip_corr +0.98, vx 0) —
the 36-dim obs does not identify the teacher's phase, so the test measured
the OBSERVATION, not the network. Two follow-up modes added (C5):

  --clock      append [sin, cos] of the teacher's true phase to obs (38-dim).
               Oracle capability probe: if this clone tracks, network + a2
               band can express c6 and the issue is purely observability.
               NOT a candidate policy (uses privileged input).
  --frames N   stack the last N obs frames (36N-dim). The deployable version
               of the same question under the frozen obs contract + history.

Modes are exclusive. Default (no flag) = original single-frame run.
Clock/frames clones are probe-evaluated in-script (per-mu rollouts); the
frozen eval subprocess only runs for the default mode (36-dim contract).

Usage (from pengu_mujoco/):
  python rl_grid4/bc_express.py --clock
  python rl_grid4/bc_express.py --frames 3
"""
import argparse
import collections as _c
import math
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
OBS_RAW = 36


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


class FeatureMaker:
    """obs -> BC input, per mode. Stateful for frames; reset per episode."""

    def __init__(self, mode, frames=1):
        self.mode = mode
        self.frames = frames
        self.dim = {"raw": OBS_RAW, "clock": OBS_RAW + 2,
                    "frames": OBS_RAW * frames}[mode]
        self._buf = None

    def reset(self):
        self._buf = None

    def __call__(self, obs, t):
        if self.mode == "raw":
            return np.asarray(obs, np.float32)
        if self.mode == "clock":
            ph = 2 * math.pi * C6["freq"] * max(0.0, t - T_HOLD)
            return np.concatenate([obs, [math.sin(ph), math.cos(ph)]]
                                  ).astype(np.float32)
        if self._buf is None:
            self._buf = _c.deque([np.asarray(obs, np.float32)] * self.frames,
                                 maxlen=self.frames)
        else:
            self._buf.append(np.asarray(obs, np.float32))
        return np.concatenate(self._buf)


def collect(episodes, dart, seed, fm):
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
        fm.reset()
        for i in range(n_steps):
            t = i * env.control_dt
            gc.apply_ctrl(env.data, act_ids, t)
            ctrl_des = env.data.ctrl[env.aid].copy()
            a_teacher = np.clip((ctrl_des - env.ctrl_mid) / env.ctrl_half, -1, 1)
            X.append(fm(obs, t)); Y.append(a_teacher.astype(np.float32))
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


def train_bc(X, Y, epochs, outdir, seed, fm):
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from grid4_rl_env import Grid4RLEnv
    torch.manual_seed(seed)
    torch.set_num_threads(4)

    class AugSpace(gym.ObservationWrapper):
        def __init__(self, env, dim):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (dim,),
                                                    np.float32)
            self._dim = dim

        def observation(self, obs):
            out = np.zeros(self._dim, np.float32)
            out[:len(obs)] = obs
            return out

    venv = DummyVecEnv([lambda: AugSpace(
        Grid4RLEnv(seed=seed, crank_band=CRANK_BAND), fm.dim)])
    model = PPO("MlpPolicy", venv, device="cpu", seed=seed,
                n_steps=64, batch_size=64,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-1.0),
                verbose=0)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    Xt = torch.as_tensor(X); Yt = torch.as_tensor(Y)
    n = len(Xt)
    idx = np.arange(n)
    mse = float("nan")
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
        mse = tot / n
        print(f"[bc] epoch {epoch + 1}/{epochs}  mse {mse:.5f}", flush=True)
    path = os.path.join(outdir, "bc_c6.zip")
    model.save(path)
    venv.close()
    return path, mse


def probe_eval(ckpt, fm, mus=(0.1, 0.2, 0.3, 0.4), seeds=3):
    """In-script rollout table (handles clock/frames feature injection).
    Also reports per-dim action std so mean-regression is caught immediately."""
    from stable_baselines3 import PPO
    from grid4_rl_env import Grid4RLEnv
    model = PPO.load(ckpt, device="cpu")
    for mu in mus:
        rows = []
        for s in range(seeds):
            env = Grid4RLEnv(eval_mode=True, mu_fixed=mu, episode_s=EP_S,
                             seed=s, crank_band=CRANK_BAND)
            obs, _ = env.reset(seed=s)
            fm.reset()
            acts = []
            n = 0; info = {}
            for i in range(int(EP_S / env.control_dt)):
                t = i * env.control_dt
                act, _ = model.predict(fm(obs, t), deterministic=True)
                acts.append(act)
                obs, _r, term, trunc, info = env.step(act)
                n += 1
                if term or trunc:
                    break
            ep = info.get("ep", {})
            rows.append((n * env.control_dt, ep.get("fell", float("nan")),
                         ep.get("vx", float("nan")),
                         ep.get("torso_roll_rms_deg", float("nan")),
                         ep.get("hip_corr", float("nan")),
                         float(np.mean(np.std(np.asarray(acts), axis=0)))))
        arr = np.asarray(rows)
        print(f"[probe] mu={mu}: dur {arr[:,0].mean():.1f}s fell {arr[:,1].mean():.1f} "
              f"vx {arr[:,2].mean():+.3f} torso_rms {arr[:,3].mean():.1f} "
              f"hip_corr {arr[:,4].mean():+.2f} act_std {arr[:,5].mean():.3f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--dart", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clock", action="store_true",
                    help="oracle phase input (capability probe)")
    ap.add_argument("--frames", type=int, default=1,
                    help="stack N obs frames (deployable variant)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-eval", action="store_true")
    a = ap.parse_args()
    if a.clock and a.frames > 1:
        raise SystemExit("--clock and --frames are exclusive")
    mode = "clock" if a.clock else ("frames" if a.frames > 1 else "raw")
    fm = FeatureMaker(mode, a.frames)
    out = a.out or os.path.join(_HERE, "runs",
                                {"raw": "bc_c6", "clock": "bc_c6_clock",
                                 "frames": f"bc_c6_f{a.frames}"}[mode])
    os.makedirs(out, exist_ok=True)
    print(f"[mode] {mode} (input dim {fm.dim}) -> {out}", flush=True)

    X, Y = collect(a.episodes, a.dart, a.seed, fm)
    np.savez_compressed(os.path.join(out, "dataset.npz"), X=X, Y=Y)
    ckpt, mse = train_bc(X, Y, a.epochs, out, a.seed, fm)
    print(f"[bc] saved {ckpt} (final mse {mse:.5f})", flush=True)
    probe_eval(ckpt, fm)
    if mode == "raw" and not a.skip_eval:
        subprocess.run([sys.executable,
                        os.path.join(_HERE, "eval_grid4_policy.py"), ckpt,
                        "--repeats", "5", "--crank-band", "0.0", "1.9"],
                       check=False)


if __name__ == "__main__":
    main()
