"""Frozen evaluation for GRID-4 RL policies (comparability contract).

Protocol: 24 s trials, mu in {0.1, 0.2, 0.3, 0.4} exact +-5% jitter, 5 seeded
repeats, eval_mode env (pose jitter only -- no obs noise / delay / DR).
Measurement window starts at T_SETTLE (2 s) to skip the standing transient.

Per trial: survive, net_fwd_speed (net displacement projected on the mean
body-heading over the window / window), heading_align (cos between net
displacement and mean heading), torso roll RMS, root roll RMS, effective kappa
(least-squares slope of torso world roll on root/hip-axis roll), single_frac,
mean stance-foot scrub.  Pass rule (frozen): survive AND heading_align > 0.5
AND net_fwd > 0.05.

Three-tier classification per checkpoint (engineering readout; thresholds are
reported alongside so they can be re-cut):
  (1) no-walk      -- pass_rate < 0.6 at any mu   -> NO standing on torso claims
  (2) walk, torso silent -- walks, median torso roll RMS < --torso-thresh (deg)
  (3) walk, torso active -- walks, median torso roll RMS >= threshold

Baselines: results/grid4_report/c3|c6/topupK5.csv (designed sweep, K=5;
exists at mu=0.1 and 0.3 -- envelope comparison lives on those two columns).

Usage (from pengu_mujoco/):
  python rl/eval_grid4_policy.py rl/runs/grid4/e2_s0/final.zip [more.zip ...]
      [--mus 0.1,0.2,0.3,0.4] [--repeats 5] [--dur 24] [--torso-thresh 5]
      [--out eval.csv]
"""
import argparse
import csv
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

T_SETTLE = 2.0


def run_trial(model_sb3, env, mu, trial_seed, dur):
    obs, _ = env.reset(seed=trial_seed)
    n_steps = int(dur / env.control_dt)
    t_settle_i = int(T_SETTLE / env.control_dt)

    xy0 = None
    heads = []
    torso_rolls = []
    root_rolls = []
    single = 0
    scrub_sum = 0.0
    survived = True
    i = 0
    for i in range(n_steps):
        act, _ = model_sb3.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(act)
        if i == t_settle_i:
            xy0 = env.data.xpos[env.root][:2].copy()
        if i >= t_settle_i:
            R = env.data.xmat[env.root].reshape(3, 3)
            f = R @ env.fwd_local
            fh = np.array([f[0], f[1]])
            n = np.linalg.norm(fh)
            if n > 1e-9:
                heads.append(fh / n)
            torso_rolls.append(env.torso_roll())
            root_rolls.append(env.root_roll())
            con = env._foot_contacts()
            if sum(con.values()) == 1:
                single += 1
        if term:
            survived = False
            break
        if trunc:
            break
    n_meas = max(1, len(torso_rolls))

    xy1 = env.data.xpos[env.root][:2].copy()
    if xy0 is None:
        xy0 = xy1
    disp = xy1 - xy0
    window = max(1e-6, (min(i + 1, n_steps) - t_settle_i) * env.control_dt)
    head_mean = np.mean(heads, axis=0) if heads else np.array([0.0, 1.0])
    hm = head_mean / max(1e-9, np.linalg.norm(head_mean))
    net_fwd = float(disp @ hm) / window
    dn = np.linalg.norm(disp)
    heading_align = float(disp @ hm) / dn if dn > 1e-6 else 0.0

    hip = env.hip_alternation()
    tr = np.asarray(torso_rolls)
    rr = np.asarray(root_rolls)
    var = float(np.var(rr))
    eff_kappa = float(np.cov(tr, rr)[0, 1] / var) if var > 1e-8 and len(tr) > 10 else float("nan")
    passed = survived and heading_align > 0.5 and net_fwd > 0.05
    return {
        "mu": round(mu, 4), "survived": int(survived), "pass": int(passed),
        "net_fwd": round(net_fwd, 4), "heading_align": round(heading_align, 4),
        "torso_roll_rms_deg": round(math.degrees(float(np.sqrt(np.mean(tr ** 2)))), 2),
        # RMS alone cannot tell a swing from a held lean (RMS^2 = mean^2 + var),
        # and a policy can post a high RMS while standing still. gen01's
        # kernel_off scored torso_roll_rms 41.3 and eff_kappa 4.22 -- which reads
        # as strong torso use -- with net_fwd 0.0094 and 0/20 pass: a parked lean.
        # The mean and the rate separate the two; frames confirmed it.
        "torso_roll_mean_deg": round(math.degrees(float(np.mean(tr))), 2),
        "torso_roll_rate_rms_dps": round(math.degrees(float(np.sqrt(np.mean(
            (np.diff(tr) / env.control_dt) ** 2)))), 1) if len(tr) > 1 else "",
        "root_roll_rms_deg": round(math.degrees(float(np.sqrt(np.mean(rr ** 2)))), 2),
        "eff_kappa": round(eff_kappa, 3) if np.isfinite(eff_kappa) else "",
        "single_frac": round(single / n_meas, 3),
        "hip_diff_rms_deg": round(hip["hip_diff_rms_deg"], 2),
        "hip_corr": round(hip["hip_corr"], 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--mus", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dur", type=float, default=24.0)
    ap.add_argument("--torso-thresh", type=float, default=5.0,
                    help="deg, torso roll RMS cut between silent/active")
    ap.add_argument("--out", default=None)
    ap.add_argument("--trial-seed-base", type=int, default=0,
                    help="offset for trial seeds; use a nonzero value for a "
                         "confirmation eval independent of a selection eval")
    ap.add_argument("--crank-band", nargs=2, type=float, default=None,
                    metavar=("MID", "HALF"),
                    help="must match the band the policy was trained with")
    ap.add_argument("--no-slew", action="store_true",
                    help="disable the sv1 servo slew clamp (legacy sv0 repro)")
    a = ap.parse_args()
    mus = [float(x) for x in a.mus.split(",")]

    from stable_baselines3 import PPO
    from grid4_rl_env import Grid4RLEnv

    rows = []
    for ckpt in a.ckpts:
        model = PPO.load(ckpt, device="cpu")
        per_mu_pass = {}
        torso_rmss = []
        for mu in mus:
            kw = dict(slew_vmax=0.0) if a.no_slew else {}
            env = Grid4RLEnv(eval_mode=True, mu_fixed=mu, episode_s=a.dur, seed=0,
                             crank_band=tuple(a.crank_band) if a.crank_band else None,
                             **kw)
            trials = []
            for rep in range(a.repeats):
                r = run_trial(model, env, mu, trial_seed=a.trial_seed_base + 1000 * rep + int(mu * 100), dur=a.dur)
                r["ckpt"] = os.path.relpath(ckpt, _HERE)
                r["rep"] = rep
                trials.append(r)
                rows.append(r)
            pr = np.mean([t["pass"] for t in trials])
            per_mu_pass[mu] = pr
            torso_rmss += [t["torso_roll_rms_deg"] for t in trials if t["pass"]]
            nf = [t["net_fwd"] for t in trials]
            print(f"  mu={mu:.1f}  pass={pr:.1f}  net_fwd mean={np.mean(nf):.3f} "
                  f"min={np.min(nf):.3f}  torso_rms med="
                  f"{np.median([t['torso_roll_rms_deg'] for t in trials]):.1f}deg", flush=True)
        walks = all(p >= 0.6 for p in per_mu_pass.values())
        if not walks:
            tier = "1 no-walk (zero standing on torso claims)"
        else:
            med = float(np.median(torso_rmss)) if torso_rmss else 0.0
            tier = (f"3 walk, torso ACTIVE (med RMS {med:.1f}deg >= {a.torso_thresh})"
                    if med >= a.torso_thresh else
                    f"2 walk, torso silent (med RMS {med:.1f}deg < {a.torso_thresh})")
        print(f"[{os.path.basename(os.path.dirname(ckpt))}] tier {tier}", flush=True)

    base = os.path.dirname(os.path.abspath(a.ckpts[0]))
    if os.path.basename(base) == "ckpts":               # run/ckpts/x.zip -> run root
        base = os.path.dirname(base)
    out = a.out or os.path.join(base, "eval_frozen.csv")
    fields = ["ckpt", "rep", "mu", "survived", "pass", "net_fwd", "heading_align",
              "torso_roll_rms_deg", "torso_roll_mean_deg", "torso_roll_rate_rms_dps",
              "root_roll_rms_deg", "eff_kappa", "single_frac",
              "hip_diff_rms_deg", "hip_corr"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} trials)", flush=True)


if __name__ == "__main__":
    main()
