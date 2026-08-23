"""Is the c6 champion a maximum of the frozen RL reward, or just a point on it?

Perturbs each of c6's five gait parameters by +-10% and +-25%, one at a time,
replays each through Grid4RLEnv's reward accounting exactly as
score_designed_gait does, and reports the walk-window total per step against
unperturbed c6. If any single-parameter step earns MORE, c6 is not a local
maximum of this reward in that direction.

Usage (from pengu_mujoco/):
  python rl_grid4/probe_c6_localmax.py --mu 0.1 --seeds 3
"""
import argparse, os, sys, itertools
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

BASE = dict(freq=1.77, hip_phi=270.0, leg_amp=105.0, hip_amp=28.0, hip_off=10.0)
FRACS = (-0.25, -0.10, +0.10, +0.25)


def one(job):
    key, frac, mu, seed = job
    import score_designed_gait as S
    p = dict(BASE)
    if key is not None:
        p[key] = BASE[key] * (1.0 + frac)
    S.C6 = p
    try:
        r = S.run_designed(mu, seed=seed)
    except Exception as e:
        return (key, frac, seed, float("nan"), float("nan"), str(e)[:40])
    w = r.get("walk", {})
    tot = sum(v for k, v in w.items() if k.startswith("r_"))
    return (key, frac, seed, tot, w.get("vx", float("nan")), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    jobs = [(None, 0.0, a.mu, s) for s in range(a.seeds)]
    jobs += [(k, f, a.mu, s) for k in BASE for f in FRACS for s in range(a.seeds)]
    print(f"mu={a.mu}  {len(jobs)} rollouts, {a.workers} workers")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, jobs))
    agg = {}
    for key, frac, seed, tot, vx, err in res:
        agg.setdefault((key, frac), []).append((tot, vx, err))
    import statistics as st
    def mean(xs): 
        xs = [x for x in xs if x == x]
        return st.mean(xs) if xs else float("nan")
    b = mean([t for t, v, e in agg[(None, 0.0)]])
    bv = mean([v for t, v, e in agg[(None, 0.0)]])
    print(f"\n{'param':<10}{'frac':>7}{'value':>10}{'total/step':>12}{'delta':>9}{'vx':>8}")
    print(f"{'c6 (base)':<10}{'':>7}{'':>10}{b:>12.3f}{'':>9}{bv:>8.3f}")
    better = []
    for k in BASE:
        for f in FRACS:
            t = mean([x for x, v, e in agg[(k, f)]])
            v = mean([v for x, v, e in agg[(k, f)]])
            d = t - b
            flag = "  <-- HIGHER" if d > 0 else ""
            if d > 0: better.append((k, f, d))
            print(f"{k:<10}{f:>+7.0%}{BASE[k]*(1+f):>10.2f}{t:>12.3f}{d:>+9.3f}{v:>8.3f}{flag}")
    print()
    if better:
        print(f"{len(better)} of {len(BASE)*len(FRACS)} single-parameter steps earn MORE than c6:")
        for k, f, d in sorted(better, key=lambda x: -x[2]):
            print(f"   {k} {f:+.0%}  {d:+.3f}/step")
    else:
        print(f"No single-parameter step of +-10%/+-25% beats c6 "
              f"({len(BASE)*len(FRACS)} probes) -- c6 is a local max along these axes.")


if __name__ == "__main__":
    main()
