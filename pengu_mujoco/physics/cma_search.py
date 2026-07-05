"""
cma_search.py - JOINT CMA-ES gait search (no DOF locked), cross-check for the
6-DOF grid in gait_sweep.py. Same model / condition / metrics (reuses
gait_sweep.run_trial), so the two methods can verify each other.

Optimizes all gait DOF together: freq, leg_amp, hip_amp, torso_amp, hip_phi,
torso_phi (25deg forward pitch is fixed via CONDITION). Objective = path-based
per-stride speed, ONLY if the gait is "valid" by Ben's two criteria (feet truly
clear the ground + clean single-support alternation); invalid/fallen -> penalty.

Run from pengu_mujoco/:
  PENGU_MODEL=v3 python physics/cma_search.py [maxfev]
"""
import os
import sys
import csv
import time
import numpy as np
import mujoco
import cma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gait_sweep as gs

# (name, lo, hi) -- the 6 joint DOF (pitch fixed in CONDITION)
VARS = [("freq", 1.0, 2.0), ("leg_amp", 40.0, 110.0), ("hip_amp", 0.0, 20.0),
        ("torso_amp", 0.0, 20.0), ("hip_phi", 0.0, 360.0), ("torso_phi", 0.0, 360.0)]
NAMES = [v[0] for v in VARS]
LO = np.array([v[1] for v in VARS]); HI = np.array([v[2] for v in VARS])


def main():
    maxfev = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    ids = gs.make_ids(model)
    outdir = os.path.join(gs._HERE, "..", "results", "gait_sweep")
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"cma_{gs.MODEL}_{gs.CONDITION['name']}.csv")
    logf = open(log_path, "w", newline="")
    w = csv.writer(logf)
    w.writerow(NAMES + ["valid", "path_speed", "single_frac", "clear_L", "clear_R", "mu_req_p95", "J"])
    nfev = [0]; best = {"J": -1e9}

    def obj(x):
        p = dict(gs.BASE)
        for n, v in zip(NAMES, LO + np.clip(x, 0, 1) * (HI - LO)):
            p[n] = float(v)
        r = gs.run_trial(model, data, ids, p)
        J = r["path_speed"] if (r["valid"] and r["survived"]) else (-1.0 + 0.1 * r["path"])
        nfev[0] += 1
        w.writerow([round(p[n], 3) for n in NAMES] +
                   [r["valid"], r["path_speed"], r["single_frac"], r["clear_L"],
                    r["clear_R"], r["mu_req_p95"], round(J, 4)])
        logf.flush()
        if J > best["J"]:
            best.update(r); best["J"] = J; best["params"] = dict(p)
            print(f"  [{nfev[0]:4d}] NEW BEST J={J:.4f} valid={r['valid']} "
                  f"pathspd={r['path_speed']:.3f} single={r['single_frac']:.2f} mu={r['mu_req_p95']} | "
                  + " ".join(f"{n}={p[n]:.1f}" for n in NAMES))
        return -J

    x0 = np.array([(gs.BASE.get(n, (l + h) / 2) - l) / (h - l) for n, l, h in VARS])
    x0 = np.clip(x0, 0.05, 0.95)
    print(f"# CMA-ES joint search  model={gs.MODEL} cond={gs.CONDITION['name']} vars={NAMES} maxfev={maxfev}")
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"bounds": [0, 1], "maxfevals": maxfev,
                                            "popsize": 12, "verb_disp": 0, "seed": 1})
    es.optimize(obj)
    logf.close()
    bp = best["params"]
    print(f"\n# DONE evals={nfev[0]} wall={(time.time()-t0)/60:.1f}min  log={log_path}")
    print(f"# BEST J={best['J']:.4f} valid={best['valid']} path_speed={best['path_speed']:.3f} "
          f"single_frac={best['single_frac']:.2f} mu_req={best['mu_req_p95']} "
          f"stride L/R={best['stride_L']}/{best['stride_R']} clear L/R={best['clear_L']}/{best['clear_R']}")
    print("# BEST PARAMS: " + "  ".join(f"{n}={bp[n]:.2f}" for n in NAMES))


if __name__ == "__main__":
    main()
