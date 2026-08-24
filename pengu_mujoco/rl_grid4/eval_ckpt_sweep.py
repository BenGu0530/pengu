"""Checkpoint-sweep evaluation: is final.zip the right checkpoint at all?

Machine D found late-training vx oscillates between fast/fragile and
slow/safe attractors, so the last checkpoint may be far from the best one.
This sweeps EVERY saved checkpoint of the given runs through the frozen eval
and reports best-by-eval vs final.

Selection metric (fixed, stated): primary = total pass count over the mu
grid; tie-break = mean net_fwd over all trials. No other ranking.

Usage (from pengu_mujoco/), one line per machine:
  python rl_grid4/eval_ckpt_sweep.py rl_grid4/runs/e2/s*/stageB --repeats 3
Optional: --mus 0.1,0.2,0.3,0.4  --crank-band MID HALF (a2 runs only)
Output: <run>/ckpt_sweep.csv per run + a printed summary table.
"""
import argparse
import csv
import glob
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def ckpt_steps(path):
    base = os.path.basename(path)
    if base == "final.zip":
        return 10 ** 9
    return int(base.split("_")[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs containing ckpts/")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--mus", default="0.1,0.2,0.3,0.4")
    ap.add_argument("--crank-band", nargs=2, type=float, default=None)
    ap.add_argument("--no-slew", action="store_true",
                    help="disable the sv1 servo slew clamp (legacy sv0 repro)")
    a = ap.parse_args()

    summary = []
    for run in a.runs:
        run = run.rstrip("/")
        zips = sorted(glob.glob(os.path.join(run, "ckpts", "*.zip")),
                      key=ckpt_steps)
        if not zips:
            print(f"[skip] no ckpts in {run}")
            continue
        rows = []
        for z in zips:
            out = z + ".evaltmp.csv"
            cmd = [sys.executable, os.path.join(_HERE, "eval_grid4_policy.py"),
                   z, "--repeats", str(a.repeats), "--mus", a.mus, "--out", out]
            if a.crank_band:
                cmd += ["--crank-band", str(a.crank_band[0]), str(a.crank_band[1])]
            if a.no_slew:
                cmd += ["--no-slew"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            with open(out) as f:
                trials = list(csv.DictReader(f))
            os.remove(out)
            n_pass = sum(int(t["pass"]) for t in trials)
            mean_nf = sum(float(t["net_fwd"]) for t in trials) / max(1, len(trials))
            steps = ckpt_steps(z)
            label = "final" if steps == 10 ** 9 else f"{steps // 1000}k"
            rows.append({"ckpt": label, "steps": steps, "n_pass": n_pass,
                         "mean_net_fwd": round(mean_nf, 4)})
            print(f"  {run} {label:>7}: pass {n_pass:2d}/"
                  f"{len(trials)}  net_fwd {mean_nf:+.3f}", flush=True)
        outcsv = os.path.join(run, "ckpt_sweep.csv")
        with open(outcsv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ckpt", "steps", "n_pass",
                                              "mean_net_fwd"])
            w.writeheader(); w.writerows(rows)
        best = max(rows, key=lambda r: (r["n_pass"], r["mean_net_fwd"]))
        final = next((r for r in rows if r["ckpt"] == "final"), rows[-1])
        summary.append((run, best, final))
        print(f"[{run}] best={best['ckpt']} (pass {best['n_pass']}, "
              f"nf {best['mean_net_fwd']}) vs final (pass {final['n_pass']}, "
              f"nf {final['mean_net_fwd']}) -> {outcsv}", flush=True)

    print("\n=== best-vs-final summary ===")
    for run, best, final in summary:
        gain = best["n_pass"] - final["n_pass"]
        print(f"{run}: best {best['ckpt']} pass {best['n_pass']} "
              f"(final {final['n_pass']}, delta {gain:+d}), "
              f"net_fwd {best['mean_net_fwd']} vs {final['mean_net_fwd']}")


if __name__ == "__main__":
    main()
