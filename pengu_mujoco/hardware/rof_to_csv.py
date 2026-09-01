"""rof_to_csv.py — the DP800 power logs as readable, labelled CSV.

Differences from Downloads/pengudata/rof2csv.py, which should not be used on this data:
  * every physical line is kept. rof2csv.py's regex rejects the samples written while the
    output is off ("0.000mV,0.000uA,0.000uW,") and then numbers time over the SURVIVING
    rows, so p_mu0-12_1 comes out with everything after line 94 shifted 146 periods early
    and p_mu0-45_3 loses 4075 of 4177 rows.
  * unit prefixes are read per field, because mixed lines exist
    ("15.999V, 0.290A,-28.926mW,").
  * each sample is labelled off / idle / walk, told whether it is against the 2 A current
    limit, and given its burst number, so the file can be read without re-deriving any of
    that.

Time: the recorder stores no timestamp and no period. `sample` is the exact row index --
that is the only real time base. `t_s` is `sample * PERIOD_S`, and PERIOD_S is INFERRED
(0.8 s) by matching burst lengths against the mocap walking durations of the three
single-burst files, which give 0.79 / 0.80 / 0.90 s. It is not the DP800's 1 s default.

    python hardware/rof_to_csv.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rof                                   # noqa: E402

SRC = os.environ.get("PENGU_ROF_DIR", "/Users/ben/Downloads/归档/c6_powersupply_data")
OUT = "../HardwareData/cot_mocap_0829/analysis/power_csv"
PERIOD_S = 0.8            # inferred, see module docstring
I_LIMIT = 2.0             # the supply's current limit
V_SET = 16.0

# filename -> (surface friction, run number, the mocap take it pairs with)
PAIR = {"p_mu0-12_1.ROF": (0.12, 1, "mu0.12_COT take 1"),
        "p_mu0-12_2.ROF": (0.12, 2, "mu0.12_COT take 2"),
        "p_mu0-12_3.ROF": (0.12, 3, "mu0.12_COT take 3"),
        "p_mu0-12_4.ROF": (0.12, 4, "mu0.12_COT take 4"),
        "p_mu0-45_1.ROF": (0.45, 1, "mu0.45_COT take 1"),
        "p_mu0-45_2.ROF": (0.45, 2, "mu0.45_COT take 2"),
        "p_mu0-45_3.ROF": (0.45, 3, "mu0.45_COT take 3"),
        "test4.ROF":      (None, None, "bench idle, torque off")}

COLS = ["file", "mu", "run", "paired_mocap_take", "sample", "t_s",
        "volts_V", "amps_A", "watts_W", "watts_VxA",
        "state", "burst", "at_current_limit", "in_constant_current"]


def rows_for(r):
    mu, run, take = PAIR.get(r.name, (None, None, ""))
    bursts = r.bursts()
    burst_of = np.full(r.n, -1)
    for k, (s, e) in enumerate(bursts):
        burst_of[s:e + 1] = k
    out = []
    for i in range(r.n):
        if r.off[i]:
            state = "output_off"
        elif burst_of[i] >= 0:
            state = "walking"
        else:
            state = "idle_holding"
        out.append({
            "file": r.name, "mu": mu if mu is not None else "", "run": run if run else "",
            "paired_mocap_take": take,
            "sample": i, "t_s": round(i * PERIOD_S, 2),
            "volts_V": round(float(r.v[i]), 4),
            "amps_A": round(float(r.a[i]), 4),
            "watts_W": round(float(r.w[i]), 4),
            "watts_VxA": round(float(r.v[i] * r.a[i]), 4),
            "state": state,
            "burst": burst_of[i] if burst_of[i] >= 0 else "",
            "at_current_limit": int(r.rail[i]),
            "in_constant_current": int(r.cc[i]),
        })
    return out, bursts


def main():
    os.makedirs(OUT, exist_ok=True)
    allrows, summary = [], []
    for f in sorted(os.listdir(SRC)):
        if not f.lower().endswith(".rof"):
            continue
        r = rof.Rof(os.path.join(SRC, f))
        rws, bursts = rows_for(r)
        allrows += rws
        with open(os.path.join(OUT, f.replace(".ROF", ".csv")), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(rws)
        mu, run, take = PAIR.get(f, (None, None, ""))
        for k, (s, e) in enumerate(bursts):
            b = r.burst_stats(s, e)
            summary.append(dict(
                file=f, mu=mu if mu is not None else "", run=run if run else "",
                paired_mocap_take=take, burst=k,
                sample_first=s, sample_last=e, n_samples=b["n"],
                duration_s=round(b["n"] * PERIOD_S, 1),
                mean_W=round(b["P_W"], 2), sem_W=round(b["P_sem_W"], 2),
                peak_W=round(float(r.w[s:e + 1].max()), 2),
                min_V=round(b["v_min"], 2), max_A=round(b["a_max"], 3),
                pct_at_current_limit=round(100 * b["rail_frac"], 1),
                pct_in_constant_current=round(100 * b["cc_frac"], 1)))
        idle = r.idle_mask()
        print(f"{f:18s} {r.n:5d} lines  {len(bursts)} burst(s)  "
              f"idle {np.median(r.w[idle]) if idle.any() else float('nan'):5.2f} W  "
              f"off {int(r.off.sum()):4d}")

    with open(os.path.join(OUT, "all_power.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(allrows)
    with open(os.path.join(OUT, "burst_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)

    readme = os.path.join(OUT, "README.md")
    with open(readme, "w") as fh:
        fh.write(f"""# Power logs as CSV — DP800 recorder, 2026-08-29

One CSV per `.ROF`, plus `all_power.csv` (everything) and `burst_summary.csv`
(one row per walking burst).

## Columns

| column | meaning |
|---|---|
| `file` | source .ROF |
| `mu` | surface friction, from the filename |
| `run` | run number within that surface |
| `paired_mocap_take` | the OptiTrack take this pairs with (by filename index) |
| `sample` | row index in the original file — **the only exact time base** |
| `t_s` | `sample x {PERIOD_S} s`. The period is **inferred**, not stored: the three
  single-burst files give 0.79 / 0.80 / 0.90 s against their mocap walking durations.
  The DP800 default is 1 s and was evidently not used. |
| `volts_V`, `amps_A`, `watts_W` | as the meter reported them, in volts, amps, watts |
| `watts_VxA` | `volts x amps`, for comparison. Per sample it differs from `watts_W` by
  up to 6 W because the meter samples the three fields at different instants inside one
  record period while the load swings 5→32 W at the gait frequency. The **means** agree
  to 0.01–0.26 W. Use `watts_W`. |
| `state` | `output_off` (supply off, reads 0), `idle_holding` (motors powered and holding,
  ~5.0 W), `walking` (inside a current burst, 16–22 W) |
| `burst` | index of the walking burst this sample belongs to |
| `at_current_limit` | 1 if amps >= {I_LIMIT} — the supply is in constant-current and
  **clipping the robot's peak demand** |
| `in_constant_current` | 1 if volts < 15.95, i.e. the supply left constant-voltage. An
  independent indicator of the same thing; voltage sags as low as 9.94 V |

## What to look at

Peak power reads a suspiciously round 32.00 W = {V_SET} V x {I_LIMIT} A in every walking
run. That is the set-point product, not delivered power: **3–26% of the samples inside a
burst are against the current limit**, so any energy integrated from these logs is a
**lower bound**. Raising the limit and re-running is the only fix.

Idle draw with the motors powered and holding is ~5.0 W. With torque off (`test4.ROF`)
it is 3.2 W. Walking is 16–22 W.
""")
    print(f"\nwrote {len(allrows)} rows, {len(summary)} bursts -> {OUT}/")


if __name__ == "__main__":
    main()
