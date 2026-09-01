"""pick_robust_gait.py — choose a gait for measuring the sim2real gap, not for winning.

The speed champions sit on knife edges: c6's 1.67/95 walks 0.379 m/s while 1.66/95 and
1.67/85 both fall. Landing on a cell like that requires the hardware to reproduce the
command to a precision it does not have (cranks track at 0.91, plus phase lag), so any
hardware-vs-sim difference gets swamped by which neighbouring cell the robot actually
landed in. For a gap measurement the right gait is the opposite: slow, and surrounded on
every axis by cells that also walk.

Selection:
  * pass at the target mu, heading >= HEAD_MIN (walks forward, not sideways)
  * freq <= FREQ_MAX (slow -- also keeps the crank rate inside the measured motor
    envelope of ~420 deg/s)
  * a 5-D neighbourhood score: of the (up to) 10 immediate neighbours along freq,
    hip_phi, leg_amp, hip_amp and hip_off, how many also pass
  * cross-mu survival is reported, not filtered: the lab floor is not the swept mu

    python grid5/pick_robust_gait.py            # c6 (kappa=2) at mu=0.3
    CFG=c3 MU=0.3 python grid5/pick_robust_gait.py
"""
import csv
import math
import os

CFG = os.environ.get("CFG", "c6")
MU = float(os.environ.get("MU", "0.3"))
FREQ_MAX = float(os.environ.get("FREQ_MAX", "1.30"))
HEAD_MIN = float(os.environ.get("HEAD_MIN", "0.90"))
W_MOTOR = 420.0          # measured no-load crank ceiling on this robot (2026-08-28)
TOP = int(os.environ.get("TOP", "12"))

F = f"results/gait_sweep/sweep_grid4_{CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
STEP = {"freq": 0.01, "hip_phi": 10.0, "leg_amp": 10.0, "hip_amp": 4.0, "hip_off": 10.0}
KEYS = list(STEP)


def key(d):
    return tuple(round(float(d[k]), 3) for k in KEYS)


def main():
    by_mu = {}
    with open(F) as fh:
        for d in csv.DictReader(fh):
            mu = round(float(d["mu"]), 3)
            by_mu.setdefault(mu, {})[key(d)] = (
                float(d["pass_rate"]),
                float(d["net_fwd_mean"]),
                float(d["head_mean"] or "nan"),
            )
    cells = by_mu[MU]
    print(f"=== {CFG} at mu={MU}: {len(cells)} cells, "
          f"{sum(1 for v in cells.values() if v[0] > 0)} pass ===")
    print(f"    filters: freq <= {FREQ_MAX}, head_mean >= {HEAD_MIN}, "
          f"crank peak (pi*f*A) <= {W_MOTOR:.0f} deg/s\n")

    cand = []
    for k, (p, net, head) in cells.items():
        if p <= 0 or k[0] > FREQ_MAX:
            continue
        if not (head >= HEAD_MIN):
            continue
        if math.pi * k[0] * k[2] > W_MOTOR:
            continue
        # 5-D neighbourhood: +-1 grid step along each axis
        nb = tot = 0
        for i, ax in enumerate(KEYS):
            for s in (-1, +1):
                kk = list(k)
                kk[i] = round(kk[i] + s * STEP[ax], 3)
                v = cells.get(tuple(kk))
                if v is None:
                    continue
                tot += 1
                nb += 1 if v[0] > 0 else 0
        if tot == 0:
            continue
        cand.append((nb / tot, tot, net, head, k))

    cand.sort(key=lambda x: (-x[0], -x[2]))
    print(f"{'nbhd':>6} {'n':>3} {'net_fwd':>8} {'head':>6}  "
          f"{'freq':>5} {'phi':>4} {'leg':>4} {'hip':>4} {'off':>4}  {'crank':>6}  "
          f"{'mu0.1':>6} {'mu0.5':>6} {'mu0.7':>6}")
    print("-" * 100)
    for frac, tot, net, head, k in cand[:TOP]:
        cross = []
        for m in (0.1, 0.5, 0.7):
            v = by_mu.get(m, {}).get(k)
            cross.append(f"{v[1]:.3f}" if v and v[0] > 0 else "  --  ")
        print(f"{frac:6.2f} {tot:3d} {net:8.3f} {head:6.3f}  "
              f"{k[0]:5.2f} {k[1]:4.0f} {k[2]:4.0f} {k[3]:4.0f} {k[4]:4.0f}  "
              f"{math.pi * k[0] * k[2]:6.0f}  {cross[0]:>6} {cross[1]:>6} {cross[2]:>6}")
    if not cand:
        print("    nothing passed the filters -- loosen FREQ_MAX or HEAD_MIN")


if __name__ == "__main__":
    main()
