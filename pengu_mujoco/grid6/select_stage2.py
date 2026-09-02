"""select_stage2.py — pick what stage 2 refines, from what stage 1 measured.

Stage 1 scanned c1's robust region at 0.05 Hz. Stage 2 goes back over the survivors at
0.01 Hz. Three filters decide which coarse cells earn that, and all three are applied
here rather than being baked into stage 1's grid, so any of them can be changed and
re-selected without re-simulating anything.

  1. PASSES        pass_rate > 0.5, i.e. the robot stayed upright, faced where it went
                   (heading > 0.5) and advanced faster than 0.05 m/s. Unchanged from
                   GRID-4, so the two campaigns stay comparable.

  2. FITS THE MOTORS   peak crank rate pi*f*A_leg and peak hip rate 2*pi*f*A_hip must
                   both stay under 354 deg/s. Measured on the robot 2026-08-30: twelve
                   points, air and ground pooled, commanded 424-613 deg/s, every one
                   executed at 354 +- 4. Below 380 the servo tracks at 0.99 with a
                   constant 19-25 ms lag; above it the amplitude ratio collapses
                   0.93 -> 0.81 -> 0.69 and the lag grows 39 -> 63 -> 88 ms. A gait
                   above the ceiling is not the gait the map selected, so refining it
                   to 0.01 Hz would be false precision.

  3. IS ROBUST     its neighbours pass too. Neighbourhood = +-1 step in frequency
                   (+-0.05) and +-1 step in hip_phi (+-10, wrapping through zero), same
                   leg_amp / hip_amp / hip_off / mu. A cell whose neighbours fail is a
                   spike, and a spike that survives only at one grid point is not
                   something the hardware can be aimed at.

What stage 2 then sweeps is the frequency neighbourhood of each survivor: the nine
values from f-0.04 to f+0.04 that stage 1 skipped. The coarse point itself is already
measured and is not repeated.

    python grid6/select_stage2.py            -> grid6/cells_c1f.csv
env: MIN_NBRS (default 4 of 4), CEILING (354.0), FMIN/FMAX clamp for the refinement.
"""
import csv
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CEILING = float(os.environ.get("CEILING", "354.0"))     # deg/s, measured
MIN_NBRS = int(os.environ.get("MIN_NBRS", "4"))         # of the 4 neighbours
FMIN, FMAX = 1.00, 2.00
STAGE1 = os.path.join(ROOT, "results", "gait_sweep",
                      "sweep_grid6_c1_c1r_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")


def phi_step(p, d):
    return float((p + d) % 360.0)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else STAGE1
    if not os.path.exists(src):
        raise SystemExit(f"stage 1 output not found: {src}")

    passed = set()          # (f, phi, leg, hip, off, mu) that passed
    rows = 0
    with open(src) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for r in rd:
            if len(r) < 9:
                continue
            rows += 1
            if float(r[6]) > 0.5:
                passed.add((round(float(r[0]), 2), round(float(r[1]), 0),
                            round(float(r[2]), 0), round(float(r[3]), 0),
                            round(float(r[4]), 0), round(float(r[5]), 1)))
    print(f"stage 1: {rows:,} rows, {len(passed):,} passing")

    # filter 2 + 3
    keep, lost_motor, lost_nbr = [], 0, 0
    for c in sorted(passed):
        f, phi, leg, hip, off, mu = c
        if math.pi * f * leg > CEILING or 2 * math.pi * f * hip > CEILING:
            lost_motor += 1
            continue
        n = 0
        for df in (-0.05, 0.05):
            if (round(f + df, 2), phi, leg, hip, off, mu) in passed:
                n += 1
        for dp in (-10.0, 10.0):
            if (f, phi_step(phi, dp), leg, hip, off, mu) in passed:
                n += 1
        if n < MIN_NBRS:
            lost_nbr += 1
            continue
        keep.append(c)
    print(f"  within the {CEILING:.0f} deg/s envelope: {len(passed) - lost_motor:,} "
          f"({lost_motor:,} dropped)")
    print(f"  and robust ({MIN_NBRS}/4 neighbours pass): {len(keep):,} "
          f"({lost_nbr:,} dropped as spikes)")

    # the refinement grid: the frequencies stage 1 stepped over, around each survivor.
    # mu is dropped here -- the sweep runs every mu for every cell it is given, and a
    # cell that survives at one friction level is worth refining at all of them.
    cells = set()
    for f, phi, leg, hip, off, mu in keep:
        for k in range(-4, 5):
            if k == 0:
                continue                       # the coarse point is already measured
            ff = round(f + 0.01 * k, 2)
            if FMIN <= ff <= FMAX and math.pi * ff * leg <= CEILING:
                cells.add((ff, phi, leg, hip, off))

    out = os.path.join(HERE, "cells_c1f.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"])
        for c in sorted(cells):
            w.writerow(list(c))
    print(f"\nwrote {out}")
    print(f"  {len(cells):,} cells x 4 mu = {len(cells) * 4:,} rows")
    if not cells:
        print("  NOTHING SURVIVED. Stage 2 has no work: either no cell in the robust\n"
              "  region fits the 354 deg/s ceiling, or every one that does is a spike.\n"
              "  That is a result, not a failure -- report it and stop.")
    else:
        by = defaultdict(int)
        for f, phi, leg, hip, off in cells:
            by[(leg, round(f, 2))] += 1
        print(f"  leg_amp values present: {sorted({c[2] for c in cells})}")
        print(f"  freq span: {min(c[0] for c in cells):.2f}-{max(c[0] for c in cells):.2f}")


if __name__ == "__main__":
    main()
