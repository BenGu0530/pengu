"""region_cells.py — build the two targeted cell lists for the GRID-6 c1 campaign.

The full lattice is 1,036,800 cells and 3.45 days on this box. Almost all of it is
known-dead: pooled over friction, c1's passing fraction runs from 49% at hip_phi 270
down to under 0.7% for every value from 70 to 220 (GRID-4 c1, 1.8 M rows). Scanning
that dead arc at 0.01 Hz resolution buys nothing, so this campaign spends its time in
two passes instead.

  stage 1  the robust region at coarse frequency
           freq 1.25-2.00 @0.05, hip_phi 230->40 through zero, everything else full.
           103,680 cells, 414,720 rows, about 8 h on 11 shards.

  stage 2  written later by select_stage2.py, not here: the survivors of stage 1 that
           also fit the motors, refined to 0.01 Hz.

Axis choices and the evidence for each:

  freq      1.25-2.00 @0.05. Passing cells in the GRID-5 c1 partial span 1.21-2.00 with
            p5-p95 at 1.23-1.90. Every value stays on the GRID-5 0.01 lattice, so any
            cell here can be compared one-for-one against the slide-tuned c1 map.
  hip_phi   230,240..350,0,10,20,30,40 (18 of 36). One connected window through zero;
            the excluded arc 50-220 holds 0.4% of all passing cells.
  leg_amp   all 10. Passers are spread across the whole 75-165 range and the champions
            sit at the top of it.
  hip_amp   all 6. Only six values; trimming saves nothing worth the risk.
  hip_off   all 6. The least-known axis -- the GRID-5 c1 partial covers only 0 and 10,
            and GRID-4 puts the best pass rates at 20-30. Not a place to guess.

Nothing here encodes the motor limit. The sweep records every cell and the 354 deg/s
envelope is applied afterwards, so it can be re-drawn without re-simulating.

    python grid6/region_cells.py            -> grid6/cells_c1r.csv
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FREQS = [round(1.25 + 0.05 * i, 2) for i in range(16)]          # 1.25 .. 2.00
HIP_PHIS = [230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 310.0, 320.0,
            330.0, 340.0, 350.0, 0.0, 10.0, 20.0, 30.0, 40.0]   # the window, through 0
LEG_AMPS = [75.0, 85.0, 95.0, 105.0, 115.0, 125.0, 135.0, 145.0, 155.0, 165.0]
HIP_AMPS = [12.0, 16.0, 20.0, 24.0, 28.0, 32.0]
HIP_OFFS = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
N_MU = 4


def main():
    out = os.path.join(HERE, "cells_c1r.csv")
    n = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["freq", "hip_phi", "leg_amp", "hip_amp", "hip_off"])
        # same nesting as the full sweep's cells(): hip_off outermost. Shards stride over
        # this order, so keeping it means a shard's workload mixes cheap and expensive
        # cells the same way the full sweep's does.
        for ho in HIP_OFFS:
            for f in FREQS:
                for hp in HIP_PHIS:
                    for la in LEG_AMPS:
                        for ha in HIP_AMPS:
                            w.writerow([f, hp, la, ha, ho])
                            n += 1
    full = 80 * 36 * 10 * 6 * 6
    print(f"wrote {out}")
    print(f"  {n:,} cells x {N_MU} mu = {n * N_MU:,} rows "
          f"({100 * n / full:.1f}% of the full {full:,}-cell lattice)")
    print(f"  freq {FREQS[0]}-{FREQS[-1]} @0.05 ({len(FREQS)})   "
          f"hip_phi {len(HIP_PHIS)} of 36   leg {len(LEG_AMPS)}  hip {len(HIP_AMPS)}  "
          f"off {len(HIP_OFFS)}")


if __name__ == "__main__":
    main()
