#!/usr/bin/env python
"""Run ONE hip_phi slice of a GRID-5 config (portable port of delta/c2_driver.py).

Same mechanism: grid5_sweep is reused byte-identical; after import we shrink
HIP_PHIS to one value and retag, so the slice gets its own CSV + manifest and the
bitmap resume indexes the restricted grid consistently (loader and loop share the
same axis arrays — this is the bitmap-safe way to slice; filtering cells() while
keeping the full-config CSV is NOT safe under bitmap resume).

    CONFIG=c2 python slice_phi.py <phi> [count|initcsv|--rows]   # + N_SHARDS/SHARD_ID

Output: results/gait_sweep/sweep_grid5_<cfg>_phiNNN_<axes>.csv — identical naming
to the NCSA Delta slices, so delta/merge_phi.py rebuilds the canonical config CSV
from any mix of Delta- and fleet-produced slices.
"""
import os, sys

ROWS_PER_SLICE = 115_200
VALID_PHIS = [float(p) for p in range(0, 360, 10)]

def die(m):
    sys.stderr.write(f"slice_phi: {m}\n"); raise SystemExit(2)

if len(sys.argv) < 2:
    die("usage: CONFIG=cN slice_phi.py <phi> [count|initcsv|--rows]")
try:
    PHI = float(sys.argv[1])
except ValueError:
    die(f"phi must be a number, got {sys.argv[1]!r}")
if PHI not in VALID_PHIS:
    die(f"phi={PHI} not on the grid (0,10,...,350)")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grid5_sweep as g5

if g5.SMOKE:
    if PHI not in set(float(x) for x in g5.HIP_PHIS):
        die(f"smoke grid has phis {list(g5.HIP_PHIS)}")
else:
    assert list(g5.MUS) == [0.1, 0.3, 0.5, 0.7], \
        f"slice tooling assumes the 4-mu axis, got {list(g5.MUS)}"

g5.HIP_PHIS = np.array([PHI])                       # 36 -> 1 (the whole trick)
g5.TAG = f"grid5_{g5.CONFIG}_phi{int(PHI):03d}" + ("_smoke" if g5.SMOKE else "")

sys.argv = [sys.argv[0]] + [a for a in sys.argv[2:] if a != "--rows"]
if "--rows" in sys.argv[1:] or (len(sys.argv) > 1 and sys.argv[1] == "--rows"):
    pass
if len(sys.argv) > 1 and sys.argv[1] not in ("count", "initcsv"):
    die(f"unknown subcommand {sys.argv[1]!r}")

if __name__ == "__main__":
    g5.main()
