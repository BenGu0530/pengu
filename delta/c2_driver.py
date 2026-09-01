#!/usr/bin/env python3
"""
Run ONE hip_phi slice of GRID-5 config C2, on NCSA Delta.

Why a driver instead of an edited sweep
---------------------------------------
`grid5_sweep.py` is reused BYTE-IDENTICAL. Its `cells()` (:139-145) and `main()`
(:210+) read the axis arrays from module globals at call time, so restricting the
sweep to one hip_phi value is a two-line patch applied after import -- no fork of
the sweep logic, no drift from the code that produced the existing GRID-5 data.

    g5.HIP_PHIS = np.array([phi])     # 36 values -> 1
    g5.TAG      = "grid5_c2_phiNNN"   # so each slice gets its own CSV

`check_manifest()` (:194-210) validates protocol/config/K/mujoco_version/slip and
NOT the axis set, so a restricted run is accepted.

One CSV per slice, on purpose
-----------------------------
`gs._load_done()` re-reads the whole CSV in EVERY shard at startup. With 36 slices
in one file that is a ~600MB read x 128 shards = ~77GB of Lustre traffic per job.
Per-slice files are ~18MB. Merge afterwards with delta/merge_phi.py.

Isolation
---------
Refuses to run unless DELTA_ISOLATED_TREE exists at the tree root, and asserts every
imported grid5 module resolves inside this tree. A copy of this file dropped into
the live repo cannot write to the live results directory.

Usage
-----
    python c2_driver.py <phi>                # run the slice (honours N_SHARDS/SHARD_ID)
    python c2_driver.py <phi> count          # print cell/row counts, write nothing
    python c2_driver.py <phi> initcsv        # create header + manifest, then exit
    python c2_driver.py <phi> --rows         # rows already done for this slice
"""

import os
import sys

ROWS_PER_SLICE = 115_200          # 80 freq x 10 leg_amp x 6 hip_amp x 6 hip_off x 4 mu
VALID_PHIS = [float(p) for p in range(0, 360, 10)]      # grid5_sweep.py:80

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
GRID5 = os.path.join(REPO, "pengu_mujoco", "grid5")


def die(msg):
    sys.stderr.write(f"c2_driver: {msg}\n")
    raise SystemExit(2)


def guard_isolated():
    """Structural refusal to run anywhere but the duplicated tree."""
    sentinel = os.path.join(REPO, "DELTA_ISOLATED_TREE")
    if not os.path.exists(sentinel):
        die(f"missing {sentinel}\n"
            f"  This driver only runs inside the isolated duplicate tree.\n"
            f"  It must never write into the live PenguMujoco repo, where the\n"
            f"  laptop fleet's sweep is running (c8 as of 2026-09-01) and its\n"
            f"  revivers would pick up stray files.")
    if not os.path.isdir(GRID5):
        die(f"no grid5 sources at {GRID5}")


def parse_phi(argv):
    if len(argv) < 2:
        die("usage: c2_driver.py <phi> [count|initcsv|--rows]")
    try:
        phi = float(argv[1])
    except ValueError:
        die(f"phi must be a number, got {argv[1]!r}")
    if phi not in VALID_PHIS:
        die(f"phi={phi} is not on the grid. Valid: 0,10,...,350")
    return phi


def slice_csv_path(phi):
    """Where this slice's CSV lives -- derived exactly as grid5_sweep.main() does."""
    axnames = "freq_hip_phi_leg_amp_hip_amp_hip_off_mu"
    return os.path.join(REPO, "pengu_mujoco", "results", "gait_sweep",
                        f"sweep_grid5_c2_phi{int(phi):03d}_{axnames}.csv")


def rows_done(phi):
    p = slice_csv_path(phi)
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)          # minus header


def main():
    guard_isolated()
    phi = parse_phi(sys.argv)
    sub = sys.argv[2] if len(sys.argv) > 2 else None

    # Cheap query paths that must not import mujoco (used by c2_ctl.sh in a loop).
    if sub == "--rows":
        print(rows_done(phi))
        return
    if sub == "--csv":
        print(slice_csv_path(phi))
        return

    # grid5_sweep.py reads CONFIG at import time (:64) and asserts it is c1..c10.
    os.environ["CONFIG"] = "c2"

    sys.path.insert(0, GRID5)
    import numpy as np
    import grid5_sweep as g5

    # The sweep's own guards (grid5_sweep.py:48-50) only check that the string
    # "grid5" appears in the module path -- true of the live tree too. Assert the
    # stronger property: everything came from THIS tree.
    for name, mod in (("grid5_sweep", g5), ("gait_config", g5.gc), ("gait_sweep", g5.gs)):
        p = os.path.abspath(mod.__file__)
        if not p.startswith(REPO + os.sep):
            die(f"{name} was imported from OUTSIDE this tree:\n  {p}\n"
                f"  expected under {REPO}")

    if g5.CONFIG != "c2":
        die(f"CONFIG is {g5.CONFIG!r}, expected 'c2'")

    # --- the whole patch ---
    g5.HIP_PHIS = np.array([phi])
    g5.TAG = f"grid5_c2_phi{int(phi):03d}"

    combos = list(g5.cells())
    n_rows = len(combos) * len(g5.MUS)
    if n_rows != ROWS_PER_SLICE:
        die(f"slice row count is {n_rows}, expected {ROWS_PER_SLICE}. "
            f"The grid definition changed -- re-do the budget math before running.")

    # main() takes its subcommand from sys.argv[1]; hand it what it expects.
    sys.argv = [sys.argv[0]] + ([sub] if sub else [])
    g5.main()


if __name__ == "__main__":
    main()
