"""vec3.py — 3-vector arithmetic without numpy's generic-path overhead.

Why this exists. Profiling one GRID-5 trial (1.487 s) showed mj_step accounting for
0.266 s and `_tilt_about` for 1.241 s -- 83% of the trial spent measuring two tilt
angles, not simulating. The kappa PID calls it twice per 1 ms step, so np.cross ran
38,000 times per trial, and for 3-element arrays numpy's moveaxis /
normalize_axis_tuple bookkeeping costs far more than the nine multiplications it
guards. 8.71 us per call against 0.62 us for the explicit form.

These compute the same products, in the same order, so results are BIT-IDENTICAL to
the numpy calls they replace; verified on 200,000 random and 50,000 unit 3-vector
pairs with zero mismatches. Nothing here may be "simplified" into a different
association order without re-running that check -- the sweep's value depends on
grid5-v2 rows being reproducible.

Why its own module rather than a helper inside torso_control: gait_sweep.py:34
prepends the PARENT directory to sys.path, so every bare import after that line is
shadowed by the root-level copy of the same module. A module name that does not
exist in the parent tree cannot be shadowed.
"""
import math
import numpy as _np


def cross3(a, b):
    a0, a1, a2 = a[0], a[1], a[2]
    b0, b1, b2 = b[0], b[1], b[2]
    return _np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def norm3(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
