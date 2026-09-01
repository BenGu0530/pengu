"""mocap.py — OptiTrack Motive 1.25 marker exports, loaded into sim world axes.

These takes export MARKER POSITIONS ONLY. There are no rigid-body quaternion columns
despite the `_RB` names, so body attitude has to be fitted from the markers (rigid.py).

Two traps this module handles so nothing downstream has to:

  * The rigid body is spelled `pengu_rightthugh_RB` (t-h-u-g-h) in every afternoon file,
    which is all 11 of the mu=0.45 and _COT takes. A loader matching the exact string
    silently loses the right thigh on half the session.
  * Absolute time comes from the header's `Capture Start Time`, never the filename --
    `...take2 - markerconfig1 2026-08-29 11.53.45 AM_000.csv` has a header start of
    12.04.59 PM, because `_000` is a re-export.

Axis convention. The capture volume is (confirmed with the lab, 2026-08-30):

    Motive +y = up
    Motive +z = forward, towards the force plates
    Motive +x = left, towards the cabinets

The sim and grid5/imu_frame_probe are Z-up with +y forward and +x to the right, so the
mapping applied ONCE here and nowhere else is

    sim_x (right)   = -Motive_x
    sim_y (forward) =  Motive_z
    sim_z (up)      =  Motive_y

det = +1, i.e. a proper rotation; getting that wrong flips the sign of every roll.

(An earlier version of this file used (Z, X, Y), which put forward on sim_x and left on
sim_y. That is still right-handed and still has up on the correct axis, so it did not
affect T/A/J, the tilts, or any distance -- those depend only on the vertical axis, on a
hinge axis fitted from the data, and on horizontal norms. It would have mattered for
euler_zxy, which is only used in the "wrong definition" comparison column.)

Positions are ZEROED at load: the first frame with all markers present becomes the
origin, so every trajectory starts at (0, 0, 0) and heights are relative to the starting
stance rather than to the Motive floor plane.
"""
import csv
import os
import re
from datetime import datetime

import numpy as np

BODY_ALIAS = {"pengu_rightthugh_RB": "pengu_rightthigh_RB"}
BODIES = ["pengu_torso_RB", "pengu_leftthigh_RB", "pengu_rightthigh_RB",
          "pengu_leftfoot_RB", "pengu_rightfoot_RB"]
SHORT = {"pengu_torso_RB": "torso", "pengu_leftthigh_RB": "Lthigh",
         "pengu_rightthigh_RB": "Rthigh", "pengu_leftfoot_RB": "Lfoot",
         "pengu_rightfoot_RB": "Rfoot"}

# Motive (X, Y, Z) -> sim world (x, y, z) = (-X_m, Z_m, Y_m)
AXIS = np.array([[-1., 0., 0.],
                 [0., 0., 1.],
                 [0., 1., 0.]])
assert abs(np.linalg.det(AXIS) - 1.0) < 1e-12, "axis map must be a proper rotation"


class Take:
    def __init__(self, path, max_frames=None, stride=1):
        self.path = path
        self.file = os.path.basename(path)
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            rows = []
            for i, r in enumerate(rd):
                rows.append(r)
                if i > 12:
                    break
            hdr = rows[0]
            self.meta = dict(zip(hdr[0::2], hdr[1::2]))
            # data header is the row whose first cell is 'Frame'
            hrow = next(i for i, r in enumerate(rows) if r and r[0] == "Frame")
            names = rows[hrow - 4]          # the 'Name' row
            axes = rows[hrow]               # Frame, Time, X, Y, Z, ...

        self.fps = float(self.meta["Capture Frame Rate"])
        assert self.meta["Length Units"] == "Meters", self.meta["Length Units"]
        assert self.meta["Coordinate Space"] == "Global"
        self.take_name = self.meta.get("Take Name", "")
        self.capture_start = datetime.strptime(
            self.meta["Capture Start Time"], "%Y-%m-%d %I.%M.%S.%f %p")
        self.mu = 0.12 if "mu0.12" in self.file else (0.45 if "mu0.45" in self.file else None)
        self.is_cot = "_COT" in self.file or "COT" in self.file
        m = re.search(r"take\s*(\d+)", self.file)
        self.take_no = int(m.group(1)) if m else None
        self.markerconfig = 2 if "markerconfig2" in self.file else (1 if "markerconfig1" in self.file else None)
        self.slug = (f"mu{str(self.mu).replace('.','')}"
                     f"{'_COT' if self.is_cot else ''}_take{self.take_no}")

        # column -> (body, marker, axis)
        cols = {}
        for j in range(2, len(axes)):
            nm = names[j] if j < len(names) else ""
            if ":" not in nm:
                continue
            body, marker = nm.split(":", 1)
            body = BODY_ALIAS.get(body.strip(), body.strip())
            cols.setdefault(body, {}).setdefault(marker.strip(), {})[axes[j]] = j
        self.cols = cols

        # read the data block
        raw = []
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            for _ in range(hrow + 1):
                next(rd)
            for k, r in enumerate(rd):
                if stride > 1 and k % stride:
                    continue
                raw.append(r)
                if max_frames and len(raw) >= max_frames:
                    break
        n = len(raw)
        self.n = n
        self.t = np.array([float(r[1]) for r in raw])

        self.xyz, self.names = {}, {}
        for body in BODIES:
            if body not in cols:
                self.xyz[SHORT[body]] = None
                continue
            mk = sorted(cols[body])
            arr = np.full((n, len(mk), 3), np.nan)
            for mi, name in enumerate(mk):
                jx, jy, jz = cols[body][name]["X"], cols[body][name]["Y"], cols[body][name]["Z"]
                for i, r in enumerate(raw):
                    if r[jx]:
                        arr[i, mi] = (float(r[jx]), float(r[jy]), float(r[jz]))
            self.xyz[SHORT[body]] = arr @ AXIS.T        # Motive -> sim world
            self.names[SHORT[body]] = mk

        # zero on the first fully-observed frame, so trajectories start at the origin
        self.origin = np.zeros(3)
        tor = self.xyz.get("torso")
        if tor is not None:
            full = np.where(~np.isnan(tor[:, :, 0]).any(axis=1))[0]
            if len(full):
                self.origin = tor[full[0]].mean(axis=0)
                for k in self.xyz:
                    if self.xyz[k] is not None:
                        self.xyz[k] = self.xyz[k] - self.origin

    def occupancy(self):
        out = {}
        for k, a in self.xyz.items():
            out[k] = float(np.isnan(a[:, :, 0]).mean()) if a is not None else 1.0
        return out

    def centroid(self, body):
        """Mean of the visible markers. NaN where fewer than 3 are visible."""
        a = self.xyz[body]
        vis = ~np.isnan(a[:, :, 0])
        c = np.nanmean(a, axis=1)
        c[vis.sum(1) < 3] = np.nan
        return c


def scan(d):
    """List the take files in a directory, newest-header-first ordering by capture time."""
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".csv")]
    return sorted(fs)
