"""rof.py — Rigol DP800 ASCII .ROF recorder logs, parsed without losing a row.

Why not Downloads/pengudata/rof2csv.py: its line regex requires bare V/A/W units, so
the samples the supply writes while its output is OFF -- "0.000mV,0.000uA,0.000uW," --
fail to match and are SKIPPED. Its time column is then built as row_index * period over
the surviving rows, so every sample after a dropped block is shifted earlier by the
length of that block. p_mu0-12_1.ROF drops 146 rows from the MIDDLE of the file;
p_mu0-45_3.ROF drops 4075 of 4177. Mixed rows exist too ("15.999V, 0.290A,-28.926mW,"),
so the unit prefix has to be read per field, not per line.

Here every physical line keeps its index. An output-off sample is a real sample -- the
recorder's clock kept running -- it just reads zero. That index is the only time base
the file has: there are no timestamps in the format, and the supply's RTC was unset
(every file is stamped 2010-01-01), so the filesystem mtimes are useless as well.

The sampling period is not stored either. fit_period() infers it from the mocap takes
rather than assuming the 1 s default.
"""
import os
import re

import numpy as np

UNIT = {"": 1.0, "m": 1e-3, "u": 1e-6, "n": 1e-9, "k": 1e3}
FIELD = re.compile(r"\s*(-?\d+(?:\.\d+)?)\s*([munk]?)([VAW])\s*")

V_SET = 16.0          # supply set-point, from the idle samples (15.997-16.001 in CV)
CC_V = 15.95          # below this the supply has left constant-voltage
RAIL_A = 1.9995       # the 2.000 A current limit
I_ON = 0.55           # burst threshold; see bursts() for where this comes from


class Rof:
    """One .ROF file. Arrays are per PHYSICAL LINE -- nothing is dropped."""

    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        raw = open(path, "rb").read().decode("ascii", "replace")
        v, a, w, self.malformed = [], [], [], []
        for n, line in enumerate(raw.splitlines()):
            if not line.strip():
                continue
            vals = {}
            for num, pfx, unit in FIELD.findall(line):
                vals[unit] = float(num) * UNIT[pfx]
            if len(vals) != 3:
                self.malformed.append((n, line))
                continue
            v.append(vals["V"]); a.append(vals["A"]); w.append(vals["W"])
        if len(self.malformed) > 5:
            raise ValueError(f"{self.name}: {len(self.malformed)} malformed lines, "
                             f"first {self.malformed[0]}")
        self.v = np.array(v); self.a = np.array(a); self.w = np.array(w)
        self.n = len(self.v)
        self.idx = np.arange(self.n)
        self.off = (np.abs(self.v) < 1e-3) & (np.abs(self.a) < 1e-6)
        self.rail = self.a >= RAIL_A
        self.cc = (self.v < CC_V) & ~self.off

    # ---- burst detection -------------------------------------------------------
    def bursts(self, i_on=I_ON, gap=3, min_len=4):
        """Contiguous runs of a > i_on, bridging holes of <= `gap` samples.

        i_on is not a guess: idle draw with the motors holding is 0.311-0.357 A, the
        torque-off bench file reads 0.200 A, and walking runs 0.8-2.0 A. The session
        histogram is empty between 0.4 and 0.6.
        """
        on = self.a > i_on
        out, s = [], None
        hole = 0
        for i, flag in enumerate(on):
            if flag:
                if s is None:
                    s = i
                hole = 0
            elif s is not None:
                hole += 1
                if hole > gap:
                    e = i - hole
                    if e - s + 1 >= min_len:
                        out.append((s, e))
                    s, hole = None, 0
        if s is not None and self.n - hole - s >= min_len:
            out.append((s, self.n - 1 - hole))
        return out

    def idle_mask(self, pad=2):
        """Samples that are neither in a burst (padded) nor output-off."""
        m = ~self.off
        for s, e in self.bursts():
            m[max(0, s - pad):min(self.n, e + 1 + pad)] = False
        return m

    def summary(self):
        b = self.bursts()
        idle = self.idle_mask()
        return dict(
            file=self.name, n=self.n, n_off=int(self.off.sum()),
            n_malformed=len(self.malformed),
            idle_A=float(np.median(self.a[idle])) if idle.any() else float("nan"),
            idle_W=float(np.median(self.w[idle])) if idle.any() else float("nan"),
            n_bursts=len(b),
            burst_lens=[e - s + 1 for s, e in b],
            a_max=float(self.a.max()), v_min=float(self.v[~self.off].min()) if (~self.off).any() else float("nan"),
            rail_frac_file=float(self.rail.mean()),
            # the meter samples V, A and W at different instants inside one record
            # period, and the load swings 5->32 W at the gait frequency, so per-sample
            # W != V*A even though the means agree. Report both.
            w_va_rms_diff=float(np.sqrt(np.mean((self.w - self.v * self.a) ** 2))),
            w_va_mean_diff=float(np.mean(self.w) - np.mean(self.v * self.a)),
        )

    def burst_stats(self, s, e):
        sl = slice(s, e + 1)
        w = self.w[sl]
        n = len(w)
        return dict(
            i0=s, i1=e, n=n,
            P_W=float(w.mean()), P_va_W=float((self.v[sl] * self.a[sl]).mean()),
            P_std_W=float(w.std(ddof=1)) if n > 1 else 0.0,
            P_sem_W=float(w.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            P_p99_unrailed=float(np.percentile(w[~self.rail[sl]], 99)) if (~self.rail[sl]).any() else float("nan"),
            rail_frac=float(self.rail[sl].mean()),
            cc_frac=float(self.cc[sl].mean()),
            a_max=float(self.a[sl].max()), v_min=float(self.v[sl].min()),
        )


def load_dir(d):
    return [Rof(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if f.lower().endswith(".rof")]


def fit_period(pairs):
    """Least squares for a single global period: n_samples_k * period ~ duration_k.

    `pairs` is [(n_samples, duration_s), ...] from burst/bout matches whose mocap side
    is not truncated at either end. Returns (period, residuals, r2).
    """
    if not pairs:
        return float("nan"), [], float("nan")
    n = np.array([p[0] for p in pairs], float)
    d = np.array([p[1] for p in pairs], float)
    period = float((n @ d) / (n @ n))
    resid = d - n * period
    ss_tot = float(((d - d.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return period, resid.tolist(), r2
