"""run_attitude.py — T / A / J for every take, on the walking segments only.

The takes are dirty by construction: Ben lifted the robot mid-take and it fell during
some runs, so a metric averaged over a whole take is meaningless. Segmentation runs
first and everything downstream is computed on walking frames alone.

    python hardware/run_attitude.py            # all takes
    PENGU_MOCAP_DIR=... python hardware/run_attitude.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attitude as at                      # noqa: E402
import mocap                               # noqa: E402

D = os.environ.get("PENGU_MOCAP_DIR", "/Users/ben/Downloads/归档/c6_COMdata")
OUT = os.environ.get("PENGU_OUT", "../HardwareData/cot_mocap_0829/analysis")
TAG = os.environ.get("PENGU_TAG", "")
STRIDE = 4                                  # 360 -> 90 Hz, plenty for a 1.7 Hz gait
BLEND_SKIP = 4.0                            # firmware T_BLEND, dropped from each bout

# Ben, 2026-08-29: drop the first and last TRIM seconds of every take outright. The head
# is the staged start (T_RAMP 4 + T_SETTLE 6 = 10 s of standing before the gait blends in)
# and the tail is where the robot was picked up or fell. Cheaper and less arguable than
# trusting the segmenter at the edges -- at the cost of losing any take shorter than 2*TRIM.
TRIM = float(os.environ.get("PENGU_TRIM_S", "10.0"))
TRIM_HEAD = float(os.environ.get("PENGU_TRIM_HEAD", TRIM))
TRIM_TAIL = float(os.environ.get("PENGU_TRIM_TAIL", TRIM))
if TRIM_HEAD >= 10.0:
    # the trim already removes T_RAMP+T_SETTLE, so skipping the blend again would throw
    # away good data twice
    BLEND_SKIP = 0.0


def smooth(x, n):
    k = np.ones(n) / n
    m = np.isfinite(x)
    y = np.where(m, x, 0.0)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(m.astype(float), k, mode="same")
    return np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)


def band_power(x, t, f0, half=0.5, win=1.0):
    """Running power of x in [f0-half, f0+half], as a rolling std of the band-passed
    signal. A direct readout of whether the actuators are swinging, which speed is not:
    a carried robot moves fast with the legs still, a slipping one is still with the
    legs swinging."""
    ok = np.isfinite(x)
    y = np.where(ok, x - np.nanmean(x[ok]) if ok.any() else 0.0, 0.0)
    dt = float(np.median(np.diff(t)))
    F = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), dt)
    F[(f < f0 - half) | (f > f0 + half)] = 0
    b = np.fft.irfft(F, n=len(y))
    n = max(3, int(win / dt))
    return np.sqrt(smooth(b ** 2, n))


def segment(r, f_cmd):
    """walking / idle / carried / fallen per frame."""
    t = r["t"]
    ct = r["c_torso"]
    dt = float(np.median(np.diff(t)))
    zT = smooth(ct[:, 2], max(3, int(0.5 / dt)))
    sp = np.full(len(t), np.nan)
    v = np.gradient(ct[:, :2], t, axis=0)
    sp = smooth(np.linalg.norm(v, axis=1), max(3, int(0.5 / dt)))
    zF = np.full(len(t), np.nan)
    if "c_Lfoot" in r and "c_Rfoot" in r:
        zF = smooth(np.fmin(r["c_Lfoot"][:, 2], r["c_Rfoot"][:, 2]), max(3, int(0.5 / dt)))
    hp = band_power(r["hip_diff"], t, f_cmd)

    zT_ref = np.nanmedian(zT[t < 3.0]) if (t < 3.0).any() else np.nanmedian(zT)
    zF_ref = np.nanmedian(zF[t < 3.0]) if np.isfinite(zF).any() else 0.05
    hp_thr = 0.25 * np.nanmax(hp) if np.isfinite(hp).any() else np.inf

    lab = np.array(["idle"] * len(t), dtype=object)
    carried = zF > zF_ref + 0.06
    fallen = zT < 0.6 * zT_ref
    walking = (hp > hp_thr) & ~carried & ~fallen
    lab[walking] = "walk"
    lab[carried] = "carried"
    lab[fallen] = "fallen"
    return lab, dict(zT=zT, zF=zF, sp=sp, hp=hp, zT_ref=zT_ref, zF_ref=zF_ref, hp_thr=hp_thr)


def bouts(lab, t, kind="walk", min_s=3.0):
    out, s = [], None
    for i, l in enumerate(lab):
        if l == kind and s is None:
            s = i
        elif l != kind and s is not None:
            if t[i - 1] - t[s] >= min_s:
                out.append((s, i - 1))
            s = None
    if s is not None and t[-1] - t[s] >= min_s:
        out.append((s, len(t) - 1))
    return out


def main():
    files = mocap.scan(D)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in files:
        try:
            tk = mocap.Take(p, stride=STRIDE)
        except Exception as e:                                   # noqa: BLE001
            print(f"!! {os.path.basename(p)}: {e}")
            continue
        f_cmd = 1.67 if tk.mu == 0.12 else 1.92
        r = at.analyse(tk)
        if r is None:
            print(f"!! {tk.slug}: missing a body, skipped")
            continue
        lab, feat = segment(r, f_cmd)
        keep = (r["t"] >= TRIM_HEAD) & (r["t"] <= r["t"][-1] - TRIM_TAIL)
        if keep.sum() < 90:
            print(f"{tk.slug:22s} dur={r['t'][-1]:6.1f}s  -- too short for head {TRIM_HEAD:.0f}s + tail {TRIM_TAIL:.0f}s, dropped")
            continue
        lab = np.where(keep, lab, "trim")
        bs = bouts(lab, r["t"])
        occ = tk.occupancy()
        frac = {k: float(np.mean(lab[keep] == k)) for k in ("walk", "idle", "carried", "fallen")}
        print(f"{tk.slug:22s} dur={r['t'][-1]:6.1f}s  walk={frac['walk']:.2f} "
              f"idle={frac['idle']:.2f} carried={frac['carried']:.2f} fallen={frac['fallen']:.2f}"
              f"  bouts={len(bs)}")
        for bi, (s, e) in enumerate(bs):
            t0 = r["t"][s] + BLEND_SKIP
            w = (r["t"] >= t0) & (r["t"] <= r["t"][e])
            if w.sum() < 90:
                continue
            A, T, J = r["A"][w], r["T"][w], r["J"][w]
            ok = np.isfinite(A) & np.isfinite(T)
            if ok.sum() < 90:
                continue
            fh, pf = at.dom_freq(r["hip_diff"][w], r["t"][w])
            ct = r["c_torso"][w]
            g = np.isfinite(ct[:, 0])
            d_net = float(np.linalg.norm(ct[g][-1, :2] - ct[g][0, :2])) if g.sum() > 2 else np.nan
            d_path = float(np.nansum(np.linalg.norm(np.diff(ct[g][:, :2], axis=0), axis=1))) if g.sum() > 2 else np.nan
            dur = float(r["t"][w][-1] - r["t"][w][0])
            sl = np.polyfit(A[ok], T[ok], 1)[0]
            rows.append(dict(
                slug=tk.slug, file=os.path.basename(p), mu=tk.mu, cot=int(tk.is_cot),
                take=tk.take_no, bout=bi, t0=round(float(r["t"][w][0]), 2),
                t1=round(float(r["t"][w][-1]), 2), dur_s=round(dur, 2), n=int(ok.sum()),
                f_cmd=f_cmd, f_meas=round(fh, 3), f_powfrac=round(pf, 3),
                hip_diff_p2p=round(float(np.nanmax(r["hip_diff"][w]) - np.nanmin(r["hip_diff"][w])), 2),
                T_rms=round(at.rms(T), 3), A_rms=round(at.rms(A), 3), J_rms=round(at.rms(J), 3),
                A_L_rms=round(at.rms(r["A_L"][w]), 3), A_R_rms=round(at.rms(r["A_R"][w]), 3),
                A_LR_diff=round(at.rms(r["A_L"][w] - r["A_R"][w]), 3),
                A_LR_corr=round(float(np.corrcoef(r["A_L"][w][ok], r["A_R"][w][ok])[0, 1]), 3),
                TA_rms=round(at.rms(T) / at.rms(A), 3), TA_slope=round(float(sl), 3),
                TA_corr=round(float(np.corrcoef(A[ok], T[ok])[0, 1]), 3),
                JA_rms=round(at.rms(J) / at.rms(A), 3),
                T_mean=round(float(np.nanmean(T)), 2), A_mean=round(float(np.nanmean(A)), 2),
                d_net_m=round(d_net, 3), d_path_m=round(d_path, 3),
                speed=round(d_net / dur, 4) if dur > 0 else np.nan,
                occ_torso=round(occ["torso"], 4), occ_Lthigh=round(occ["Lthigh"], 4),
                occ_Rthigh=round(occ["Rthigh"], 4),
                resid_torso_mm=round(r["cal"]["torso"]["resid_med_mm"], 2),
                axle_sratio=round(r["cal"]["axle"]["s_ratio"], 3),
            ))
    if rows:
        with open(os.path.join(OUT, f"attitude_0829{TAG}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} bouts -> {OUT}/attitude_0829{TAG}.csv")


if __name__ == "__main__":
    main()
