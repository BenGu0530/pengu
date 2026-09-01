"""attitude.py — T, A and J for one take, in the same terms the robot's controller uses.

Definitions (matching torso_control.TorsoKappaPID):
    A = tilt of the lower body (hip axis) about the torso hinge axis
    T = tilt of the torso       about the same axis
    J = torso joint angle,  T = A + J
The controller's objective is T = kappa * A, so T/A is its scoreboard; the robot itself
never measures A -- it reconstructs it as A_hat = T - s*J from the torso IMU and the
torso encoder. Here A is measured directly from the thigh markers, which is the point.

The hinge axis is built geometrically, not fitted: the hip joints rotate about a LATERAL
axis (the thighs swing fore/aft), so the horizontal direction perpendicular to that axle
is the robot's forward direction, which is exactly the axis the torso rolls about.
Fitting torso-vs-thigh relative rotation instead gives s2/s1 = 0.21 because that
relative motion contains the hip DOF as well as the torso DOF -- two joints, not one.
"""
import math

import numpy as np

import rigid

_Z = np.array([0.0, 0.0, 1.0])
BODIES = ("torso", "Lthigh", "Rthigh")


def _up_body(R, quiet):
    """Body-frame vector that points world-up during the quiet reference window."""
    ok = quiet & np.isfinite(R[:, 0, 0])
    if ok.sum() < 5:
        ok = np.isfinite(R[:, 0, 0])
    u = np.einsum("nji,j->ni", R[ok], _Z).mean(axis=0)
    return u / np.linalg.norm(u)


def analyse(tk, quiet_s=3.0):
    """Returns a dict of per-frame series plus the calibration metadata."""
    t = tk.t
    quiet = t < quiet_s
    out = {"t": t, "slug": tk.slug, "mu": tk.mu, "cal": {}}

    fits = {}
    for b in BODIES:
        if tk.xyz[b] is None:
            return None
        ref, meta = rigid.reference_shape(tk.xyz[b], quiet)
        R, c, resid, k = rigid.kabsch(tk.xyz[b], ref)
        fits[b] = dict(R=R, c=c, resid=resid, k=k)
        out["cal"][b] = dict(ref_n=meta["n"], resid_med_mm=float(1000 * np.nanmedian(resid)),
                             resid_p99_mm=float(1000 * np.nanpercentile(resid, 99)),
                             frac_k3=float(np.mean(k == 3)), frac_used=float(np.mean(k >= 3)))
    for b in ("Lfoot", "Rfoot"):
        if tk.xyz[b] is not None:
            out[f"c_{b}"] = tk.centroid(b)

    # hip axle: the single hinge shared by the two thighs
    axle_L, am = rigid.fit_shared_axis(fits["Lthigh"]["R"], fits["Rthigh"]["R"])
    if axle_L is None:
        return None
    out["cal"]["axle"] = am
    axle_w = np.einsum("nij,j->ni", fits["Lthigh"]["R"], axle_L)     # world, per frame

    # hinge axis = horizontal direction perpendicular to the axle = robot forward
    h = np.cross(axle_w, _Z)
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    # sign it along net travel so +roll means the same thing in every take
    ct = fits["torso"]["c"]
    good = np.isfinite(ct[:, 0])
    disp = ct[good][-1] - ct[good][0]
    if np.dot(np.nanmean(h[good], axis=0), disp) < 0:
        h = -h
        axle_w = -axle_w
    out["h"] = h
    out["axle_w"] = axle_w

    for b in BODIES:
        u = _up_body(fits[b]["R"], quiet)
        out["cal"][b]["u_body"] = u.tolist()
        out[f"tilt_{b}"] = np.degrees(rigid.tilt_about(fits[b]["R"], u, h))
        # each body's own z-x-y roll, kept only to demonstrate why it is the wrong metric
        Rb = fits[b]["R"]
        own = np.full(len(t), np.nan)
        fin = np.where(np.isfinite(Rb[:, 0, 0]))[0]
        for i in fin:
            own[i] = math.degrees(rigid.euler_zxy(Rb[i])[2])
        out[f"ownroll_{b}"] = own

    out["T"] = out["tilt_torso"]
    out["A_L"] = out["tilt_Lthigh"]
    out["A_R"] = out["tilt_Rthigh"]
    out["A"] = 0.5 * (out["A_L"] + out["A_R"])
    out["J"] = out["T"] - out["A"]

    # hip differential angle about the axle -- the direct readout of leg swing
    Rel = np.einsum("nji,njk->nik", fits["Lthigh"]["R"], fits["Rthigh"]["R"])
    fin = np.isfinite(Rel[:, 0, 0])
    rv = np.full((len(t), 3), np.nan)
    rv[fin] = rigid.rotvec(Rel[fin])
    out["hip_diff"] = np.degrees(rv @ axle_L)

    out["c_torso"] = fits["torso"]["c"]
    out["fits"] = fits
    return out


def dom_freq(x, t, lo=0.5, hi=3.5):
    """Dominant frequency of a series, band-limited. Returns (f, power_fraction)."""
    ok = np.isfinite(x)
    if ok.sum() < 64:
        return float("nan"), float("nan")
    y = x[ok] - np.nanmean(x[ok])
    dt = float(np.median(np.diff(t[ok])))
    n = len(y)
    f = np.fft.rfftfreq(n, dt)
    P = np.abs(np.fft.rfft(y * np.hanning(n))) ** 2
    band = (f >= lo) & (f <= hi)
    if not band.any():
        return float("nan"), float("nan")
    i = np.argmax(np.where(band, P, 0))
    return float(f[i]), float(P[i] / P[1:].sum())


def rms(x):
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")
