"""rigid.py — body attitude from markers, and roll measured the way the robot measures it.

The one decision that dominates this file: roll is the tilt ABOUT THE TORSO HINGE AXIS,
not each body's own Euler roll. Measured on mu0.12_COT take1, the same data gives

    tilt_about a shared hinge:   A_L vs A_R differ by 0.91 deg rms, r = 0.968
    each body's own z-x-y roll:  A_L vs A_R differ by 11.9 deg   (each is only 6.7 rms)

because the hip joints rotate about a LATERAL axis: the legs swing antiphase, so each
thigh's own forward axis swings antiphase too, and a roll referred to that axis picks up
the swing as a spurious antiphase signal. Tilt about a common axis is immune -- rotating
about an axis that lies in the projection plane does not move the projection.

This is the same trap `torso_control.py`'s module docstring already records for the sim
("a world-frame lateral-tilt measure ... is WRONG ... because rotating a tilted hinge
mixes roll and yaw"), and the mocap side has to inherit that decision so the numbers are
comparable to `pid.torso_roll` / `pid.axis_roll`.

euler_zxy / grav_roll are COPIED verbatim from grid5/imu_frame_probe.py rather than
imported: that module runs `import mujoco` and builds a sweep model at import time.
"""
import math

import numpy as np

_Z = np.array([0.0, 0.0, 1.0])


# ---- copied verbatim from grid5/imu_frame_probe.py -------------------------------
def euler_zxy(R):
    """Intrinsic z-x-y: R = Rz(yaw) @ Rx(pitch) @ Ry(roll). World y = forward, so
    pitch = forward dip (about lateral x), roll = lateral lean (about forward y)."""
    pitch = math.asin(max(-1.0, min(1.0, float(R[2, 1]))))
    roll = math.atan2(-float(R[2, 0]), float(R[2, 2]))
    yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return yaw, pitch, roll


def grav_roll(R):
    """Gravity-vector roll: g in the frame's own coords, lean about the forward axis.
    Pitch-immune by construction."""
    g = -R[2, :]
    return math.atan2(float(g[0]), -float(g[2]))


# ---- vectorised form of torso_control._tilt_about --------------------------------
def tilt_about(R, u_body, h):
    """Signed tilt (rad) of each frame's calibrated up-vector away from vertical,
    measured ABOUT axis h. R:(n,3,3), u_body:(3,), h:(n,3) or (3,).

    Exact at any attitude: rotation about h moves neither the projection of the up
    vector nor of world-z within the plane perpendicular to h.
    """
    h = np.atleast_2d(h)
    if h.shape[0] == 1:
        h = np.repeat(h, R.shape[0], axis=0)
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    v = R @ u_body                                        # (n,3)
    v_p = v - (np.sum(v * h, axis=1, keepdims=True)) * h
    z_p = _Z - (h @ _Z)[:, None] * h
    nv = np.linalg.norm(v_p, axis=1); nz = np.linalg.norm(z_p, axis=1)
    ok = (nv > 1e-9) & (nz > 1e-9)
    v_p = np.where(ok[:, None], v_p / np.where(nv[:, None] == 0, 1, nv[:, None]), 0)
    z_p = np.where(ok[:, None], z_p / np.where(nz[:, None] == 0, 1, nz[:, None]), 0)
    s = np.sum(np.cross(z_p, v_p) * h, axis=1)
    c = np.sum(z_p * v_p, axis=1)
    out = np.arctan2(s, c)
    out[~ok] = np.nan
    return out


# ---- Kabsch ----------------------------------------------------------------------
def kabsch(P, ref):
    """Fit ref -> P per frame. P:(n,m,3) with NaN gaps, ref:(m,3).

    Returns R:(n,3,3), c:(n,3) centroid of the used markers, resid:(n,) rms in metres,
    k:(n,) number of markers used. Frames with k<3 give NaN.
    """
    n, m, _ = P.shape
    R = np.full((n, 3, 3), np.nan)
    c = np.full((n, 3), np.nan)
    resid = np.full(n, np.nan)
    vis = ~np.isnan(P[:, :, 0])
    k = vis.sum(1)

    # batch the full-visibility frames, loop only the ragged ones
    full = k == m
    groups = [(np.where(full)[0], np.arange(m))] if full.any() else []
    for i in np.where(~full & (k >= 3))[0]:
        groups.append((np.array([i]), np.where(vis[i])[0]))

    for idx, cols in groups:
        if len(cols) < 3:
            continue
        Q = P[np.ix_(idx, cols)]                       # (g, mc, 3)
        A = ref[cols]                                  # (mc, 3)
        qc = Q.mean(axis=1, keepdims=True)
        ac = A.mean(axis=0, keepdims=True)
        Qd = Q - qc
        Ad = A - ac
        H = np.einsum("gmi,mj->gij", Qd, Ad)
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(np.einsum("gij,gjk->gik", U, Vt)))
        D = np.zeros((len(idx), 3, 3))
        D[:, 0, 0] = D[:, 1, 1] = 1.0
        D[:, 2, 2] = d
        Rg = np.einsum("gij,gjk,gkl->gil", U, D, Vt)
        R[idx] = Rg
        c[idx] = qc[:, 0, :]
        pred = np.einsum("gij,mj->gmi", Rg, Ad) + qc
        resid[idx] = np.sqrt(np.nanmean(np.sum((Q - pred) ** 2, axis=2), axis=1))
    return R, c, resid, k


def reference_shape(P, quiet):
    """Per-marker median shape over `quiet` frames where every marker is visible.

    Must be taken BEFORE the walk command: it defines the zero of T and A, so it sets
    the DC offset of everything downstream.
    """
    vis = ~np.isnan(P[:, :, 0])
    cand = np.where(quiet & vis.all(1))[0]
    if len(cand) < 5:
        cand = np.where(vis.all(1))[0]
    if len(cand) == 0:
        return None, dict(ok=False, n=0)
    Q = P[cand]
    ref = np.median(Q, axis=0)
    ref = ref - ref.mean(axis=0)
    # one Procrustes pass so the median is not biased by drift within the window
    R, _, resid, _ = kabsch(Q, ref)
    good = np.isfinite(resid)
    if good.sum() >= 5:
        aligned = np.einsum("nji,nmj->nmi", R[good], Q[good] - Q[good].mean(axis=1, keepdims=True))
        ref = np.median(aligned, axis=0)
        ref = ref - ref.mean(axis=0)
    return ref, dict(ok=True, n=int(len(cand)), resid_med=float(np.nanmedian(resid)))


def rotvec(R):
    """Rotation vectors of a stack of rotation matrices, without scipy."""
    tr = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.arccos(tr)
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2],
                   R[:, 0, 2] - R[:, 2, 0],
                   R[:, 1, 0] - R[:, 0, 1]], axis=1)
    nrm = np.linalg.norm(ax, axis=1, keepdims=True)
    ax = np.divide(ax, np.where(nrm == 0, 1, nrm))
    return ax * ang[:, None]


def fit_shared_axis(R_a, R_b, min_deg=5.0):
    """Axis (in frame a) of the single hinge joining bodies a and b.

    The relative rotation R_a^T R_b is a rotation about the joint axis, so its rotation
    vectors all lie along that axis; SVD of the stack recovers it. The singular-value
    ratio is the quality check -- a real single hinge gives s2/s1 well under 0.15.
    """
    ok = np.isfinite(R_a[:, 0, 0]) & np.isfinite(R_b[:, 0, 0])
    Rel = np.einsum("nji,njk->nik", R_a[ok], R_b[ok])
    rv = rotvec(Rel)
    ang = np.linalg.norm(rv, axis=1)
    use = ang > math.radians(min_deg)
    if use.sum() < 20:
        return None, dict(ok=False, n=int(use.sum()))
    X = rv[use]
    U, S, Vt = np.linalg.svd(X - 0 * X.mean(0), full_matrices=False)
    axis = Vt[0] / np.linalg.norm(Vt[0])
    return axis, dict(ok=True, n=int(use.sum()), s_ratio=float(S[1] / S[0]),
                      s=[float(v) for v in S])
