"""footfall.py — distance walked, measured by where the feet actually land.

Ben, 2026-08-29: "走了多远只看脚 —— 左脚着地到下一次右脚着地再到左脚着地".

Every other candidate for "distance travelled" is contaminated:

    torso centroid path   inflated 1.7x by the torso's own kappa=2 rocking
    hip midpoint path     still carries the per-step lateral wobble
    smoothed hip path     depends on the smoothing window you pick
    net displacement      charges the gait for heading drift, and collapses to ~0
                          if the robot walks back on itself

Footfalls have none of those problems. Each touchdown is a place the robot physically put
its foot; summing the distance between consecutive touchdowns of alternating feet counts
every turn (the steps just point in a slowly rotating direction) while oscillation that
returns to where it started contributes nothing. It is also the standard gait measure,
so the number means the same thing as it does in the locomotion literature.

Caveat this cannot fix: on a slippery surface the foot slides during stance, so the
landing point is not where the foot ends up. Landing-to-landing therefore measures the
step the robot *took*, not the ground it *kept*. Both are reported.
"""
import numpy as np


def touchdowns(z, t, lo_q=20, hi_q=80, frac=0.30, min_gap_s=0.20):
    """Frame indices where the foot comes down.

    The threshold is taken from the foot's own height distribution rather than an
    absolute number, because the floor height differs between capture volumes (the
    afternoon takes were run on a different patch of floor).
    """
    ok = np.isfinite(z)
    if ok.sum() < 50:
        return np.array([], int)
    lo, hi = np.percentile(z[ok], [lo_q, hi_q])
    if hi - lo < 5e-3:                       # foot never lifted -- no gait
        return np.array([], int)
    thr = lo + frac * (hi - lo)
    below = z < thr
    dt = float(np.median(np.diff(t)))
    gap = max(1, int(min_gap_s / dt))
    idx = []
    for i in range(1, len(z)):
        if not np.isfinite(z[i]) or not np.isfinite(z[i - 1]):
            continue
        if below[i] and not below[i - 1]:                 # downward crossing
            if not idx or i - idx[-1] >= gap:
                idx.append(i)
    return np.array(idx, int)


def stride_distance(cL, cR, t):
    """Distance walked, summed over STRIDES -- same foot, consecutive landings.

    Not left-to-right: the two feet are ~8 cm apart laterally, so a left-to-right step
    vector has that separation baked into it and stepping in place would accumulate two
    stance widths per cycle. Measured on mu012_COT_take1 that inflated the distance to
    5.32 m while the body advanced 0.83 m. Same-foot landing to same-foot landing is zero
    for in-place stepping and equals the ground actually covered otherwise, turns
    included -- the stride vectors simply rotate as the robot curves.

    Returns (distance_m, per_foot, meta); distance is the mean of the two feet's totals,
    which are independent measurements of the same quantity and so also a self-check.
    """
    out, per = {}, {}
    for name, c in (("L", cL), ("R", cR)):
        td = touchdowns(c[:, 2], t)
        lens = []
        for i0, i1 in zip(td[:-1], td[1:]):
            p0, p1 = c[i0, :2], c[i1, :2]
            if np.isfinite(p0).all() and np.isfinite(p1).all():
                lens.append(float(np.linalg.norm(p1 - p0)))
        per[name] = dict(n_td=len(td), n_strides=len(lens), total=float(sum(lens)),
                         lens=lens)
    nL, nR = per["L"]["n_strides"], per["R"]["n_strides"]
    if nL < 2 or nR < 2:
        return float("nan"), per, dict(ok=False, nL=nL, nR=nR)
    tot = 0.5 * (per["L"]["total"] + per["R"]["total"])
    disagree = abs(per["L"]["total"] - per["R"]["total"]) / max(tot, 1e-9)
    return tot, per, dict(ok=True, nL=nL, nR=nR, LR_disagree=float(disagree))


def step_distance(cL, cR, t):
    """Sum of landing-to-landing distances, alternating feet.

    Returns (total_m, steps, meta). `steps` is a list of (t_from, t_to, foot_landing,
    length_m) so a bad step can be inspected rather than silently averaged in.
    """
    tdL = touchdowns(cL[:, 2], t)
    tdR = touchdowns(cR[:, 2], t)
    if len(tdL) < 2 or len(tdR) < 2:
        return float("nan"), [], dict(ok=False, nL=len(tdL), nR=len(tdR))

    ev = sorted([(i, "L") for i in tdL] + [(i, "R") for i in tdR])
    # keep only alternating events: two landings of the same foot in a row means one
    # touchdown of the other foot was missed (occlusion), and the step across that hole
    # would be two steps counted as one
    seq = [ev[0]]
    dropped = 0
    for e in ev[1:]:
        if e[1] == seq[-1][1]:
            dropped += 1
            seq[-1] = e                     # keep the later one
        else:
            seq.append(e)

    steps = []
    for (i0, f0), (i1, f1) in zip(seq[:-1], seq[1:]):
        p0 = (cL if f0 == "L" else cR)[i0, :2]
        p1 = (cL if f1 == "L" else cR)[i1, :2]
        if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        steps.append((float(t[i0]), float(t[i1]), f1, float(np.linalg.norm(p1 - p0))))
    tot = float(sum(s[3] for s in steps))
    return tot, steps, dict(ok=True, nL=len(tdL), nR=len(tdR), n_steps=len(steps),
                            dropped_same_foot=dropped)


def stance_slip(c, td, t, win_s=0.15):
    """How far the foot moves during the first `win_s` of stance -- a direct slip readout."""
    dt = float(np.median(np.diff(t)))
    n = max(2, int(win_s / dt))
    out = []
    for i in td:
        j = min(len(c) - 1, i + n)
        p0, p1 = c[i, :2], c[j, :2]
        if np.isfinite(p0).all() and np.isfinite(p1).all():
            out.append(float(np.linalg.norm(p1 - p0)))
    return np.array(out)


def step_speed(body_xy, cL, cR, t, foot="both"):
    """Speed from the body's displacement between consecutive footfalls.

    The method the lab uses, and it is better than any smoothing I could pick: sampling
    the body position once per step samples it at the SAME GAIT PHASE every time, so the
    within-stride oscillation (the torso's kappa=2 rocking, the pelvis wobble) contributes
    the same offset to every sample and cancels in the difference. No filter window has to
    be chosen.

    Returns (dict of totals, per-step list). Each step contributes a displacement and a
    dt, so both the "average the per-step velocities" and the "total distance over total
    time" forms are available -- they differ if the steps are unevenly spaced, and both
    are reported.
    """
    tds = []
    if foot in ("both", "L"):
        tds += [(i, "L") for i in touchdowns(cL[:, 2], t)]
    if foot in ("both", "R"):
        tds += [(i, "R") for i in touchdowns(cR[:, 2], t)]
    tds.sort()
    if len(tds) < 3:
        return dict(ok=False, n_steps=len(tds)), []

    steps = []
    for (i0, f0), (i1, f1) in zip(tds[:-1], tds[1:]):
        p0, p1 = body_xy[i0], body_xy[i1]
        dt = float(t[i1] - t[i0])
        if dt <= 0 or not (np.isfinite(p0).all() and np.isfinite(p1).all()):
            continue
        d = float(np.linalg.norm(p1 - p0))
        steps.append(dict(t0=float(t[i0]), t1=float(t[i1]), foot=f1,
                          d_m=d, dt_s=dt, v_mps=d / dt))
    if len(steps) < 2:
        return dict(ok=False, n_steps=len(steps)), steps

    d = np.array([s["d_m"] for s in steps])
    dt = np.array([s["dt_s"] for s in steps])
    v = d / dt
    return dict(ok=True, n_steps=len(steps),
                dist_m=float(d.sum()), time_s=float(dt.sum()),
                v_pooled=float(d.sum() / dt.sum()),      # total distance / total time
                v_mean=float(v.mean()),                  # mean of per-step velocities
                v_median=float(np.median(v)),
                v_sd=float(v.std(ddof=1)),
                step_d_median=float(np.median(d)),
                step_dt_median=float(np.median(dt))), steps
