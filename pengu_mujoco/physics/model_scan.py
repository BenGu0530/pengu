"""
model_scan.py - Full static + dynamic audit of the Pengu MJCF model.

Answers "is this model good/precise enough?" with numbers instead of eyeballing
the XML. Read-only: never writes the model, only loads penguV2/scene.xml and
probes it. Run from pengu_mujoco/:

    /home/ben/miniconda3/envs/mujoco/bin/python physics/model_scan.py

Sections:
  1. Model overview + solver/timestep options (the un-set <option> defaults)
  2. Per-body mass / inertia validity (positive + triangle inequality)
  3. Joints: ranges / which are unlimited
  4. Actuators: gains, ctrlrange, forcerange
  5. Closed-loop (crank-slider) constraint residual at init + after settle
  6. Standing balance: whole-body CoM vs foot support polygon (margin)
  7. Contacts at stand: count, self-collisions, penetration depth
  8. Static stability: drift / tilt over a 5 s stand-hold
  9. Left/right symmetry: mirror a symmetric pose, compare feet
 10. Timestep convergence: is the natural-frequency walk numerically converged?
"""
import os
import sys
import math
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gait_config as gc
from gait_config import XML_PATH, build_ids, set_initial_pose, STAND_HIP_DEG

FLOOR_GEOM = "floor"
ROOT_BODY = "leftthighmotor"
TORSO_BODY = "easytorso"
LFOOT_BODY = "right_foot0080___fillet13"
RFOOT_BODY = "right_foot0080"

CONNECT_PAIRS = [
    ("closing_crank3-L_1", "closing_crank3-L_2"),
    ("closing_crank3-L_1_z", "closing_crank3-L_2_z"),
    ("closing_crank3-R_1", "closing_crank3-R_2"),
    ("closing_crank3-R_1_z", "closing_crank3-R_2_z"),
]

H = "=" * 72


def _id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def _hold_stand(data, aid):
    data.ctrl[aid["hip-L"]] = math.radians(STAND_HIP_DEG)
    data.ctrl[aid["hip-R"]] = math.radians(STAND_HIP_DEG)
    data.ctrl[aid["crank1-L"]] = 0.0
    data.ctrl[aid["crank1-R"]] = 0.0
    data.ctrl[aid["torso"]] = 0.0


def whole_body_com(model, data):
    """Mass-weighted CoM of all robot bodies (skip world=0), world frame."""
    m = model.body_mass[1:]
    x = data.xipos[1:]
    return (m[:, None] * x).sum(0) / m.sum()


# ---- tiny convex hull + point-in-polygon (no scipy dependency) ----
def _hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _signed_margin(poly, q):
    """Signed distance from point q to polygon boundary.
    + = inside (margin), - = outside (how far out). Handles 1/2-point degens."""
    n = len(poly)
    if n == 1:
        return -np.linalg.norm(q - poly[0])
    if n == 2:
        a, b = poly
        ab = b - a
        t = np.clip(np.dot(q - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        return -np.linalg.norm(q - (a + t * ab))
    # polygon: min distance to edges, signed by inside test
    inside = True
    min_d = 1e9
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(q - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        min_d = min(min_d, np.linalg.norm(q - (a + t * ab)))
        if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) < 0:
            inside = False
    return min_d if inside else -min_d


def sec1_overview(model):
    print(H)
    print("1. MODEL OVERVIEW + SOLVER OPTIONS")
    print(H)
    print(f"  bodies={model.nbody}  dofs(nv)={model.nv}  joints={model.njnt} "
          f" actuators={model.nu}  eq_constraints={model.neq}  geoms={model.ngeom}")
    total = model.body_mass[1:].sum()
    print(f"  total mass = {total:.4f} kg")
    o = model.opt
    integ = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}.get(int(o.integrator), o.integrator)
    cone = {0: "pyramidal", 1: "elliptic"}.get(int(o.cone), o.cone)
    jac = {0: "dense", 1: "sparse", 2: "auto"}.get(int(o.jacobian), o.jacobian)
    print(f"  timestep   = {o.timestep*1e3:.3f} ms  ({1/o.timestep:.0f} Hz)   "
          f"[MuJoCo DEFAULT - no <option> in XML]" )
    print(f"  integrator = {integ}    cone = {cone}    jacobian = {jac}")
    print(f"  solver iterations={o.iterations}  tolerance={o.tolerance:.0e}  "
          f"ls_iter={o.ls_iterations}")
    print(f"  gravity = {o.gravity}")


def sec2_inertia(model):
    print("\n" + H)
    print("2. PER-BODY MASS / INERTIA VALIDITY")
    print(H)
    print(f"  {'body':28s} {'mass[kg]':>9s}  {'principal inertia [kg*m^2]':>34s}  ok?")
    bad = 0
    for i in range(1, model.nbody):
        name = model.body(i).name
        m = model.body_mass[i]
        I = model.body_inertia[i]  # principal moments (diagonalized by compiler)
        pos = I.min() > 0
        a, b, c = sorted(I)
        tri = (a + b) >= c - 1e-12  # triangle inequality on principal moments
        ok = pos and tri and m > 0
        bad += not ok
        flag = "OK" if ok else ("NEG!" if not pos else ("TRI!" if not tri else "M0!"))
        print(f"  {name:28s} {m:9.5f}  [{I[0]:.3e} {I[1]:.3e} {I[2]:.3e}]  {flag}")
    print(f"  --> {bad} invalid bodies" if bad else "  --> all inertias physical (PD + triangle inequality)")


def sec3_joints(model):
    print("\n" + H)
    print("3. JOINTS (ranges / unlimited)")
    print(H)
    jtype = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    unlimited = []
    for i in range(model.njnt):
        name = model.joint(i).name
        t = jtype.get(int(model.jnt_type[i]), "?")
        lim = bool(model.jnt_limited[i])
        rng = model.jnt_range[i]
        if t == "free":
            print(f"  {name:24s} {t:6s}  (6-dof floating base)")
            continue
        rs = f"[{math.degrees(rng[0]):+.1f}, {math.degrees(rng[1]):+.1f}] deg" if t == "hinge" \
            else f"[{rng[0]:+.4f}, {rng[1]:+.4f}] m"
        print(f"  {name:24s} {t:6s}  limited={int(lim)}  {rs if lim else 'UNLIMITED'}")
        if not lim and t in ("hinge", "slide"):
            unlimited.append(name)
    if unlimited:
        print(f"  --> UNLIMITED joints: {', '.join(unlimited)}")


def sec4_actuators(model):
    print("\n" + H)
    print("4. ACTUATORS (gains / ctrlrange / forcerange)")
    print(H)
    for i in range(model.nu):
        name = model.actuator(i).name
        gp = model.actuator_gainprm[i, 0]
        bp = model.actuator_biasprm[i]
        cr = model.actuator_ctrlrange[i] if model.actuator_ctrllimited[i] else None
        fr = model.actuator_forcerange[i] if model.actuator_forcelimited[i] else None
        crs = f"[{cr[0]:+.2f},{cr[1]:+.2f}]" if cr is not None else "UNLIMITED"
        frs = f"[{fr[0]:+.2f},{fr[1]:+.2f}] Nm" if fr is not None else "UNLIMITED"
        print(f"  {name:10s} kp={gp:6.1f}  kv={-bp[2]:6.2f}  ctrl={crs:18s} force={frs}")


def sec5_loop(model, data, aid, jadr):
    print("\n" + H)
    print("5. CLOSED-LOOP (crank-slider) CONSTRAINT RESIDUAL")
    print(H)

    def gaps():
        out = []
        for s1, s2 in CONNECT_PAIRS:
            i1 = _id(model, mujoco.mjtObj.mjOBJ_SITE, s1)
            i2 = _id(model, mujoco.mjtObj.mjOBJ_SITE, s2)
            d = np.linalg.norm(data.site_xpos[i1] - data.site_xpos[i2]) * 1e3
            out.append((s1, d))
        return out

    set_initial_pose(model, data, aid, jadr)
    g0 = gaps()
    print("  loop-closure gap at init pose (should be ~0):")
    for s, d in g0:
        print(f"    {s:22s} {d:8.4f} mm")
    # settle under gravity holding stand
    for _ in range(int(2.0 / model.opt.timestep)):
        _hold_stand(data, aid)
        mujoco.mj_step(model, data)
    g1 = gaps()
    print("  loop-closure gap after 2 s settle (constraint drift under load):")
    for s, d in g1:
        print(f"    {s:22s} {d:8.4f} mm")
    worst = max(d for _, d in g1)
    print(f"  --> worst residual = {worst:.4f} mm "
          f"({'tight' if worst < 1 else 'SOFT - loop is compliant'})")


def sec6_7_balance_contacts(model, data, aid, jadr):
    print("\n" + H)
    print("6+7. STANDING BALANCE (CoM vs support) + CONTACTS")
    print(H)
    set_initial_pose(model, data, aid, jadr)
    for _ in range(int(3.0 / model.opt.timestep)):
        _hold_stand(data, aid)
        mujoco.mj_step(model, data)

    floor = _id(model, mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
    foot_pts, self_pairs, penetrations = [], [], []
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        if floor in (g1, g2):
            foot_pts.append(c.pos[:2].copy())
        else:
            b1 = model.body(model.geom_bodyid[g1]).name
            b2 = model.body(model.geom_bodyid[g2]).name
            self_pairs.append((b1, b2))
        if c.dist < -1e-4:
            penetrations.append(c.dist)

    com = whole_body_com(model, data)
    print(f"  whole-body CoM (world) = ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f}) m")
    print(f"  ground contacts = {len(foot_pts)}   self-collision contacts = {len(self_pairs)}")
    if foot_pts:
        poly = _hull(np.array(foot_pts))
        margin = _signed_margin(poly, com[:2])
        fp = np.array(foot_pts)
        print(f"  support footprint x:[{fp[:,0].min():+.3f},{fp[:,0].max():+.3f}] "
              f"y:[{fp[:,1].min():+.3f},{fp[:,1].max():+.3f}] m  ({len(poly)}-pt hull)")
        verdict = "INSIDE (statically balanced)" if margin > 0 else "OUTSIDE support (would tip)"
        print(f"  CoM-to-support margin = {margin*1e3:+.1f} mm  --> {verdict}")
    else:
        print("  !! NO ground contacts - robot not standing on its feet")
    if self_pairs:
        from collections import Counter
        cc = Counter(tuple(sorted(p)) for p in self_pairs)
        print(f"  SELF-COLLISIONS ({len(self_pairs)} contacts):")
        for (a, b), n in cc.most_common():
            print(f"    {a} <-> {b}   x{n}")
    if penetrations:
        print(f"  max penetration depth = {min(penetrations)*1e3:.2f} mm "
              f"({len(penetrations)} contacts > 0.1 mm)")
    else:
        print("  no significant penetration (<0.1 mm)")


def sec8_stability(model, data, aid, jadr):
    print("\n" + H)
    print("8. STATIC STABILITY (5 s stand-hold drift)")
    print(H)
    set_initial_pose(model, data, aid, jadr)
    root = _id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    torso = _id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    z0 = data.xpos[root][2]
    xy0 = data.xpos[root][:2].copy()
    for _ in range(int(5.0 / model.opt.timestep)):
        _hold_stand(data, aid)
        mujoco.mj_step(model, data)
    R = data.xmat[torso].reshape(3, 3)
    up = -R[:, 1]  # torso "up" per walk_pengu convention
    roll = math.degrees(math.atan2(up[0], up[2]))
    pitch = math.degrees(math.atan2(up[1], up[2]))
    dz = data.xpos[root][2] - z0
    dxy = np.linalg.norm(data.xpos[root][:2] - xy0)
    vmax = np.abs(data.qvel).max()
    print(f"  root z drop over 5 s   = {dz*1e3:+.1f} mm")
    print(f"  root xy drift          = {dxy*1e3:.1f} mm")
    print(f"  torso roll / pitch     = {roll:+.1f} / {pitch:+.1f} deg")
    print(f"  max |qvel| at t=5s     = {vmax:.3f}  "
          f"({'settled' if vmax < 0.5 else 'STILL MOVING / unstable'})")


def sec9_symmetry(model, data, aid, jadr):
    print("\n" + H)
    print("9. LEFT/RIGHT SYMMETRY (mirror a symmetric pose, compare feet)")
    print(H)
    # air model: no gravity, no contact -> pure kinematics
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    torso = _id(model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
    lf = _id(model, mujoco.mjtObj.mjOBJ_BODY, LFOOT_BODY)
    rf = _id(model, mujoco.mjtObj.mjOBJ_BODY, RFOOT_BODY)
    for cr in (20.0, 45.0):
        set_initial_pose(model, data, aid, jadr)
        data.qpos[2] += 1.0
        mujoco.mj_forward(model, data)
        data.ctrl[aid["hip-L"]] = math.radians(STAND_HIP_DEG)
        data.ctrl[aid["hip-R"]] = math.radians(STAND_HIP_DEG)
        data.ctrl[aid["torso"]] = 0.0
        data.ctrl[aid["crank1-L"]] = math.radians(cr)
        data.ctrl[aid["crank1-R"]] = math.radians(cr)
        for _ in range(4000):
            mujoco.mj_step(model, data)
        tp = data.xpos[torso]
        L = (data.xpos[lf] - tp) * 1e3
        R = (data.xpos[rf] - tp) * 1e3
        # mirror: ideal symmetric => L.x = -R.x, L.y=R.y, L.z=R.z
        asym = np.array([L[0] + R[0], L[1] - R[1], L[2] - R[2]])
        print(f"  crank_L=crank_R={cr:.0f} deg:")
        print(f"    L foot rel torso = ({L[0]:+7.1f},{L[1]:+7.1f},{L[2]:+7.1f}) mm")
        print(f"    R foot rel torso = ({R[0]:+7.1f},{R[1]:+7.1f},{R[2]:+7.1f}) mm")
        print(f"    asymmetry (x+,y-,z-) = ({asym[0]:+6.1f},{asym[1]:+6.1f},{asym[2]:+6.1f}) mm"
              f"   {'~symmetric' if np.abs(asym).max() < 5 else 'ASYMMETRIC'}")


def sec10_convergence():
    print("\n" + H)
    print("10. TIMESTEP CONVERGENCE (is the natural-freq walk converged?)")
    print(H)
    names = ["hip-L", "hip-R", "crank1-R", "torso", "crank1-L"]
    # use the 4dof anchor that matched real machine
    gc.set_hip_amp(12.0); gc.set_crank_amp(73.0); gc.set_torso_amp(0.0)
    gc.set_walk_freq(1.32)
    root_name = ROOT_BODY
    print("  anchor 4dof_c73_h12 @1.32Hz, 18 s sim, varying dt:")
    print(f"  {'dt[ms]':>7s} {'fwd_dist[m]':>12s} {'final_z[m]':>11s} {'roll[deg]':>10s}")
    prev = None
    for dt in (4.0, 2.0, 1.0, 0.5):
        m = mujoco.MjModel.from_xml_path(XML_PATH)
        m.opt.timestep = dt * 1e-3
        d = mujoco.MjData(m)
        aid, jadr = build_ids(m)
        gc.set_initial_pose(m, d, aid, jadr)
        rid = _id(m, mujoco.mjtObj.mjOBJ_BODY, root_name)
        y0 = d.xpos[rid][1]
        R0 = d.xmat[rid].reshape(3, 3).copy()
        while d.time < 18.0:
            gc.apply_ctrl(d, aid, d.time)
            mujoco.mj_step(m, d)
        fwd = d.xpos[rid][1] - y0
        z = d.xpos[rid][2]
        R = d.xmat[rid].reshape(3, 3)
        # roll relative to spawn
        rel = R0.T @ R
        roll = math.degrees(math.atan2(rel[2, 1], rel[2, 2]))
        tag = ""
        if prev is not None:
            tag = f"  d(fwd) vs prev = {abs(fwd-prev)*1e3:5.1f} mm"
        prev = fwd
        print(f"  {dt:7.1f} {fwd:12.3f} {z:11.3f} {roll:10.1f}{tag}")
    print("  --> if fwd_dist keeps changing as dt halves, the sweep is NOT converged")


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    aid, jadr = build_ids(model)

    print("\nPENGU MODEL FULL SCAN  (model: %s)" % XML_PATH)
    sec1_overview(model)
    sec2_inertia(model)
    sec3_joints(model)
    sec4_actuators(model)
    sec5_loop(model, data, aid, jadr)
    sec6_7_balance_contacts(model, data, aid, jadr)
    sec8_stability(model, data, aid, jadr)
    # sec9 mutates model opts -> use a fresh model
    m2 = mujoco.MjModel.from_xml_path(XML_PATH)
    d2 = mujoco.MjData(m2)
    a2, j2 = build_ids(m2)
    sec9_symmetry(m2, d2, a2, j2)
    sec10_convergence()
    print("\n" + H)
    print("SCAN COMPLETE")
    print(H)


if __name__ == "__main__":
    main()
