"""harden_cad.py — turn a raw onshape-to-robot MuJoCo export into a sweep-ready model.

The recipe is the diff between models/pengu1_05cad (raw) and models/hardware_c1 (hand
hardened, 2026-09-02), applied mechanically so the next CAD export gets the same
conventions without a hand edit:

  * <option timestep=0.001 integrator=implicitfast>     the closed crank-slider loop
  * position actuators forcerange +/-4.1 N.m           XM430 stall torque
  * collision default contype=0 conaffinity=0          contact on the two foot geoms only,
    foot geoms contype=1 conaffinity=1 friction=0.9 0.005 0.0001
  * 9 -> 5 actuators (hip-L, hip-R, crank1-R, crank1-L, torso) with ctrlranges; the
    passive loop members (sliders, second cranks) stay driven by the connect equalities.
    Onshape names the right disk crank2_R and the left disk crank1-L; the actuator named
    crank1-R drives joint crank2_R, exactly as in hardware_c1.
  * scene.xml copied from hardware_c1.

Then the model is loaded and measured at the neutral stand (base z 0.18, every joint
at 0, the same pose com_ratio_of uses): total mass, COM ratio (whole-robot COM height /
easyaxis height), easytorso mass and principal inertias — against hardware_c1 as the
reference. Optional --ballast adds a point mass to easytorso at the height that makes
the COM ratio hit --com-ratio (what the 2026-09-02 hardening did by hand for +100 g);
without it the export's own mass is kept and only reported.

    python models/harden_cad.py pengu1_05cad hardware_c1_check --ballast 2.2724 --com-ratio 1.05
    python models/harden_cad.py hardware_c1v2 hardware_c1v2h
"""
import argparse
import os
import re
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "hardware_c1")
INIT_Z = 0.18
DRIVEN = [("hip-L", "hip-L", "-1.5708 1.5708"), ("hip-R", "hip-R", "-1.5708 1.5708"),
          ("crank1-R", "crank2_R", "-3.14 3.14"), ("crank1-L", "crank1-L", "-3.14 3.14"),
          ("torso", "torso", "-0.7854 0.7854")]
FOOT_CONTACT = ' contype="1" conaffinity="1" friction="0.9 0.005 0.0001"'


def once(s, old, new):
    n = s.count(old)
    assert n == 1, f"expected exactly one match, found {n}: {old[:70]!r}"
    return s.replace(old, new)


def harden(src, dst, note):
    s = open(os.path.join(src, "robot.xml")).read()
    raw = re.search(r'<mujoco model="([^"]+)">', s).group(1)
    name = os.path.basename(dst)
    s = s.replace(f'"{raw}"', f'"{name}"')
    s = once(s, '  <compiler angle="radian" meshdir="assets" autolimits="true"/>\n',
             '  <compiler angle="radian" meshdir="assets" autolimits="true"/>\n'
             f'  <!-- HARDENING by models/harden_cad.py from the raw export models/{os.path.basename(src)}:\n'
             '       explicit small timestep + implicitfast integrator for the closed crank-slider\n'
             '       loop; XM430 stall torque +/-4.1 N.m as the hard actuator limit; contact enabled\n'
             '       on the two foot geoms only; the passive members of the closed loop are driven\n'
             '       by the connect equalities, not by actuators. Same conventions as hardware_c1.\n'
             f'       {note} -->\n'
             '  <option timestep="0.001" integrator="implicitfast"/>\n')
    s = once(s, '<position kp="50" dampratio="1"/>',
             '<position kp="50" dampratio="1" forcerange="-4.1 4.1"/>')
    s = once(s, '<geom group="3"/>', '<geom group="3" contype="0" conaffinity="0"/>')
    feet = re.findall(r'<geom type="mesh" class="collision"[^>]*mesh="([^"]*foot[^"]*)"[^>]*/>', s)
    assert len(feet) == 2, f"expected two foot collision geoms, found {feet}"
    s = re.sub(r'(<geom type="mesh" class="collision"[^>]*mesh="[^"]*foot[^"]*"[^>]*?)/>',
               lambda m: m.group(1) + FOOT_CONTACT + "/>", s)
    joints = set(re.findall(r'<joint[^>]*name="([^"]+)"', s))
    missing = [j for _, j, _ in DRIVEN if j not in joints]
    assert not missing, f"driven joints not in export: {missing}; have {sorted(joints)}"
    act = ('  <!-- Driven set = penguV3 convention: hip x2 + ONE rotary crank disk per leg + torso.\n'
           '       The closed crank-slider loop is 1-DOF/leg, so driving the disk sets leg extension;\n'
           '       the rod and slider follow via the connect equalities and are left PASSIVE.\n'
           '       NOTE asymmetric onshape naming: right disk = crank2_R, left disk = crank1-L. -->\n'
           '  <actuator>\n'
           + "".join(f'    <position class="{name}" name="{a}" joint="{j}" ctrlrange="{r}"/>\n'
                     for a, j, r in DRIVEN)
           + '  </actuator>')
    s = re.sub(r'  <actuator>.*?</actuator>', act, s, count=1, flags=re.S)
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(dst, "robot.xml"), "w").write(s)
    if os.path.isdir(os.path.join(dst, "assets")):
        shutil.rmtree(os.path.join(dst, "assets"))
    shutil.copytree(os.path.join(src, "assets"), os.path.join(dst, "assets"))
    shutil.copy(os.path.join(src, "config.json"), dst)
    shutil.copy(os.path.join(REF, "scene.xml"), dst)


def neutral(model):
    import mujoco
    d = mujoco.MjData(model)
    d.qpos[:] = 0.0
    d.qpos[2] = INIT_Z
    d.qpos[3] = 1.0
    mujoco.mj_forward(model, d)
    return d


def measure(path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(path, "scene.xml"))
    d = neutral(m)
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "easyaxis")
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    out = dict(mass=float(m.body_subtreemass[1]),
               com_ratio=float(d.subtree_com[1][2]) / float(d.xpos[aid][2]),
               torso_mass=float(m.body_mass[tid]),
               torso_I=[float(x) for x in m.body_inertia[tid]],
               torso_ipos=[float(x) for x in m.body_ipos[tid]],
               com=[float(x) for x in d.subtree_com[1]], nu=int(m.nu),
               actuators=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])
    return out


def ballast(dst, target_mass, target_ratio):
    """Add (target_mass - current) to easytorso at the height that gives target_ratio."""
    import mujoco
    p = os.path.join(dst, "robot.xml")
    s = open(p).read()
    m = mujoco.MjModel.from_xml_path(os.path.join(dst, "scene.xml"))
    dm = target_mass - float(m.body_subtreemass[1])
    if abs(dm) < 1e-4:
        print(f"  ballast: mass already {target_mass:.4f} kg, nothing added")
        return
    tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "easytorso")
    d = neutral(m)
    R = d.xmat[tid].reshape(3, 3)
    up_b = R.T @ np.array([0.0, 0.0, 1.0])            # world up in easytorso's frame
    mt = re.search(r'(<inertial pos="([^"]+)" mass="([^"]+)" fullinertia="([^"]+)"/>)',
                   s[s.index('name="easytorso"'):])
    tag, pos_s, mass_s, fi_s = mt.groups()
    p0 = np.array([float(x) for x in pos_s.split()])
    m0 = float(mass_s)
    fi = [float(x) for x in fi_s.split()]              # Ixx Iyy Izz Ixy Ixz Iyz about p0
    I0 = np.array([[fi[0], fi[3], fi[4]], [fi[3], fi[1], fi[5]], [fi[4], fi[5], fi[2]]])
    aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "easyaxis")

    def ratio_with(q):
        m2 = mujoco.MjModel.from_xml_path(os.path.join(dst, "scene.xml"))
        m2.body_mass[tid] = m0 + dm
        m2.body_ipos[tid] = (m0 * p0 + dm * q) / (m0 + dm)
        mujoco.mj_setConst(m2, mujoco.MjData(m2))
        d2 = neutral(m2)
        return float(d2.subtree_com[1][2]) / float(d2.xpos[aid][2])

    lo, hi = -0.30, 0.30
    assert ratio_with(p0 + lo * up_b) < target_ratio < ratio_with(p0 + hi * up_b)
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        if ratio_with(p0 + mid * up_b) < target_ratio:
            lo = mid
        else:
            hi = mid
    q = p0 + 0.5 * (lo + hi) * up_b
    mnew = m0 + dm
    pnew = (m0 * p0 + dm * q) / mnew
    d0, dq = p0 - pnew, q - pnew
    Inew = (I0 + m0 * (np.dot(d0, d0) * np.eye(3) - np.outer(d0, d0))
            + dm * (np.dot(dq, dq) * np.eye(3) - np.outer(dq, dq)))
    fi_new = [Inew[0, 0], Inew[1, 1], Inew[2, 2], Inew[0, 1], Inew[0, 2], Inew[1, 2]]
    new_tag = (f'<inertial pos="{pnew[0]:.10g} {pnew[1]:.10g} {pnew[2]:.10g}" mass="{mnew:.8g}" '
               f'fullinertia="{" ".join(f"{x:.9g}" for x in fi_new)}"/>')
    s = once(s, tag, new_tag)
    zw = (R @ q + d.xpos[tid])[2]
    s = s.replace(" -->\n  <option", f"\n       BALLAST: {dm*1000:+.1f} g added to easytorso by harden_cad.py at world z = "
                  f"{zw*1000:.2f} mm (neutral stand) so the COM ratio is {target_ratio:.4f}; "
                  f"mass {mnew:.4f} kg. -->\n  <option", 1)
    open(p, "w").write(s)
    print(f"  ballast: {dm*1000:+.1f} g at world z {zw*1000:.2f} mm -> easytorso {mnew:.5f} kg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="raw export dir name under models/")
    ap.add_argument("dst", help="output dir name under models/")
    ap.add_argument("--ballast", type=float, default=None, help="target total mass, kg")
    ap.add_argument("--com-ratio", type=float, default=1.05)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    src, dst = os.path.join(HERE, a.src), os.path.join(HERE, a.dst)
    harden(src, dst, a.note)
    if a.ballast is not None:
        ballast(dst, a.ballast, a.com_ratio)
    new, ref = measure(dst), measure(REF)
    print(f"\n{'':22s}{a.dst:>16s}{'hardware_c1':>16s}")
    for k in ("mass", "com_ratio", "torso_mass"):
        print(f"  {k:20s}{new[k]:16.5f}{ref[k]:16.5f}")
    for i in range(3):
        print(f"  torso_I{i+1:<15d}{new['torso_I'][i]:16.6f}{ref['torso_I'][i]:16.6f}")
    print(f"  {'com xyz (m)':20s}{str([round(x, 4) for x in new['com']]):>16s}")
    print(f"  actuators ({new['nu']}): {new['actuators']}")
    print(f"\nwrote {dst}/{{robot.xml, scene.xml, assets/, config.json}}")


if __name__ == "__main__":
    main()
