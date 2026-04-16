"""
debug_pengu.py  –  Print diagnostic info about the penguin's initial state

Run from: ~/Documents/ben_gu/ben_pengu/pengu
Usage:     python debug_pengu.py
"""

import numpy as np
import mujoco

XML_PATH = "penguV2/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("=" * 60)
print("PENGUIN DIAGNOSTICS")
print("=" * 60)

# 1. Freejoint initial qpos
print("\n--- Freejoint (root) initial qpos ---")
print(f"  Position (x,y,z): {data.qpos[0:3]}")
print(f"  Quaternion (w,x,y,z): {data.qpos[3:7]}")

# 2. All body positions
print("\n--- Body positions (xipos) ---")
for i in range(model.nbody):
    name = model.body(i).name
    pos = data.xipos[i]
    print(f"  {name:30s}  x={pos[0]:+.4f}  y={pos[1]:+.4f}  z={pos[2]:+.4f}")

# 3. World COM
masses = model.body_mass
total_mass = float(np.sum(masses))
com = (masses[:, None] * data.xipos).sum(axis=0) / total_mass
print(f"\n--- World COM ---")
print(f"  COM = ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
print(f"  Total mass = {total_mass:.4f} kg")

# 4. Torso orientation
print("\n--- Body orientations (z-column of rotation matrix = 'up' vector) ---")
for bname in ["leftthighmotor", "easyaxis", "rightthighmotor", "easytorso"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid < 0:
        continue
    R = data.xmat[bid].reshape(3, 3)
    up = R[:, 2]  # z-column = local z in world frame
    fwd = R[:, 1]  # y-column
    print(f"  {bname:25s}  up={up}  fwd(y-col)={fwd}")

# 5. Feet positions
print("\n--- Foot positions ---")
for bname in ["right_foot0080", "right_foot0080___fillet13"]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    if bid >= 0:
        pos = data.xipos[bid]
        print(f"  {bname:40s}  z={pos[2]:+.4f}")

# 6. Find lowest geom
print("\n--- Lowest geom (contact with ground) ---")
min_z = float('inf')
min_name = ""
for gid in range(model.ngeom):
    name = model.geom(gid).name
    if name == "floor":
        continue
    center_z = data.geom_xpos[gid][2]
    rbound = model.geom_rbound[gid]
    lowest = center_z - rbound
    if lowest < min_z:
        min_z = lowest
        min_name = f"geom {gid} (body: {model.body(model.geom_bodyid[gid]).name})"
print(f"  Lowest point: {min_name} at z={min_z:+.4f}")

# 7. Joint info
print("\n--- Joint qpos at init ---")
for j in range(model.njnt):
    jnt = model.joint(j)
    adr = model.jnt_qposadr[j]
    if jnt.type[0] == 0:  # free joint
        print(f"  {jnt.name:25s}  [freejoint] pos={data.qpos[adr:adr+3]}  quat={data.qpos[adr+3:adr+7]}")
    else:
        print(f"  {jnt.name:25s}  qpos={data.qpos[adr]:.4f}")

# 8. Gravity
print(f"\n--- Gravity ---")
print(f"  {model.opt.gravity}")

print("\n" + "=" * 60)
print("Copy-paste this output back to Claude!")
print("=" * 60)