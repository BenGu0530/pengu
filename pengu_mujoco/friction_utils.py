"""
friction_utils.py - Runtime floor-friction override for the surface ablation.

Sits next to gait_config.py (no package / __init__.py). Run scripts from
pengu_mujoco/ and import as:
    from friction_utils import set_floor_friction, SURFACES

Physical model (see scene.xml floor geom + robot.xml foot geoms):
  Foot collision geoms have a fixed sliding friction of 0.9 (rubbery plastic).
  The FLOOR is the variable. MuJoCo uses min(geom1, geom2) for a contact pair,
  so as long as the floor mu < 0.9, the floor dominates every foot-floor contact.
  Only the sliding coefficient (friction[:,0]) is varied; torsional/rolling are
  left at MuJoCo defaults.
"""
import mujoco

# Estimated sliding-friction coefficients for the 4 ablation surfaces.
SURFACES = {
    "mocap_floor": 0.5,    # baseline (mu ~ 0.4-0.7)
    "acrylic":     0.30,
    "uhmw_pe":     0.14,
    "ptfe_ice":    0.06,   # acrylic + PTFE film, ice-analog (mu ~ 0.04-0.08)
}


def set_floor_friction(model, mu_sliding):
    """Override floor sliding friction at runtime. Returns the geom_id modified."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("floor geom not found")
    model.geom_friction[floor_id, 0] = mu_sliding
    return floor_id
