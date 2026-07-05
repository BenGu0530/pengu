"""
friction_utils.py - Runtime floor-friction override for the surface ablation.

Sits next to gait_config.py (no package / __init__.py). Run scripts from
pengu_mujoco/ and import as:
    from friction_utils import set_floor_friction, SURFACES

Physical model (see scene.xml floor geom + robot.xml foot geoms):
  Foot collision geoms have a fixed sliding friction of 0.9 (rubbery plastic).
  The FLOOR is the variable.

  IMPORTANT (fixed 2026-06): for two geoms of EQUAL priority MuJoCo combines
  friction by the ELEMENTWISE MAXIMUM, not the minimum. So merely lowering the
  floor's mu below 0.9 did NOTHING -- max(foot 0.9, floor) = 0.9 always, and the
  floor setting was silently ignored (verified: identical trajectories at
  mu=0.06/0.30/0.70). To make the floor actually dictate contact friction we
  raise the floor geom's `priority` above the feet (default 0); the higher-
  priority geom's friction then wins the contact, so effective mu = floor mu.
  Only the sliding coefficient (friction[:,0]) is varied; torsional/rolling are
  left at MuJoCo defaults.
"""
import mujoco

# Estimated sliding-friction coefficients for the 4 ablation surfaces.
SURFACES = {
    "mocap_floor": 0.7,    # baseline (mu ~ 0.4-0.7, upper end)
    "acrylic":     0.30,
    "uhmw_pe":     0.14,
    "ptfe_ice":    0.06,   # acrylic + PTFE film, ice-analog (mu ~ 0.04-0.08)
}


def set_floor_friction(model, mu_sliding):
    """Override floor sliding friction at runtime so the FLOOR dictates the
    foot-floor contact friction. Returns the floor geom_id.

    Sets the floor's sliding friction AND raises its contact `priority` above the
    feet, otherwise MuJoCo's equal-priority elementwise-max rule keeps the foot's
    0.9 and ignores the floor (see module docstring)."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        raise ValueError("floor geom not found")
    model.geom_friction[floor_id, 0] = mu_sliding
    model.geom_priority[floor_id] = 1          # floor wins the contact -> mu = floor mu
    return floor_id
