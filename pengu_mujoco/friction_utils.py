"""
friction_utils.py - Runtime floor-friction override for the surface ablation.

Sits next to gait_config.py (no package / __init__.py). Run scripts from
pengu_mujoco/ and import as:
    from friction_utils import set_floor_friction, SURFACES

Physical model (see scene.xml floor geom + robot.xml foot geoms):
  The foot rubber is very grippy -- real sliding-friction coefficient ~2.8
  (underestimate; rubber on rough surfaces routinely mu>1). The FLOOR is the
  variable, and each SURFACES value is the MEASURED foot-on-that-surface pair
  coefficient, NOT a floor-only number.

  WHY THE FLOOR DICTATES CONTACT mu (and why that is physically right here):
  Real friction is a property of the material PAIR, measured empirically -- it is
  not foot_mu combined with floor_mu by any rule. MuJoCo forces a per-geom mu +
  a combine rule, and its EQUAL-priority rule is the ELEMENTWISE MAXIMUM, which is
  non-physical here: max(foot 2.8, ice 0.06) = 2.8 would mean the robot never
  slips on ice (verified 2026-06: identical trajectories at mu=0.06/0.30/0.70).
  We raise the floor geom's `priority` above the feet (default 0) so the floor's
  mu wins the contact. Because the foot (~2.8) is far grippier than any floor
  (<=0.7), the real interface mu is capped by the slipperier surface (the floor),
  so "floor wins" ~= "the slippery member limits grip" -- the correct limit.
  => effective contact mu = SURFACES[surface]. sim2real accuracy hinges ONLY on
  those SURFACES numbers being real measured foot-on-surface coefficients; the
  foot's 2.8 only matters to confirm foot >> floor (so this approximation holds).
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
