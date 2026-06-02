"""
air_probe.py - Air-phase kinematic probe for Pengu.

Purpose:
  Static two-foot-on-ground tests told us how each DOF affects balance/tilt
  (with both feet pinned). But that is NOT what hip/crank/torso do during
  swing phase, when one foot is in the air. This probe removes gravity and
  contact, sweeps one DOF at a time, and measures each foot's position
  relative to the torso center. Slopes (mm foot motion per +1 deg DOF)
  give a clean kinematic signature for each joint.

Outputs:
  air_probe.png       - 2 x 5 subplot grid of foot motion vs each DOF
  Console summary     - numeric slopes per DOF

Conventions:
  Foot positions are reported in WORLD axes minus torso world position.
  World axes (from gait_config spawn pose):
    +x = lateral (right side of robot)
    +y = forward (robot faces +y)
    +z = up
  Because we disable gravity and pin no body, the root may rotate, but the
  (foot - torso) vector is invariant to that.
"""
import math
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")  # headless-safe: must precede the pyplot import
import matplotlib.pyplot as plt
from gait_config import (
    XML_PATH, build_ids, set_initial_pose, STAND_HIP_DEG
)


# ===================================================================
# Helpers
# ===================================================================

def make_air_model():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # Zero gravity: the robot will float freely.
    model.opt.gravity[:] = 0.0
    # Disable all contact so feet can pass through anything.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    return model


def nominal_ctrl_dict():
    """Baseline: stand pose hips, torso neutral, cranks retracted."""
    return {
        "hip-L":    math.radians(STAND_HIP_DEG),   # -25
        "hip-R":    math.radians(STAND_HIP_DEG),   # -25
        "torso":    0.0,
        "crank1-L": 0.0,
        "crank1-R": 0.0,
    }


def apply_ctrl_dict(data, act_ids, ctrl_dict):
    for k, v in ctrl_dict.items():
        data.ctrl[act_ids[k]] = v


def settle(model, data, n_steps):
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def foot_rel_torso(data, torso_id, lfoot_id, rfoot_id):
    """Return (L, R) foot positions in world axes, with origin at torso."""
    tp = data.xpos[torso_id]
    L = (data.xpos[lfoot_id] - tp).copy()
    R = (data.xpos[rfoot_id] - tp).copy()
    return L, R


# ===================================================================
# Sweep routine
# ===================================================================

def sweep_dof(model, data, act_ids, jnt_adr, body_ids, dof_name, values_deg,
              settle_steps=2000):
    """Sweep one DOF; hold others at nominal. Returns angles, L, R arrays."""
    torso_id, lfoot_id, rfoot_id = body_ids
    Ls, Rs, jnts = [], [], []

    for v_deg in values_deg:
        # Fresh pose
        set_initial_pose(model, data, act_ids, jnt_adr)
        data.qpos[2] += 1.0   # lift 1m high (contact is off anyway)
        mujoco.mj_forward(model, data)

        # Apply nominal + override
        ctrl = nominal_ctrl_dict()
        ctrl[dof_name] = math.radians(v_deg)
        apply_ctrl_dict(data, act_ids, ctrl)

        # Let position actuators + equality constraints settle
        settle(model, data, settle_steps)

        # Read
        L, R = foot_rel_torso(data, torso_id, lfoot_id, rfoot_id)
        Ls.append(L)
        Rs.append(R)
        if dof_name in jnt_adr:
            jnts.append(math.degrees(data.qpos[jnt_adr[dof_name]]))
        else:
            jnts.append(float('nan'))

    return np.array(values_deg), np.array(Ls), np.array(Rs), np.array(jnts)


# ===================================================================
# Main
# ===================================================================

def main():
    model = make_air_model()
    data = mujoco.MjData(model)
    act_ids, jnt_adr = build_ids(model)

    def _bid(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise RuntimeError(f"Body '{name}' not found in model")
        return i

    torso_id = _bid("easytorso")
    lfoot_id = _bid("right_foot0080___fillet13")   # slider-L host = LEFT foot
    rfoot_id = _bid("right_foot0080")              # slider-R host = RIGHT foot
    body_ids = (torso_id, lfoot_id, rfoot_id)

    # ---- Baseline (nominal ctrl) ----
    set_initial_pose(model, data, act_ids, jnt_adr)
    data.qpos[2] += 1.0
    mujoco.mj_forward(model, data)
    apply_ctrl_dict(data, act_ids, nominal_ctrl_dict())
    settle(model, data, 3000)
    L0, R0 = foot_rel_torso(data, torso_id, lfoot_id, rfoot_id)

    print("=" * 70)
    print("AIR-PHASE KINEMATIC PROBE")
    print("=" * 70)
    print(f"Baseline (nominal ctrl: hip=-25, torso=0, crank=0):")
    print(f"  L foot rel torso = ({L0[0]*1000:+6.1f}, {L0[1]*1000:+6.1f}, "
          f"{L0[2]*1000:+6.1f})  [mm]")
    print(f"  R foot rel torso = ({R0[0]*1000:+6.1f}, {R0[1]*1000:+6.1f}, "
          f"{R0[2]*1000:+6.1f})  [mm]")
    print(f"  (world axes: x=lateral, y=forward, z=up)")

    # ---- Sweeps ----
    sweep_defs = {
        "hip-L":    np.linspace(-40, -10, 7),    # stand=-25, +/-15 range
        "hip-R":    np.linspace(-40, -10, 7),
        "torso":    np.linspace(-15,  15, 7),
        "crank1-L": np.linspace(  0,  60, 7),
        "crank1-R": np.linspace(  0,  60, 7),
    }

    results = {}
    for dof, vals in sweep_defs.items():
        print(f"\nSweeping {dof:10s} over {vals[0]:+.0f} .. {vals[-1]:+.0f} deg"
              f" ({len(vals)} pts)...")
        a, L, R, j = sweep_dof(model, data, act_ids, jnt_adr, body_ids,
                               dof, vals, settle_steps=2000)
        results[dof] = (a, L, R, j)
        for vv, ll, rr in zip(a, L, R):
            print(f"  {dof}={vv:+6.1f}:  "
                  f"L=({ll[0]*1000:+6.1f},{ll[1]*1000:+6.1f},{ll[2]*1000:+6.1f})  "
                  f"R=({rr[0]*1000:+6.1f},{rr[1]*1000:+6.1f},{rr[2]*1000:+6.1f})  [mm]")

    # ---- Plot ----
    fig, axes = plt.subplots(2, 5, figsize=(24, 8))
    for i, (dof, (a, L, R, j)) in enumerate(results.items()):
        ax = axes[0, i]
        ax.plot(a, L[:, 0] * 1000, 'r-o', label='dx (lat)')
        ax.plot(a, L[:, 1] * 1000, 'g-o', label='dy (fwd)')
        ax.plot(a, L[:, 2] * 1000, 'b-o', label='dz (up)')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_title(f'{dof}  -> L foot rel torso')
        ax.set_xlabel(f'{dof} cmd [deg]')
        ax.set_ylabel('L foot rel torso [mm]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        ax = axes[1, i]
        ax.plot(a, R[:, 0] * 1000, 'r-o', label='dx (lat)')
        ax.plot(a, R[:, 1] * 1000, 'g-o', label='dy (fwd)')
        ax.plot(a, R[:, 2] * 1000, 'b-o', label='dz (up)')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.set_title(f'{dof}  -> R foot rel torso')
        ax.set_xlabel(f'{dof} cmd [deg]')
        ax.set_ylabel('R foot rel torso [mm]')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig('air_probe.png', dpi=130, bbox_inches='tight')
    print("\n[Saved] air_probe.png")

    # ---- Numeric slope summary ----
    print("\n" + "=" * 70)
    print("SLOPE SUMMARY  (mm foot motion per +1 deg DOF, linear fit endpoints)")
    print("=" * 70)
    print("x = lateral (robot right)   y = forward   z = up")
    print()
    for dof, (a, L, R, j) in results.items():
        da = a[-1] - a[0]
        Lsl = (L[-1] - L[0]) / da * 1000
        Rsl = (R[-1] - R[0]) / da * 1000
        print(f"  {dof:10s}")
        print(f"    L foot:  dx={Lsl[0]:+6.2f}  dy={Lsl[1]:+6.2f}  dz={Lsl[2]:+6.2f}   mm/deg")
        print(f"    R foot:  dx={Rsl[0]:+6.2f}  dy={Rsl[1]:+6.2f}  dz={Rsl[2]:+6.2f}   mm/deg")
    print("=" * 70)
    print("\nInterpretation notes:")
    print("  * |slope| < ~0.1 mm/deg -> DOF essentially doesn't move that foot")
    print("  * Large dy slope -> DOF swings the foot forward/backward (good for")
    print("    step-taking)")
    print("  * Large dz slope -> DOF lifts/lowers the foot (good for clearance)")
    print("  * Large dx slope -> DOF moves the foot laterally")


if __name__ == "__main__":
    main()