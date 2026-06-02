"""
physics_probe.py - Run physics identification experiments headlessly.

Experiment #3: Static torso direction mapping (which way does body roll?)
Experiment #1: Torso frequency sweep (find resonance)

Outputs:
  - exp3_direction.png
  - exp1_resonance.png
  - Printed summary
"""

import math
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")  # headless-safe: must precede the pyplot import
import matplotlib.pyplot as plt
from gait_config import XML_PATH, build_ids, set_initial_pose


# ===================== Experiment #3: Direction mapping =====================
def run_direction_mapping():
    """
    Hold torso_ctrl at a ramped +15 deg, see which way body rolls.
    Keeps hip and crank at zero, no oscillation on torso.

    Roll metric reasoning:
      At spawn, body +x ~ world +x, body +y ~ world +y (init_pitch rotates
      about world y, which does NOT couple body x into world z).
      A pure body-frame roll about body +x rotates body +y toward world +z
      or -z, and rotates body +z toward world -y or +y.
      So the roll-sensitive entries of the body->world rotation matrix are:
        mat[1,0] = bx[y] = world-y component of body +x  (small, ~0 for pure roll)
        mat[1,2] = bz[y] = world-y component of body +z  (key roll indicator)
        mat[2,1] = by[z] = world-z component of body +y  (key roll indicator)
      The original logging of mat[2,0]=bx[z] and mat[2,1]=by[z] conflated
      pitch and roll; here we log all of them for disambiguation.
    """
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    act_ids, jnt_adr = build_ids(model)
    set_initial_pose(model, data, act_ids, jnt_adr)

    # Timing: 2s hold zero, 2s ramp to +15 deg, 5s hold +15 deg
    T_HOLD_ZERO = 2.0
    T_RAMP = 2.0
    T_HOLD_POS = 5.0
    T_END = T_HOLD_ZERO + T_RAMP + T_HOLD_POS
    TARGET_DEG = 15.0

    log = {"t": [], "ctrl": [], "jnt": [],
           # pitch-sensitive (kept for continuity with earlier run)
           "bx_z": [], "by_z": [],
           # roll-sensitive (new)
           "bx_y": [], "bz_y": [],
           "base_x": [], "base_y": [], "base_z": []}

    while data.time < T_END:
        t = data.time
        # Torso ctrl schedule: 0 -> ramp -> +15 -> hold
        if t < T_HOLD_ZERO:
            torso_cmd_deg = 0.0
        elif t < T_HOLD_ZERO + T_RAMP:
            alpha = (t - T_HOLD_ZERO) / T_RAMP
            torso_cmd_deg = TARGET_DEG * alpha
        else:
            torso_cmd_deg = TARGET_DEG

        # Apply: zero everything except torso
        data.ctrl[act_ids["hip-L"]] = 0.0
        data.ctrl[act_ids["hip-R"]] = 0.0
        data.ctrl[act_ids["crank1-L"]] = 0.0
        data.ctrl[act_ids["crank1-R"]] = 0.0
        data.ctrl[act_ids["torso"]] = math.radians(torso_cmd_deg)

        # Log every ~20ms
        if len(log["t"]) == 0 or t - log["t"][-1] >= 0.02:
            quat = np.array([data.qpos[3], data.qpos[4],
                             data.qpos[5], data.qpos[6]])
            mat = np.zeros(9)
            mujoco.mju_quat2Mat(mat, quat)
            mat = mat.reshape(3, 3)
            log["t"].append(t)
            log["ctrl"].append(torso_cmd_deg)
            log["jnt"].append(math.degrees(data.qpos[jnt_adr["torso"]]))
            # Pitch-sensitive (body axis -> world z)
            log["bx_z"].append(mat[2, 0])
            log["by_z"].append(mat[2, 1])
            # Roll-sensitive (body axis -> world y)
            log["bx_y"].append(mat[1, 0])
            log["bz_y"].append(mat[1, 2])
            log["base_x"].append(data.qpos[0])
            log["base_y"].append(data.qpos[1])
            log["base_z"].append(data.qpos[2])

        mujoco.mj_step(model, data)

    return log


def plot_direction_mapping(log):
    t = np.array(log["t"])

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    ax = axes[0]
    ax.plot(t, log["ctrl"], 'k--', lw=2, label='torso_ctrl cmd')
    ax.plot(t, log["jnt"], 'b-', lw=2, label='torso_jnt actual')
    ax.set_ylabel('Torso angle [deg]')
    ax.set_title('Exp #3: Direction mapping - static torso at +15 deg',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Roll-sensitive axes (the new, correct metric)
    ax = axes[1]
    ax.plot(t, log["bx_y"], 'r-', lw=2, label='bx[y] = mat[1,0]')
    ax.plot(t, log["bz_y"], 'b-', lw=2, label='bz[y] = mat[1,2]')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Body-axis y component')
    ax.set_title('ROLL-sensitive metric (body axis -> world y)',
                 fontweight='bold', color='darkred')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pitch-sensitive axes (for comparison / sanity)
    ax = axes[2]
    ax.plot(t, log["bx_z"], 'r-', lw=2, label='bx[z] = mat[2,0]')
    ax.plot(t, log["by_z"], 'g-', lw=2, label='by[z] = mat[2,1]')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel('Body-axis z component')
    ax.set_title('Pitch-sensitive metric (body axis -> world z) - for reference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    ax.plot(t, log["base_x"], 'c-', lw=2, label='base_x (lateral)')
    ax.plot(t, log["base_y"], 'm-', lw=2, label='base_y (forward)')
    ax.plot(t, log["base_z"], 'y-', lw=2, label='base_z (height)')
    ax.set_ylabel('Position [m]')
    ax.set_xlabel('Time [s]')
    ax.set_title('Base position drift during static torso hold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('exp3_direction.png', dpi=150, bbox_inches='tight')
    print("[Saved] exp3_direction.png")

    # Print summary
    final_bx_y = log["bx_y"][-1]
    final_bz_y = log["bz_y"][-1]
    final_bx_z = log["bx_z"][-1]
    final_by_z = log["by_z"][-1]
    final_base_x = log["base_x"][-1]
    final_base_y = log["base_y"][-1]
    final_base_z = log["base_z"][-1]
    final_jnt = log["jnt"][-1]

    def _classify(v, label_pos, label_neg, thresh=0.05):
        if v > thresh:
            return label_pos
        if v < -thresh:
            return label_neg
        return "~0 (no signal)"

    print("\n" + "=" * 60)
    print("EXP #3 SUMMARY - Direction mapping (corrected metrics)")
    print("=" * 60)
    print(f"Final torso_ctrl: +15.00 deg")
    print(f"Final torso_jnt:  {final_jnt:+.2f} deg")
    print()
    print("--- ROLL-sensitive (body axis -> world y) ---")
    print(f"  bx[y] = {final_bx_y:+.3f}  "
          f"({_classify(final_bx_y, 'body +x tilts toward world +y', 'body +x tilts toward world -y')})")
    print(f"  bz[y] = {final_bz_y:+.3f}  "
          f"({_classify(final_bz_y, 'body +z tilts toward world +y', 'body +z tilts toward world -y')})")
    print()
    print("--- PITCH-sensitive (body axis -> world z), for reference ---")
    print(f"  bx[z] = {final_bx_z:+.3f}  "
          f"({_classify(final_bx_z, 'body +x tilts UP', 'body +x tilts DOWN')})")
    print(f"  by[z] = {final_by_z:+.3f}  "
          f"({_classify(final_by_z, 'body +y tilts UP', 'body +y tilts DOWN')})")
    print()
    print("--- Base position ---")
    print(f"  base_x = {final_base_x:+.3f} m  (lateral drift)")
    print(f"  base_y = {final_base_y:+.3f} m  (forward drift)")
    print(f"  base_z = {final_base_z:+.3f} m  (height)")
    print()
    print("Interpretation guide:")
    print("  Body +x is the lateral (left-right) axis.")
    print("  A pure roll about body +x rotates body +z into world +/-y.")
    print("  If bz[y] > 0, the top of the body tips toward world +y side.")
    print("  Pair this with the viewer visual (walk_pengu.py STATIC_TORSO_DEG)")
    print("  to confirm which side is 'left' vs 'right' in your mental model.")
    print("=" * 60)


# ===================== Experiment #1: Resonance sweep =====================
def run_frequency_sweep():
    """
    Sweep torso sine frequency from 0.3 to 3.0 Hz.
    At each frequency, drive torso at amp=5 deg, hold ~5 full cycles,
    then measure the actual joint amplitude / ctrl amplitude ratio.
    Peak ratio -> resonance.
    """
    FREQS_HZ = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.64, 1.8,
                2.0, 2.2, 2.5, 2.8, 3.0]
    AMP_DEG = 5.0
    T_SETTLE = 2.0
    N_MEASURE_CYCLES = 4   # measure over 4 cycles

    results = []  # list of dicts

    for freq in FREQS_HZ:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        data = mujoco.MjData(model)
        act_ids, jnt_adr = build_ids(model)
        set_initial_pose(model, data, act_ids, jnt_adr)

        period = 1.0 / freq
        T_MEASURE = N_MEASURE_CYCLES * period
        T_END = T_SETTLE + T_MEASURE

        log_t = []
        log_ctrl = []
        log_jnt = []

        while data.time < T_END:
            t = data.time
            phase = 2 * math.pi * freq * t
            torso_cmd_deg = AMP_DEG * math.sin(phase)

            data.ctrl[act_ids["hip-L"]] = 0.0
            data.ctrl[act_ids["hip-R"]] = 0.0
            data.ctrl[act_ids["crank1-L"]] = 0.0
            data.ctrl[act_ids["crank1-R"]] = 0.0
            data.ctrl[act_ids["torso"]] = math.radians(torso_cmd_deg)

            if len(log_t) == 0 or t - log_t[-1] >= 0.005:
                log_t.append(t)
                log_ctrl.append(torso_cmd_deg)
                log_jnt.append(math.degrees(data.qpos[jnt_adr["torso"]]))

            mujoco.mj_step(model, data)

        # Only use the measurement window (after settle)
        t_arr = np.array(log_t)
        ctrl_arr = np.array(log_ctrl)
        jnt_arr = np.array(log_jnt)
        mask = t_arr >= T_SETTLE
        jnt_window = jnt_arr[mask]
        ctrl_window = ctrl_arr[mask]

        # Amplitude = (max - min) / 2
        cmd_amp = (ctrl_window.max() - ctrl_window.min()) / 2.0
        act_amp = (jnt_window.max() - jnt_window.min()) / 2.0
        gain = act_amp / cmd_amp if cmd_amp > 1e-6 else 0.0

        # Cross-correlation to estimate phase lag
        # Normalize both signals, then find lag that maximizes correlation
        t_window = t_arr[mask]
        ctrl_norm = (ctrl_window - ctrl_window.mean()) / (ctrl_window.std() + 1e-9)
        jnt_norm = (jnt_window - jnt_window.mean()) / (jnt_window.std() + 1e-9)
        corr = np.correlate(jnt_norm, ctrl_norm, mode='full')
        lags = np.arange(-len(ctrl_norm) + 1, len(ctrl_norm))
        best_lag_idx = np.argmax(corr)
        best_lag_samples = lags[best_lag_idx]
        dt = t_window[1] - t_window[0] if len(t_window) > 1 else 0.001
        lag_seconds = best_lag_samples * dt
        lag_deg = (lag_seconds / period) * 360.0

        results.append({
            "freq": freq,
            "cmd_amp": cmd_amp,
            "act_amp": act_amp,
            "gain": gain,
            "lag_deg": lag_deg,
        })
        print(f"  freq={freq:5.2f} Hz | cmd_amp={cmd_amp:.2f} | "
              f"act_amp={act_amp:5.2f} | gain={gain:.3f} | lag={lag_deg:+6.1f} deg")

    return results


def plot_frequency_sweep(results):
    freqs = [r["freq"] for r in results]
    gains = [r["gain"] for r in results]
    lags = [r["lag_deg"] for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    ax.plot(freqs, gains, 'b-o', lw=2, markersize=7)
    ax.axhline(1.0, color='k', lw=0.5, ls=':', label='unity gain')
    ax.set_ylabel('Gain (act_amp / cmd_amp)')
    ax.set_title('Exp #1: Torso frequency sweep - gain vs frequency',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Mark peak
    peak_idx = int(np.argmax(gains))
    ax.axvline(freqs[peak_idx], color='r', lw=1, ls='--',
               label=f'peak at {freqs[peak_idx]:.2f} Hz')
    ax.legend()

    ax = axes[1]
    ax.plot(freqs, lags, 'g-o', lw=2, markersize=7)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axhline(-90, color='r', lw=0.5, ls=':', label='-90 (resonance)')
    ax.set_ylabel('Phase lag [deg]')
    ax.set_xlabel('Drive frequency [Hz]')
    ax.set_title('Phase lag (actual relative to cmd)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('exp1_resonance.png', dpi=150, bbox_inches='tight')
    print("[Saved] exp1_resonance.png")

    print("\n" + "=" * 60)
    print("EXP #1 SUMMARY - Resonance sweep")
    print("=" * 60)
    print(f"Peak gain:   {gains[peak_idx]:.3f} at {freqs[peak_idx]:.2f} Hz")
    print(f"Gain at 1.64 Hz (your current WALK_FREQ): "
          f"{next(r['gain'] for r in results if abs(r['freq']-1.64)<0.01):.3f}")
    print(f"Lag at 1.64 Hz: "
          f"{next(r['lag_deg'] for r in results if abs(r['freq']-1.64)<0.01):+.1f} deg")
    print("\nInterpretation:")
    print("  If peak gain > 1.5 and lag at peak is near -90 deg,")
    print("    the system is underdamped and has a clear resonance.")
    print("  If gain is flat ~1 and lag grows with freq,")
    print("    no resonance, just a low-pass actuator.")
    print("  If gain drops monotonically,")
    print("    you're already above resonance everywhere.")
    print("=" * 60)


# ===================== Main =====================
if __name__ == "__main__":
    print("\n[Exp #3] Running direction mapping...")
    log3 = run_direction_mapping()
    plot_direction_mapping(log3)

    print("\n[Exp #1] Running frequency sweep (this takes ~1 min)...")
    results1 = run_frequency_sweep()
    plot_frequency_sweep(results1)

    print("\n[Done] Both experiments complete.")
    print("  View: exp3_direction.png, exp1_resonance.png")