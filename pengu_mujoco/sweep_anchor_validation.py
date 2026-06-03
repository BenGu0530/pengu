# Validates that sim reproduces known real-robot walking anchors.
# Real-machine ground-truth configs (from pengu_mujoco/videos/ filenames):
#   Isolated single-signal modes (verify sim's natural freq):
#     - hip_only:        hip_amp=11°,  crank=0,    torso=0    (freq swept)
#     - crank_only:      hip=0,        crank=104°, torso=0    (freq swept)
#   4-dof combined modes (hip + crank, no torso; real machine ran freq=1.25):
#     - 4dof_c73_h12:    hip=12°,      crank=73°,  torso=0
#     - 4dof_c77_h17:    hip=17°,      crank=77°,  torso=0
#   5-dof all-signal modes (real machine ran freq=1.25):
#     - 5dof_t1_c73_h12: hip=12°,      crank=73°,  torso=1°
#     - 5dof_t9_c73_h12: hip=12°,      crank=73°,  torso=9°
#
# Run from pengu_mujoco/:
#   conda activate mujoco
#   nohup python sweep_anchor_validation.py > sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Tail progress:
#   tail -f results/sweep_anchor_validation_<latest>/progress.log
"""
sweep_anchor_validation.py - Independent 1D fine-frequency sweeps to check
whether sim reproduces the known real-robot walking anchors (sim-to-real
validation, NOT exploration).

Real-machine validation across isolated and combined modes: the ISOLATED modes
(hip_only, crank_only) verify sim's natural walking frequency for a single
signal; the COMBINED modes (4dof hip+crank, 5dof hip+crank+torso) verify
multi-signal coordination at the real-machine amplitudes.

One independent 1D sweep per anchor (NO grid): for each anchor we hold the three
amplitudes fixed at the real-robot values and sweep frequency at a fine 0.01 Hz
step to locate sim's natural walking frequency. Even though the combined modes
ran at freq=1.25 on the real machine, we sweep freq to find sim's natural
resonance for direct comparison. Uses the CURRENT gait controller
(gait_config.apply_ctrl, same as walk_pengu.py) and the explicit friction model
(foot mu 0.9 fixed, floor mu set per surface via friction_utils). One fresh
model load per trial.
"""
import os
import csv
import math
import time
import socket
import platform
import subprocess
from datetime import datetime

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")          # headless only, never opens a window
import matplotlib.pyplot as plt

import gait_config as gc
from friction_utils import set_floor_friction, SURFACES

# ===================================================================
#  SWEEP CONFIG  (edit here)
# ===================================================================
# Anchors come from real-robot data — known walking configurations.
# Sweep frequency at fine step around each anchor to find sim's natural freq.

ANCHORS = [
    # Isolated single-signal modes — sweep freq to find sim's natural freq
    {
        "name": "hip_only",
        "hip_amp_deg":   11.0,
        "crank_amp_deg":  0.0,
        "torso_amp_deg":  0.0,
    },
    {
        "name": "crank_only",
        "hip_amp_deg":    0.0,
        "crank_amp_deg":104.0,
        "torso_amp_deg":  0.0,
    },
    # 4-dof combined modes (hip + crank, no torso) — real machine used freq=1.25
    # but we sweep freq to find sim's natural resonance for direct comparison
    {
        "name": "4dof_c73_h12",
        "hip_amp_deg":   12.0,
        "crank_amp_deg": 73.0,
        "torso_amp_deg":  0.0,
    },
    {
        "name": "4dof_c77_h17",
        "hip_amp_deg":   17.0,
        "crank_amp_deg": 77.0,
        "torso_amp_deg":  0.0,
    },
    # 5-dof all-signal modes — real machine used freq=1.25
    {
        "name": "5dof_t1_c73_h12",
        "hip_amp_deg":   12.0,
        "crank_amp_deg": 73.0,
        "torso_amp_deg":  1.0,
    },
    {
        "name": "5dof_t9_c73_h12",
        "hip_amp_deg":   12.0,
        "crank_amp_deg": 73.0,
        "torso_amp_deg":  9.0,
    },
]

FREQ_MIN  = 1.0    # Hz
FREQ_MAX  = 2.2    # Hz
FREQ_STEP = 0.01   # fine step — Pengu is non-linear, regime changes are narrow

SURFACE   = "mocap_floor"   # μ = 0.7 baseline, single surface for this validation
SIM_DURATION = 20.0          # 5s stand + 2s transition + 13s walk window
N_TRIALS  = 1                # validation, not statistics

# ===================================================================
#  Constants
# ===================================================================
XML_PATH = "penguV2/scene.xml"     # relative, like walk_pengu.py (run from pengu_mujoco/)
ROOT_BODY = "leftthighmotor"       # floating-base body (per walk_pengu.py / sweep_freq_surface.py)
FALL_Z = 0.05                      # root z below this => fallen
ROLL_SETTLE = 2.0                  # ignore this many seconds of walk before measuring roll
PROGRESS_THRESH = 0.3              # m of fwd progress for the "low-roll-with-progress" filter

CSV_FIELDS = [
    "anchor_name", "hip_amp", "crank_amp", "torso_amp", "freq_hz", "mu_floor",
    "survived", "fall_time", "walk_time",
    "dist_xy", "dist_fwd", "dist_lat",
    "mean_speed_fwd", "torso_roll_amp_deg", "pitch_offset_deg", "pitch_amp_deg", "error_msg",
]


def _git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_trial(idx, n_total, anchor, freq, sim_duration, log):
    """Run one (anchor, freq) trial. Catches ALL exceptions -> error_msg.
    Returns a dict with the CSV_FIELDS keys."""
    res = {
        "anchor_name": anchor["name"],
        "hip_amp": anchor["hip_amp_deg"],
        "crank_amp": anchor["crank_amp_deg"],
        "torso_amp": anchor["torso_amp_deg"],
        "freq_hz": round(freq, 4),
        "mu_floor": float("nan"),
        "survived": False, "fall_time": float("nan"), "walk_time": 0.0,
        "dist_xy": 0.0, "dist_fwd": 0.0, "dist_lat": 0.0,
        "mean_speed_fwd": float("nan"),
        "torso_roll_amp_deg": float("nan"),
        "pitch_offset_deg": float("nan"), "pitch_amp_deg": float("nan"),
        "error_msg": "",
    }
    t_wall0 = time.perf_counter()
    sim_t = 0.0
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        floor_id = set_floor_friction(model, SURFACES[SURFACE])
        res["mu_floor"] = float(model.geom_friction[floor_id, 0])

        data = mujoco.MjData(model)
        act_ids, jnt_adr = gc.build_ids(model)

        # Isolated-signal anchor: set the three amplitudes + frequency BEFORE the
        # loop. apply_ctrl/compute_gait read these as module globals every step.
        gc.set_hip_amp(anchor["hip_amp_deg"])
        gc.set_crank_amp(anchor["crank_amp_deg"])
        gc.set_torso_amp(anchor["torso_amp_deg"])
        gc.set_walk_freq(freq)
        gc.set_initial_pose(model, data, act_ids, jnt_adr)   # calls mj_forward

        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        if root_id < 0:
            raise RuntimeError(f"root body '{ROOT_BODY}' not found")

        # Roll / pitch reference: world-up expressed in the root body frame at
        # spawn, then re-expressed in world each step. With no rotation since
        # spawn this is [0,0,1]; as the root leans it tilts away. Measuring
        # RELATIVE to spawn makes the -30 deg spawn pitch the zero reference.
        #   roll(t)  = atan2(up[0], up[2])  -> lateral lean about forward (y)
        #   pitch(t) = atan2(up[1], up[2])  -> fwd/back lean about lateral (x)
        #     + = leaning MORE forward than spawn,  - = recovering toward upright
        R0 = data.xmat[root_id].reshape(3, 3).copy()
        up_local = R0.T @ np.array([0.0, 0.0, 1.0])

        walk_start = gc.T_HOLD + gc.T_TRANSITION
        roll_min = roll_max = None
        pitch_min = pitch_max = None
        pos_ws = None
        last_pos = data.xpos[root_id][:2].copy()

        while data.time < sim_duration:
            gc.apply_ctrl(data, act_ids, data.time)
            mujoco.mj_step(model, data)
            sim_t = data.time
            p = data.xpos[root_id]
            last_pos = p[:2].copy()

            if p[2] < FALL_Z:                        # fell
                res["fall_time"] = float(data.time)
                break
            if pos_ws is None and data.time >= walk_start:
                pos_ws = p[:2].copy()
            # roll + pitch off the spawn-relative up vector; skip the first
            # ROLL_SETTLE s of walk so the gait settles before we measure.
            if data.time >= walk_start + ROLL_SETTLE:
                R = data.xmat[root_id].reshape(3, 3)
                up = R @ up_local
                roll = math.degrees(math.atan2(up[0], up[2]))
                roll_min = roll if roll_min is None else min(roll_min, roll)
                roll_max = roll if roll_max is None else max(roll_max, roll)
                # pitch sign: + = leaning MORE forward than spawn,
                #             - = recovering toward upright (or past it)
                pitch = math.degrees(math.atan2(up[1], up[2]))
                pitch_min = pitch if pitch_min is None else min(pitch_min, pitch)
                pitch_max = pitch if pitch_max is None else max(pitch_max, pitch)

        survived = math.isnan(res["fall_time"])
        res["survived"] = bool(survived)
        end_time = sim_t if survived else res["fall_time"]
        res["walk_time"] = max(0.0, end_time - walk_start)

        if pos_ws is not None:
            dx = float(last_pos[0] - pos_ws[0])
            dy = float(last_pos[1] - pos_ws[1])
            res["dist_lat"] = dx
            res["dist_fwd"] = dy
            res["dist_xy"] = math.hypot(dx, dy)
            if res["walk_time"] > 1e-6:
                res["mean_speed_fwd"] = dy / res["walk_time"]
        if roll_min is not None:
            res["torso_roll_amp_deg"] = (roll_max - roll_min) / 2.0
        if pitch_min is not None:
            # Metric A: most extreme signed deviation from spawn pitch
            res["pitch_offset_deg"] = (pitch_max if abs(pitch_max) >= abs(pitch_min)
                                       else pitch_min)
            # Metric B: pitch oscillation amplitude (peak-to-peak / 2)
            res["pitch_amp_deg"] = (pitch_max - pitch_min) / 2.0

    except Exception as e:                            # never let one trial kill the sweep
        res["error_msg"] = f"{type(e).__name__}: {e}"

    wall = time.perf_counter() - t_wall0
    line = (f"[{idx}/{n_total}] anchor={anchor['name']:<11} freq={freq:.3f} "
            f"survived={res['survived']} dist={res['dist_fwd']:.3f} "
            f"roll={res['torso_roll_amp_deg']:.2f}  ({sim_t:.1f}s sim, {wall:.1f}s wall)")
    if res["error_msg"]:
        line += f"  ERROR: {res['error_msg']}"
    log(line)
    return res


def _fmt_row(r):
    return (f"freq={r['freq_hz']:.3f} Hz  dist_fwd={r['dist_fwd']:+.3f} m  "
            f"roll={r['torso_roll_amp_deg']:.2f}°  poff={r['pitch_offset_deg']:+.2f}°  "
            f"pamp={r['pitch_amp_deg']:.2f}°  survived={r['survived']}")


def write_summary(path, results):
    lines = []
    lines.append("=" * 72)
    lines.append("ANCHOR VALIDATION SWEEP - SUMMARY")
    lines.append("=" * 72)
    n = len(results)
    n_err = sum(1 for r in results if r["error_msg"])
    n_surv = sum(1 for r in results if r["survived"])
    lines.append(f"surface={SURFACE}  trials={n}  survived={n_surv}  "
                 f"fell={n - n_surv - n_err}  errored={n_err}")
    lines.append("rankings labelled '(survived only)' exclude fallen/errored trials.")
    lines.append("")

    for anchor in ANCHORS:
        name = anchor["name"]
        rows = [r for r in results if r["anchor_name"] == name]
        if not rows:
            lines.append(f"[{name}] (no trials)")
            lines.append("")
            continue
        surv = [r for r in rows if r["survived"]]
        lines.append("-" * 72)
        lines.append(f"[{name}]  hip={anchor['hip_amp_deg']:.1f}°  "
                     f"crank={anchor['crank_amp_deg']:.1f}°  torso={anchor['torso_amp_deg']:.1f}°")
        lines.append(f"  trials={len(rows)}  survived={len(surv)}  fell={len(rows) - len(surv)}")
        if surv:
            sf = [r["freq_hz"] for r in surv]
            lines.append(f"  survived freq range: {min(sf):.3f}-{max(sf):.3f} Hz")
        else:
            lines.append("  survived freq range: (none survived)")

        # top-10 by fwd distance (candidate natural-freq points), survivors only
        top_dist = sorted(surv, key=lambda r: r["dist_fwd"], reverse=True)[:10]
        lines.append("  top-10 by dist_fwd (candidate natural-freq points, survived only):")
        if top_dist:
            for r in top_dist:
                lines.append(f"      {_fmt_row(r)}")
        else:
            lines.append("      (no survivors)")

        # top-10 by lowest roll among trials with real fwd progress (eyeball criterion)
        prog = [r for r in surv
                if r["dist_fwd"] > PROGRESS_THRESH
                and not math.isnan(r["torso_roll_amp_deg"])]
        top_roll = sorted(prog, key=lambda r: r["torso_roll_amp_deg"])[:10]
        lines.append(f"  top-10 by LOW roll among dist_fwd>{PROGRESS_THRESH} m "
                     f"(walk progress AND low roll):")
        if top_roll:
            for r in top_roll:
                lines.append(f"      {_fmt_row(r)}")
        else:
            lines.append(f"      (no survivors with dist_fwd>{PROGRESS_THRESH} m)")
        lines.append("")

    # ---- cross-anchor summary ----
    lines.append("=" * 72)
    lines.append("CROSS-ANCHOR")
    lines.append("=" * 72)
    all_surv = [r for r in results if r["survived"]]
    if all_surv:
        best_d = max(all_surv, key=lambda r: r["dist_fwd"])
        lines.append(f"  best by dist_fwd (survived only): anchor={best_d['anchor_name']}  "
                     f"{_fmt_row(best_d)}")
    else:
        lines.append("  best by dist_fwd: (no survivors)")
    prog_all = [r for r in all_surv
                if r["dist_fwd"] > PROGRESS_THRESH
                and not math.isnan(r["torso_roll_amp_deg"])]
    if prog_all:
        best_lr = min(prog_all, key=lambda r: r["torso_roll_amp_deg"])
        lines.append(f"  best low-roll-with-progress: anchor={best_lr['anchor_name']}  "
                     f"{_fmt_row(best_lr)}")
    else:
        lines.append(f"  best low-roll-with-progress: (no survivors with dist_fwd>{PROGRESS_THRESH} m)")
    lines.append("")

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def write_plot(path, results):
    """One subplot per anchor: dist_fwd vs freq (left axis) and torso_roll_amp
    vs freq (right axis). Survival shown by marker color (green=survived,
    red=fell) on the dist_fwd line."""
    n = len(ANCHORS)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4.2 * n), squeeze=False)
    for i, anchor in enumerate(ANCHORS):
        name = anchor["name"]
        rows = sorted([r for r in results if r["anchor_name"] == name],
                      key=lambda r: r["freq_hz"])
        freqs = [r["freq_hz"] for r in rows]
        dists = [r["dist_fwd"] for r in rows]
        rolls = [r["torso_roll_amp_deg"] for r in rows]
        colors = ["tab:green" if r["survived"] else "tab:red" for r in rows]

        ax = axes[i][0]
        l1, = ax.plot(freqs, dists, "-", color="tab:blue", lw=1.5, label="dist_fwd [m]")
        ax.scatter(freqs, dists, c=colors, s=22, zorder=3,
                   edgecolors="k", linewidths=0.3)
        ax.set_ylabel("dist_fwd [m]", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="0.7", lw=0.8)

        ax2 = ax.twinx()
        l2, = ax2.plot(freqs, rolls, "-", color="tab:orange", lw=1.5,
                       label="torso_roll_amp [deg]")
        ax2.set_ylabel("torso_roll_amp [deg]", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax.set_title(f"{name}  (hip={anchor['hip_amp_deg']:.0f}° "
                     f"crank={anchor['crank_amp_deg']:.0f}° "
                     f"torso={anchor['torso_amp_deg']:.0f}°)  "
                     f"green=survived / red=fell",
                     fontweight="bold", loc="left")
        if i == n - 1:
            ax.set_xlabel("freq [Hz]")
        ax.legend(handles=[l1, l2], loc="upper right", fontsize=9)
    fig.suptitle(f"Pengu anchor validation sweep (surface={SURFACE})", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"sweep_anchor_validation_{ts}")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "results.csv")
    log_path = os.path.join(outdir, "progress.log")
    sum_path = os.path.join(outdir, "summary.txt")
    plot_path = os.path.join(outdir, "summary_plot.png")

    logf = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    # restore ALL mutated globals at the end no matter what
    orig_freq  = gc.WALK_FREQ
    orig_hip   = gc.WALK_HIP_AMP_DEG
    orig_crank = gc.WALK_CRANK_AMP_DEG
    orig_torso = gc.WALK_TORSO_AMP_DEG

    results = []
    freqs = np.round(np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP / 2.0, FREQ_STEP), 4)
    n_total = len(ANCHORS) * len(freqs) * N_TRIALS
    t_start = time.perf_counter()

    try:
        log(f"# sweep_anchor_validation  start={ts}")
        log(f"# host={socket.gethostname()}  python={platform.python_version()}  mujoco={mujoco.__version__}")
        log(f"# git_head={_git_head()}  outdir={outdir}")
        log(f"# config: FREQ {FREQ_MIN}-{FREQ_MAX} step {FREQ_STEP} ({len(freqs)} freqs) | "
            f"anchors={[a['name'] for a in ANCHORS]} | surface={SURFACE} | "
            f"N_TRIALS={N_TRIALS} | SIM_DURATION={SIM_DURATION}s | total={n_total} trials")

        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            writer.writeheader()
            cf.flush()
            idx = 0
            for anchor in ANCHORS:
                for f in freqs:
                    for _ in range(N_TRIALS):
                        idx += 1
                        r = run_trial(idx, n_total, anchor, float(f), SIM_DURATION, log)
                        results.append(r)
                        writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
                        cf.flush()                    # incremental: safe if we crash later

        try:
            write_summary(sum_path, results)
            log(f"# wrote {sum_path}")
        except Exception as e:
            log(f"# summary FAILED: {type(e).__name__}: {e}")
        try:
            write_plot(plot_path, results)
            log(f"# wrote {plot_path}")
        except Exception as e:
            log(f"# plot FAILED: {type(e).__name__}: {e}")

        wall = time.perf_counter() - t_start
        log(f"# DONE  {len(results)} trials  wall={wall:.1f}s  outputs in {outdir}")
    finally:
        # restore every global we touched, regardless of success/failure
        gc.set_walk_freq(orig_freq)
        gc.set_hip_amp(orig_hip)
        gc.set_crank_amp(orig_crank)
        gc.set_torso_amp(orig_torso)
        logf.close()


if __name__ == "__main__":
    main()
