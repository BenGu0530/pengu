"""
sweep_freq_surface.py - Headless frequency x surface sweep for Pengu.

Replaces the broken backup_scripts/sweep_freq_pengu.py. Uses the CURRENT gait
(gait_config.apply_ctrl, same as walk_pengu.py) and the explicit friction model:
foot mu = 0.9 (fixed), floor mu = the variable (set per surface at runtime via
friction_utils.set_floor_friction). One fresh model load per trial.

# Run from pengu_mujoco/:
#   conda activate mujoco
#   nohup python sweep_freq_surface.py > sweep_$(date +%Y%m%d_%H%M%S).log 2>&1 &
# Tail progress with:
#   tail -f results/sweep_freq_surface_<latest>/progress.log
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
from friction_utils import set_floor_friction, SURFACES as SURFACE_MU

# ===================================================================
#  SWEEP CONFIG  (edit here)
# ===================================================================
FREQ_MIN = 1.0       # Hz
FREQ_MAX = 2.2       # Hz
FREQ_STEP = 0.05     # 25 frequencies
SURFACES = ["mocap_floor", "acrylic", "uhmw_pe", "ptfe_ice"]  # 4 surfaces
N_TRIALS = 1         # single trial per (surface, freq) - exploratory, not statistical
SIM_DURATION = 20.0  # seconds per trial (5s stand + 2s transition + 13s walk)

# ===================================================================
#  Constants
# ===================================================================
XML_PATH = "penguV2/scene.xml"     # relative, like walk_pengu.py (run from pengu_mujoco/)
ROOT_BODY = "leftthighmotor"       # floating-base body (per walk_pengu.py)
FALL_Z = 0.05                      # root z below this => fallen
ROLL_SETTLE = 2.0                  # ignore this many seconds of walk before measuring roll

CSV_FIELDS = [
    "idx", "surface", "mu_floor", "freq_hz",
    "survived", "fall_time", "walk_time",
    "dist_xy", "dist_fwd", "dist_lat",
    "mean_speed_fwd", "mean_speed_lat",
    "torso_roll_amp_deg", "error_msg",
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


def run_trial(idx, n_total, surface, freq, sim_duration, log):
    """Run one (surface, freq) trial. Catches ALL exceptions -> error_msg.
    Returns a dict with the CSV_FIELDS keys."""
    res = {
        "idx": idx, "surface": surface, "mu_floor": float("nan"),
        "freq_hz": round(freq, 4), "survived": False, "fall_time": float("nan"),
        "walk_time": 0.0, "dist_xy": 0.0, "dist_fwd": 0.0, "dist_lat": 0.0,
        "mean_speed_fwd": float("nan"), "mean_speed_lat": float("nan"),
        "torso_roll_amp_deg": float("nan"), "error_msg": "",
    }
    t_wall0 = time.perf_counter()
    sim_t = 0.0
    try:
        model = mujoco.MjModel.from_xml_path(XML_PATH)
        floor_id = set_floor_friction(model, SURFACE_MU[surface])
        res["mu_floor"] = float(model.geom_friction[floor_id, 0])

        data = mujoco.MjData(model)
        act_ids, jnt_adr = gc.build_ids(model)
        gc.set_walk_freq(freq)                       # mutate current gait's frequency
        gc.set_initial_pose(model, data, act_ids, jnt_adr)   # calls mj_forward

        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
        if root_id < 0:
            raise RuntimeError(f"root body '{ROOT_BODY}' not found")

        # Roll reference: world-up expressed in the root body frame at spawn.
        # roll(t) = lateral lean of the root's up axis about the forward (y) axis,
        # measured RELATIVE to spawn (so the -30 deg spawn pitch is the zero ref).
        R0 = data.xmat[root_id].reshape(3, 3).copy()
        up_local = R0.T @ np.array([0.0, 0.0, 1.0])

        walk_start = gc.T_HOLD + gc.T_TRANSITION
        roll_min = roll_max = None
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
            if data.time >= walk_start + ROLL_SETTLE:
                R = data.xmat[root_id].reshape(3, 3)
                up = R @ up_local
                roll = math.degrees(math.atan2(up[0], up[2]))
                roll_min = roll if roll_min is None else min(roll_min, roll)
                roll_max = roll if roll_max is None else max(roll_max, roll)

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
                res["mean_speed_lat"] = dx / res["walk_time"]
        if roll_min is not None:
            res["torso_roll_amp_deg"] = (roll_max - roll_min) / 2.0

    except Exception as e:                            # never let one trial kill the sweep
        res["error_msg"] = f"{type(e).__name__}: {e}"

    wall = time.perf_counter() - t_wall0
    line = (f"[{idx}/{n_total}] surface={surface:<11} freq={freq:.3f} "
            f"survived={res['survived']} ({sim_t:.1f}s sim, {wall:.1f}s wall)")
    if res["error_msg"]:
        line += f"  ERROR: {res['error_msg']}"
    log(line)
    return res


def write_summary(path, results):
    lines = []
    lines.append("=" * 70)
    lines.append("FREQUENCY x SURFACE SWEEP - SUMMARY")
    lines.append("=" * 70)
    n = len(results)
    n_err = sum(1 for r in results if r["error_msg"])
    n_surv = sum(1 for r in results if r["survived"])
    lines.append(f"trials: {n}   survived: {n_surv}   fell: {n - n_surv - n_err}   errored: {n_err}")
    lines.append("")
    for surface in SURFACES:
        rows = [r for r in results if r["surface"] == surface]
        if not rows:
            lines.append(f"[{surface}] (no trials)")
            continue
        surv = [r for r in rows if r["survived"]]
        mu = rows[0]["mu_floor"]
        lines.append(f"[{surface}]  mu_floor={mu:.3f}  trials={len(rows)}  survived={len(surv)}")
        if surv:
            best_d = max(surv, key=lambda r: r["dist_fwd"])
            best_s = max(surv, key=lambda r: (r["mean_speed_fwd"] if not math.isnan(r["mean_speed_fwd"]) else -1e9))
            rolls = [r["torso_roll_amp_deg"] for r in surv if not math.isnan(r["torso_roll_amp_deg"])]
            lines.append(f"    survived freqs: {min(r['freq_hz'] for r in surv):.2f}-{max(r['freq_hz'] for r in surv):.2f} Hz")
            lines.append(f"    best fwd dist : {best_d['dist_fwd']:+.3f} m @ {best_d['freq_hz']:.2f} Hz")
            lines.append(f"    best fwd speed: {best_s['mean_speed_fwd']:+.3f} m/s @ {best_s['freq_hz']:.2f} Hz")
            if rolls:
                lines.append(f"    roll amp range: {min(rolls):.2f}-{max(rolls):.2f} deg")
        errs = [r for r in rows if r["error_msg"]]
        if errs:
            lines.append(f"    ERRORS ({len(errs)}): e.g. {errs[0]['error_msg']}")
        lines.append("")
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text + "\n")
    return text


def write_plot(path, results):
    metrics = [
        ("survived", "survived (1/0)", lambda r: 1 if r["survived"] else 0),
        ("mean_speed_fwd", "fwd speed [m/s]", lambda r: r["mean_speed_fwd"]),
        ("torso_roll_amp_deg", "roll amp [deg]", lambda r: r["torso_roll_amp_deg"]),
    ]
    nrows = len(SURFACES)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.2 * nrows), squeeze=False)
    for i, surface in enumerate(SURFACES):
        rows = sorted([r for r in results if r["surface"] == surface], key=lambda r: r["freq_hz"])
        freqs = [r["freq_hz"] for r in rows]
        mu = rows[0]["mu_floor"] if rows else float("nan")
        for j, (_, ylabel, fn) in enumerate(metrics):
            ax = axes[i][j]
            ys = [fn(r) for r in rows]
            ax.plot(freqs, ys, "-o", ms=4)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel)
            if i == nrows - 1:
                ax.set_xlabel("freq [Hz]")
            if j == 0:
                ax.set_title(f"{surface} (mu={mu:.2f})", fontweight="bold", loc="left")
    fig.suptitle("Pengu frequency x surface sweep", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("results", f"sweep_freq_surface_{ts}")
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

    orig_freq = gc.WALK_FREQ          # restore at the end no matter what
    results = []
    freqs = np.round(np.arange(FREQ_MIN, FREQ_MAX + FREQ_STEP / 2.0, FREQ_STEP), 3)
    n_total = len(SURFACES) * len(freqs) * N_TRIALS
    t_start = time.perf_counter()

    try:
        log(f"# sweep_freq_surface  start={ts}")
        log(f"# host={socket.gethostname()}  python={platform.python_version()}  mujoco={mujoco.__version__}")
        log(f"# git_head={_git_head()}  outdir={outdir}")
        log(f"# config: FREQ {FREQ_MIN}-{FREQ_MAX} step {FREQ_STEP} ({len(freqs)} freqs) | "
            f"surfaces={SURFACES} | N_TRIALS={N_TRIALS} | SIM_DURATION={SIM_DURATION}s | total={n_total} trials")

        with open(csv_path, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            writer.writeheader()
            cf.flush()
            idx = 0
            for surface in SURFACES:
                for f in freqs:
                    for _ in range(N_TRIALS):
                        idx += 1
                        r = run_trial(idx, n_total, surface, float(f), SIM_DURATION, log)
                        results.append(r)
                        writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
                        cf.flush()                    # incremental: results safe if we crash later

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
        gc.set_walk_freq(orig_freq)                   # restore global gait state
        logf.close()


if __name__ == "__main__":
    main()
