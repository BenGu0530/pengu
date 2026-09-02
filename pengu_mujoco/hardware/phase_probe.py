"""phase_probe.py — why the gait comes out "out of phase" on the robot.

The firmware commands and the simulation command the SAME waveform (checked line by line,
2026-08-30):

    crank_L = 0.5*A_leg*(1 + sin(p))            crank_R = 0.5*A_leg*(1 + sin(p + pi))
    hip_L   = off + A_hip*max(0, sin(p+pi+PHI)) hip_R   = off + A_hip*max(0, sin(p + PHI))

so any phase error is in EXECUTION, not in the formula. Each of the four leg servos closes
its own loop, and each has a different velocity demand, so each lags its command by a
different amount. The gait's defining quantity -- the hip-to-crank phase offset hip_phi --
is therefore NOT what was commanded once the motors are in the loop.

This script measures that. It fits, at the commanded gait frequency, the fundamental of
every commanded channel and of the matching measured channel, and reports

  * per motor : executed/commanded amplitude ratio, and the lag in degrees of gait cycle
                and in milliseconds
  * per pair  : the phase offset as COMMANDED and as EXECUTED, and the difference.
                `hip_R - crank_R` is hip_phi itself.
  * hip shape : an ideal half-rectified sine has DC/h1 = 0.6366 and h2/h1 = 0.4244. If the
                servo cannot return fast enough the trough fills in, DC rises and h2/h1
                falls -- the hip stops being a half-rectified swing at all.
  * slew      : peak measured joint rate against the commanded peak rate, per motor. The
                bench ceiling measured 2026-08-28 was 420-440 deg/s.

Nothing here needs a new sketch: it reads the CSV the 't' key already streams.

    # hardware
    python hardware/phase_probe.py ../HardwareData/phase/air_n.csv --freq 1.43 --phi 260
    # simulation, identical fit, for the same gait
    python hardware/phase_probe.py --sim --freq 1.43 --phi 260 --leg 95 --hip 28 \
           --off 30 --mu 0.5 --com 1.05 --kappa 0

Columns of the hardware CSV (the 't' header):
    w,t,alpha,goal_slL,pos_slL,goal_slR,pos_slR,goal_hipL,pos_hipL,goal_hipR,pos_hipR,
    goal_torso,pos_torso,mA_torso,imu_roll,imu_pitch,axis,dt_ms
The log also carries 1 Hz "t=..." and 5 Hz "roll=..." lines; only lines starting with
"w," are used, and short/garbled lines (serial drops) are dropped and counted.
"""
import argparse
import math
import os
import sys

import numpy as np

COLS = ["t", "alpha", "goal_slL", "pos_slL", "goal_slR", "pos_slR",
        "goal_hipL", "pos_hipL", "goal_hipR", "pos_hipR",
        "goal_torso", "pos_torso", "mA_torso", "imu_roll", "imu_pitch", "axis", "dt_ms",
        "i_lim_mA", "clamp_deg", "roll_rate", "kd", "gx", "gy", "gz", "freq", "leg_amp", "hip_amp"]

# (label, commanded column, measured column, "leg" | "hip" | "torso")
CHANNELS = [("crank L", "goal_slL", "pos_slL", "leg"),
            ("crank R", "goal_slR", "pos_slR", "leg"),
            ("hip   L", "goal_hipL", "pos_hipL", "hip"),
            ("hip   R", "goal_hipR", "pos_hipR", "hip"),
            ("torso  ", "goal_torso", "pos_torso", "torso")]

# phase offsets that define the gait; hip_R - crank_R is hip_phi
PAIRS = [("hip R - crank R  (= hip_phi)", "goal_hipR", "pos_hipR", "goal_slR", "pos_slR"),
         ("hip L - crank L  (= hip_phi)", "goal_hipL", "pos_hipL", "goal_slL", "pos_slL"),
         ("crank R - crank L (=180)", "goal_slR", "pos_slR", "goal_slL", "pos_slL"),
         ("hip R - hip L     (=180)", "goal_hipR", "pos_hipR", "goal_hipL", "pos_hipL")]

HALFREC_DC = 1.0 / math.pi / 0.5          # DC / h1 of max(0, sin)  = 0.6366
HALFREC_H2 = (2.0 / (3.0 * math.pi)) / 0.5  # h2 / h1               = 0.4244


# --------------------------------------------------------------------------- fitting
def fit_harmonics(t, y, f, n_harm=3):
    """Least squares y ~ c0 + sum_k [a_k cos(2pi k f t) + b_k sin(2pi k f t)].

    Returns dict with dc, and per harmonic amplitude/phase where the convention is
    y_k = M_k * sin(2 pi k f t + psi_k), psi in degrees.
    Non-uniform sampling is fine: f is known, so this is a plain regression, not an FFT.
    """
    M = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        w = 2 * math.pi * k * f * t
        M += [np.cos(w), np.sin(w)]
    M = np.column_stack(M)
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    resid = y - M @ coef
    out = {"dc": float(coef[0]),
           "resid_rms": float(np.sqrt(np.mean(resid ** 2))),
           "rms": float(np.sqrt(np.mean((y - y.mean()) ** 2)))}
    for k in range(1, n_harm + 1):
        a, b = float(coef[2 * k - 1]), float(coef[2 * k])
        out[f"amp{k}"] = math.hypot(a, b)
        out[f"psi{k}"] = math.degrees(math.atan2(a, b))
    return out


def wrap180(d):
    return (d + 180.0) % 360.0 - 180.0


def wrap360(d):
    return d % 360.0


def refine_freq(t, y, f0, span=0.06, n=241):
    """Best-fit fundamental frequency near f0, by minimising the residual."""
    grid = np.linspace(f0 * (1 - span), f0 * (1 + span), n)
    best, bf = None, f0
    for f in grid:
        r = fit_harmonics(t, y, f, 1)["resid_rms"]
        if best is None or r < best:
            best, bf = r, f
    return float(bf)


def peak_rate(t, y):
    """Robust peak |dy/dt| in deg/s: p99 of the finite difference, to ignore single-sample
    encoder glitches."""
    dt = np.diff(t)
    ok = dt > 1e-4
    if not ok.any():
        return float("nan")
    r = np.abs(np.diff(y)[ok] / dt[ok])
    return float(np.percentile(r, 99))


# --------------------------------------------------------------------------- loading
def load_csv(path):
    """Read the telemetry rows, taking the column names from the log itself.

    The sketches print a header line starting `w,t,alpha,...` at every WALK, and the column
    set has changed several times (17 -> 18 -> 24 -> 27 -> 23 as knobs came and went).
    Trusting a hard-coded list silently mislabels every column when it drifts, so the header
    in the file wins. COLS is the fallback for the captures made before the header was
    printed on every bout; those are padded with zeros if they are short.
    """
    keep, bad, other, cols = [], 0, 0, None
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip().strip("\x00")
            if not line.startswith("w,"):
                other += 1
                continue
            f = line[2:].split(",")
            if f and f[0] == "t":                      # this is a header, not a row
                if cols is not None and f != cols:
                    sys.exit(f"{os.path.basename(path)} holds two different column sets; "
                             f"split it before analysing")
                cols = f
                other += 1
                continue
            if cols is None:                           # pre-header capture
                while len(f) < len(COLS):
                    f = f + ["0"]
                if len(f) != len(COLS):
                    bad += 1
                    continue
            elif len(f) != len(cols):
                bad += 1
                continue
            if keep and len(f) != len(keep[0]):    # a header appeared mid-file and moved
                bad += 1                           # the goalposts; do not build a ragged
                continue                           # array out of two different formats
            try:
                keep.append([float(x) for x in f])
            except ValueError:
                bad += 1
    if not keep:
        sys.exit(f"no 'w,' telemetry rows in {path} -- did the robot ever reach WALK?")
    names = cols if cols is not None else COLS
    a = np.array(keep)
    d = {c: a[:, i] for i, c in enumerate(names)}
    for c in COLS:                                     # anything the log did not carry
        if c not in d:
            d[c] = np.zeros(len(a))
    # The firmware drives the left motors with the opposite sign (home - magL, home -
    # hipL) and logs the signed command, so every LEFT channel is the negative of the
    # physical magnitude that sim reports. Negating a signal shifts its fundamental phase
    # by 180 deg, which would show up as a bogus 180 in the L-vs-R rows. Both the command
    # and the measurement of a given motor carry the same flip, so hip_phi is unaffected
    # either way; this only makes the left-vs-right comparison mean what it says.
    for c in ("goal_slL", "pos_slL", "goal_hipL", "pos_hipL"):
        d[c] = -d[c]
    d["_n_bad"] = bad
    d["_n_other"] = other
    return d


def trim(d, skip_head, skip_tail, alpha_min=0.999):
    t = d["t"]
    m = d["alpha"] >= alpha_min
    if not m.any():
        sys.exit("alpha never reached 1.0 -- the blend never finished, record is too short")
    t0 = t[m].min() + skip_head
    t1 = t[m].max() - skip_tail
    m &= (t >= t0) & (t <= t1)
    out = {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    out["_win"] = (float(t0), float(t1))
    return out


# --------------------------------------------------------------------------- segmenting
def cycle_table(d, f, t0=14.0):
    """One row per gait cycle, with a state label.

    A walking record is not one experiment. gnd_n_k0 (2026-08-30) ran 61 s and contained
    47 walking cycles, 12 where the robot was barely moving (held or stood up), and 9 lying
    on its side -- and a fit over the whole thing reports the falls as if they were gait.
    The two labels below come straight from what those segments look like:

      DOWN  : any of
                - the roll sits past 25 deg AND the torso joint has stopped moving (<6 deg
                  of travel in a whole cycle) because its command is pinned on the clamp;
                - the torso command is on the clamp for the WHOLE cycle. pengu-4 had two
                  cycles at 100% rail with 1.1 and 9.9 deg of joint travel that the roll
                  rule let through as walking;
                - the PITCH sits more than 35 deg off its walking value. pengu-10 ended
                  with five cycles at +70 to +85 deg of pitch -- flat on its back -- while
                  the roll stayed under 12, so a roll-only rule called every one of them
                  a walking cycle. The robot falls backwards; the classifier was blind to
                  the entire axis it falls about.
      quiet : the lower body swings less than 15 deg peak-to-peak -- held, or standing.

    Everything else is `walk`. Only runs of >=4 consecutive walk cycles are used, so a
    single cycle between two falls never enters a phase difference.
    """
    t = d["t"]
    # The BNO's pitch carries a fixed offset and, when the roll is large, its Euler
    # decomposition wraps through +-180 and the channel is unusable. Both are handled here:
    # the baseline is the centre of the dominant cluster (one refinement pass, so a record
    # that ends fallen still baselines on the walking part), and a wrapped channel disables
    # the pitch rule instead of firing it on everything.
    pit = d.get("imu_pitch")
    pit_ok = pit is not None and float(np.max(pit) - np.min(pit)) < 170.0
    if pit_ok:
        pit0 = float(np.median(pit))
        near = np.abs(pit - pit0) < 35.0
        if near.sum() > 10:
            pit0 = float(np.median(pit[near]))
    else:
        pit0 = 0.0
    rows = []
    for k in range(int((t[-1] - t0) * f) + 1):
        a, b = t0 + k / f, t0 + (k + 1) / f
        m = (t >= a) & (t < b)
        if m.sum() < 5:
            continue
        roll, J, ax, cmd = d["imu_roll"][m], d["pos_torso"][m], d["axis"][m], d["goal_torso"][m]
        w = 2 * math.pi * f * t[m]
        M = np.column_stack([np.ones(m.sum()), np.cos(w), np.sin(w)])
        c, *_ = np.linalg.lstsq(M, roll, rcond=None)
        jp2p = float(J.max() - J.min())
        axp2p = float(ax.max() - ax.min())
        rail = float((np.abs(cmd) > 24.5).mean())
        dpit = float(np.mean(pit[m] - pit0)) if pit_ok else 0.0
        st = ("DOWN" if ((abs(roll.mean()) > 25 and jp2p < 6)
                         or rail > 0.99
                         or abs(dpit) > 35.0)
              else ("quiet" if axp2p < 15 else "walk"))
        rows.append(dict(k=k, t=a, mask=m, state=st, roll_mean=float(roll.mean()),
                         roll_p2p=float(roll.max() - roll.min()), axis_p2p=axp2p, J_p2p=jp2p,
                         rail=rail, dpitch=dpit, pit_ok=pit_ok,
                         amp=math.hypot(c[1], c[2]), psi=math.degrees(math.atan2(c[1], c[2]))))
    return rows


def walk_runs(rows, min_len=4):
    """Consecutive `walk` cycles, in runs of at least min_len."""
    out, cur = [], []
    for r in rows:
        if r["state"] == "walk" and (not cur or r["k"] == cur[-1]["k"] + 1):
            cur.append(r)
        else:
            if len(cur) >= min_len:
                out.append(cur)
            cur = [r] if r["state"] == "walk" else []
    if len(cur) >= min_len:
        out.append(cur)
    return out


def report_cycles(d, f, t0=14.0):
    rows = cycle_table(d, f, t0)
    if not rows:
        return None
    runs = walk_runs(rows)
    n = {s: sum(r["state"] == s for r in rows) for s in ("walk", "quiet", "DOWN")}
    print(f"\n-- per cycle ------------------------------------------------------------")
    print(f"{len(rows)} cycles: {n['walk']} walk, {n['quiet']} quiet/held, {n['DOWN']} down"
          f"   ->  {len(runs)} run(s) of >=4 consecutive walk cycles, "
          f"{sum(len(r) for r in runs)} cycles used")
    dpsi = [abs(wrap180(r[i]["psi"] - r[i - 1]["psi"]))
            for r in runs for i in range(1, len(r))]
    w = [c for r in runs for c in r]
    if not w:
        print("no clean walking run -- nothing below is computed")
        return None
    amp = np.array([c["amp"] for c in w])
    print(f"{'roll phase drift per cycle':32s}{np.mean(dpsi) if dpsi else float('nan'):8.0f} deg"
          f"   (sim locks to 1-6)")
    print(f"{'roll fundamental amplitude':32s}{amp.mean():8.1f} deg   CV {amp.std()/amp.mean():.2f}"
          f"   (sim CV 0.02-0.04)")
    for nm, key in (("lower-body roll p2p", "axis_p2p"), ("torso joint p2p", "J_p2p")):
        v = np.array([c[key] for c in w])
        print(f"{nm:32s}{v.mean():8.1f} deg   spread {v.min():.0f}-{v.max():.0f}")
    print(f"{'torso command at the clamp':32s}"
          f"{100*np.mean([c['rail'] for c in w]):8.0f} %")
    if float(np.max(d["i_lim_mA"])) > 0:
        cap = np.array([d["i_lim_mA"][c["mask"]].mean() for c in w])
        clp = np.array([d["clamp_deg"][c["mask"]].mean() for c in w])
        print(f"{'torso limits in force':32s}{cap.mean():8.0f} mA   "
              f"clamp +-{clp.mean():.0f} deg")
    mask = np.zeros(len(d["t"]), bool)
    for c in w:
        mask |= c["mask"]
    return mask


# --------------------------------------------------------------------------- sim path
def sim_rollout(args):
    """Run the sim and return the same column dict, in degrees, sampled at 20 Hz."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    sys.path[:0] = [os.path.join(root, "grid5"), root]   # grid5's gait_config must win
    os.environ.setdefault("PENGU_MODEL", "1.31")
    import mujoco
    import gait_config as gc
    import gait_sweep as gs
    import grid5_sweep as g5
    from friction_utils import set_floor_friction
    from torso_control import TorsoKappaPID

    model = mujoco.MjModel.from_xml_path(gs.XML)
    data = mujoco.MjData(model)
    lean = gc.STAND_HIP_DEG
    gc.STAND_HIP_DEG = 0.0
    g5.apply_com_variant(model, args.com)
    pid = TorsoKappaPID(model, kappa=args.kappa, measure_after=0.0)
    gc.TORSO_CONTROLLER = pid
    gc.STAND_HIP_DEG = lean
    set_floor_friction(model, args.mu)
    gs.FLOOR_MU = args.mu
    gs.CONDITION["hip_off"] = args.off
    gs._set_gait(dict(freq=args.freq, hip_phi=args.phi, leg_amp=args.leg, hip_amp=args.hip))
    act, jadr = gc.build_ids(model)
    gc.set_initial_pose(model, data, act, jadr)

    aid = {n: act[n] for n in ("hip-L", "hip-R", "crank1-R", "crank1-L", "torso")}
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
           for n, j in (("hip-L", "hip-L"), ("hip-R", "hip-R"), ("crank1-R", "crank2_R"),
                        ("crank1-L", "crank1-L"), ("torso", "torso"))}
    qadr = {n: model.jnt_qposadr[i] for n, i in jid.items()}
    root_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, gs.ROOT_BODY)

    gc.T_HOLD = 1e9
    t0 = None
    nxt, rows = 0.0, []
    D = math.degrees
    while True:
        if t0 is None:
            tt = data.time
            if (tt >= gs.QUIET_MIN_T and float(np.max(np.abs(data.qvel))) < gs.QUIET_QVEL) \
                    or tt >= gs.QUIET_MAX_T:
                t0 = tt
                gc.T_HOLD = tt
        gc.apply_ctrl(data, act, data.time)
        mujoco.mj_step(model, data)
        if t0 is None:
            continue
        if data.time > t0 + gc.T_TRANSITION + args.dur:
            break
        if data.time >= nxt:
            nxt = data.time + 0.05
            # roll must be the tilt about the torso hinge, the same definition the
            # controller nulls and the same one the mocap analysis validated. The world
            # x-axis Euler roll used here originally is the FORE-AFT tilt in this
            # convention (sim is z-up with +y forward), which is a different quantity --
            # it read 8.4 deg rms where the controller's own readout said 1.9.
            R = data.xmat[root_b].reshape(3, 3)
            roll = D(pid.torso_roll(data))
            pitch = D(-math.asin(max(-1.0, min(1.0, R[2, 0]))))
            walking = data.time >= t0 + gc.T_TRANSITION
            rows.append([data.time - t0, 1.0 if walking else 0.0,
                         D(data.ctrl[aid["crank1-L"]]), D(data.qpos[qadr["crank1-L"]]),
                         D(data.ctrl[aid["crank1-R"]]), D(data.qpos[qadr["crank1-R"]]),
                         D(data.ctrl[aid["hip-L"]]), D(data.qpos[qadr["hip-L"]]),
                         D(data.ctrl[aid["hip-R"]]), D(data.qpos[qadr["hip-R"]]),
                         D(data.ctrl[aid["torso"]]), D(data.qpos[qadr["torso"]]),
                         0.0, roll, pitch, D(pid.axis_roll(data)), 50.0, 3210.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, args.freq, args.leg, args.hip])
    gc.T_HOLD = 5.0
    a = np.array(rows)
    d = {c: a[:, i] for i, c in enumerate(COLS)}
    d["_n_bad"] = d["_n_other"] = 0
    return d


# --------------------------------------------------------------------------- report
def report(d, args, src):
    t = d["t"]
    dur = float(t[-1] - t[0])
    f_nom = args.freq
    f_fit = refine_freq(t, d["goal_slL"], f_nom)
    f = f_nom if args.use_nominal else f_fit
    ncyc = dur * f

    print(f"\n=== {src} ===")
    print(f"gait      freq {f_nom:.2f} Hz   hip_phi {args.phi:.0f}   leg {args.leg:.0f}   "
          f"hip {args.hip:.0f}   off {args.off:.0f}")
    print(f"window    {d['_win'][0]:.1f} - {d['_win'][1]:.1f} s   {dur:.1f} s   "
          f"{ncyc:.1f} cycles   {len(t)} samples ({len(t)/max(dur,1e-9):.1f} Hz, "
          f"{len(t)/max(ncyc,1e-9):.1f}/cycle)")
    if d["_n_bad"] or d["_n_other"]:
        print(f"dropped   {d['_n_bad']} malformed 'w,' rows, {d['_n_other']} non-CSV lines")
    # A hand-tuning capture holds several settings in one file. Fitting across a parameter
    # change is meaningless, so say so rather than quietly averaging over it.
    if "hip_phi" in d and np.any(d["leg_amp"] > 0):
        combos = {(round(a, 2), round(b, 1), round(c, 1), round(e, 0))
                  for a, b, c, e in zip(d["freq"], d["leg_amp"], d["hip_amp"], d["hip_phi"])}
        if len(combos) > 1:
            print(f"!! this capture holds {len(combos)} different parameter settings. The fits "
                  f"below\n!! pool all of them and mean nothing. Split by the freq / leg_amp / "
                  f"hip_amp / hip_phi\n!! columns first, or analyse one setting at a time.")
            for c in sorted(combos)[:8]:
                print(f"     freq {c[0]:.2f}  leg {c[1]:.0f}  swing {c[2]:.0f}  phi {c[3]:.0f}")
    print(f"fit freq  {f_fit:.4f} Hz from the command channel "
          f"({100*(f_fit/f_nom-1):+.2f}% vs commanded; large means the wrong --freq)")
    if "dt_ms" in d and np.nanmax(d["dt_ms"]) > 0:
        print(f"loop      telemetry period {np.median(d['dt_ms']):.1f} ms median, "
              f"{np.percentile(d['dt_ms'],95):.1f} p95")

    fits = {}
    for lab, gcol, pcol, kind in CHANNELS:
        fits[gcol] = fit_harmonics(t, d[gcol], f)
        fits[pcol] = fit_harmonics(t, d[pcol], f)

    print("\n-- per motor: does the command arrive, and when? -------------------------")
    print(f"{'motor':9s}{'cmd amp':>9s}{'exe amp':>9s}{'ratio':>7s}"
          f"{'lag deg':>9s}{'lag ms':>8s}{'cmd rate':>10s}{'exe rate':>10s}{'fitres':>8s}")
    for lab, gcol, pcol, kind in CHANNELS:
        g, p = fits[gcol], fits[pcol]
        lag = wrap180(g["psi1"] - p["psi1"])
        cmd_rate = peak_rate(t, d[gcol])
        exe_rate = peak_rate(t, d[pcol])
        ratio = p["amp1"] / g["amp1"] if g["amp1"] > 1e-6 else float("nan")
        print(f"{lab:9s}{g['amp1']:9.2f}{p['amp1']:9.2f}{ratio:7.3f}"
              f"{lag:9.1f}{1000*lag/360.0/f:8.1f}{cmd_rate:10.0f}{exe_rate:10.0f}"
              f"{p['resid_rms']:8.2f}")
    print("lag > 0 means the joint arrives AFTER the command. Rates are deg/s, p99 of the")
    print("sampled finite difference; the bench velocity ceiling measured 420-440 deg/s.")

    print("\n-- the gait's own phase offsets: commanded vs executed -------------------")
    print(f"{'pair':32s}{'commanded':>11s}{'executed':>10s}{'shift':>8s}{'ms':>8s}"
          f"{'phi_cmd':>9s}{'phi_exe':>9s}")
    for lab, ag, ap, bg, bp in PAIRS:
        cmd = wrap360(fits[ag]["psi1"] - fits[bg]["psi1"])
        exe = wrap360(fits[ap]["psi1"] - fits[bp]["psi1"])
        sh = wrap180(exe - cmd)
        # a hip-vs-crank offset of X is hip_phi = X + 180: the crank's fundamental carries
        # the built-in pi of 0.5*(1+sin(p+pi)) that the hip's max(0,sin) does not.
        phi = (f"{wrap360(cmd+180):9.0f}{wrap360(exe+180):9.1f}"
               if "hip_phi" in lab else " " * 18)
        print(f"{lab:32s}{cmd:11.1f}{exe:10.1f}{sh:8.1f}{1000*sh/360.0/f:8.1f}{phi}")
    print(f"phi_exe is the hip_phi the robot ACTUALLY ran; it was commanded {args.phi:.0f}.")
    print("hip_phi's grid step is 10 deg, so a shift of 10 means a different sweep cell.")

    print("\n-- hip waveform shape (ideal half-rectified sine: DC/h1 0.637, h2/h1 0.424) --")
    print(f"{'channel':14s}{'DC':>8s}{'h1':>8s}{'h2':>8s}{'h3':>8s}{'DC/h1':>8s}{'h2/h1':>8s}")
    for lab, gcol, pcol, kind in CHANNELS:
        if kind != "hip":
            continue
        for nm, col in ((f"{lab} cmd", gcol), (f"{lab} exe", pcol)):
            q = fits[col]
            dc = q["dc"] - args.off * (1 if q["dc"] > 0 else -1)   # remove the lean offset
            print(f"{nm:14s}{dc:8.2f}{q['amp1']:8.2f}{q['amp2']:8.2f}{q['amp3']:8.2f}"
                  f"{dc/q['amp1'] if q['amp1']>1e-6 else float('nan'):8.3f}"
                  f"{q['amp2']/q['amp1'] if q['amp1']>1e-6 else float('nan'):8.3f}")

    print("\n-- body ------------------------------------------------------------------")
    rf = fit_harmonics(t, d["imu_roll"], f)
    print(f"roll   mean {d['imu_roll'].mean():7.2f}  rms {rf['rms']:6.2f}  "
          f"fundamental {rf['amp1']:6.2f} deg at psi {rf['psi1']:7.1f}")
    print(f"pitch  mean {d['imu_pitch'].mean():7.2f}  rms "
          f"{np.std(d['imu_pitch']):6.2f}   (hip_off commanded {args.off:.0f} deg -- the "
          f"SIGN of this against sim says whether the lean goes the same way)")
    if np.nanmax(np.abs(d["mA_torso"])) > 0:
        print(f"torso current p95 {np.percentile(np.abs(d['mA_torso']),95):.0f} mA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", help="captured serial log (omit with --sim)")
    ap.add_argument("--sim", action="store_true", help="run the simulation instead")
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--phi", type=float, required=True)
    ap.add_argument("--leg", type=float, default=95.0)
    ap.add_argument("--hip", type=float, default=28.0)
    ap.add_argument("--off", type=float, default=30.0)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--com", type=float, default=1.05)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=20.0, help="sim window after the blend")
    ap.add_argument("--skip-head", type=float, default=2.0,
                    help="[s] dropped after alpha reaches 1")
    ap.add_argument("--skip-tail", type=float, default=1.0)
    ap.add_argument("--use-nominal", action="store_true", default=True)
    a = ap.parse_args()
    if a.sim:
        d = sim_rollout(a)
        src = f"SIMULATION  mu={a.mu} COM={a.com} kappa={a.kappa}"
    else:
        if not a.csv:
            sys.exit("give a CSV or --sim")
        d = load_csv(a.csv)
        src = f"HARDWARE  {os.path.basename(a.csv)}"
    d = trim(d, a.skip_head, a.skip_tail)
    report(d, a, src)
    mask = report_cycles(d, a.freq)
    if mask is not None and 0 < mask.sum() < len(mask):
        print("\n### the fits above cover the whole window, falls included. Repeating them"
              "\n### on the clean walking cycles only:")
        report({k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                for k, v in d.items()}, a, src + "  [clean walking cycles]")


if __name__ == "__main__":
    main()
