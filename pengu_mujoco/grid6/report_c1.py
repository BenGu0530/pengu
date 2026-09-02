"""report_c1.py — what the two GRID-6 stages found, in one file.

Reads whichever of the two stage CSVs exist and writes
results/grid6_report/c1_asbuilt.md. Numbers only; no verdicts.

Per friction level it gives the passing count, the champion, and the champion that the
robot could actually execute -- those are usually different cells, and the gap between
them is the point of the campaign. The envelope is the measured 354 deg/s crank ceiling
(2026-08-30, twelve points, air and ground pooled); peak crank rate is pi*f*A_leg and
peak hip rate 2*pi*f*A_hip.
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "grid6_report")
CEILING = 354.0
STAGES = [("stage 1 (0.05 Hz, robust region)", "c1r"),
          ("stage 2 (0.01 Hz, survivors)", "c1f")]


def load(tag):
    p = os.path.join(ROOT, "results", "gait_sweep",
                     f"sweep_grid6_c1_{tag}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv")
    if not os.path.exists(p):
        return None, p
    out = []
    with open(p) as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for r in rd:
            if len(r) < 9:
                continue
            out.append(tuple(float(x) for x in r[:9]))
    return out, p


def fmt(c):
    return f"{c[0]:.2f} / {c[1]:.0f} / {c[2]:.0f} / {c[3]:.0f} / {c[4]:.0f}"


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ["# GRID-6 c1 on the as-built model", "",
         "Model `models/hardware_c1`: CAD export of the physical lowered-counterweight",
         "build, hardened to the 5-actuator convention, ballasted with 100 g at",
         "z = 212.63 mm so total mass is 2.2724 kg and the neutral-stand COM ratio is",
         "1.0500 -- both matching the slide-tuned c1 of GRID-5. What differs is the torso's",
         "rotational inertia: 1.76x / 1.46x / 5.97x the slide-tuned variant's, because",
         "splitting the mass physically spreads it about its own centre while sliding an",
         "inertial position does not.", "",
         "A gait passes if the robot stays upright, holds heading alignment above 0.5 and",
         f"advances faster than 0.05 m/s. The executable subset additionally keeps peak",
         f"crank rate pi*f*A_leg and peak hip rate 2*pi*f*A_hip under {CEILING:.0f} deg/s,",
         "the servo ceiling measured on the robot on 2026-08-30.", ""]

    for label, tag in STAGES:
        rows, path = load(tag)
        L += [f"## {label}", ""]
        if rows is None:
            L += [f"Not present: `{os.path.basename(path)}`", ""]
            continue
        mus = sorted({r[5] for r in rows})
        L += [f"{len(rows):,} rows.", "",
              "| mu | rows | pass | pass% | best net | best cell | executable best | its cell |",
              "|---|---|---|---|---|---|---|---|"]
        for mu in mus:
            sub = [r for r in rows if r[5] == mu]
            ok = [r for r in sub if r[6] > 0.5]
            env = [r for r in ok
                   if math.pi * r[0] * r[2] <= CEILING and 2 * math.pi * r[0] * r[3] <= CEILING]
            b = max(ok, key=lambda r: r[8]) if ok else None
            e = max(env, key=lambda r: r[8]) if env else None
            L.append(f"| {mu} | {len(sub):,} | {len(ok):,} | {100*len(ok)/max(len(sub),1):.2f}% | "
                     + (f"{b[8]:.4f} | {fmt(b)} | " if b else "— | — | ")
                     + (f"{e[8]:.4f} | {fmt(e)} |" if e else "none | — |"))
        L.append("")
        # what the envelope costs, per mu
        L += ["Cost of the motor envelope (best executable / best overall):", ""]
        for mu in mus:
            ok = [r for r in rows if r[5] == mu and r[6] > 0.5]
            env = [r for r in ok
                   if math.pi * r[0] * r[2] <= CEILING and 2 * math.pi * r[0] * r[3] <= CEILING]
            if ok and env:
                L.append(f"- mu {mu}: {max(e[8] for e in env)/max(o[8] for o in ok):.2f}"
                         f"  ({len(env):,} of {len(ok):,} passing cells are executable)")
            elif ok:
                L.append(f"- mu {mu}: no passing cell is executable ({len(ok):,} pass)")
        L.append("")

    p = os.path.join(OUT, "c1_asbuilt.md")
    open(p, "w").write("\n".join(L) + "\n")
    print("\n".join(L[-30:]))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
