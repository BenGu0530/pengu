#!/usr/bin/env bash
# Per-generation table. Includes the columns that separate a real waddle from
# the known false positives, and a reward-budget breakdown so a candidate's
# behaviour can be traced to which term it is harvesting.
set -u
cd "$(dirname "$0")/.."
GEN=${1:?usage: report_gen.sh <gen-name>}
PY=${PY:-../.sweep_venv/bin/python}
GEN="$GEN" "$PY" - <<'PYEOF'
import csv, glob, os
import numpy as np
GEN = os.environ["GEN"]
OUT = f"overnight/{GEN}"

def num(r, k, d=float("nan")):
    try: return float(r[k])
    except Exception: return d

print(f"\n=== {GEN} ===")
hdr = (f"{'cand':<16}{'vtail':>7}{'vswg':>6}{'fall':>6}{'eplen':>7}"
       f"{'roll':>7}{'rollμ':>7}{'|μ|/R':>7}{'rate':>6}{'hipc':>7}{'asym':>7}"
       f"{'sing':>6}{'ePass':>7}{'eNfwd':>7}")
print(hdr); print("-" * len(hdr))
rows = []
for dg in sorted(glob.glob(f"runs/overnight/{GEN}/*/diag.csv")):
    cand = os.path.basename(os.path.dirname(dg))
    d = list(csv.DictReader(open(dg)))
    if not d: continue
    L = d[-1]
    tail = [num(x, "vx") for x in d[-4:]]
    tail = [v for v in tail if v == v]
    vtail = sum(tail)/len(tail) if tail else float("nan")
    vswg = max(tail)-min(tail) if len(tail) > 1 else float("nan")
    roll, rollm = num(L, "torso_roll_rms_deg"), num(L, "torso_roll_mean_deg")
    ratio = abs(rollm)/roll if roll and roll == roll and rollm == rollm else float("nan")
    # Flags must come from the EVAL rollout, not the training diag. gen01's
    # kernel_off had training rate 138 deg/s and tripped nothing, while its
    # deterministic eval was a torso parked at -47 deg moving 1.5-3.0 deg/s.
    ev = f"{OUT}/{cand}.eval.csv"
    ep = en = e_roll = e_mean = e_rate = e_ratio = float("nan")
    if os.path.exists(ev):
        er = list(csv.DictReader(open(ev)))
        if er:
            ep = sum(int(float(x.get("pass", 0))) for x in er)/len(er)
            en = sum(num(x, "net_fwd") for x in er)/len(er)
            g = lambda k: [num(x, k) for x in er if num(x, k) == num(x, k)]
            e_roll = float(np.mean(g("torso_roll_rms_deg"))) if g("torso_roll_rms_deg") else float("nan")
            e_mean = float(np.mean(g("torso_roll_mean_deg"))) if g("torso_roll_mean_deg") else float("nan")
            e_rate = float(np.mean(g("torso_roll_rate_rms_dps"))) if g("torso_roll_rate_rms_dps") else float("nan")
            if e_roll == e_roll and e_roll > 1e-6 and e_mean == e_mean:
                e_ratio = abs(e_mean)/e_roll
    # prefer eval-side torso columns where present
    if e_roll == e_roll: roll, rollm, ratio = e_roll, e_mean, e_ratio
    rate_use = e_rate if e_rate == e_rate else num(L, "torso_roll_rate_rms_dps")
    rows.append((cand, vtail, vswg, num(L,"fall_rate"), num(L,"ep_len"), roll, rollm,
                 ratio, rate_use, num(L,"hip_corr"),
                 num(L,"stride_asym"), num(L,"single_frac"), ep, en, L))
for r in sorted(rows, key=lambda r: -(r[13] if r[13]==r[13] else -1)):
    f = lambda v,w,d: (f"{v:>{w}.{d}f}" if v==v else f"{'-':>{w}}")
    flags = []
    if r[7] == r[7] and r[7] > 0.7: flags.append("HELD-LEAN")
    if r[8] == r[8] and r[8] < 30 and r[5] > 15: flags.append("STATIC-TORSO")
    if r[13] == r[13] and r[13] < 0.03 and r[5] > 15: flags.append("FAKE-TORSO")
    if r[2] == r[2] and r[2] > 0.05: flags.append("unstable")
    if abs(r[10]) > 0.25 if r[10]==r[10] else False: flags.append("ASYM")
    print(f"{r[0]:<16}{f(r[1],7,3)}{f(r[2],6,3)}{f(r[3],6,2)}{f(r[4],7,0)}"
          f"{f(r[5],7,1)}{f(r[6],7,1)}{f(r[7],7,2)}{f(r[8],6,0)}{f(r[9],7,2)}"
          f"{f(r[10],7,3)}{f(r[11],6,2)}{f(r[12],7,2)}{f(r[13],7,3)}"
          + ("   " + ",".join(flags) if flags else ""))

print("\nreward budget at the last diag row (per step):")
comp = ["r_track","r_progress","r_back","r_energy","r_swing","r_scrub","r_smooth","r_fall"]
print(f"{'cand':<16}" + "".join(f"{c.replace('r_',''):>10}" for c in comp) + f"{'pos_sum':>9}")
for r in rows:
    L = r[14]
    v = {c: num(L, c) for c in comp}
    pos = sum(x for x in v.values() if x == x and x > 0)
    print(f"{r[0]:<16}" + "".join(f"{v[c]:>10.3f}" if v[c]==v[c] else f"{'-':>10}" for c in comp)
          + f"{pos:>9.3f}")

print("""
HELD-LEAN     |roll_mean|/roll_RMS > 0.7 -- torso parked to one side, not swinging.
              This is the failure that reads as a success in the summary numbers.
STATIC-TORSO  high roll RMS but roll RATE < 30 deg/s -- leaning, barely moving.
FAKE-TORSO    high roll RMS in eval but net_fwd < 0.03 -- the exact false positive
              Ben flagged: reads as strong torso use, goes nowhere.
ASYM          |stride_asym| > 0.25 -- one leg taking much longer strides, i.e.
              curving or compensating for a held lean.
unstable      vswing > 0.05 over the last 1M steps; the vtail is a draw, not a value.
Frames: overnight/<gen>/frames/<cand>_strip.png  (t = 3, 6, 9 s stacked)""")
PYEOF
