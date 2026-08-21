#!/usr/bin/env bash
# One table over every run under runs/: last diag row + frozen-eval means.
# Frozen-recipe runs and tuning arms are labelled so they are never read as one set.
#   bash summarize_tune.sh            # table to stdout
#   bash summarize_tune.sh > tune_summary.txt
set -u
cd "$(dirname "$0")"
PY=${PY:-../.sweep_venv/bin/python}
"$PY" - <<'PYEOF'
import csv, glob, os, re
def num(r, k, d=float("nan")):
    try: return float(r[k])
    except Exception: return d

rows = []
for dg in sorted(glob.glob("runs/*/diag.csv")):
    run = os.path.basename(os.path.dirname(dg))
    if "smoke" in run: continue
    d = list(csv.DictReader(open(dg)))
    if not d: continue
    L = d[-1]
    # arm = the reward override baked into the tag, if any
    m = re.search(r"r2((?:ns)?(?:-[a-z_]+[0-9.]+)*)", run)
    arm = (m.group(1) or "").lstrip("-") if m else ""
    kind = "frozen" if not arm else "tune"
    # still-climbing check: change over the last diag interval
    dv = num(L, "vx") - num(d[-2], "vx") if len(d) > 1 else float("nan")
    ev = glob.glob(os.path.join("runs", run, "eval_*.csv"))
    e_pass = e_nf = float("nan")
    if ev:
        er = list(csv.DictReader(open(ev[0])))
        if er:
            e_pass = sum(int(float(x.get("pass", 0))) for x in er) / len(er)
            e_nf = sum(num(x, "net_fwd") for x in er) / len(er)
    rows.append(dict(run=run, kind=kind, arm=arm or "-",
                     steps=num(L, "steps") / 1e6, vx=num(L, "vx"), dvx=dv,
                     fall=num(L, "fall_rate"), eplen=num(L, "ep_len"),
                     roll=num(L, "torso_roll_rms_deg"),
                     rollm=num(L, "torso_roll_mean_deg"),
                     rate=num(L, "torso_roll_rate_rms_dps"),
                     hip=num(L, "hip_corr"), asym=num(L, "stride_asym"),
                     cmd=num(L, "vx_cmd"), epass=e_pass, enf=e_nf))

rows.sort(key=lambda r: (r["kind"], -r["vx"] if r["vx"] == r["vx"] else 0))
h = (f"{'run':<34}{'kind':<7}{'Msteps':>7}{'vx':>8}{'dvx':>7}{'cmd':>6}{'fall':>6}"
     f"{'eplen':>7}{'roll':>7}{'rollμ':>7}{'rate':>6}{'hipc':>7}{'asym':>7}"
     f"{'ePass':>7}{'eNfwd':>7}")
print(h); print("-" * len(h))
last = None
for r in rows:
    if last and r["kind"] != last: print()
    last = r["kind"]
    f = lambda k, w, d: (f"{r[k]:>{w}.{d}f}" if r[k] == r[k] else f"{'-':>{w}}")
    flag = " <climbing" if r["dvx"] == r["dvx"] and r["dvx"] > 0.02 else ""
    print(f"{r['run']:<34}{r['kind']:<7}{f('steps',7,2)}{f('vx',8,3)}{f('dvx',7,3)}"
          f"{f('cmd',6,2)}{f('fall',6,2)}{f('eplen',7,0)}{f('roll',7,1)}{f('rollm',7,1)}"
          f"{f('rate',6,0)}{f('hip',7,2)}{f('asym',7,3)}{f('epass',7,2)}{f('enf',7,3)}"
          + flag)
print("""
dvx    = change in vx over the last diag interval; '<climbing' means the run had
         not converged when it stopped -- read its vx as a lower bound.
rollμ  = mean torso roll. |rollμ| ~ roll means a held lean, |rollμ| << roll a waddle.
rate   = RMS d(roll)/dt [deg/s]. ~0 with high roll = leaning, not moving.
asym   = stride length asymmetry (L-R)/(L+R). 0 = equal strides = walking straight.
ePass/eNfwd = frozen-eval pass fraction and mean net_fwd, all mu pooled.
kind   = frozen recipe vs reward-tuning arm. Do not pool the two.""")
PYEOF
