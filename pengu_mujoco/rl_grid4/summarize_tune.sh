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
    #末值单点不可用：vx 在后段会剧烈震荡（seed 0 在相邻两个 diag 点之间从
    # 0.193 掉到 0.098）。改用后段窗口的均值和跨度：swing 大 = 还在摇摆，
    # 该 run 的任何单点数都不可信。
    tail = [num(x, "vx") for x in d[-4:]]
    tail = [v for v in tail if v == v]
    vtail = sum(tail) / len(tail) if tail else float("nan")
    vswing = (max(tail) - min(tail)) if len(tail) > 1 else float("nan")
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
                     vtail=vtail, vswing=vswing,
                     fall=num(L, "fall_rate"), eplen=num(L, "ep_len"),
                     roll=num(L, "torso_roll_rms_deg"),
                     rollm=num(L, "torso_roll_mean_deg"),
                     rate=num(L, "torso_roll_rate_rms_dps"),
                     hip=num(L, "hip_corr"), asym=num(L, "stride_asym"),
                     cmd=num(L, "vx_cmd"), epass=e_pass, enf=e_nf))

rows.sort(key=lambda r: (r["kind"], -r["vx"] if r["vx"] == r["vx"] else 0))
h = (f"{'run':<34}{'kind':<7}{'Msteps':>7}{'vx':>8}{'vtail':>8}{'vswing':>8}{'cmd':>6}"
     f"{'fall':>6}{'eplen':>7}{'roll':>7}{'hipc':>7}{'asym':>7}{'ePass':>7}{'eNfwd':>7}")
print(h); print("-" * len(h))
last = None
for r in rows:
    if last and r["kind"] != last: print()
    last = r["kind"]
    f = lambda k, w, d: (f"{r[k]:>{w}.{d}f}" if r[k] == r[k] else f"{'-':>{w}}")
    flag = ""
    if r["vswing"] == r["vswing"] and r["vswing"] > 0.05: flag = "  <unstable"
    elif r["dvx"] == r["dvx"] and r["dvx"] > 0.02: flag = "  <climbing"
    print(f"{r['run']:<34}{r['kind']:<7}{f('steps',7,2)}{f('vx',8,3)}{f('vtail',8,3)}"
          f"{f('vswing',8,3)}{f('cmd',6,2)}{f('fall',6,2)}{f('eplen',7,0)}{f('roll',7,1)}"
          f"{f('hip',7,2)}{f('asym',7,3)}{f('epass',7,2)}{f('enf',7,3)}" + flag)
print("""
vtail  = mean vx over the last 4 diag points (1M steps). Use this, not the final
         point: vx oscillates late in training (seed 0 went 0.193 -> 0.098 between
         two adjacent diag points while fall went 0.78 -> 0.28).
vswing = max-min of those 4 points. >0.05 means the run is still swinging between
         a fast-and-falling and a slow-and-safe policy; no single number from it
         is trustworthy.
dvx    = change in vx over the last diag interval; '<climbing' means the run had
         not converged when it stopped -- read its vx as a lower bound.
rollμ  = mean torso roll. |rollμ| ~ roll means a held lean, |rollμ| << roll a waddle.
rate   = RMS d(roll)/dt [deg/s]. ~0 with high roll = leaning, not moving.
asym   = stride length asymmetry (L-R)/(L+R). 0 = equal strides = walking straight.
ePass/eNfwd = frozen-eval pass fraction and mean net_fwd, all mu pooled.
kind   = frozen recipe vs reward-tuning arm. Do not pool the two.""")
PYEOF
