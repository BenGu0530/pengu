#!/usr/bin/env bash
# Decide which ablation arms deserve extra seeds, instead of believing a single
# seed. An arm counts as a candidate only if its effect is larger than the
# frozen baseline's own seed-to-seed spread -- with 4 baseline seeds ranging
# 0.063-0.224 at 3M, anything smaller than that range is indistinguishable from
# seed noise.
#
# Refuses to rank at all if any run involved had not converged, because a
# truncated run's vx is a lower bound and differences between lower bounds mean
# nothing.
#
#   bash triage_arms.sh                # report + ready-to-paste commands
#   MIN_STEPS=5500000 bash triage_arms.sh
set -u
cd "$(dirname "$0")"
PY=${PY:-../.sweep_venv/bin/python}
MIN_STEPS=${MIN_STEPS:-5500000} "$PY" - <<'PYEOF'
import csv, glob, os, re, statistics as st

MIN = float(os.environ.get("MIN_STEPS", 5.5e6))

def num(r, k, d=float("nan")):
    try: return float(r[k])
    except Exception: return d

runs = {}
for dg in sorted(glob.glob("runs/*/diag.csv")):
    name = os.path.basename(os.path.dirname(dg))
    if "smoke" in name or name.endswith("_3M") or "cut" in name:
        continue
    d = list(csv.DictReader(open(dg)))
    if not d:
        continue
    L = d[-1]
    steps = num(L, "steps")
    if steps < MIN:                      # only fully-run arms are comparable
        continue
    m = re.match(r"gate0_r2((?:ns)?(?:-[a-z_]+[0-9.]+)*)a1e1c2_s(\d+)$", name)
    if not m:
        continue
    arm = (m.group(1) or "").lstrip("-") or "FROZEN"
    seed = int(m.group(2))
    dv = num(L, "vx") - num(d[-2], "vx") if len(d) > 1 else float("nan")
    runs.setdefault(arm, []).append(dict(
        name=name, seed=seed, vx=num(L, "vx"), dvx=dv, fall=num(L, "fall_rate"),
        hip=num(L, "hip_corr"), roll=num(L, "torso_roll_rms_deg"),
        asym=num(L, "stride_asym"), steps=steps))

if "FROZEN" not in runs:
    raise SystemExit(f"no completed frozen baseline run at >= {MIN/1e6:.1f}M steps; "
                     "nothing to compare against yet")

base = runs.pop("FROZEN")
bvx = sorted(r["vx"] for r in base)
lo, hi, med = bvx[0], bvx[-1], st.median(bvx)
bhip = st.median(r["hip"] for r in base)

print(f"frozen baseline: {len(base)} seed(s) at >= {MIN/1e6:.1f}M steps")
for r in sorted(base, key=lambda r: r["seed"]):
    print(f"    s{r['seed']}  vx {r['vx']:+.3f}  dvx {r['dvx']:+.3f}  "
          f"fall {r['fall']:.2f}  hip_corr {r['hip']:+.2f}")
print(f"    median vx {med:+.3f}   seed range [{lo:+.3f}, {hi:+.3f}]  "
      f"(width {hi-lo:.3f})   median hip_corr {bhip:+.2f}")

climbing = [r["name"] for r in base if r["dvx"] == r["dvx"] and r["dvx"] > 0.02]
for a, rs in runs.items():
    climbing += [r["name"] for r in rs if r["dvx"] == r["dvx"] and r["dvx"] > 0.02]
if climbing:
    print("\n*** NOT RANKING ***")
    print("These runs were still climbing when they stopped, so their vx is a")
    print("lower bound and comparing them is meaningless. Extend them first:")
    for n in climbing:
        print(f"    {n}")
    print("\n(everything below is printed for information only)")

print(f"\n{'arm':<14}{'seeds':>6}{'vx':>8}{'vs base':>9}{'hip_corr':>10}{'fall':>7}"
      f"{'asym':>8}   verdict")
print("-" * 86)
cands = []
for arm, rs in sorted(runs.items(), key=lambda kv: -st.median(r["vx"] for r in kv[1])):
    vx = st.median(r["vx"] for r in rs)
    hip = st.median(r["hip"] for r in rs)
    d = vx - med
    outside = vx < lo or vx > hi
    flipped = (hip < 0) != (bhip < 0)
    why = []
    if outside: why.append("outside seed range")
    if flipped: why.append("hip_corr sign flip")
    verdict = ("ADD SEEDS: " + ", ".join(why)) if why else "within seed noise, skip"
    if why and len(rs) < 3:
        cands.append(arm)
    print(f"{arm:<14}{len(rs):>6}{vx:>8.3f}{d:>+9.3f}{hip:>+10.2f}"
          f"{st.median(r['fall'] for r in rs):>7.2f}"
          f"{st.median(r['asym'] for r in rs):>8.3f}   {verdict}")

print("\nRule: an arm is a candidate only if its median vx falls OUTSIDE the")
print("frozen seeds' own range, or its hip_corr flips sign (in-phase vs")
print("alternating gait). A difference smaller than the baseline's seed spread")
print("cannot be separated from seed noise on n=1.")

if not cands:
    print("\nNo arm clears the bar. Nothing to follow up.")
else:
    print(f"\n{len(cands)} arm(s) worth more seeds. Paste to run seeds 1 and 2:\n")
    ov = {}
    for line in open("ablate_arms.txt"):
        line = line.split("#")[0].strip()
        if line:
            p = line.split()
            ov[p[0].replace("_", "").replace("no", "", 1) if False else p[0]] = " ".join(p[1:])
    # map tag-arm (e.g. track0) back to the declared override
    decl = {}
    for line in open("ablate_arms.txt"):
        line = line.split("#")[0].strip()
        if not line: continue
        nm, *rest = line.split()
        key, _, val = rest[0].partition("=")
        decl[f"{key}{float(val):g}"] = (nm, " ".join(rest))
    for arm in cands:
        nm, rwargs = decl.get(arm, (arm, None))
        if rwargs is None:
            print(f"  # {arm}: could not map back to ablate_arms.txt, run by hand")
            continue
        print(f"  SEEDS=\"1 2\" ARMS=<(echo '{nm} {rwargs}') bash run_tune_queue.sh")
    print("\n  # or all of them in one queue:")
    names = " ".join(decl[a][0] for a in cands if a in decl)
    print(f"  grep -E '^({'|'.join(decl[a][0] for a in cands if a in decl)}) ' ablate_arms.txt "
          f"> followup_arms.txt && SEEDS=\"1 2\" ARMS=followup_arms.txt bash run_tune_queue.sh")
PYEOF
