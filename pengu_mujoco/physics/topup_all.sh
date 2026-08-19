#!/usr/bin/env bash
# Upgrade EVERY K=1 passing row (pass_rate > 0) of a config to true K=5:
# runs repeats r=1..4 with the frozen seeds and merges — identical to a native K=5
# sweep for those rows. Non-passing rows stay K=1 (documented in the report).
#
#   bash physics/topup_all.sh c4 [n_shards]           # resume-safe, re-run any time
#   SWEEP_NICE=19 bash physics/topup_all.sh c6        # shared box
#
# Output: results/gait_sweep/sweep_grid4_cN_topupK5.csv  (12-col, rows override base)
set -u
CFG="${1:?usage: topup_all.sh cN [n_shards]}"
cd "$(dirname "$0")/.."

BASE="results/gait_sweep/sweep_grid4_${CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
[ -f "$BASE" ] || gunzip -k "$BASE.gz" 2>/dev/null || { echo "FATAL: no $BASE(.gz)"; exit 1; }
SEL="results/gait_sweep/topup_sel_${CFG}.csv"
OUT="results/gait_sweep/sweep_grid4_${CFG}_topupK5.csv"
LOG="results/gait_sweep/${CFG}_topup.log"

# python: 3.8.x mujoco only (same gate as run_sweep.sh)
PY=""
for c in "${GRID3_PY:-}" "$PWD/.sweep_venv/bin/python" python3 python; do
  [ -z "$c" ] && continue
  if "$c" -c "import mujoco, numpy; assert mujoco.__version__.startswith('3.8')" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -z "$PY" ] && { echo "FATAL: no mujoco-3.8 python (run physics/run_sweep.sh once to build .sweep_venv)"; exit 1; }
echo "using python: $PY"

# selection = all pass>0 rows of the base map
"$PY" - "$BASE" "$SEL" <<'PYEOF'
import sys, csv
base, sel = sys.argv[1], sys.argv[2]
n = 0
with open(base) as f, open(sel, "w", newline="") as g:
    w = csv.writer(g); w.writerow(["freq","hip_phi","leg_amp","hip_amp","hip_off","mu"])
    for r in csv.DictReader(f):
        try:
            if float(r["pass_rate"]) > 0:
                w.writerow([r["freq"], r["hip_phi"], r["leg_amp"], r["hip_amp"], r["hip_off"], r["mu"]]); n += 1
        except (ValueError, KeyError):
            pass
print(f"selection: {n} passing rows")
PYEOF

[ -f "$OUT" ] || head -1 "$BASE" > "$OUT"        # header first: avoids concurrent-header race
CORES="$( { getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4; } )"
N="${2:-$(( CORES > 3 ? CORES - 2 : 1 ))}"
for s in $(seq 0 $((N - 1))); do
  CONFIG="$CFG" TARGET_K=5 N_SHARDS=$N SHARD_ID=$s \
    nohup nice -n "${SWEEP_NICE:-0}" "$PY" physics/topup_k.py "$BASE" "$SEL" "$OUT" >> "$LOG" 2>&1 &
done
done0=$(( $(wc -l < "$OUT") - 1 ))
target=$(( $(wc -l < "$SEL") - 1 ))
echo "launched $CFG topup: $N shards, $done0/$target rows done (resume-safe; re-run to revive)"
echo "watch:  wc -l $OUT      # target $((target + 1)) incl header"
echo "ship:   gzip -kf $OUT && git add -f $OUT.gz && git pull --rebase && git commit -m 'GRID-4 $CFG topupK5 (all passers)' && git push"
