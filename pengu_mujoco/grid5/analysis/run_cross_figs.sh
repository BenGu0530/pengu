#!/bin/bash
# GRID-5 cross-config figures, all complete configs. Reconstructed from the 2026-09-03
# session (which produced results/grid5_report/cross/ for c1 c3 c4 c5 c6 and left no memo)
# from the file timestamps and each script's usage line; recorded here so the next run is
# a command, not an archaeology.
#
#   bash grid5/analysis/run_cross_figs.sh [out_dir] [configs...]
#   default: results/grid5_report/cross  c1 c2 c3 c4 c5 c10 c6
#
# c6 MUST be listed last: it carries the extra mu 0.9 and load5.compatible() takes the
# first config's mu axis as the shared one (extension allowed, prefix is a hard error).
# Needs the npz caches (results/grid5_report/cache/grid5_<cfg>.npz); a missing one is
# built on first load (~25 s per config from the .csv.gz).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-results/grid5_report/cross}"
shift || true
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=(c1 c2 c3 c4 c5 c10 c6)
P=${PENGU_PY:-/opt/anaconda3/envs/pengu/bin/python}
F=grid5/analysis/figs
mkdir -p "$OUT"
echo "configs: ${CFGS[*]}   out: $OUT   $(date)"

run() { echo; echo "== $*"; "$P" "$@" 2>&1 | grep -v "^\s*$" | tail -4; }

# batch 1 (09-03 15:57-16:07)
run $F/robust_region.py   --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/robust_region.py   --round grid5 --configs "${CFGS[@]}" --out "$OUT" --frac
run $F/com_ladder.py      --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/cone_util.py       --configs "${CFGS[@]}" --out "$OUT"
run $F/fall_phase.py      --configs "${CFGS[@]}" --out "$OUT"
run $F/speed_rank.py      --round grid5 --configs "${CFGS[@]}" --out "$OUT" --mu 0.1
run $F/speed_vs_mu.py     --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/speed_vs_mu.py     --round grid5 --configs "${CFGS[@]}" --out "$OUT" --tier robust
run $F/speed_vs_mu.py     --round grid5 --configs "${CFGS[@]}" --out "$OUT" --top 100
run $F/speed_vs_mu.py     --round grid5 --configs "${CFGS[@]}" --out "$OUT" --top-frac 1.0 --tier robust
# batch 2 (09-03 16:54-17:06)
run $F/passfrac_vs_mu.py      --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/thickness_vs_mu.py     --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/cot_frontier.py        --configs "${CFGS[@]}" --out "$OUT"
run $F/cot_vs_mu.py           --round grid5 --configs "${CFGS[@]}" --out "$OUT"
run $F/tstart_vs_mu.py        --configs "${CFGS[@]}" --out "$OUT"
run $F/imu_roll.py            --configs "${CFGS[@]}" --out "$OUT"
run $F/lat_disp.py            --configs "${CFGS[@]}" --out "$OUT"
run $F/nonpasser_breakdown.py --configs "${CFGS[@]}" --out "$OUT"
run $F/slip_vs_roll.py        --configs "${CFGS[@]}" --out "$OUT"
run $F/speed_vs_mu.py --round grid5 --configs "${CFGS[@]}" --out "$OUT" --top-frac 0.001
run $F/speed_vs_mu.py --round grid5 --configs "${CFGS[@]}" --out "$OUT" --top-frac 0.01
run $F/speed_vs_mu.py --round grid5 --configs "${CFGS[@]}" --out "$OUT" --top-frac 0.05

rm -rf "$OUT/bw"          # style5.finish() writes greyscale twins; Ben 2026-09-09: not wanted
echo; echo "done $(date)"; ls "$OUT"/*.png | wc -l | xargs echo "png:"
