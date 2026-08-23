#!/usr/bin/env bash
# The control the 9 e2 ablation arms are missing.
#
# runs/e2/s* are the two-stage warm-started protocol, which C7 dropped, so they
# are not a matched baseline for single-stage from-scratch + curriculum arms.
# Mac coordination note 2026-08-22: seed 0 is a byte-duplicate of Mac's
# runs/e2x2hf4b/a1p1 (r3d), already trained and evaled there (stand, 0/5 all mu),
# and training is byte-deterministic given (seed, config), so s0 is dropped.
# Seeds 1/2/3 here pool with Mac's s0 for a 4-seed spread. This trains the
# frozen reward under --mode e2:
# matched control, s0/s1/s2 together give the seed spread C1 asks for, so an
# ablation delta can be read against it instead of against nothing.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../.sweep_venv/bin/python}
GEN=e2_base
OUT="overnight/$GEN"
mkdir -p "$OUT/frames" "$OUT/logs"
echo "=== $GEN  $(date '+%F %T') ==="
pids=()
for s in 1 2 3; do
  ( nice -n 10 "$PY" -u train_grid4.py --mode e2 --seed "$s" --curriculum \
      --steps 3000000 --n-envs 4 --name "overnight/$GEN/base_s$s" \
      > "$OUT/logs/base_s$s.train.log" 2>&1 ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "  training done $(date '+%T')"
for s in 1 2 3; do
  name="base_s$s"; rd="runs/overnight/$GEN/$name"
  [ -f "$rd/ckpts/final.zip" ] || { echo "  $name: no final.zip"; continue; }
  nice -n 10 "$PY" -u eval_ckpt_sweep.py "$rd" --repeats 3 > "$OUT/logs/$name.sweep.log" 2>&1
  lbl=$(grep -oE "best=[0-9]+k" "$OUT/logs/$name.sweep.log" | tail -1 | cut -d= -f2)
  ck=""
  if [ -n "$lbl" ]; then
    st=$(( ${lbl%k} * 1000 ))
    [ -f "$rd/ckpts/ckpt_${st}_steps.zip" ] && ck="$rd/ckpts/ckpt_${st}_steps.zip"
  fi
  [ -n "$ck" ] || { echo "  WARNING $name: could not resolve '$lbl', using final"; ck="$rd/ckpts/final.zip"; }
  echo "  $name: selected $(basename "$ck")"; echo "$ck" > "$OUT/$name.selected_ckpt.txt"
  nice -n 10 "$PY" -u eval_grid4_policy.py "$ck" --mus 0.1,0.2,0.3,0.4 --repeats 5 \
      --trial-seed-base 50000 --out "$OUT/$name.eval.csv" > "$OUT/logs/$name.confirm.log" 2>&1
  MUJOCO_GL=egl nice -n 10 "$PY" -u render_grid4_policy.py "$ck" --mu 0.1 --dur 12 \
      --out "$OUT/$name.mu0.1.mp4" > "$OUT/logs/$name.render.log" 2>&1
  if [ -f "$OUT/$name.mu0.1.mp4" ]; then
    FF=$("$PY" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
    for t in 3 6 9; do "$FF" -y -ss $t -i "$OUT/$name.mu0.1.mp4" -frames:v 1 \
        "$OUT/frames/${name}_t${t}.png" -loglevel error 2>/dev/null; done
    "$FF" -y -i "$OUT/frames/${name}_t3.png" -i "$OUT/frames/${name}_t6.png" \
      -i "$OUT/frames/${name}_t9.png" -filter_complex "[0][1][2]vstack=3" \
      "$OUT/frames/${name}_strip.png" -loglevel error 2>/dev/null
  fi
done
bash overnight/report_gen.sh "$GEN" > "$OUT/report.txt" 2>&1
echo "=== $GEN complete $(date '+%F %T') ==="
