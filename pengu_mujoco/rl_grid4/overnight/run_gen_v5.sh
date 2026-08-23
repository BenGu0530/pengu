#!/usr/bin/env bash
# v5: the cfg line after <name> is passed to train_grid4.py VERBATIM, so an arm
# can set --crank-band / --shape / --rw independently. v4 forced everything
# through --rw, which cannot express the band or the shape.
#
# v4: --mode e2 (the ice arm, mu ~ U(0.1,0.4)). v1-v3 ran --mode gate0 (mu
# fixed 0.7) while evaluating at mu 0.1-0.4 -- trained and tested on different
# friction. Gate 0 had PASSED on 2026-08-21 and the protocol had already moved
# to e2; the gate0 line was carried over from an earlier command and never
# re-examined. Everything from gen01 through gen03 is affected.
#
# One generation of the overnight reward search: train N candidate reward
# configs concurrently, then eval and render each so the result can be looked
# at, not only read.
#
# The looking matters. Several known local optima are invisible in the summary
# numbers -- e.g. a torso held over to one side with the legs compensating
# every step reads as "fast, high torso roll RMS, survives", which is what a
# good result also looks like. torso_roll_mean_deg and torso_roll_rate_rms_dps
# separate those two, and the frames confirm it.
#
#   bash run_gen.sh gen01 cfgs/gen01.txt
#
# cfg file format, one candidate per line:  <name> <verbatim train args...>
#   w03   --crank-band 0.0 1.9 --rw straight=0.3
#   pen   --crank-band 0.0 1.9 --shape penalty --rw fall=250
set -u
cd "$(dirname "$0")/.."                     # rl_grid4/
GEN=${1:?usage: run_gen.sh <gen-name> <cfg-file>}
CFG=${2:?usage: run_gen.sh <gen-name> <cfg-file>}
PY=${PY:-../.sweep_venv/bin/python}
STEPS=${STEPS:-3000000}
NENV=${NENV:-4}
SEED=${SEED:-0}
MUS=${MUS:-0.1,0.2,0.3,0.4}
DEMO_MU=${DEMO_MU:-0.1}
OUT="overnight/$GEN"
mkdir -p "$OUT/frames" "$OUT/logs"

echo "=== $GEN  $(date '+%F %T') ==="
grep -vE '^\s*(#|$)' "$CFG" | while read -r name rest; do
  echo "  $name : $rest"
done

# ---- train all candidates concurrently ----
pids=()
while read -r name rest; do
  case "$name" in ''|'#'*) continue ;; esac
  ( nice -n 10 "$PY" -u train_grid4.py --mode e2 --seed "$SEED" --curriculum \
      --steps "$STEPS" --n-envs "$NENV" --name "overnight/$GEN/$name" $rest \
      > "$OUT/logs/$name.train.log" 2>&1 ) &
  pids+=($!)
done < <(grep -vE '^\s*(#|$)' "$CFG")
echo "  training ${#pids[@]} candidates ..."
for p in "${pids[@]}"; do wait "$p"; done
echo "  training done $(date '+%T')"

# ---- C2 protocol (Ben, rl_open_concerns Resolutions 2026-08-22): select each
# run's checkpoint by frozen eval, then CONFIRM on independent trial seeds. The
# confirmation numbers are the reportable ones. final.zip samples the late
# oscillation at an arbitrary phase (best-vs-final pass deltas +2..+10 of 12).
# Everything downstream -- eval, render, frames -- uses the SELECTED checkpoint.
# ---- select, confirm, render each ----
while read -r name rest; do
  case "$name" in ''|'#'*) continue ;; esac
  rd="runs/overnight/$GEN/$name"
  [ -f "$rd/ckpts/final.zip" ] || { echo "  $name: no final.zip, skipped"; continue; }
  # 1. selection sweep over every saved checkpoint (3 reps, selection seeds)
  nice -n 10 "$PY" -u eval_ckpt_sweep.py "$rd" --repeats 2 \
      > "$OUT/logs/$name.sweep.log" 2>&1
  # eval_ckpt_sweep prints a LABEL (e.g. "2000k"), not a path. v2 fed that label
  # straight to [ -f ] and silently fell back to final.zip, so C2 selection never
  # actually happened. Map the label to the checkpoint file instead.
  lbl=$(grep -oE "best=[0-9]+k" "$OUT/logs/$name.sweep.log" | tail -1 | cut -d= -f2)
  ck=""
  if [ -n "$lbl" ]; then
    steps=$(( ${lbl%k} * 1000 ))
    for c in "$rd/ckpts/ckpt_${steps}_steps.zip" "$rd/ckpts/final.zip"; do
      [ -f "$c" ] && { ck="$c"; break; }
    done
  fi
  [ -n "$ck" ] && [ -f "$ck" ] || ck="$rd/ckpts/final.zip"
  [ "$(basename "$ck")" = "final.zip" ] && [ -n "$lbl" ] && \
    echo "  $name: WARNING selection label $lbl did not resolve, using final.zip"
  echo "  $name: selected $(basename "$ck")"
  echo "$ck" > "$OUT/$name.selected_ckpt.txt"
  # 2. confirmation on INDEPENDENT trial seeds -- these are the reportable numbers
  nice -n 10 "$PY" -u eval_grid4_policy.py "$ck" --mus "$MUS" --repeats 5 \
      --trial-seed-base 50000 \
      --out "$OUT/$name.eval.csv" > "$OUT/logs/$name.eval.log" 2>&1
  # 3. render the SELECTED checkpoint, not final
  MUJOCO_GL=egl nice -n 10 "$PY" -u render_grid4_policy.py "$ck" --mu "$DEMO_MU" \
      --dur 12 --out "$OUT/$name.mu${DEMO_MU}.mp4" > "$OUT/logs/$name.render.log" 2>&1
  if [ -f "$OUT/$name.mu${DEMO_MU}.mp4" ]; then
    FF=$("$PY" -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
    for t in 3 6 9; do
      "$FF" -y -ss $t -i "$OUT/$name.mu${DEMO_MU}.mp4" -frames:v 1 \
        "$OUT/frames/${name}_t${t}.png" -loglevel error 2>/dev/null
    done
    "$FF" -y -i "$OUT/frames/${name}_t3.png" -i "$OUT/frames/${name}_t6.png" \
      -i "$OUT/frames/${name}_t9.png" -filter_complex "[0][1][2]vstack=3" \
      "$OUT/frames/${name}_strip.png" -loglevel error 2>/dev/null
  fi
  echo "  $name: eval + render done"
done < <(grep -vE '^\s*(#|$)' "$CFG")

bash overnight/report_gen.sh "$GEN"
echo "=== $GEN complete $(date '+%F %T') ==="
