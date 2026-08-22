#!/usr/bin/env bash
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
# cfg file format, one candidate per line:  <name> <rw overrides...>
# Only existing reward weights may be changed (including to 0). No new terms.
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
  ( nice -n 10 "$PY" -u train_grid4.py --mode gate0 --seed "$SEED" --curriculum \
      --steps "$STEPS" --n-envs "$NENV" --name "overnight/$GEN/$name" --rw $rest \
      > "$OUT/logs/$name.train.log" 2>&1 ) &
  pids+=($!)
done < <(grep -vE '^\s*(#|$)' "$CFG")
echo "  training ${#pids[@]} candidates ..."
for p in "${pids[@]}"; do wait "$p"; done
echo "  training done $(date '+%T')"

# ---- eval + render each ----
while read -r name rest; do
  case "$name" in ''|'#'*) continue ;; esac
  ck="runs/overnight/$GEN/$name/ckpts/final.zip"
  [ -f "$ck" ] || { echo "  $name: no final.zip, skipped"; continue; }
  nice -n 10 "$PY" -u eval_grid4_policy.py "$ck" --mus "$MUS" --repeats 5 \
      --out "$OUT/$name.eval.csv" > "$OUT/logs/$name.eval.log" 2>&1
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
