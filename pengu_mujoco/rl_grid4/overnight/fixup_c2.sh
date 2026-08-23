#!/usr/bin/env bash
# Repair generations run under run_gen_v2.sh, where the checkpoint-selection
# step silently fell back to final.zip: eval_ckpt_sweep prints a LABEL ("2000k")
# and v2 fed that straight to [ -f ], which always failed.
#
# The sweep itself was fine, so ckpt_sweep.csv already holds the selection. This
# re-runs only the CONFIRM (independent trial seeds) and the render, against the
# checkpoint the sweep actually picked.
#
#   bash fixup_c2.sh retro_a retro_b retro_c
set -u
cd "$(dirname "$0")/.."
PY=${PY:-../.sweep_venv/bin/python}
MUS=${MUS:-0.1,0.2,0.3,0.4}
DEMO_MU=${DEMO_MU:-0.1}

for GEN in "$@"; do
  OUT="overnight/$GEN"
  [ -d "$OUT" ] || { echo "[skip] $GEN missing"; continue; }
  for rd in runs/overnight/$GEN/*/; do
    name=$(basename "$rd")
    sw="$rd/ckpt_sweep.csv"
    [ -f "$sw" ] || { echo "  $GEN/$name: no ckpt_sweep.csv, skipped"; continue; }
    # selection metric as fixed in eval_ckpt_sweep: n_pass, tie-break mean_net_fwd
    best=$("$PY" - "$sw" <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("steps")]
if rows:
    b = max(rows, key=lambda r: (int(r["n_pass"]), float(r["mean_net_fwd"])))
    print(b["steps"])
PYEOF
)
    [ -n "$best" ] || { echo "  $GEN/$name: empty sweep"; continue; }
    ck="$rd/ckpts/ckpt_${best}_steps.zip"
    [ -f "$ck" ] || ck="$rd/ckpts/final.zip"
    echo "  $GEN/$name: selected $(basename "$ck")"
    echo "$ck" > "$OUT/$name.selected_ckpt.txt"
    nice -n 10 "$PY" -u eval_grid4_policy.py "$ck" --mus "$MUS" --repeats 5 \
        --trial-seed-base 50000 --out "$OUT/$name.eval.csv" \
        > "$OUT/logs/$name.confirm.log" 2>&1
    MUJOCO_GL=egl nice -n 10 "$PY" -u render_grid4_policy.py "$ck" --mu "$DEMO_MU" \
        --dur 12 --out "$OUT/$name.mu${DEMO_MU}.mp4" \
        > "$OUT/logs/$name.render.log" 2>&1
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
  done
  bash overnight/report_gen.sh "$GEN" > "$OUT/report.txt" 2>&1
  echo "[fixup] $GEN done"
done
echo "[fixup] all done $(date '+%F %T')"
