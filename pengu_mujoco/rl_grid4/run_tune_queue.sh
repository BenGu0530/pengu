#!/usr/bin/env bash
# Run the reward arms in ablate_arms.txt (default) one after another, N at a time,
# each followed automatically by the frozen eval. Resumable: an arm whose
# eval CSV already exists is skipped, so re-running the script continues where
# it stopped after a reboot or a Ctrl-C.
#
#   bash run_tune_queue.sh                       # defaults: 3 concurrent, 6M steps, seed 0
#   JOBS=2 STEPS=3000000 SEEDS="0 1" bash run_tune_queue.sh
#   ARMS=tune_arms.txt bash run_tune_queue.sh   # magnitude tuning, only after ablations
#
# Every arm is an independent single-factor change from the frozen baseline;
# none is selected in response to another's result. Tags carry the override, so
# tuning runs can never be pooled with frozen-recipe runs.
set -u
cd "$(dirname "$0")"

ARMS=${ARMS:-ablate_arms.txt}
JOBS=${JOBS:-3}
STEPS=${STEPS:-6000000}
SEEDS=${SEEDS:-0}
PY=${PY:-../.sweep_venv/bin/python}
CORES=$( (getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8) )
NENV=${NENV:-$(( (CORES - 2) / JOBS ))}
[ "$NENV" -lt 2 ] && NENV=2
mkdir -p runs logs

[ -x "$PY" ] || { echo "python not found at $PY"; exit 1; }
"$PY" -c "import stable_baselines3, gymnasium" 2>/dev/null || {
  echo "installing deps into $PY ..."; "$PY" -m pip install -q gymnasium "stable-baselines3>=2.0"; }

echo "queue: $(grep -cvE '^\s*(#|$)' "$ARMS") arms x $(echo $SEEDS | wc -w) seed(s)"
echo "       ${JOBS} concurrent, ${NENV} envs each, ${STEPS} steps, ${CORES} cores"
echo

launch() {                       # launch <name> <seed> <rw args...>
  local name=$1 seed=$2; shift 2
  local log="logs/${name}_s${seed}.log"
  echo "[$(date +%H:%M:%S)] start ${name} seed ${seed}   (rw: $*)"
  nice -n 10 "$PY" -u train_grid4.py --mode gate0 --seed "$seed" --curriculum \
      --steps "$STEPS" --n-envs "$NENV" --rw "$@" > "$log" 2>&1
  # tag is printed by the trainer; recover the run dir from it
  local dir
  dir=$(grep -oE 'runs/[A-Za-z0-9._-]+' "$log" | head -1)
  if [ -n "$dir" ] && [ -f "$dir/ckpts/final.zip" ]; then
    echo "[$(date +%H:%M:%S)] eval  ${name} seed ${seed}  -> $dir"
    nice -n 10 "$PY" -u eval_grid4_policy.py "$dir/ckpts/final.zip" \
        --out "$dir/eval_frozen.csv" >> "$log" 2>&1
  else
    echo "[$(date +%H:%M:%S)] FAILED ${name} seed ${seed}  (see $log)"
  fi
}

n=0
while read -r name rest; do
  case "$name" in ''|'#'*) continue ;; esac
  for seed in $SEEDS; do
    # resume: skip an arm that already produced an eval
    if compgen -G "runs/*${name}*/eval_frozen.csv" > /dev/null 2>&1; then
      echo "[skip] ${name} seed ${seed} already evaluated"; continue
    fi
    launch "$name" "$seed" $rest &
    n=$((n + 1))
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 20; done
  done
done < "$ARMS"

wait
echo
echo "[$(date +%H:%M:%S)] queue done, ${n} runs"
bash summarize_tune.sh 2>/dev/null || true
