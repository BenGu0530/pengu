#!/usr/bin/env bash
# pipeline.sh — the two-stage GRID-6 c1 campaign, unattended.
#
#   stage 1   c1's robust region at 0.05 Hz          ~414,720 rows, ~8 h on 11 shards
#   select    keep what passes, fits the motors, and is not a spike
#   stage 2   refine the survivors to 0.01 Hz
#   report    one file with what was found
#
# Everything runs on models/hardware_c1 — the hardware model, hardened to 5 actuators
# and ballasted to the ladder's 2.2724 kg at COM ratio 1.0500. Rows are tagged
# grid6_c1_c1r / grid6_c1_c1f so they can never be mistaken for the slide-tuned c1 map
# that PSC and the XPS laptop are producing.
#
#   bash grid6/pipeline.sh [shards]        default 11 (coprime with 60, no tail idling)
#
# Safe to re-run: both stages resume by cell index, so an interrupted run picks up where
# it stopped. Stop it with:  touch results/gait_sweep/WATCHDOG_OFF && pkill -f grid6_sweep
set -u
cd "$(dirname "$0")/.."                       # repo root pengu_mujoco/

N="${1:-11}"
PY=/opt/anaconda3/envs/pengu/bin/python3.11
OUT=results/gait_sweep
LOG=$OUT/grid6_pipeline.log
S1=$OUT/sweep_grid6_c1_c1r_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
S2=$OUT/sweep_grid6_c1_c1f_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv

say() { echo "$(date '+%F %T')  $*" | tee -a "$LOG"; }

run_stage() {                                  # $1 = cell list, $2 = human label
  local cells="$1" label="$2"
  say "=== $label: launching $N shards on $(basename "$cells") ==="
  CONFIG=c1 GRID6_CELLS="$cells" "$PY" grid6/grid6_sweep.py initcsv >>"$LOG" 2>&1
  for s in $(seq 0 $((N-1))); do
    CONFIG=c1 GRID6_CELLS="$cells" N_SHARDS=$N SHARD_ID=$s \
      nohup "$PY" -u grid6/grid6_sweep.py >>"$OUT/grid6_${label}_run.log" 2>&1 &
  done
  sleep 20
  say "$label: $(pgrep -f 'grid6_sweep.py' | wc -l | tr -d ' ') shards up"
  # poll until every shard has exited; report progress so a tail -f is informative
  while [ "$(pgrep -f 'grid6_sweep.py' | wc -l | tr -d ' ')" -gt 0 ]; do
    sleep 300
    if [ -f "$OUT/WATCHDOG_OFF" ]; then
      say "$label: WATCHDOG_OFF present — stopping the pipeline"; pkill -f grid6_sweep.py
      exit 3
    fi
  done
  say "$label: all shards exited"
  # A dead shard also makes pgrep return 0, which is indistinguishable from "finished"
  # unless the output is counted. Delta lost six hours to exactly this failure mode.
  local csv="$OUT/sweep_grid6_c1_${label}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
  local want=$(( ($(wc -l < "$cells") - 1) * 4 ))
  local got=$(( $(wc -l < "$csv") - 1 ))
  say "$label: $got / $want rows"
  if [ "$got" -lt "$want" ]; then
    say "$label: SHORT by $((want-got)) rows — reviving once"
    for s in $(seq 0 $((N-1))); do
      CONFIG=c1 GRID6_CELLS="$cells" N_SHARDS=$N SHARD_ID=$s \
        nohup "$PY" -u grid6/grid6_sweep.py >>"$OUT/grid6_${label}_run.log" 2>&1 &
    done
    sleep 20
    while [ "$(pgrep -f 'grid6_sweep.py' | wc -l | tr -d ' ')" -gt 0 ]; do sleep 300; done
    got=$(( $(wc -l < "$csv") - 1 ))
    say "$label: after revival $got / $want rows"
    [ "$got" -lt "$want" ] && say "$label: STILL SHORT — the report covers what exists"
  fi
}

say "################ GRID-6 c1 pipeline, $N shards ################"
say "model: models/hardware_c1 (hardware model, 2.2724 kg, COM ratio 1.0500)"
rm -f "$OUT/WATCHDOG_OFF"

# ---------------------------------------------------------------- stage 1
"$PY" grid6/region_cells.py | tee -a "$LOG"
run_stage grid6/cells_c1r.csv c1r
say "stage 1 rows: $(( $(wc -l < "$S1") - 1 ))"

# ---------------------------------------------------------------- selection
say "=== selecting stage 2 ==="
"$PY" grid6/select_stage2.py "$S1" 2>&1 | tee -a "$LOG"
if [ ! -s grid6/cells_c1f.csv ] || [ "$(wc -l < grid6/cells_c1f.csv)" -le 1 ]; then
  say "nothing survived the pass + motor-envelope + robustness filters; stage 2 skipped"
else
  run_stage grid6/cells_c1f.csv c1f
  say "stage 2 rows: $(( $(wc -l < "$S2") - 1 ))"
fi

# ---------------------------------------------------------------- report
say "=== report ==="
"$PY" grid6/report_c1.py 2>&1 | tee -a "$LOG"
say "################ pipeline finished ################"
