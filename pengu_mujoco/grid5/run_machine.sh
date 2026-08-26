#!/usr/bin/env bash
# GRID-5 per-machine queue runner: one line per machine, works through its config
# queue in order, resume-safe, self-reviving. Usage (after git pull):
#
#   nohup bash grid5/run_machine.sh <machine> > results/gait_sweep/machine_$(hostname).log 2>&1 &
#   tail -f results/gait_sweep/grid5_c*_run.log        # live sweep progress
#
# machine -> queue (Phase A config first, Phase B second; edit here to reassign):
#   naomio : c4 c2 c9 (strongest box, carries the extra Phase-A config)
#   rml3   : c5 c10          mac : c3 c8 (Mac shares CPU with other projects)
#   rml2   : c6 c7 (full CPU — Ben 2026-08-26: sweeping first, RL track paused)
#   laptop : c1  (weak box: one config; pick up leftovers when done)
#
# Per config: initcsv+manifest+shards via run_sweep.sh, grid5 watchdog installed for
# reboot survival, then poll .done every 5 min (reviving dead shards). A config whose
# .done already exists is skipped, so re-running this line after a crash/reboot is
# always safe. When the whole queue is done it prints DONE and exits.
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/

M="${1:?usage: run_machine.sh mac|rml3|naomio|rml2|laptop}"
NICE=0; NSH=""
case "$M" in
  naomio) QUEUE="c4 c2 c9" ;;
  rml3)   QUEUE="c5 c10" ;;
  mac)    QUEUE="c3 c8" ;;
  rml2)   QUEUE="c6 c7" ;;                   # full CPU (sweeping first — Ben)
  laptop) QUEUE="c1" ;;
  *) echo "unknown machine '$M' (mac|rml3|naomio|rml2|laptop)"; exit 2 ;;
esac
echo "== GRID-5 queue on '$M': $QUEUE  (nice=$NICE shards=${NSH:-auto})"

for CFG in $QUEUE; do
  CSV="results/gait_sweep/sweep_grid5_${CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
  if [ -f "$CSV.done" ]; then
    echo "== $CFG already complete (.done), skipping"; continue
  fi
  echo "== $(date '+%F %T') starting $CFG"
  rm -f results/gait_sweep/WATCHDOG_OFF
  SWEEP_NICE=$NICE bash grid5/sweep_watchdog.sh install "$CFG" $NSH >/dev/null 2>&1 || true
  nice -n "$NICE" bash grid5/run_sweep.sh "$CFG" $NSH
  while [ ! -f "$CSV.done" ]; do
    sleep 300
    LIVE=$(pgrep -f 'grid5_sweep[.]py' | wc -l | tr -d ' ')
    if [ "$LIVE" -lt 1 ] && [ ! -f "$CSV.done" ]; then
      echo "== $(date '+%F %T') $CFG: no live shards, reviving"
      nice -n "$NICE" bash grid5/run_sweep.sh "$CFG" $NSH
    fi
  done
  ROWS=$(( $(wc -l < "$CSV") - 1 ))
  echo "== $(date '+%F %T') $CFG COMPLETE ($ROWS rows). Ship: gzip -kf $CSV && git add -f $CSV.gz ${CSV%.csv}.manifest.json"
done
echo "== $(date '+%F %T') queue '$M' DONE"
