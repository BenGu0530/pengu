#!/usr/bin/env bash
# Generic one-shot GRID-4 continuation: pull -> restore committed snapshot -> resume
# -> watchdog. Works for any config on any machine, any number of times.
#
#   bash physics/resume_config.sh c1                # plain
#   SWEEP_NICE=19 bash physics/resume_config.sh c1  # shared box (children inherit nice)
#
# Resume is by axis-tuple; seeded randomization means continuing another machine's
# rows here produces exactly the rows it would have produced.
set -u
CFG="${1:?usage: resume_config.sh cN}"
cd "$(dirname "$0")/.."
echo "== repo: $PWD  config: $CFG"

echo "== pulling latest =="
git pull --ff-only || { echo "git pull failed — resolve manually"; exit 1; }

CSV="results/gait_sweep/sweep_grid4_${CFG}_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv"
mkdir -p results/gait_sweep
# tracked snapshot: force it back to the COMMITTED version (index tricks can't smuggle rows)
git checkout HEAD -- "$CSV.gz" 2>/dev/null || true

snap_rows=0; loc_rows=0
[ -f "$CSV.gz" ] && snap_rows=$(( $(gunzip -c "$CSV.gz" | wc -l) - 1 ))
[ -f "$CSV" ]    && loc_rows=$(( $(wc -l < "$CSV") - 1 ))
echo "== $CFG rows: committed snapshot=$snap_rows  local=$loc_rows"
if [ "$snap_rows" -gt "$loc_rows" ]; then
  gunzip -c "$CSV.gz" > "$CSV"; echo "== unpacked snapshot -> live CSV ($snap_rows rows)"
else
  echo "== keeping local CSV ($loc_rows rows)"
fi
if ! head -1 "$CSV" | grep -q '^freq,'; then
  echo "FATAL: $CSV has no header — repair per docs/grid4_fleet_memo.md pre-flight 4"; exit 1
fi

rm -f results/gait_sweep/WATCHDOG_OFF
echo "== launching $CFG (nice=${SWEEP_NICE:-0}) =="
nice -n "${SWEEP_NICE:-0}" bash physics/run_sweep.sh "$CFG"
echo "== arming watchdog =="
SWEEP_NICE="${SWEEP_NICE:-0}" bash physics/sweep_watchdog.sh install "$CFG"

echo
echo "== watch:  wc -l $CSV      # target 1,818,001 incl header"
echo "== ship-back when full:"
echo "   touch results/gait_sweep/WATCHDOG_OFF"
echo "   awk 'NF' $CSV > t && mv t $CSV && gzip -kf $CSV"
echo "   git add -f $CSV.gz && git pull --rebase && git commit -m 'GRID-4 $CFG complete' && git push"
