#!/usr/bin/env bash
# One-shot c2 continuation (machine F / rml3, or any box). Does EVERYTHING:
#   repo cd -> git pull -> unpack newest committed c2 snapshot -> header check
#   -> resume sweep (skips all rows already present) -> arm watchdog.
#
#   bash physics/resume_c2.sh          # run from anywhere, any number of times
#
# Safe to re-run: resume is by axis-tuple; re-running just revives dead shards.
# Randomization is seeded per (cell,mu,rep), so continuing another machine's
# rows here produces exactly what that machine would have produced.
set -u
cd "$(dirname "$0")/.."                                   # repo root pengu_mujoco/
echo "== repo: $PWD"

echo "== pulling latest =="
git pull --ff-only || { echo "git pull failed — resolve manually"; exit 1; }

CSV=results/gait_sweep/sweep_grid4_c2_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
mkdir -p results/gait_sweep

# use whichever has MORE rows: local live CSV vs committed snapshot
snap_rows=0; loc_rows=0
[ -f "$CSV.gz" ] && snap_rows=$(( $(gunzip -c "$CSV.gz" | wc -l) - 1 ))
[ -f "$CSV" ]    && loc_rows=$(( $(wc -l < "$CSV") - 1 ))
echo "== c2 rows: committed snapshot=$snap_rows  local=$loc_rows"
if [ "$snap_rows" -gt "$loc_rows" ]; then
  gunzip -c "$CSV.gz" > "$CSV"
  echo "== unpacked snapshot -> live CSV ($snap_rows rows)"
else
  echo "== keeping local CSV ($loc_rows rows)"
fi

# header is load-bearing for resume — refuse to continue without it
if ! head -1 "$CSV" | grep -q '^freq,'; then
  echo "FATAL: $CSV has no header — repair per docs/grid4_fleet_memo.md pre-flight 4"
  exit 1
fi

rm -f results/gait_sweep/WATCHDOG_OFF                     # re-enable watchdog (c6 ship-back set it)

echo "== launching c2 (resumes automatically) =="
bash physics/run_sweep.sh c2

echo "== arming watchdog =="
bash physics/sweep_watchdog.sh install c2

echo
echo "== done. watch progress with:"
echo "   wc -l $CSV        # target 1,818,001 incl header"
echo "== when it reaches the target, ship back:"
echo "   touch results/gait_sweep/WATCHDOG_OFF"
echo "   awk 'NF' $CSV > t && mv t $CSV && gzip -kf $CSV"
echo "   git add -f $CSV.gz && git pull --rebase && git commit -m 'GRID-4 c2 complete' && git push"
