#!/usr/bin/env bash
# Guard for fine3: a stray narrow-config sweep (subset of cells) is running
# alongside the authoritative full-phase sweep and would write a PREMATURE .done
# sentinel, tricking run_grid.sh into not resuming the full sweep after a reboot.
# While the full-sweep PID is alive we keep deleting any .done; once it exits we
# only (re)create .done if the CSV actually covers all cells. No signals used
# (kill is blocked cross-sandbox), liveness via /proc.
set -u
cd /home/ben/Documents/cmu/pengu/pengu_mujoco || exit 1
FULL_PID="${1:?usage: fine3_guard.sh <full_sweep_pid>}"
CSV=results/gait_sweep/sweep_v3_p25_fine3_freq_hip_phi_leg_amp_hip_amp_torso_amp_torso_phi.csv
DONE="$CSV.done"
NCELLS=26928
while [ -d "/proc/$FULL_PID" ]; do
    [ -f "$DONE" ] && rm -f "$DONE"
    sleep 5
done
# full sweep exited: allow .done only if coverage is complete (unique 6-tuples)
UNIQ=$(tail -n +2 "$CSV" 2>/dev/null | cut -d, -f1-6 | sort -u | wc -l)
if [ "${UNIQ:-0}" -ge "$NCELLS" ]; then
    touch "$DONE"
    echo "$(date '+%F %T') fine3 complete: uniq=$UNIQ -> restored .done"
else
    echo "$(date '+%F %T') fine3 full pid gone but uniq=$UNIQ < $NCELLS; left .done removed for resume"
fi
