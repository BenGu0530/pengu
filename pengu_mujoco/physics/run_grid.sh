#!/usr/bin/env bash
# Auto-resume launcher for the SHARDED fine3c low-freq gait grid (physics/gait_sweep.py).
# ~3.97M cells split across N_SHARDS workers (global_index % N_SHARDS == SHARD_ID,
# disjoint -> no duplicate work). Resume-safe: each shard skips done cells; per-shard
# sentinel <csv>.shard<i>of<N>.done; master <csv>.done written by the last shard.
# Safe to call from @reboot AND a periodic cron watchdog (per-shard pidfile guards).
set -u
cd /home/ben/Documents/cmu/pengu/pengu_mujoco || exit 1

# Serialize launches: prevent the @reboot run and the */10 cron watchdog (or a
# manual run) from racing and double-launching the same shard. Re-exec under flock.
LOCK=/tmp/pengu_fine3c.lock
if [ "${_FINE3C_LOCKED:-}" != "1" ]; then
    exec env _FINE3C_LOCKED=1 flock -n "$LOCK" "$0" "$@" || exit 0
fi

export PENGU_MODEL=v3
PY=/home/ben/miniconda3/envs/mujoco/bin/python
N_SHARDS=16
CSV=results/gait_sweep/sweep_v3_p25_fine3c_freq_hip_phi_leg_amp_hip_amp_torso_amp_torso_phi.csv
LOG=results/fine3c_autoresume.log
mkdir -p results results/gait_sweep

# whole sweep already complete?
[ -f "$CSV.done" ] && exit 0

# create the CSV header exactly once so worker shards never race it
[ -f "$CSV" ] || "$PY" physics/gait_sweep.py initcsv >> "$LOG" 2>&1

alive() {  # $1 = pidfile; true if it points at a live gait_sweep worker
    local p; p=$(cat "$1" 2>/dev/null) || return 1
    [ -n "$p" ] && [ -r "/proc/$p/cmdline" ] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q gait_sweep.py
}

for s in $(seq 0 $((N_SHARDS-1))); do
    PIDF="results/gait_sweep/.fine3c_shard${s}.pid"
    alive "$PIDF" && continue                                   # still running
    [ -f "${CSV}.shard${s}of${N_SHARDS}.done" ] && continue     # already finished
    SHARD_ID=$s N_SHARDS=$N_SHARDS nohup "$PY" physics/gait_sweep.py >> "$LOG" 2>&1 &
    echo $! > "$PIDF"
    echo "$(date '+%F %T') launched shard $s/$N_SHARDS pid $!" >> "$LOG"
    sleep 1
done
