#!/usr/bin/env bash
# Layer 1 of the watchdog: run one hip_phi slice across N shards inside a SLURM
# job, and restart any shard that dies before the slice is complete.
#
#   bash delta/c2_node.sh <phi> [n_shards]
#
# Shards coordinate exactly the way the existing sweep does: N_SHARDS/SHARD_ID
# env vars (grid5_sweep.py:230-231), every shard appending to one CSV with
# f.flush() after each row (grid5_sweep.py:298), resume by axis-tuple via
# gs._load_done(). So a killed shard costs at most the row it was mid-way through.
#
# Default shard count is ALL cores on the node. Unlike the laptop fleet
# (run_sweep.sh uses cores-2 to leave the machine usable) a compute node has
# nothing else to do, and the job is billed for everything it holds whether it is
# used or not -- leaving cores idle is burning allocation for nothing. This is true
# under either charge unit: if Delta bills node-hours we hold the whole node
# regardless, and if it bills core-hours we asked for all 128 anyway.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
PHI="${1:?usage: c2_node.sh <phi> [n_shards]}"
N="${2:-${SLURM_CPUS_ON_NODE:-8}}"

PY="${DELTA_PY:-${DELTA_WORK:-/projects/beht/bgu}/mjx_venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

DRIVER="$HERE/c2_driver.py"
PHI3=$(printf "%03d" "$PHI")
LOG_DIR="$REPO/pengu_mujoco/results/gait_sweep"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/c2_phi${PHI3}_run.log"

ROWS_TARGET=115200
POLL=60                 # seconds between liveness checks
MAX_RESTARTS=5          # per shard; a shard failing repeatedly is a real bug

# Optional wall-clock deadline, epoch seconds. c2_burn.sh sets it so a slice stops
# CLEANLY shortly before the job's time limit rather than being killed mid-flight.
# Why it matters: a job killed at its limit is charged in full but never runs its
# packaging step, and the outer loop never records what the slice cost. Stopping
# early is nearly free -- rows are flushed one at a time (grid5_sweep.py:298) and
# resume is by axis-tuple via gs._load_done(), so at most the row in progress is
# lost and the next job picks the slice up where this one left off.
#
# Exit codes: 0 = slice complete, 1 = broken, 3 = stopped on deadline (partial,
# resumable, NOT an error).
DEADLINE="${C2_DEADLINE:-0}"
past_deadline() { [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== slice phi=$PHI  shards=$N  job=${SLURM_JOB_ID:-none}  node=$(hostname) ==="
log "python: $PY"
"$PY" -c "import mujoco;print('mujoco', mujoco.__version__)" 2>&1 | tee -a "$LOG"

# Header + manifest must exist before any shard appends (check_manifest refuses
# to write without it). Idempotent: skips if the CSV is already there.
"$PY" "$DRIVER" "$PHI" initcsv >>"$LOG" 2>&1 || { log "initcsv FAILED"; exit 1; }

rows_now() { "$PY" "$DRIVER" "$PHI" --rows 2>/dev/null || echo 0; }
START_ROWS=$(rows_now)
log "resuming from $START_ROWS / $ROWS_TARGET rows"
if [ "$START_ROWS" -ge "$ROWS_TARGET" ]; then
    log "slice already complete -- nothing to do"
    exit 0
fi
if past_deadline; then
    log "deadline already passed -- not launching shards for phi=$PHI"
    exit 3
fi

declare -a PIDS RESTARTS SHARD_STATE
launch() {   # launch <shard_id>
    local s=$1
    N_SHARDS=$N SHARD_ID=$s nohup "$PY" -u "$DRIVER" "$PHI" >>"$LOG" 2>&1 &
    PIDS[$s]=$!
    SHARD_STATE[$s]=running
}

for s in $(seq 0 $((N - 1))); do
    RESTARTS[$s]=0
    launch "$s"
done
log "launched $N shards"

T0=$(date +%s)
while :; do
    sleep "$POLL"

    ROWS=$(rows_now)
    if [ "$ROWS" -ge "$ROWS_TARGET" ]; then
        log "slice COMPLETE at $ROWS rows"
        break
    fi

    if past_deadline; then
        log "DEADLINE reached at $ROWS/$ROWS_TARGET rows -- stopping shards cleanly"
        for s in $(seq 0 $((N - 1))); do kill "${PIDS[$s]}" 2>/dev/null; done
        sleep 5
        for s in $(seq 0 $((N - 1))); do kill -9 "${PIDS[$s]}" 2>/dev/null; done
        wait 2>/dev/null || true
        log "=== slice phi=$PHI stopped on deadline: $(rows_now) rows (resumable) ==="
        exit 3
    fi

    alive=0; finished=0; failed=0
    for s in $(seq 0 $((N - 1))); do
        case "${SHARD_STATE[$s]}" in
            done|failed) [ "${SHARD_STATE[$s]}" = done ] && finished=$((finished+1)) \
                                                        || failed=$((failed+1)); continue ;;
        esac

        if kill -0 "${PIDS[$s]}" 2>/dev/null; then
            alive=$((alive + 1)); continue
        fi

        # The shard is gone -- but WHY? A shard that has processed every cell in
        # its stripe exits 0, which is success, not a crash. Restarting those
        # (the original bug) burns the restart budget at the end of every slice
        # and can exhaust it before a genuine crash needs it. `wait` on a dead
        # child returns its status immediately; each pid is waited exactly once
        # because the state machine never revisits a done/failed shard.
        wait "${PIDS[$s]}" 2>/dev/null; rc=$?
        if [ "$rc" -eq 0 ]; then
            SHARD_STATE[$s]=done
            finished=$((finished + 1))
        elif [ "${RESTARTS[$s]}" -lt "$MAX_RESTARTS" ]; then
            RESTARTS[$s]=$(( RESTARTS[s] + 1 ))
            log "shard $s CRASHED (rc=$rc) -> restart ${RESTARTS[$s]}/$MAX_RESTARTS"
            launch "$s"
            alive=$((alive + 1))
        else
            SHARD_STATE[$s]=failed
            failed=$((failed + 1))
            log "shard $s CRASHED (rc=$rc) and is out of restarts -- giving up on it"
        fi
    done

    ELAPSED=$(( $(date +%s) - T0 ))
    RATE=0
    [ "$ELAPSED" -gt 0 ] && RATE=$(( (ROWS - START_ROWS) * 3600 / ELAPSED ))
    log "rows=$ROWS/$ROWS_TARGET  running=$alive done=$finished failed=$failed  ${RATE} rows/hr"

    if [ "$alive" -eq 0 ]; then
        if [ "$failed" -eq 0 ]; then
            # Every shard exited cleanly. Either the slice is finished (the row
            # check at the top of the next pass will catch it) or the stripes do
            # not cover the grid -- a real bug worth saying out loud.
            log "all $N shards exited cleanly at $ROWS/$ROWS_TARGET rows"
            [ "$ROWS" -ge "$ROWS_TARGET" ] && { log "slice COMPLETE at $ROWS rows"; break; }
            log "  BUT the slice is short by $(( ROWS_TARGET - ROWS )) rows."
            log "  Shards believe they are done while cells are missing -- this is a"
            log "  sharding bug, not a crash. Do NOT just resubmit; investigate."
            exit 1
        fi
        log "no shards left: $finished finished cleanly, $failed exhausted their"
        log "  $MAX_RESTARTS restarts, slice at $ROWS/$ROWS_TARGET -- giving up"
        exit 1
    fi
done

wait 2>/dev/null || true
FINAL=$(rows_now)
log "=== slice phi=$PHI finished: $FINAL rows ==="
[ "$FINAL" -ge "$ROWS_TARGET" ] || exit 1
