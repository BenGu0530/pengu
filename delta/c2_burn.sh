#!/usr/bin/env bash
# Layer 2 of the watchdog, "burn" variant: work through C2's slices back-to-back
# inside ONE job until the job's own time limit is nearly up, then stop cleanly.
#
#   bash delta/c2_burn.sh [n_shards]
#
# WHY ONE LONG JOB INSTEAD OF ONE JOB PER SLICE
# ---------------------------------------------
# Delta charges core-hours, so the allocation -- not wall clock -- is what runs
# out. `beht-delta-cpu` had 431 core-hours left on 2026-09-01, which at PSC's
# measured ~152 core-hours/slice is under three of C2's thirty-six slices. The
# balance will be gone long before any queue limit is, so the campaign is not
# "36 jobs" but "one job that runs until the money stops".
#
# Chaining inside a single allocation buys three things over one-job-per-slice:
#   - no queue wait between slices (each wait is dead time you paid to schedule)
#   - the job's --time IS the budget cap: at N cores, T hours costs exactly N*T
#     core-hours, so sizing --time to the remaining balance spends it precisely
#     instead of leaving a slice-sized remainder unspendable
#   - the run ends on a partial slice rather than refusing to start one, and a
#     partial slice is not waste: rows are flushed one at a time and resume is by
#     axis-tuple, so the next allocation continues exactly where this stopped
#
# WHAT IT COSTS TO BE STOPPED MID-SLICE
# -------------------------------------
# At most one row per shard (the one in flight). That is why burning to zero is
# safe here and would not be for a workload that only produces output at the end.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
STATE="$HERE/state"
RESULTS="$REPO/pengu_mujoco/results/gait_sweep"

N="${1:-${SLURM_CPUS_ON_NODE:-8}}"

PHI_FIRST=0
PHI_LAST=350
PHI_STEP=10
ROWS_TARGET=115200
MARGIN=600              # seconds of headroom before the job's limit: shard
                        # shutdown + gzip of an ~18MB CSV, with room to spare

PY="${DELTA_PY:-${DELTA_WORK:-/projects/beht/bgu}/mjx_venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

mkdir -p "$STATE" "$RESULTS"
BURN_LOG="$STATE/burn.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$BURN_LOG"; }

[ -f "$REPO/DELTA_ISOLATED_TREE" ] || { log "ERROR: $REPO is not the isolated tree"; exit 1; }

# ---- when must we stop? ---------------------------------------------------
# Ask SLURM rather than trusting a number passed in: --time can be overridden at
# submit (c2_ctl.sh burn sizes it from the live balance), and a requeue restarts
# the clock. If SLURM will not say, run without a deadline and let the outer
# time limit kill us -- degraded, but not wrong.
job_end_epoch() {
    if [ -n "${SLURM_JOB_END_TIME:-}" ] && [ "${SLURM_JOB_END_TIME}" -gt 0 ] 2>/dev/null; then
        echo "$SLURM_JOB_END_TIME"; return
    fi
    local e
    e=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null \
        | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
    if [ -n "$e" ]; then date -d "$e" +%s 2>/dev/null && return; fi
    echo 0
}

END_EPOCH=$(job_end_epoch)
if [ "${END_EPOCH:-0}" -gt 0 ]; then
    DEADLINE=$(( END_EPOCH - MARGIN ))
else
    DEADLINE=0
fi
export C2_DEADLINE="$DEADLINE"

T_JOB_START=$(date +%s)
log "=== BURN start  job=${SLURM_JOB_ID:-none}  node=$(hostname)  shards=$N ==="
if [ "$DEADLINE" -gt 0 ]; then
    log "job ends $(date -d "@$END_EPOCH" '+%F %T'), stopping at $(date -d "@$DEADLINE" '+%F %T') (${MARGIN}s margin)"
    log "budget implied by --time: $(awk -v n="$N" -v s="$((DEADLINE - T_JOB_START))" 'BEGIN{printf "%.1f", n*s/3600}') core-hours"
else
    log "WARNING: could not read the job end time -- running with no deadline."
    log "  The slice in flight will be killed at the time limit instead of stopping"
    log "  cleanly. Its rows are still on disk and resumable; only the packaging"
    log "  step is lost."
fi

rows_of() { "$PY" "$HERE/c2_driver.py" "$1" --rows 2>/dev/null || echo 0; }

SLICES_DONE=0
SLICES_PARTIAL=0
RC_FINAL=0

for PHI in $(seq $PHI_FIRST $PHI_STEP $PHI_LAST); do
    R=$(rows_of "$PHI")
    if [ "$R" -ge "$ROWS_TARGET" ]; then
        log "phi=$PHI already complete ($R rows) -- skipping"
        continue
    fi

    if [ "$DEADLINE" -gt 0 ] && [ "$(date +%s)" -ge "$DEADLINE" ]; then
        log "deadline reached before phi=$PHI -- stopping"
        break
    fi

    log "--- slice phi=$PHI (from $R rows) ---"
    T0=$(date +%s)
    bash "$HERE/c2_node.sh" "$PHI" "$N"
    RC=$?
    T1=$(date +%s)

    ELAPSED=$(( T1 - T0 ))
    CORE_H=$(awk -v c="$N" -v e="$ELAPSED" 'BEGIN{printf "%.1f", c*e/3600.0}')
    NODE_H=$(awk -v n="${SLURM_JOB_NUM_NODES:-1}" -v e="$ELAPSED" 'BEGIN{printf "%.3f", n*e/3600.0}')
    R_END=$(rows_of "$PHI")

    # Same columns as c2_phi.slurm writes, so c2_ctl.sh reads either path.
    echo "${SLURM_JOB_ID:-local} ${PHI} ${N} ${SLURM_JOB_NUM_NODES:-1} ${ELAPSED} ${CORE_H} ${NODE_H} $(date -Iseconds) rc=${RC} rows=${R_END}" \
        >> "$STATE/costs.txt"

    case "$RC" in
      0)
        SLICES_DONE=$(( SLICES_DONE + 1 ))
        log "phi=$PHI COMPLETE: $R_END rows, ${ELAPSED}s, ${CORE_H} core-hours"
        CSV=$(ls "$RESULTS"/sweep_grid5_c2_phi$(printf '%03d' "$PHI")*.csv 2>/dev/null | head -1)
        [ -n "$CSV" ] && gzip -kf "$CSV" && log "  packaged $(basename "$CSV").gz ($(stat -c %s "$CSV.gz" 2>/dev/null) bytes)"
        ;;
      3)
        SLICES_PARTIAL=$(( SLICES_PARTIAL + 1 ))
        log "phi=$PHI STOPPED ON DEADLINE at $R_END/$ROWS_TARGET rows, ${CORE_H} core-hours"
        log "  This is the intended way to run out of allocation. The slice resumes"
        log "  from $R_END rows whenever there is budget again."
        # Package the PARTIAL slice too. c2_ctl.sh fetch pulls only *.csv.gz, so
        # skipping this silently strands the rows the run ended on -- which for a
        # job designed to stop mid-slice is guaranteed to happen every time.
        CSV=$(ls "$RESULTS"/sweep_grid5_c2_phi$(printf '%03d' "$PHI")*.csv 2>/dev/null | head -1)
        [ -n "$CSV" ] && gzip -kf "$CSV" && log "  packaged partial $(basename "$CSV").gz ($(stat -c %s "$CSV.gz" 2>/dev/null) bytes)"
        break
        ;;
      *)
        # A real failure. Do NOT move to the next slice: whatever is broken would
        # be broken there too, and the job would spend the rest of the allocation
        # discovering that repeatedly.
        RC_FINAL=$RC
        log "phi=$PHI FAILED (rc=$RC) after ${CORE_H} core-hours -- NOT continuing."
        log "  Read $RESULTS/c2_phi$(printf '%03d' "$PHI")_run.log before resubmitting."
        break
        ;;
    esac
done

T_JOB_END=$(date +%s)
JOB_ELAPSED=$(( T_JOB_END - T_JOB_START ))
JOB_CORE_H=$(awk -v c="$N" -v e="$JOB_ELAPSED" 'BEGIN{printf "%.1f", c*e/3600.0}')

log "=== BURN done: $SLICES_DONE complete, $SLICES_PARTIAL partial, rc=$RC_FINAL ==="
log "job wall clock ${JOB_ELAPSED}s on $N cores = ${JOB_CORE_H} core-hours"
log "reconcile: sacct -X -j ${SLURM_JOB_ID:-?} -o JobID,AllocCPUS,ElapsedRaw,Elapsed,State"
log "then:      accounts     (lags; compare against the number above)"

exit "$RC_FINAL"
