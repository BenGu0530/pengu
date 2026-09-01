#!/usr/bin/env bash
# Layer 3 of the watchdog: campaign controller with a BUDGET GUARD.
#
#   bash delta/c2_ctl.sh burn          size a job from the LIVE balance (prints only)
#   bash delta/c2_ctl.sh burn --go     ...and submit it: run until the account is dry
#   bash delta/c2_ctl.sh start [phi]   submit ONE slice at a time instead (default phi=0)
#   bash delta/c2_ctl.sh status        what is done, what it cost, what is left
#   bash delta/c2_ctl.sh next          (called by a finishing job) submit the next slice
#   bash delta/c2_ctl.sh stop          stop after the current job; submit nothing more
#   bash delta/c2_ctl.sh resume        clear the stop flag and continue
#   bash delta/c2_ctl.sh spent         allocation consumed by this campaign, from sacct
#   bash delta/c2_ctl.sh unit          evidence for which unit `accounts` reports
#   bash delta/c2_ctl.sh fetch         print the rsync command to run on the Mac
#
# Why a guard at all
# ------------------
# `beht-delta-cpu` had 431 of 1142 Hours left on 2026-09-01 and is SHARED WITH
# NINE OTHER PEOPLE. Delta charges 1 SU per core-hour -- a node-exclusive hour is
# 128 SU (NCSA Delta user guide, Job Accounting) -- so at PSC's measured ~152
# core-hours per slice the balance buys under three of C2's thirty-six slices.
# An unattended chain would drain the whole project account in three jobs.
#
# The guard therefore starts at a cap that permits ONE slice. That is not
# pessimism, it is the honest size of the budget: going further is a decision to
# spend other people's allocation, and it should require typing a new number into
# state/budget.txt rather than happening while nobody is watching.
#
# Accounting is from sacct (AllocCPUS x ElapsedRaw), NOT from `accounts`, which
# lags and would stall every job transition. That makes the guard's arithmetic an
# INDEPENDENT estimate of the real balance -- reconcile it with `accounts` after
# the first job (see `status` and `unit`).

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
STATE="$HERE/state"
JOBS="$STATE/jobs.txt"              # "<jobid> <phi>" per line
BUDGET_F="$STATE/budget.txt"        # cap for the whole campaign, in $UNIT
EST_F="$STATE/est_per_slice.txt"    # updated from observed cost
UNIT_F="$STATE/unit.txt"            # core-hour | node-hour
RESERVE_F="$STATE/reserve.txt"      # core-hours to leave unspent
CORES_F="$STATE/cores.txt"          # cores per burn job
STOP="$STATE/STOP"
LOG="$STATE/ctl.log"

PHI_FIRST=0
PHI_LAST=350
PHI_STEP=10
N_SLICES=36
ROWS_TARGET=115200
ACCOUNT=beht-delta-cpu

# One slice, and no more, without a deliberate decision. See the header.
DEFAULT_BUDGET=160
# PSC's measured C1 cost (152.6 and 159.1 core-hours). An UPPER bound for C2:
# com=1.20 falls more than com=1.05, and EPYC 7763 beats 7742. Calibration replaces it.
DEFAULT_EST=156
# Documented: NCSA Delta user guide, Job Accounting -- "a node-exclusive job that
# runs on a compute node for one hour will be charged 128 SUs (128 cores x 1 hour)".
# `bash c2_ctl.sh unit` re-checks this against the balance actually observed.
DEFAULT_UNIT=core-hour

# Core-hours deliberately left unspent by `burn`. Zero by decision: the account is
# the lab's, the work is worth more finished than the balance is worth held, and a
# partial slice is fully resumable. Set it to 3-5 if you want to keep the ability
# to run one more calibration job after the balance is gone -- PSC's memo names
# "no headroom left to diagnose" as a real cost.
DEFAULT_RESERVE=0
# Cores per burn job. NOT 128, and this is a judgement call worth reading:
# allocation is charged per core-hour, so what matters is rows PER CORE-HOUR, not
# rows per second. PSC measured 904 rows/hr/core on 8 cores and 754 on 128 -- the
# full node loses ~17% to memory bandwidth and shared-filesystem contention. With
# wall clock nowhere near binding (431 core-hours is 13h at 32 cores, against a
# 2-day partition limit) a narrower job buys more data for the same money.
# 32 sits between the two measured points. Raise it if the queue makes you wait,
# lower it if calibration shows the per-core rate still climbing.
DEFAULT_CORES=32
PARTITION_MAX_HOURS=48

mkdir -p "$STATE"
[ -f "$JOBS" ]     || : > "$JOBS"
[ -f "$BUDGET_F" ] || echo "$DEFAULT_BUDGET" > "$BUDGET_F"
[ -f "$EST_F" ]    || echo "$DEFAULT_EST"    > "$EST_F"
[ -f "$UNIT_F" ]   || echo "$DEFAULT_UNIT"   > "$UNIT_F"
[ -f "$RESERVE_F" ] || echo "$DEFAULT_RESERVE" > "$RESERVE_F"
[ -f "$CORES_F" ]   || echo "$DEFAULT_CORES"   > "$CORES_F"

UNIT=$(tr -d '[:space:]' < "$UNIT_F")

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
die() { log "ERROR: $*"; exit 1; }

PY="${DELTA_PY:-${DELTA_WORK:-/projects/beht/bgu}/mjx_venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3)"

rows_of() { "$PY" "$HERE/c2_driver.py" "$1" --rows 2>/dev/null || echo 0; }

# ---- allocation spent -----------------------------------------------------
# Two independent sources, and we take the LARGER. A budget guard that
# under-counts overspends; one that over-counts merely stops early.
#
#   sacct     authoritative, but slurmdbd records are PURGED eventually --
#             after which this silently returns 0 for old jobs.
#   costs.txt written by each job at exit (c2_phi.slurm), so it outlives the
#             sacct record. Slightly under-counts its own job (written before
#             the job actually exits).
#
# Which TRES to multiply by depends on the unit: core-hours bill AllocCPUS,
# node-hours bill AllocNodes.
spent_from_sacct() {
    local ids col
    ids=$(awk 'NF{printf "%s,", $1}' "$JOBS" | sed 's/,$//')
    if [ -z "$ids" ]; then echo 0; return; fi
    [ "$UNIT" = "node-hour" ] && col=2 || col=1
    sacct -X -n -P -j "$ids" -o AllocCPUS,AllocNodes,ElapsedRaw 2>/dev/null \
      | awk -F'|' -v c="$col" '$c ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { s += $c * $3 / 3600 }
                               END { printf "%.0f", s + 0 }'
}

# costs.txt columns: jobid phi cores nodes elapsed core_hours node_hours ts rc
spent_from_file() {
    [ -f "$STATE/costs.txt" ] || { echo 0; return; }
    local col
    [ "$UNIT" = "node-hour" ] && col=7 || col=6
    awk -v c="$col" '{ s += $c } END { printf "%.0f", s + 0 }' "$STATE/costs.txt"
}

spent() {
    local a b
    a=$(spent_from_sacct); b=$(spent_from_file)
    [ "$a" -ge "$b" ] && echo "$a" || echo "$b"
}

completed_slices() {
    local n=0 phi
    for phi in $(seq $PHI_FIRST $PHI_STEP $PHI_LAST); do
        [ "$(rows_of "$phi")" -ge "$ROWS_TARGET" ] && n=$((n + 1))
    done
    echo "$n"
}

next_phi() {    # first slice that is not complete; empty if all done
    local phi
    for phi in $(seq $PHI_FIRST $PHI_STEP $PHI_LAST); do
        if [ "$(rows_of "$phi")" -lt "$ROWS_TARGET" ]; then echo "$phi"; return; fi
    done
    echo ""
}

# Refine the per-slice estimate from what slices actually cost, so the guard gets
# more accurate as the campaign runs instead of trusting PSC's C1 number forever.
refresh_estimate() {
    local done_n su
    done_n=$(completed_slices)
    [ "$done_n" -lt 1 ] && return
    su=$(spent)
    [ "$su" -lt 1 ] && return
    awk -v s="$su" -v n="$done_n" 'BEGIN{ printf "%.0f", s / n }' > "$EST_F"
}

submit() {      # submit <phi>
    local phi=$1 out jid
    # The #SBATCH --output path in c2_phi.slurm is relative, and sbatch resolves
    # it against the submitting CWD -- which differs between "Ben on a login node"
    # and "a finishing job calling next". Pin it.
    mkdir -p "$STATE/logs"
    cd "$REPO" || die "cannot cd to $REPO"
    out=$(sbatch --parsable "$HERE/c2_phi.slurm" "$phi" 2>&1) || {
        log "sbatch FAILED for phi=$phi: $out"
        log "  If this ran from a compute node and sbatch is not permitted there,"
        log "  run 'bash delta/c2_ctl.sh next' from a login node instead."
        log "  If the reason is QOSGrpBillingMinutes, the account is out of allocation."
        return 1
    }
    jid="${out%%;*}"
    echo "$jid $phi" >> "$JOBS"
    log "submitted phi=$phi as job $jid"
}

cmd_status() {
    local budget sp est done_n nxt remain can
    budget=$(cat "$BUDGET_F"); sp=$(spent); est=$(cat "$EST_F")
    done_n=$(completed_slices); nxt=$(next_phi)
    remain=$(( budget - sp ))
    can=$(awk -v r="$remain" -v e="$est" 'BEGIN{ printf "%d", (e>0? int(r/e) : 0) }')

    echo "=== C2 campaign status (NCSA Delta, $ACCOUNT) ==="
    echo "  slices complete : $done_n / $N_SLICES   (phi $PHI_FIRST..$PHI_LAST step $PHI_STEP)"
    echo "  next slice      : ${nxt:-<none, all done>}"
    echo "  jobs submitted  : $(awk 'NF' "$JOBS" | wc -l | tr -d ' ')"
    echo
    echo "  charge unit     : $UNIT   ($UNIT_F)"
    echo "  budget cap      : $budget $UNIT   ($BUDGET_F)"
    echo "  spent           : $sp $UNIT   (sacct $(spent_from_sacct) / recorded $(spent_from_file), larger wins)"
    echo "  remaining       : $remain $UNIT"
    echo "  est per slice   : $est $UNIT   -> room for ~$can more slice(s)"
    [ -f "$STOP" ] && echo "  STOP flag       : SET (nothing will be submitted)"
    echo
    echo "  This account is shared with nine other people. 'remaining' above is"
    echo "  the guard's cap, NOT the project balance -- run 'accounts' for that,"
    echo "  and 'bash delta/c2_ctl.sh unit' to check the two agree."
    echo
    printf "  %-6s %-10s %s\n" "phi" "rows" "state"
    local phi r
    for phi in $(seq $PHI_FIRST $PHI_STEP $PHI_LAST); do
        r=$(rows_of "$phi")
        if   [ "$r" -ge "$ROWS_TARGET" ]; then printf "  %-6s %-10s %s\n" "$phi" "$r" "done"
        elif [ "$r" -gt 0 ];              then printf "  %-6s %-10s %s\n" "$phi" "$r" "PARTIAL"
        fi
    done
}

cmd_unit() {
    # The one number that decides whether this campaign is affordable at all.
    # Documented answer and observed answer, side by side -- PSC taught that the
    # documented number and the measured number can differ by 40%, and that the
    # balance command can lag by hours, so neither source is trusted alone.
    echo "=== charge unit: documented vs observed ==="
    echo
    echo "DOCUMENTED (NCSA Delta user guide, Job Accounting):"
    echo "  \"a node-exclusive job that runs on a compute node for one hour will be"
    echo "   charged 128 SUs (128 cores x 1 hour)\"   => 1 Hour = 1 CORE-hour"
    echo "  state/unit.txt currently says: $UNIT"
    echo
    echo "BASELINE (written by delta_setup.sh before anything was charged):"
    if [ -f "$STATE/balance_at_setup.txt" ]; then
        sed 's/^/  /' "$STATE/balance_at_setup.txt"
    else
        echo "  (none -- delta_setup.sh had not run, or `accounts` was unavailable)"
    fi
    echo
    echo "NOW:"
    accounts 2>&1 | sed 's/^/  /'
    echo
    echo "THIS CAMPAIGN'S JOBS, both readings (sacct does not lag):"
    local ids
    ids=$(awk 'NF{printf "%s,", $1}' "$JOBS" | sed 's/,$//')
    if [ -z "$ids" ]; then
        echo "  (no jobs recorded in state/jobs.txt yet -- calibration jobs are not"
        echo "   recorded there; add its id by hand if you want it counted)"
    else
        sacct -X -n -P -j "$ids" -o JobID,AllocCPUS,AllocNodes,ElapsedRaw,State \
          | awk -F'|' '{ ch += $2*$4/3600; nh += $3*$4/3600;
                         printf "  job %-12s %4s cores %2s nodes %7ss  %s\n", $1,$2,$3,$4,$5 }
                       END { printf "\n  total: %.1f core-hours   %.3f node-hours\n", ch, nh }'
    fi
    echo
    echo "  Compare the drop from BASELINE to NOW against those two totals."
    echo "  Whichever it matches is the unit. If the balance has not moved at all,"
    echo "  wait -- NCSA's accounting can lag behind sacct by hours (on PSC the"
    echo "  equivalent command lagged 1-2h and nearly caused a 128x misreading)."
    echo "  If they disagree once the balance has settled, fix state/unit.txt"
    echo "  BEFORE submitting anything else; the guard's arithmetic depends on it."
}

# ---- burn: spend what is left, precisely ----------------------------------
# `accounts` prints one row per account: <name> <balance> <deposited> <project>.
balance_of() {  # balance_of <file-or-empty-for-live>
    local src="${1:-}"
    if [ -n "$src" ]; then cat "$src" 2>/dev/null; else accounts 2>/dev/null; fi \
      | awk -v a="$ACCOUNT" '$1 == a && $2 ~ /^[0-9]+$/ { print $2; exit }'
}

to_core_hours() {   # to_core_hours <amount-in-$UNIT>
    [ "$UNIT" = "node-hour" ] && awk -v x="$1" 'BEGIN{printf "%.0f", x*128}' || echo "$1"
}

# Two independent readings, and we take the SMALLER. Over-estimating the balance
# overspends someone else's allocation; under-estimating just leaves a little on
# the table, and a partial slice resumes for free.
#   live       `accounts` -- authoritative but LAGS (PSC's equivalent lagged 1-2h,
#              which nearly caused a 128x misreading of the charge unit)
#   derived    baseline recorded before the first job, minus what sacct says this
#              campaign has spent since. Does not lag, but only sees OUR jobs --
#              nine other people share this account and can spend it underneath us.
remaining_core_hours() {
    local live derived base spent_ch out
    live=$(balance_of); base=$(balance_of "$STATE/balance_at_setup.txt")
    [ -n "$live" ] && live=$(to_core_hours "$live")
    if [ -n "$base" ]; then
        spent_ch=$(spent_from_sacct)
        derived=$(awk -v b="$(to_core_hours "$base")" -v s="$spent_ch" 'BEGIN{printf "%.0f", b-s}')
    fi
    if   [ -n "$live" ] && [ -n "${derived:-}" ]; then
        out=$(awk -v a="$live" -v b="$derived" 'BEGIN{print (a<b? a : b)}')
    elif [ -n "$live" ];             then out="$live"
    elif [ -n "${derived:-}" ];      then out="$derived"
    else                                  out=""
    fi
    echo "$out"
}

hms() { awk -v h="$1" 'BEGIN{ t=int(h*3600); printf "%02d:%02d:00", int(t/3600), int((t%3600)/60) }'; }

cmd_burn() {
    local go="" ; [ "${1:-}" = "--go" ] && go=1
    local cores reserve rem_raw rem wall wall_capped tl live base

    cores=$(tr -d '[:space:]' < "$CORES_F")
    reserve=$(tr -d '[:space:]' < "$RESERVE_F")
    live=$(balance_of); base=$(balance_of "$STATE/balance_at_setup.txt")
    rem_raw=$(remaining_core_hours)

    echo "=== C2 burn plan ($ACCOUNT, $UNIT) ==="
    echo "  balance now (accounts, lags) : ${live:-<unavailable>} $UNIT"
    echo "  baseline at setup            : ${base:-<none recorded>} $UNIT"
    echo "  spent by this campaign(sacct): $(spent_from_sacct) $UNIT"
    if [ -z "$rem_raw" ]; then
        echo
        echo "  Cannot read a balance from either source. Run this on a login node"
        echo "  where 'accounts' exists, or run delta_setup.sh first so a baseline"
        echo "  exists. Refusing to size a job against an unknown budget."
        return 1
    fi
    rem=$(awk -v r="$rem_raw" -v v="$reserve" 'BEGIN{ x=r-v; printf "%d", (x>0? x : 0) }')
    echo "  remaining (smaller of the two): $rem_raw core-hours"
    echo "  reserve kept back             : $reserve core-hours   ($RESERVE_F)"
    echo "  spendable                     : $rem core-hours"
    echo

    if [ "$rem" -le 0 ]; then
        echo "  Nothing left to spend. Everything already produced is on disk and"
        echo "  every partial slice resumes by axis-tuple whenever budget returns."
        return 0
    fi

    # Allocation is charged per core-hour, so total rows depend on rows PER
    # CORE-HOUR, not on how fast the job finishes. Narrower jobs measured faster
    # per core on PSC; wall clock has room to spare either way.
    echo "  cores -> wall clock for the same $rem core-hours:"
    local c
    for c in 8 16 32 64 128; do
        printf "    %3s cores : %6.1f h" "$c" "$(awk -v r="$rem" -v c="$c" 'BEGIN{print r/c}')"
        awk -v r="$rem" -v c="$c" -v m="$PARTITION_MAX_HOURS" \
            'BEGIN{ printf "%s\n", (r/c > m ? "   EXCEEDS the 2-day partition limit" : "") }'
    done
    echo "    (PSC measured 904 rows/hr/core on 8 cores vs 754 on 128 -- about 17%"
    echo "     more data per core-hour on the narrower job. Set $CORES_F to change.)"
    echo

    wall=$(awk -v r="$rem" -v c="$cores" 'BEGIN{print r/c}')
    wall_capped=$(awk -v w="$wall" -v m="$PARTITION_MAX_HOURS" 'BEGIN{print (w>m? m : w)}')
    tl=$(hms "$wall_capped")

    echo "  CHOSEN: $cores cores x $tl  =  $(awk -v c="$cores" -v w="$wall_capped" 'BEGIN{printf "%.0f", c*w}') core-hours"
    awk -v w="$wall" -v m="$PARTITION_MAX_HOURS" 'BEGIN{ if (w>m)
        printf "  NOTE: capped at the partition limit -- %.0f core-hours will be left over.\n         Run burn again afterwards, or raise cores.\n", (w-m)*'"$cores"' }'
    echo "  est slices from this: ~$(awk -v c="$cores" -v w="$wall_capped" -v e="$(cat "$EST_F")" 'BEGIN{printf "%.1f", c*w/e}') (at $(cat "$EST_F") core-hours/slice)"
    echo "  next incomplete slice: $(next_phi)"
    echo
    echo "  The job works through slices back-to-back and stops cleanly ~10 min"
    echo "  before its limit. It ends mid-slice on purpose -- that is what spending"
    echo "  the balance to zero looks like, and the partial slice resumes later."
    echo

    if [ -z "$go" ]; then
        echo "  Nothing submitted. To actually spend it:  bash $0 burn --go"
        return 0
    fi

    mkdir -p "$STATE/logs"
    cd "$REPO" || die "cannot cd to $REPO"
    local out jid
    out=$(sbatch --parsable --ntasks-per-node="$cores" --time="$tl" \
          "$HERE/c2_burn.slurm" 2>&1) || {
        log "sbatch FAILED: $out"
        log "  QOSGrpBillingMinutes in the reason means the account is already out."
        return 1
    }
    jid="${out%%;*}"
    echo "$jid burn" >> "$JOBS"
    log "BURN submitted as job $jid: $cores cores x $tl (~$(awk -v c="$cores" -v w="$wall_capped" 'BEGIN{printf "%.0f", c*w}') core-hours)"
}

case "${1:-status}" in
    start)
        [ -f "$STOP" ] && rm -f "$STOP"
        phi="${2:-$PHI_FIRST}"
        log "campaign start at phi=$phi, budget cap $(cat "$BUDGET_F") $UNIT"
        submit "$phi"
        ;;
    next)
        if [ -f "$STOP" ]; then
            log "STOP flag set -- not submitting"; exit 0
        fi
        refresh_estimate
        budget=$(cat "$BUDGET_F"); sp=$(spent); est=$(cat "$EST_F"); nxt=$(next_phi)
        if [ -z "$nxt" ]; then
            log "ALL $N_SLICES SLICES COMPLETE -- campaign finished"; exit 0
        fi
        if [ $(( sp + est )) -gt "$budget" ]; then
            log "BUDGET GUARD: spent=${sp} + est=${est} > cap=${budget} $UNIT"
            log "  stopping cleanly before phi=$nxt."
            log "  Everything done so far is on disk and resumable."
            log "  Raising the cap spends an account shared with nine other people."
            log "  When that is agreed:  echo <new cap> > $BUDGET_F && bash $0 resume"
            exit 0
        fi
        submit "$nxt"
        ;;
    status)  cmd_status ;;
    unit)    cmd_unit ;;
    burn)    cmd_burn "${2:-}" ;;
    spent)   spent; echo ;;
    fetch)
        # Delta has rsync and scp (Bridges-2 had neither), but the laptop sits
        # behind CMU NAT and cannot be reached from here, so the pull still has to
        # be started on the laptop. rsync verifies what it transfers, so unlike the
        # PSC route there is no separate sha256 pass to remember.
        echo "Run this ON YOUR MAC (not here):"
        echo
        echo "  cd \"\$HOME/Documents/CMU/Robomechanics Lab/PenguMujoco_delta\" && \\"
        echo "  mkdir -p pengu_mujoco/results/gait_sweep && \\"
        # --include/--exclude rather than a glob in the remote path: two patterns
        # inside one pair of quotes would be read as a single filename with a space.
        echo "  rsync -avP \\"
        echo "    --include='sweep_grid5_c2_phi*.csv.gz' --include='*.manifest.json' --exclude='*' \\"
        echo "    bgu@login.delta.ncsa.illinois.edu:$REPO/pengu_mujoco/results/gait_sweep/ \\"
        echo "    pengu_mujoco/results/gait_sweep/ && \\"
        echo "  gunzip -kf pengu_mujoco/results/gait_sweep/*.csv.gz && \\"
        echo "  python delta/merge_phi.py"
        echo
        echo "Ready to pull:"
        ls -la "$REPO/pengu_mujoco/results/gait_sweep"/*.csv.gz 2>/dev/null \
          | awk '{print "  ", $5, $9}' || echo "   (none yet)"
        ;;
    stop)    touch "$STOP"; log "STOP set -- current job will finish, nothing new submitted" ;;
    resume)  rm -f "$STOP"; log "STOP cleared"; exec "$0" next ;;
    *) echo "usage: $0 {start [phi]|burn [--go]|status|next|stop|resume|spent|unit|fetch}"; exit 2 ;;
esac
