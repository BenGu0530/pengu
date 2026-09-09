#!/usr/bin/env bash
# Pull the GRID-6 results back from PSC. RUN THIS ON THE LAPTOP, not on PSC.
#
#   bash psc/fetch_grid6.sh                    # into pengu_mujoco/results/
#   bash psc/fetch_grid6.sh /some/other/dir
#
# Bridges-2 has neither rsync nor scp (not a PATH problem -- not installed, and
# no module provides them), and it cannot push: compute nodes have no outbound
# internet and the laptop sits behind CMU NAT. So the transfer is tar over ssh,
# always initiated from this side. tar does not verify what it moved, which is
# why this script checks row counts afterwards instead of trusting the pipe.
#
# Run the four --merge steps on the PSC login node first (they cost 0 SU):
#   cd $PROJECT/pengu_hw/pengu_mujoco
#   CONFIG=c1 python grid6/realism_check.py --merge
#   CONFIG=c6 python grid6/realism_check.py --merge
#   python grid6/hw_sweep.py --mu 0.12 --merge
#   python grid6/hw_sweep.py --mu 0.45 --merge

set -uo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/pengu_mujoco/results}"
REMOTE="/ocean/projects/cis250009p/bgu/pengu_hw/pengu_mujoco/results"

# Expected data rows, derived from the inputs rather than guessed:
#   realism_c1  192,152 cells x 2 mu x 1 variant (act)
#   realism_c6  117,670 cells x 2 mu x 2 variants (act, both)
#   hwact_*     111,540 grid cells, one row each (the 4 rollouts per cell are
#               aggregated into that row)
declare -a NAMES=(grid6_report/realism_c1.csv
                  grid6_report/realism_c6.csv
                  grid6_hw/hwact_mu012.csv
                  grid6_hw/hwact_mu045.csv)
declare -a WANT=(384304 470680 111540 111540)

echo "dest   : $DEST"
echo "remote : $REMOTE"
mkdir -p "$DEST/grid6_report" "$DEST/grid6_hw"

echo
echo "=== pulling (gzipped in flight; the CSVs are ~190 MB raw) ==="
ssh psc "cd '$REMOTE' && tar czf - \
    grid6_report/realism_c1.csv grid6_report/realism_c6.csv \
    grid6_hw/hwact_mu012.csv grid6_hw/hwact_mu045.csv" \
  | tar xzvf - -C "$DEST"
rc=$?
if [ "$rc" -ne 0 ]; then
    echo
    echo "TRANSFER FAILED (rc=$rc)."
    echo "  If ssh appeared to hang with no output, the multiplexed connection"
    echo "  had expired and it was waiting for a password it could not show"
    echo "  through the pipe. Run 'ssh psc' once in a normal terminal, then retry."
    exit "$rc"
fi

echo
echo "=== verifying row counts ==="
bad=0
for i in "${!NAMES[@]}"; do
    f="$DEST/${NAMES[$i]}"
    if [ ! -f "$f" ]; then
        printf "  %-34s MISSING\n" "${NAMES[$i]}"; bad=1; continue
    fi
    n=$(( $(wc -l < "$f") - 1 ))
    if [ "$n" -eq "${WANT[$i]}" ]; then
        printf "  %-34s %'d rows  ok\n" "${NAMES[$i]}" "$n"
    else
        printf "  %-34s %'d rows  EXPECTED %'d  (short %'d)\n" \
               "${NAMES[$i]}" "$n" "${WANT[$i]}" "$(( ${WANT[$i]} - n ))"
        bad=1
    fi
done

echo
if [ "$bad" -eq 0 ]; then
    echo "all four files complete"
else
    echo "SOME FILES ARE SHORT."
    echo "  A merge only includes the shards that exist. Check the merge output"
    echo "  on PSC for a 'shards missing' warning, re-run those array indices,"
    echo "  merge again, and re-fetch."
    exit 1
fi
