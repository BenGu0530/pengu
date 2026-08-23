#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q "all done" overnight/queue_e2.log 2>/dev/null; do sleep 120; done
bash overnight/run_base_seeds.sh > overnight/e2_base.log 2>&1
