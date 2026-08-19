# GRID-4 topup memo — upgrade a config's passing rows to true K=5

The K=1 map is done for c3–c6; single-trial champions inflate (top-order statistics).
This job re-scores **every row that passed at K=1** with the full 5 seeded repeats —
numerically identical to having swept that row at K=5 from the start (validated).
Paper numbers will quote these K=5 values.

## Assignment

| config | machine | scope | est. time |
|---|---|---|---|
| **c4** | Mac (running since 2026-08-19) | ~391k passing rows | ~4–5 days |
| **c6** | any idle box (E or C) | ~43k passing rows | **~6–12 h** |
| c3 / c5 | later, same command | ~50k / ~178k rows | after c6 |
| c1 / c2 | after their maps land | — | — |

## Run (one line, resume-safe)

```bash
cd ~/pengu/pengu_mujoco            # adjust to this machine's repo path
git pull
bash physics/topup_all.sh c6       # 2nd arg = shard count (default cores-2)
# shared box: SWEEP_NICE=19 bash physics/topup_all.sh c6
```

The script: picks/validates a mujoco-3.8 python → unpacks the committed base map →
builds the pass>0 selection → launches sharded workers. Interrupting is safe;
**re-running the same line revives it** (done rows are skipped). No watchdog is
armed for topups — if the machine reboots, just run the line again.

## Verify it started

```bash
tail -2 results/gait_sweep/c6_topup.log     # startup line: com=1.3100 ... K 1->5
wc -l results/gait_sweep/sweep_grid4_c6_topupK5.csv    # growing
```

## Ship-back when the row target is reached (the launch line prints it)

```bash
OUT=results/gait_sweep/sweep_grid4_c6_topupK5.csv
gzip -kf "$OUT" && git add -f "$OUT.gz"
git pull --rebase && git commit -m "GRID-4 c6 topupK5 (all passers)" && git push
```

Notes
- Output rows OVERRIDE the base K=1 rows in analysis (grid4_report will consume them).
- Rows that failed at K=1 stay single-trial by design — the scope is "everything that
  showed life"; stated in the report's methods note.
- Do not run two machines on the same config's topup (same one-CSV rule as sweeps).
