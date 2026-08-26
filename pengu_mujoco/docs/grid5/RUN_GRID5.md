# RUN_GRID5 — operations guide for the round-2 sweep

What GRID-5 is: the 10-config co-design sweep (kappa{0,2} x COM{1.05,1.10,1.20,
1.31,1.40}), staged start + extended metrics, 2,142,720 rows/config. The frozen
protocol and all evidence: `docs/grid5_design.md`. Planning memo:
`docs/grid5/session_2026-08-26_planning_memo.md`. Quick launch card:
`docs/grid5/fleet_launch.md`. Analysis/plotting brief (separate
plot-only session): `docs/grid5/PLOT_GRID5.md`. All sweep code lives in `grid5/` (the GRID-4
pipeline in `physics/` is the untouched backup — never edit it).

## 1. Deployment status (launched 2026-08-26, night)

| machine | queue | status |
|---|---|---|
| naomio (32 cores) | c4 -> c2 -> c9 | RUNNING (c4, 30 shards) |
| rml2 (16 cores) | c6 -> c7 | RUNNING (c6, 14 shards; measured ~51k rows/h -> ~41 h/config) |
| rml3 (16 cores) | c5 -> c10 | RUNNING (c5, 14 shards) |
| mac | c3 -> c8 | TO DEPLOY (Ben) |
| laptop | c1 | TO DEPLOY (Ben) |

Phase A = c1..c6 (leading config on every machine + c2 second on naomio);
Phase B = c7..c10 queued behind. Queues advance automatically on `.done`.

## 2. Deploying a machine (mac / laptop — two lines)

From the repo root `pengu_mujoco/` on branch `friction-experiments`:

    git pull --ff-only
    nohup bash grid5/run_machine.sh mac    > results/gait_sweep/machine_mac.log    2>&1 &   # mac
    nohup bash grid5/run_machine.sh laptop > results/gait_sweep/machine_laptop.log 2>&1 &   # laptop

First run auto-builds `.sweep_venv` (mujoco 3.8.x, pinned) if missing — takes a
few minutes, then shards start. Nothing else to configure: shard count defaults
to cores-2; DR_K=1 (the map protocol); the manifest is written automatically.

What the one line does, per config in the queue:
initcsv + `manifest.json` -> N shards (`run_sweep.sh`, resume-safe by axis-tuple)
-> installs the grid5 watchdog (@reboot + 10-min cron; grid5-tagged lines only,
never touches grid4 crontab entries) -> polls `.done` every 5 min, reviving dead
shards -> on `.done`, prints the ship-back line and starts the next config.
Re-running the same line after any crash or reboot is always safe: `.done`
configs are skipped, partial configs resume exactly where every machine left off
(rows are seeded per (cell, mu, repeat) — machine-independent).

## 3. Monitoring

    tail -f results/gait_sweep/machine_<name>.log      # queue log (config-level)
    tail -f results/gait_sweep/grid5_<cfg>_run.log     # shard log (cell progress)
    wc -l results/gait_sweep/sweep_grid5_<cfg>_*.csv   # rows done; full = 2,142,721 incl. header

Throughput anchor (measured on rml2, 14 shards): ~51,000 rows/h -> ~41 h per
config; naomio at 30 shards should run roughly 2x that. Watchdog activity:
`results/gait_sweep/watchdog5.log`.

## 4. Stopping / restarting

Stop ON PURPOSE (the watchdog will otherwise revive shards):

    touch results/gait_sweep/WATCHDOG_OFF
    pkill -f 'grid5_sweep[.]py' ; pkill -f 'run_machine.sh'

Restart later: `rm -f results/gait_sweep/WATCHDOG_OFF`, then re-run the machine's
one line. NEVER edit a running .sh (bash reads incrementally — the 18.5 h lesson);
editing a running .py is safe. Do not mix DR_K values into one CSV.

## 5. When a config completes (ship-back)

The queue log prints the exact line; in general:

    awk 'NF' <csv> > t && mv t <csv>
    gzip -kf <csv>
    git add -f <csv>.gz <csv-stem>.manifest.json    # results/ and *.csv are gitignored
    git commit    # message by Ben; CONFIRM THE BRANCH before pushing (friction-experiments)

Ship the manifest WITH the CSV — grid5 analysis tools refuse a CSV whose manifest
does not match (protocol, K, slip constants, mujoco version). Fleet rules apply:
commit local run data before pulling; announce any `results/` layout change in a
memo BEFORE pushing it.

## 6. Troubleshooting

- "manifest missing — run initcsv first": the CSV exists but its manifest.json
  does not (e.g. copied by hand). Run `CONFIG=<cfg> python grid5/grid5_sweep.py
  initcsv` from `grid5/`, or re-run the machine line.
- "MANIFEST MISMATCH ... refusing to write": the process settings (K, slip
  constants, mujoco version) differ from the artifact's manifest. Do NOT force —
  fix the environment (usually a stray DR_K env var or wrong mujoco).
- CSV with no header: resume silently recovers 0 rows and re-runs everything —
  the launcher and watchdog refuse to run on a header-less CSV; repair per
  `docs/grid4_fleet_memo.md` pre-flight 4 (same trap as GRID-4).
- Two machines on ONE config: safe ONLY with disjoint shard ids (same N_SHARDS,
  different SHARD_ID sets, both after one initcsv). Default setup = one config
  per machine; prefer that.
- Reassigning queues (e.g. laptop finished c1): edit the queue table at the top
  of `grid5/run_machine.sh`, commit, pull on the target machine, re-run its line.

## 7. After the maps (not yet started)

Per config: hot-region topup K=1->5 (`grid5/topup_k.py`, resume-safe, shardable),
then the frozen selection chain (three champion tracks T-speed/T-cot/T-slip,
+50000-seed confirmation, neighborhood fine scan) via `grid5/grid5_select.py`
(to be written before analysis begins — see grid5_design.md for the frozen
procedure). Robust-region reporting is four-tier: surv-only / pass / strict
heading>=0.9 / clean-pass slip<=0.05, all recomputable from the saved records.
