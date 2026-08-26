# GRID-5 fleet launch — one line per machine

Prereq (every machine): repo cloned, on the working branch, then just:

    git pull --ff-only

First run auto-builds `.sweep_venv` (mujoco 3.8.x) if missing — same as GRID-4.

## The one line, per machine

| machine | queue (in order) | command |
|---|---|---|
| **naomio** (strongest) | c4 → c2 → c9 | `nohup bash grid5/run_machine.sh naomio > results/gait_sweep/machine_naomio.log 2>&1 &` |
| **rml3** | c5 → c10 | `nohup bash grid5/run_machine.sh rml3   > results/gait_sweep/machine_rml3.log   2>&1 &` |
| **mac** (shares CPU with other projects) | c3 → c8 | `nohup bash grid5/run_machine.sh mac    > results/gait_sweep/machine_mac.log    2>&1 &` |
| **rml2** (this box, shares RL) | c6 → c7 | `nohup bash grid5/run_machine.sh rml2   > results/gait_sweep/machine_rml2.log   2>&1 &` |
| **laptop** (weak) | c1 | `nohup bash grid5/run_machine.sh laptop > results/gait_sweep/machine_laptop.log 2>&1 &` |

Coverage: Phase A = c1..c6 (one leading on each machine; naomio carries the 6th, c2,
second in its queue). Phase B = c7..c10 queued behind. rml2 runs `nice 19` with 8
shards so the RL track keeps headroom. If the laptop finishes c1, point it at
whatever is still running elsewhere by editing the queue table at the top of
`grid5/run_machine.sh` (or just `bash grid5/run_sweep.sh <cfg>` — resume is
by axis-tuple, two machines can even share one config safely IF they use disjoint
shard ids; simplest is one config per machine).

## Checking progress

    tail -f results/gait_sweep/grid5_c4_run.log                 # shard log (per config)
    tail -f results/gait_sweep/machine_mac.log                  # queue log (per machine)
    wc -l results/gait_sweep/sweep_grid5_c4_*.csv               # rows done (of 2,142,721 incl. header)

## What the one line does

`run_machine.sh` walks its queue: per config it runs `grid5/run_sweep.sh`
(initcsv + `manifest.json` + N shards, resume-safe), installs the grid5 watchdog
(@reboot + 10-min cron — survives reboots without disturbing any grid4 crontab
lines), then polls for `.done` every 5 min and revives dead shards. Configs already
`.done` are skipped, so re-running the same line after any crash/reboot is always
safe. Stop on purpose: `touch results/gait_sweep/WATCHDOG_OFF` and kill the
`grid5_sweep.py` processes.

## When a config completes

The queue log prints the ship-back line:

    gzip -kf <csv> && git add -f <csv>.gz <csv-stem>.manifest.json
    git commit   # message by Ben; confirm branch before pushing

(`results/` and `*.csv` are gitignored — data enters the repo only via `git add -f`.
Ship the manifest with the CSV; grid5 analysis tools refuse a CSV whose manifest
does not match.)
