# CLAUDE.md — working agreement & project frame (Pengu)

## Working agreement (Ben's explicit rules — follow these over any instinct of mine)

**1. Do NOT draw conclusions.**
Report what the data shows. Do not declare a hypothesis confirmed, dead, falsified,
"in the noise", or "the story is X". Present numbers and let Ben conclude. My job is
measurement + engineering, not verdicts.

**2. Do NOT interrupt, deprioritize, or write off a sweep before ALL sweeps are done.**
- Never stop / suggest stopping a running sweep based on a mid-stream judgment.
- Never say a sweep is "useless", "wasted", "already answered", or "deep in diminishing
  returns".
- Never claim a judgment "can already be made" while scans are incomplete.
- Partial results are partial. Non-linear locomotion needs the full sweep — cutting
  corners now makes every later conclusion worthless. Let scans finish, then look.

**3. Simulation only. No real-robot pressure.**
Do not raise hardware deadlines, hardware validation risk, or "hardware will slip the
deadline" as considerations. Real-robot testing looks only at the best case and a few
good cases, chosen after the simulation table is complete. It is not a gate, not a
schedule risk, and not a reason to scope anything down. Stop mentioning it.

**4. Ben decides direction. I propose, measure, and build.**
Do not re-frame the paper spine, retire a direction, or redefine the headline metric on
my own. Surface facts + options; Ben chooses.

## Project frame (current, as of 2026-07-09)

Robot: **penguV3** only (v2 retired — fundamental model problems). MuJoCo, 5 actuators.

The study is a **co-design** question over two axes:

### Axis 1 — DESIGN: COM height / hip-axis height (≈ leg length)
Sweep values: **1.05 (human), 1.2, 1.4, 1.6 (penguin)**.
⚠️ Definition conflict to resolve with Ben: `results/BEST_GAITS.md:3` records the current
model as "COM/leg = 1.05 (penguin)", while Ben's spec says 1.05 = human, 1.6 = penguin.
Pin the exact ratio definition and where penguV3 sits on it before building variants.
Mass-distribution variants come from Onshape CAD exports (Ben).

### Axis 2 — CONTROL: torso strategy (3 conditions)
Kinematic fact: `easytorso` is a child of `easyaxis`, so the torso joint angle is
**relative to the hip axis**. With the torso motor as a fixed joint, the torso stays
perpendicular to the hip axis and **rolls with the axis**.
- **Gait 1 (human-like)**: reactive, well-tuned **PID holds the torso at absolute 0 roll
  in the world** — i.e. it rotates *counter* to the hip-axis roll. Humans keep the body
  upright and still.
- **Gait 2 (penguin-like)**: reactive **PID leans the torso toward the stance leg** —
  i.e. *with* the axis roll direction, further into it. Torso leads the gait; hypothesised
  energy recovery.
- **Gait 3 (comparison only)**: torso removed.
These are **assumptions to test with data**, not claims.
⚠️ Implementation note: the existing `gait_config.py` torso is an **open-loop sinusoid**
(`torso_amp`, `torso_phi`) on the torso joint. `torso_amp=0` is the *fixed-joint* case
(torso rolls with the axis) — it is **not** Gait 1. Gait 1/2 need new reactive PID torso
control modes.

### Minor axes (more evidence for the table)
- **Foot gap**: 80 mm (current V3) vs 100 mm (wider = penguin; narrow = human).
- **Foot geometry**: rolling contact; contact point moves as foot geometry changes.
- **Surfaces**: friction ladder (`friction_utils.SURFACES`).

### Deliverable
A **2 (gait) × 4 (COM) table** = 8 configs (+ gait 3 as control) — sweep all and compare:
does human-gait + human-COM win? does penguin-gait + penguin-COM win on low-friction
surfaces? Foot gap and extra surfaces add evidence.

### Co-design extension (professor's idea)
At each COM design point: (a) sweep again, and/or (b) control optimization, and/or
(c) **PPO black-box RL to let the gait EMERGE**, then compare the emergent gait against
the two pre-designed gaits. The `rl/` CPG-RL pipeline exists and can be repurposed per
design point.

## Engineering constraints (do not violate)
- Run from repo root `pengu_mujoco/`; flat imports, no package.
- `gait_config` global setters do **not** restore — file defaults are not live values.
- Friction priority hack is load-bearing: `geom_priority[floor]=1`.
- μ stance-gate differs by tool (`F_HI=4N` in `gait_sweep` vs `FN_MIN=1N` elsewhere) —
  don't mix μ numbers across tools.
- CSV column order: `single_frac` is column **14** (0-based 13). Verify columns before
  any analysis.
- `.gitignore` blocks `results/` and `*.csv` → data needs `git add -f`.
- Sweeps: shard with `N_SHARDS`/`SHARD_ID`, `initcsv` before sharded workers, resume is
  by axis-tuple. Reboot-safe launchers: `physics/run_grid2.sh`, `physics/run_stageB.sh`.
