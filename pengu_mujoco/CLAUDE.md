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

### NAMING RULE (Ben, explicit)
**Never label a parameter value "penguin" or "human".** Use the number only
(COM ratio 1.05 / 1.2 / 1.4 / 1.6; Gait 1 / Gait 2 / Gait 3; foot gap 80 / 100 mm).
No species framing on any axis, in code, docs, plots, or prose.

### Axis 1 — DESIGN: COM height / hip-axis height
Sweep values: **1.05, 1.2, 1.4, 1.6**.
Definition (measured, to confirm with Ben): **whole-robot COM height above the floor
divided by hip-axis (`easyaxis`) height above the floor**, at the neutral standing pose.
penguV3 today measures **1.108** on this definition (COM 0.1881 m, axis 0.1697 m,
`STAND_HIP_DEG=0`) — i.e. the current model already sits at the LOW end of the range;
variants must push the ratio UP.
Open questions for Ben before CAD:
1. Raise the ratio by **moving mass up** (long legs, top-heavy) or **shortening legs**
   (squat, short legs)? Same ratio, physically different robots. Or both (2-D design space)?
2. Which **canonical pose** defines the ratio (neutral stand vs the hip_off=30° walk posture)?
3. **Whole-robot COM** (assumed) or upper-body only?
Variants come from Onshape CAD exports (Ben).

### Axis 2 — CONTROL: torso strategy
Kinematic fact: `easytorso` is a child of `easyaxis`, so the torso joint angle is
**relative to the hip axis**; torso world roll ≈ hip-axis roll + torso joint angle.
With the torso motor as a fixed joint the torso rolls **with** the axis.

Proposed continuous parameterization — **torso follow gain κ**:
`target torso world roll = κ × (hip-axis roll)`, tracked by an outer-loop PID
(feedforward `(κ−1)×axis_roll` + Kp·e + Ki∫e + Kd·ė, D from roll rate), output clamped
to the torso ctrlrange (±45°).
- **κ = 0 → Gait 1**: torso held at absolute 0 roll in the world (counter-rotates the
  axis). Benchmark: measured torso world roll stays ≈ 0 (report `torso_roll_rms`).
- κ = 1 → the fixed-joint case (what the old `torso_amp=0` sweep slice did).
- **κ > 1 → Gait 2**: torso leans further into the roll, toward the stance leg.
- κ < 0 → torso leans away from the stance leg (control condition).
- **Gait 3**: torso removed (comparison only).
κ makes torso strategy a continuous sweepable scalar; Gait 1/2 are points on it.
Readout `torso_stance_corr` verifies whether κ>1 actually puts the torso over the stance
foot. These are **assumptions to test with data**, not claims.

Note: the torso motor is limited to **±4.1 N·m** (XM430 stall). At high COM ratios the
PID may saturate and fail to hold upright — that is a **co-design result**, not a bug;
`torso_roll_rms` exposes it.

⚠️ The existing `gait_config.py` torso is an **open-loop sinusoid** (`torso_amp`,
`torso_phi`). It implements none of Gait 1/2/3; `torso_amp=0` is the κ=1 fixed-joint
case. The κ PID mode is new code to write.

### Minor axes (more evidence for the table)
- **Foot gap**: 80 mm (current V3) vs 100 mm.
- **Foot geometry**: rolling contact; contact point moves as foot geometry changes.
- **Surfaces**: friction ladder (`friction_utils.SURFACES`).

### Deliverable
A **gait × COM table** (2 main gaits × 4 COM ratios = 8 configs, + Gait 3 as control) —
sweep all and compare which combinations do best, including on low-friction surfaces.
Foot gap and extra surfaces add evidence.

### Order of work (Ben)
1. **PID torso (κ) first** — it is the prerequisite for every cell of the table.
2. **Sweep script** ready next.
3. **RL / co-design emergence last** — not urgent.

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
