# Pengu MuJoCo — orientation for Fable

> **What this file is.** The full orientation **map** for this penguin-robot gait codebase —
> the "read this first" doc referenced as #1 in `fable_prompt.md` (the Chinese kickoff you were
> handed). Read it top-to-bottom once; it covers the research goal, the repo layout, the two
> experiment pipelines, the exact metric definitions, the coordinate conventions, and the traps.
> File references look like `path/to/file.py:123` (path relative to the repo root, `pengu_mujoco/`).
> For bulk reading beyond this map, **delegate to the `code-scout` subagent** rather than pulling
> raw files into context (see `fable_prompt.md` → "委派读取给 code-scout").
>
> **Golden rules.** (1) Run every script **from the repo root** `pengu_mujoco/` — imports are
> flat (`from gait_config import …`), there is no package/`__init__.py`. (2) The gait is
> **open-loop position control**; "best" is judged by eye from rendered video, *backed* by the
> quantitative metrics below. (3) The RL pipeline is decoupled — it does **not** import
> `gait_config.py`. (4) Before trusting a docstring, check §11 (Known doc/code drift).

---

## 1. Research north star (why this repo exists)

Source of truth: `penguin plan for summer.pdf` (in the repo root).

**The paper claim.** *Penguin-style walking — where the torso is held/swung **over the stance
foot** — is better on slippery surfaces* (it keeps the ground-reaction force inside the friction
cone). Penguins waddle this way; humans adopt the same non-plantigrade, torso-rolling pattern
**only** on slippery walkways (ref: *Motor Patterns During Walking on a Slippery Walkway*,
J. Neurophysiol. 2009). On normal ground humans hold the torso upright with no roll.

**What we're testing in sim.** The interaction of **two variables** — torso phasing and mass
distribution — with **surface friction**. The plan enumerates these sim conditions (combinations
of mass distribution × torso mode):

| torso mode | meaning |
|---|---|
| `upright` (PD hold, no swing) | torso held vertical — the "human on normal ground" control |
| `over_stance` (penguin-like) | torso swings **over the planted stance leg** |
| `over_swing` (non-plantigrade) | torso swings **over the swing leg** |
| no torso | lower body only |

crossed with **penguin vs human mass distribution** (human target: COM at 54–57 % of standing
height). For each condition we optimize the motor amplitudes/frequency to a **matched speed**,
then report: **GRFs and the friction cone** (normal vs tangential force → minimum friction
required to walk), **walking speed**, **COM motion**, and **foot roll vs foot pitch** on slippery
vs non-slippery surfaces (penguins show more foot roll + less foot pitch — a single peak in foot
motion). Hardware validation (multiple surfaces, COM measurement) is done separately by a
collaborator; our job is the simulation half.

Keep this framing in mind: the friction/surface work in `physics/friction_study.py` **is the
actual paper experiment**; the big gait sweeps and the RL work are the machinery that produces
walkable gaits to run those experiments on.

---

## 2. Repo map

```
pengu_mujoco/
├── gait_config.py          ← THE open-loop controller + all tunable gait params (shared heart)
├── friction_utils.py       ← runtime floor-friction override (surface ablation)
├── walk_pengu.py           ← interactive MuJoCo viewer + per-joint decoded readout (debug tool)
├── optimize_gait.py        ← older CMA-ES "clean forward speed" optimizer (penguV2)
│
├── penguV2/                ← model v2: crank-slider CLOSED-LOOP robot, natively pitched forward
│   ├── scene.xml, robot.xml           (closed-loop; open-loop gait + friction study use this)
│   ├── scene_rl.xml, robot_rl.xml     (RL variant: sliders are DIRECT prismatic actuators)
│   └── config.json
├── penguV3/                ← model v3: upright re-export (pitch fixed in CAD), same 5 actuators
│   ├── scene.xml, robot.xml
│   └── _*.png / _*.mp4 / _*.gif  (stance/pitch/torso still-frame studies)
│
├── physics/                ← PIPELINE A: open-loop gait sweep → optimize → report (see §6)
│   ├── gait_sweep.py       ← core measurement engine (grid sweep, footfall detection, metrics)
│   ├── cma_search.py       ← CMA-ES cross-check (reuses gait_sweep.run_trial)
│   ├── heatmaps.py         ← post-process a grid CSV into robustness "highland" heatmaps
│   ├── gait_report.py      ← 3-panel "beauty" report video for one chosen gait + GRACE metrics
│   ├── analyze_gait.py     ← deep single-gait diagnostic (friction cone, COM trajectory)
│   ├── friction_study.py   ← THE surface-ablation paper experiment (penguV2, 3 torso modes)
│   ├── grf_friction_probe.py ← debug single-run GRF/μ probe (stdout only)
│   ├── analyze_sweep.py    ← post-process the OLDER anchor-validation freq sweeps
│   ├── physics.py          ← system-ID (torso→roll direction, torso resonance)
│   ├── model_scan.py       ← read-only model audit (mass/inertia/constraints/balance)
│   ├── run_grid.sh         ← sharded/resumable/reboot-safe launcher for the big grid
│   └── fine3_guard.sh      ← watchdog protecting a long sweep's resume state
│
├── rl/                     ← PIPELINE B: CPG-RL (PPO modulating a central pattern generator, §7)
│   ├── pengu_env.py        ← the Gymnasium env (CPG-RL, bio-imitation reward, v3/v4 switch)
│   ├── train_penguin.py    ← bio-imitation training (produces ppo_penguin_v3/v4)
│   ├── train_curriculum.py ← base-reward training w/ domain-rand + speed curriculum
│   ├── train_smoke.py      ← 80k-step pipeline sanity check
│   ├── penguin_metrics.py  ← scientific "ruler": OURS-vs-real-penguin gait table
│   ├── stepping_test.py    ← per-step gait geometry (stride/cadence/footprints)
│   ├── sweep_pitch_rl.py   ← spawn-pitch robustness sweep
│   ├── render_policy.py / render_with_trajectory.py / render_views.py  ← video renderers
│   ├── models/             ← released policies: ppo_penguin_v3.zip, ppo_penguin_v4.zip
│   └── runs/               ← training logs, checkpoints, eval CSVs, videos
│
├── results/                ← all experiment outputs + the hand-maintained registry
│   ├── BEST_GAITS.md       ← ⭐ registry of winning gaits (hand-edited, read this)
│   ├── gait_sweep/         ← grid CSVs, CMA CSVs, report/analyze videos
│   ├── friction_study/     ← penguin_configs.csv (the paper table)
│   └── *_YYYYMMDD_HHMMSS.log
│
├── backup_scripts/ , backups/   ← archived earlier scripts + snapshots (ignore for new work)
└── *.log , outcmaes/            ← stray run logs + CMA-ES internal state
```

Top-level stray sweep scripts (`sweep_amp_freq.py`, `sweep_phase_freq.py`,
`sweep_anchor_validation.py`, `sweep_freq_surface.py`, `slice_2d.py`, `actual_gait_plot.py`,
`render_gait.py`, `friction_scan.py`) are the **older, pre-`physics/`** sweep generation. They
still run, but new gait work lives in `physics/`. `analyze_sweep.py` reads *these* older
anchor-validation CSVs, not the `physics/gait_sweep.py` grid.

---

## 3. The robot: two models, one controller

Both models expose the **same 5 actuators** so one controller drives either:
`hip-L, hip-R, crank1-R, crank1-L, torso` (`gait_config.py:17`).

Selected by env var **`PENGU_MODEL`** (`gait_config.py:16`), default `v2`:

| | **v2** (`penguV2/scene.xml`) | **v3** (`penguV3/scene.xml`) |
|---|---|---|
| structure | crank-slider **closed-loop** (revolute crank drives a slider via an equality constraint) | **upright re-export**, pitch fixed in CAD |
| init pitch | **−30°** (natively pitched forward) | **0°** (natively upright) |
| stand hip | **−25°** | **0°** |
| how forward pitch is imposed | built into the model | **dynamically** via `hip_off=30°` (~25° fwd pitch), eased in over the transition |
| used by | friction study, grf probe, old optimizer | the gait grid + CMA (`run_grid.sh` pins `v3`), gait_report/analyze (hardcoded v3) |

There is also **`penguV2/robot_rl.xml`** — an RL-only variant that **replaces the crank-slider
closed loop with direct prismatic slider actuators** (`hip-L, hip-R, slider-L, slider-R, torso`),
removing the equality constraint so PPO can run fast. Only `rl/pengu_env.py` uses it.

**Kinematic-tree warning** (`walk_pengu.py:1-25`): the tree root is `leftthighmotor` (an
onshape→robot artifact) and is used **everywhere as the base/root position proxy** — it is *not*
a torso. `hip-L`/`hip-R` are internal hinges along the chain, **not** biomechanical left/right
hips. The two foot bodies are confusingly named: `right_foot0080` = **Right** foot,
`right_foot0080___fillet13` = **Left** foot. This `FOOT_BODIES` dict is copy-pasted into ~5
scripts. The real torso body is `easytorso`.

**World frame** (`gait_config.py:129-134`, set in `set_initial_pose`):
`qpos[0]=x`=**lateral**, `qpos[1]=y`=**forward** (robot faces +y), `qpos[2]=z`=**up**. Every
"forward" metric reads index `[1]`; "lateral" reads `[0]`. Forward pitch is a rotation about
world-y in the free-joint quaternion `qpos[3:7]`.

---

## 4. The open-loop gait controller (`gait_config.py`) — the shared heart

This one file holds **all tunable params + the controller**; the physics pipeline imports from
it. The gait is 5 phase-coupled sinusoids of a single base phase `phase = 2π·freq·(t−T_HOLD)`:

- **Legs** (`crank1-L/R`): half-rectified sine `crank = α·amp·0.5·(1+sin)`, built-in phasing
  A=0° / B=180° (antiphase). (`gait_config.py:116-117`)
- **Hips** (`hip-L/R`): half-rectified swing about an offset, built-in C=180° / D=0°, plus an
  optional **antisymmetric lean** (`WALK_HIP_LEAN_DEG`, mirrors the real robot's `p_leanAngle`).
  (`gait_config.py:118-119`)
- **Torso**: full sine, built-in E=0°. (`gait_config.py:120`)

`apply_ctrl` (`gait_config.py:142-176`) is a **hold → smoothstep-transition → walk** schedule
(`T_HOLD` seconds standing, `T_TRANSITION` seconds blending in).

**Critical mechanism — global-mutating setters.** `set_walk_freq / set_hip_amp / set_crank_amp
/ set_torso_amp` (`gait_config.py:199-240`) and direct assignments like
`gc.PHASE_OFFSET_C_DEG = …` **reassign module globals**, and `compute_gait` reads those globals.
This is how every sweep injects a cell's parameters: mutate the globals, run one trial, repeat.
It works because cells run **sequentially in one process**. The traps this creates are in §10.

---

## 5. The metrics vocabulary (memorize this)

These names appear across both pipelines and in `results/BEST_GAITS.md`. The open-loop
definitions all live in `physics/gait_sweep.py::run_trial`, measured only in the window
`t ∈ [SETTLE=11 s, SIM=24 s]` after the gait has settled.

- **`path`** — sum of all per-step 2D step lengths (a step = 2D distance between consecutive
  same-foot touchdowns). Total distance the feet traveled; robust to turning.
- **`path_speed`** = `path / walk_time`. **Not** progress — a gait can rack up `path_speed` while
  curling or marching in place.
- **`net_fwd_speed`** = (final − first) **forward (y)** root displacement / walk_time. **Real
  progress.** Can be negative. *This metric exists because the `path_speed` #1 gait actually had
  `net_fwd = −0.12` — it was curling backward* (`results/BEST_GAITS.md`). **Always check this.**
- **`straightness`** = straight-line displacement / `path`. 1.0 = dead straight, ~0 = looping.
- **`single_frac`** — fraction of timesteps with **exactly one** foot loaded (`Fn > F_HI=4 N`).
  The single-support walking signature; ~1.0 = clean L/R alternation.
- **`μ_req` / `mu_req_p95`** — per-timestep, per-foot **friction-cone demand** `|Ft|/Fn`,
  stance-gated (only counted when the foot is loaded). `mu_req_p95` = 95th percentile = the
  minimum floor friction the gait needs to not slip. **This is the core paper metric.**
  ⚠️ stance-gate threshold differs by script: `F_HI=4 N` in `gait_sweep.py`, `FN_MIN=1 N` in
  `grf_friction_probe.py` / `analyze_gait.py` / `friction_study.py` — so μ numbers are **not
  directly comparable across tools**.
- **`valid`** — the acceptance gate: survived AND ≥2 steps per foot AND clearance>0 both feet AND
  `single_frac ≥ 0.3`. Used to filter CMA objectives and heatmap averaging.
- **`clear_L/R`, `stride_L/R`, `stride_sym`, `cadence`, `n_steps`** — swing clearance, stride
  length per foot, L/R stride asymmetry, steps/s, total steps.
- **GRACE metrics** (`gait_report.py`, **eval-only, never used to rank**): `lat_sep` (L/R
  footstep lateral separation), `com_reg` (spectral purity of the COM-x rock, 1 = pure sine),
  `weight_transfer` (+1 = COM leans onto the stance foot — the penguin signature).
- **`torso_stance_corr`** (`friction_study.py`) — +1 = torso leans over the stance foot
  (penguin-like); the direct test of the paper claim.

RL uses its own scientific ruler (`rl/penguin_metrics.py`): stride frequency, waddle roll
amplitude, sagittal lean amplitude, forward speed, **Griffin & Kram (2000) mechanical-energy
recovery %**, and lateral-KE fraction — compared against measured **king-penguin** values.

---

## 6. Pipeline A — open-loop gait: sweep → optimize → report (`physics/`)

The intended sequence (all from `pengu_mujoco/`):

```bash
# 0. (optional) audit the model — mass/inertia/constraints/balance
python physics/model_scan.py

# 1. SWEEP a parameter grid. Measures EVERY cell; there is no scalar objective.
#    The big "fine3c" grid is 6-DOF, ~3.97M cells (~20 h on 24 cores) → run sharded via:
bash physics/run_grid.sh                       # sharded, resumable, reboot-safe
#    A single-process run is just:
PENGU_MODEL=v3 python physics/gait_sweep.py

# 2. MAP the highlands: mean path_speed over VALID cells, averaged over the other dims,
#    so broad robust plateaus win over razor-thin spikes.
python physics/heatmaps.py

# 3. CROSS-CHECK with CMA-ES joint optimization (reuses gait_sweep.run_trial, so the two
#    methods verify each other). Objective J = path_speed if valid&survived else penalty.
PENGU_MODEL=v3 python physics/cma_search.py 1200        # 1200 = maxfev

# 4. DEEP-DIVE one gait: friction cone (Ft-vs-Fn scatter) + top-down COM trajectory.
PENGU_MODEL=v3 MUJOCO_GL=egl python physics/analyze_gait.py

# 5. STANDARD "beauty" report video for the chosen winner (edit the G dict at the top first):
#    3 panels (elevated-side | back | live COM-trajectory+footsteps) + GRACE metrics.
PENGU_MODEL=v3 MUJOCO_GL=egl python physics/gait_report.py

# 6. quick render of a single grid cell:
MUJOCO_GL=egl python physics/gait_sweep.py viz 1.59 110

# 7. REGISTER the winner BY HAND in results/BEST_GAITS.md (it is not auto-generated).
```

**`gait_sweep.py` internals worth knowing.** `run_trial` (`:121-223`) is the measurement core:
steps 24 s, ignores everything before `SETTLE=11 s`, and detects footfalls with a **de-bounced
state machine** (Schmitt trigger `F_HI=4 N` / `F_LO=1 N` + clearance gate + touchdown
refractory). `sweep()` does **append-per-cell CSV, resume from an existing CSV, and modulo-
sharding** via env vars `N_SHARDS` / `SHARD_ID` (`global_index % N_SHARDS == SHARD_ID`).
`AXES` (`:78-85`) is the list of `(param, values)` whose Cartesian product = the cells.
Subcommand `initcsv` writes the header exactly once (**must run before sharded workers** or they
race the header).

**Env vars that matter:** `PENGU_MODEL` (`v2`|`v3`), `N_SHARDS`/`SHARD_ID` (sharded sweep),
`MUJOCO_GL=egl` (required for any headless render: viz/report/analyze). Note `gait_report.py`
and `analyze_gait.py` **hardcode** `penguV3/scene.xml`, and `friction_study.py` /
`grf_friction_probe.py` **hardcode** `penguV2/scene.xml` — so `PENGU_MODEL` only actually
switches `gait_sweep.py`, `cma_search.py`, `physics.py`, `model_scan.py`.

**The registry** `results/BEST_GAITS.md` is hand-maintained. Current ⭐ = **`fine2_best_netfwd`**
(freq 1.59 Hz, hip_phi 180°, leg 110°, hip 20°, torso 20°, torso_phi 0°, hip_off 30°):
`net_fwd_speed 0.226 m/s`, `straightness 0.521`, `single_frac 0.999`, `μ_req p95 0.55`.

### The surface-ablation experiment (the actual paper deliverable)

`physics/friction_study.py` (penguV2) runs the plan's core test. For each of 3 torso `CONFIGS`
(`upright` / `over_stance` / `over_swing`) it **CMA-optimizes to a matched target speed
`V_TARGET=0.08 m/s`** (not max speed — so conditions are compared at equal speed), then measures
at the optimum: **`min_mu_to_walk`** (lowest floor μ on the ladder `[1.0 … 0.06]` where it still
survives and travels >0.15 m), **`mu_req_p95`** on a no-slip reference, plus speed, COM-z, foot
roll, foot pitch, and `torso_stance_corr`. Output: `results/friction_study/penguin_configs.csv`.

Floor friction is set by `friction_utils.set_floor_friction(model, μ)`. **Non-obvious physics**
(`friction_utils.py:14-20`): the feet have fixed μ=0.9, and MuJoCo combines **equal-priority**
geoms by **elementwise max**, so lowering the floor μ alone did *nothing* until the fix also
**raises `geom_priority[floor]=1`** so the floor wins the contact. If you fork friction code,
keep the priority line. `SURFACES`: `mocap_floor`=0.7 (baseline), `acrylic`=0.30, `uhmw_pe`=0.14,
`ptfe_ice`=0.06 (ice analog).

### Long-run orchestration

`run_grid.sh` — launcher/watchdog safe to call from `@reboot` + periodic cron. Re-execs under
`flock` (no double-launch), creates the CSV header once, then for each of 16 shards uses a
`/proc`-based liveness check (`kill`/signals are blocked cross-sandbox) + a `.done` sentinel to
skip running/finished shards, else `nohup`s the worker. The last shard to finish writes the
master `.done` and triggers plots. `fine3_guard.sh` — babysits an earlier `fine3` sweep: while
the full sweep is alive it deletes any premature `.done` every 5 s, and only re-creates `.done`
once true coverage is complete (counts unique 6-tuples ≥ `NCELLS`). Both are `/proc`-liveness,
signal-free by necessity.

---

## 7. Pipeline B — CPG-RL (`rl/`)

**This is CPG-RL, not raw joint RL** (Bellegarda & Ijspeert style). A PPO policy outputs 6
central-pattern-generator parameters at 50 Hz; an internal phase oscillator turns them into the
**same sinusoidal gait structure** as the open-loop controller (antiphase leg extension,
antiphase hip swing, torso roll) — but the amplitudes/frequency/phase are **modulated by the
policy** around a nominal, instead of hand-tuned constants. It runs on `penguV2/scene_rl.xml`
(the prismatic, constraint-free variant). **The RL env does not import `gait_config.py`** — the
two pipelines are decoupled; think of CPG-RL as the learned, adaptive successor to the fixed
open-loop gait.

**Env (`rl/pengu_env.py::PenguCPGEnv`):**
- **Action** `Box(-1,1,(6,))` = CPG params `[freq, leg_ext_amp, hip_amp, torso_amp, hip_phase,
  torso_phase]`, mapped `a → MID + a·RNG`. Amplitudes are low-pass smoothed every control step;
  rhythm params (freq, phases) are **latched** — only updated at leg-switch phase boundaries — so
  cadence/phasing stay constant within a half-stride.
- **Observation** `Box(∞,(28,))` = projected gravity(3) + base angular vel(3) + body-frame linear
  vel(3) + joint pos(5) + joint vel(5) + `[sin,cos]` of CPG phase(2) + last action(6) + `vx_cmd`(1).
- **Reset**: resamples `vx_cmd`, spawns at `INIT_Z=0.20`, `INIT_PITCH=−30°` (counter-pitched near
  upright, ± jitter if domain-rand), floor μ ~ U(0.3,0.9) if domain-rand.
- **Termination**: fell (z<0.08, or |roll|>60°, or |pitch|>60°) → −5 penalty; truncated at
  `episode_s·control_hz = 500` steps.

**Reward** (per control step) — always-on: `r_energy = −0.0005·Σ|F·v|`, `r_smooth =
−0.01·Σ(Δaction)²`. Then a `bio_imitate` flag switches the main block:

- **Base** (`bio_imitate=False`): speed-tracking + strong `r_progress (4.0·max(0,vx))` +
  reverse penalty + single-support bonus + scrub penalty + swing-protraction bonus + bob/roll/
  pitch dead-band penalties.
- **Bio-imitation** (`bio_imitate=True`): pulls the gait onto the **measured king-penguin
  signature** (1.27 Hz cadence, 8° roll, 2° lean), with **deliberately low** `r_progress (1.0)`
  because speed-chasing pushed the baseline to a 2 Hz sprint. Action range narrows to a penguin
  prior (freq pinned ~[1.12,1.42] Hz).

**The v3 ↔ v4 switch** (within bio mode, `propulsion` flag) — the **only** difference is one
reward term, `r_swing`:
- **v3** (`propulsion=False`): `r_swing = 0`. A clean slow **waddle**, cleanest lean, but rocks
  nearly *in place* (lateral KE ~76 %, net speed ~0.04 m/s).
- **v4** (`propulsion=True`): adds `r_swing = 1.5·clip(swing_rate,0,0.6)`, rewarding the swing
  foot protracting **forward** so it lands ahead and the body vaults over the stance foot —
  turning the rock into a real **step** (lateral KE 76 %→47 %) without a velocity command.

**Training** — stable-baselines3 **PPO**, `MlpPolicy`, **CPU-only**, 8 `SubprocVecEnv` workers.
Shared HPs: `n_steps=1024, batch_size=4096, n_epochs=5, gamma=0.99, gae_lambda=0.95, lr=3e-4,
ent_coef=0.005, clip_range=0.2, target_kl=0.03, net_arch=[256,256]`. Default 3 M steps.
- `python rl/train_smoke.py` → 80k-step pipeline check → `rl/runs/ppo_smoke.zip`.
- `python rl/train_curriculum.py [steps] [n_envs] [kind]` → base reward, domain-rand ON,
  speed-command curriculum → `rl/runs/ppo_curriculum_{kind}.zip`.
- `python rl/train_penguin.py [steps] [n_envs] [v3|v4]` → bio-imitation, domain-rand OFF →
  `rl/runs/ppo_penguin_{v3|v4}.zip` + 250k checkpoints.
- **Released policies**: `rl/models/ppo_penguin_v3.zip` (waddle) and `..._v4.zip` (propulsion).

**Eval / render** (renders need `MUJOCO_GL=egl`):
- `python rl/penguin_metrics.py <model.zip> <kind> <vx_cmd> <bio0|bio1>` → OURS-vs-penguin table
  + `rl/runs/penguin_metrics_{bio|base}.png`.
- `python rl/stepping_test.py …` → stride/cadence/footprint stats + `stepping_footprints.png`.
- `python rl/sweep_pitch_rl.py …` → spawn-pitch robustness → `pitch_sweep_rl.{csv,png}`.
- `render_policy.py` (simple gif), `render_with_trajectory.py` (side view + footstep inset mp4),
  `render_views.py` (front / 3-quarter / 360°-orbit in one rollout) → `rl/runs/*.mp4/.gif`.

---

## 8. Conventions & gotchas (the traps)

1. **Run from repo root.** Flat imports; no package.
2. **Global-mutation setters don't restore.** Sweeps mutate `gc.WALK_*`, `gc.PHASE_OFFSET_*`,
   `gc.T_HOLD/T_TRANSITION` and never reset them (`grf_friction_probe.py` restores only
   `PHASE_OFFSET_E_DEG`). Anything importing `gait_config` after a sweep sees clobbered values,
   **not** the file defaults (`WALK_FREQ=1.64`, etc.). Don't assume the file constants are live.
3. **`path_speed` ≠ progress.** High `path_speed` can hide negative `net_fwd_speed` (curling /
   turning in place). Always read `net_fwd_speed` + `straightness` together.
4. **In sweeps, `hip_phi` drives BOTH hips and leg phases A/B are zeroed** (overriding the file's
   A/B=45°). Sweeps use raw built-in leg phasing (A=0°/B=180°) + one symmetric hip phase.
5. **μ stance-gate threshold differs by script** (`F_HI=4 N` vs `FN_MIN=1 N`) → μ_req not
   comparable across tools.
6. **Friction priority hack** is load-bearing (§6). Don't drop `geom_priority[floor]=1`.
7. **Model v2 vs v3** differ in init pitch/stand-hip and how forward pitch is imposed (§3). Some
   scripts hardcode one model regardless of `PENGU_MODEL`.
8. **`initcsv` before sharded workers**, or shards race the header.
9. **Body names lie**: `hip-L/R` aren't biomech hips; `right_foot0080___fillet13` is the LEFT
   foot; `leftthighmotor` (the root proxy) isn't a torso; `easytorso` is.
10. **Judged by eye.** The quantitative winner and the "beautiful" gait aren't always the same —
    that's why `gait_report.py` renders video and `BEST_GAITS.md` is curated by hand.

---

## 9. Known doc/code drift (trust the code, not these docstrings)

- `rl/pengu_env.py:11` header says leg amp "[0.0,0.05] m"; the **effective** range is
  [0.023, 0.047] m via the `leg_mid/leg_rng` override in `__init__`.
- `rl/pengu_env.py:22-24` reward summary lists an "alive" bonus and a flat upright penalty that
  **don't exist in code** — roll/pitch are dead-band penalties; the only survival signal is the
  −5 fall penalty.
- `rl/train_curriculum.py:4-5` docstring curriculum thresholds differ from the real
  `curriculum_range` function (`:40-48`) — trust the function.
- `results/BEST_GAITS.md` note: `net_fwd_speed`/`straightness` were added in the *fine2* sweep;
  older `fine1` and the first full-grid CSV headers **lack those columns**.

---

## 10. Quick reference — where to look for X

| I want to… | Go to |
|---|---|
| understand the research goal | `penguin plan for summer.pdf`, §1 above |
| tune / read the open-loop gait | `gait_config.py` |
| find a good gait (sweep→CMA→report) | `physics/gait_sweep.py`, `cma_search.py`, `heatmaps.py`, `gait_report.py` |
| run the friction/surface paper experiment | `physics/friction_study.py` → `results/friction_study/penguin_configs.csv` |
| inspect friction cone / GRFs of one gait | `physics/analyze_gait.py`, `grf_friction_probe.py` |
| see the current best gaits | `results/BEST_GAITS.md` |
| train / eval the RL policy | `rl/train_penguin.py`, `rl/penguin_metrics.py`, `rl/models/*.zip` |
| debug a pose interactively | `walk_pengu.py` (MuJoCo viewer + decoded readout) |
| understand a metric's exact formula | §5 above, then `physics/gait_sweep.py::run_trial` |

---

*If a fact here conflicts with the code, the code wins — flag the drift and update this file.*
