# RL handoff memo — gait emergence on the COM-1.31 body (GRID-4 companion)

Audience: the agent implementing the RL phase. Everything below is ALREADY DECIDED
with Ben — do not re-litigate design choices marked FIXED; open items are marked.
Working agreement: `pengu_mujoco/CLAUDE.md` (no species labels; no AI attribution in
commits; branch `friction-experiments`; Ben draws conclusions).

## 0. Purpose (one paragraph)

The GRID-4 sweeps measured designed gaits: open-loop CPG × torso strategy κ∈{0,2} on
three COM bodies. The RL phase asks whether the interesting torso behavior **emerges**
(train from scratch) and whether it **survives** (imitate a designed gait, then
fine-tune with a torso-agnostic reward). Start with the COM-1.31 body: it has the
strongest designed-gait contrast (ice ceiling κ2/κ0 = 3.5×, see
`results/grid4_report/`).

## 1. Experiment matrix (FIXED)

| arm | init | μ per episode | vx_cmd | question |
|---|---|---|---|---|
| E1 scratch-generalist | random | U(0.1, 0.7) | 0.49 m/s | does torso use emerge? |
| E2 scratch-ice | random | U(0.1, 0.4) | 0.16 m/s | does it emerge on ice? |
| E3 distill-retention | BC of a designed κ2 gait | U(0.1, 0.4) | 0.16 m/s | does torso use survive task-only fine-tuning? |

vx_cmd values are the K=5 net_fwd_mean ceilings of the 1.31 body (c3@μ0.3 = 0.490,
c6@μ0.1 = 0.163) — deliberately NOT the inflated K=1 numbers. Multi-seed everything
(≥4 seeds/arm); PPO locomotion seed variance is the #1 local-optimum "rescue".

## 2. Environment (new file, do NOT reuse rl/pengu_env.py as-is)

`rl/pengu_env.py` is the OLD CPG-modulation env, hardcoded to penguV2 at line 108.
Copy its *machinery* (decimation, SubprocVecEnv usage, reward skeleton), replace the
substance:

- **Model**: `os.environ["PENGU_MODEL"]="1.31"` before importing `gait_config`, load
  `gait_sweep.XML` (resolves to `models/pengu1_31/scene.xml`), then
  `grid4_sweep.apply_com_variant(model, 1.31)` (+8.73 mm slide, mass 2.2724 kg —
  assert both). Engine pin: mujoco 3.8.x (`.sweep_venv`; the launcher gate enforces).
- **NO TorsoKappaPID.** The policy commands the torso directly; κ is a *readout*
  (regression), never a controller.
- **Action (5-dim, 50 Hz)**: `[-1,1]^5` → position targets within ctrlranges
  (hip-L/R ±1.5708, crank1-R→joint crank2_R ±3.14, crank1-L ±3.14, torso ±0.7854 —
  note the asymmetric crank naming, copy actuator IDs from `gait_config.build_ids`).
  Light first-order action filter allowed (document the constant).
- **Obs (25-dim)**: projected gravity(3) + base angvel(3) + base linvel in body
  frame(3) + joint pos(5) + joint vel(5) + last action(5) + vx_cmd(1).
  **No clock/phase input.** μ is NOT observable (policy must infer from
  proprioception — part of the "generalist" definition).
- **Reset**: μ ~ U(range) via `friction_utils.set_floor_friction` (floor priority
  hack is load-bearing); pose jitter yaw ±5° / pitch ±3° / lateral ±1 cm (same as
  sweep protocol).
- **Episode**: 10 s @50 Hz training; termination fall = root z<0.08 or |roll|>60° or
  |pitch|>60°; fall reward −5 (the real penalty is losing the remaining episode).
- Axis gotcha (machine-D finding): `easytorso` body +y points world-DOWN at neutral;
  the horizontal heading axis lives on root body `leftthighmotor` (+y). Use
  `gait_liveplot.py` / `com_wiper.py`'s self-calibration pattern for any body-frame
  math.

## 3. Reward (FIXED — the emergence claim depends on it)

```
r = 0.8·exp(−(vx−vx_cmd)²/0.02)            # tracking kernel (σ≈0.14 m/s)
  + 4.0·max(0,vx) + 2.0·min(0,vx)          # forward driver / backward penalty
  − 0.0005·Σ_{legs+hips}|τ·q̇|             # energy — TORSO EXCLUDED (Ben, 2026-08-20:
                                           #   removes anti-torso bias; adds no pro-torso shaping)
  − 0.01·‖aₜ−aₜ₋₁‖²                        # action rate (covers all 5 dims incl torso)
  − 5.0 on fall (terminal)
```

RED LINES: no alive bonus, no gait shaping of any kind (no cadence / roll / clearance /
single-support / slip terms), no torso-use reward. If policies circle in place, a
minimal yaw penalty MAY be added — documented as an amendment, applied to all arms.

## 4. E3 teacher/student specifics

- Teacher = a **scripted** designed gait (CPG + TorsoKappaPID κ=2 on the 1.31 body);
  its 5-dim ctrl targets are directly readable each step — no RL teacher needed.
- **Expert gait selection — do NOT blindly take c6's champion**: it walks 75° off-axis
  (heading drift finding, `docs/grid4_analysis_session_2026-08-20.md` §5). Pick from
  c6 μ=0.1 robust list (`results/grid4_report/c6/top_gaits.csv`) requiring
  `head_mean ≥ 0.9` in the map row, then verify with `physics/gait_probe.py`
  (straight, stepping, torso roll RMS ≈ 20°+). Use the staged start
  (`staged_start_probe.py` pattern) for clean rollouts.
- BC data: rollouts across μ∈[0.1,0.4] × pose jitter × small action noise (DART-style),
  ~500k (obs, ctrl) pairs @50 Hz; supervised MSE into the SAME net as E1/E2.
  Gate: the clone must reproduce the gait under the frozen eval (§6) before fine-tuning.
  Known risk: the expert is time-indexed, the student must infer phase from state; if
  cloning fails, stack 2–3 obs frames (declare as amendment).
- Fine-tune: 5–10M steps PPO, reward §3 verbatim, NO KL anchor — free drift is the point.
- **Headline figure: retention curve** — measured torso roll RMS (and effective κ) vs
  fine-tune steps, checkpoint every 250k.

## 5. Training (FIXED hyperparams, inherited from the validated CPG trainers)

SB3 PPO, MlpPolicy [256,256], lr 3e-4, γ 0.99, λ 0.95, clip 0.2, ent 0.005,
target_kl 0.03, n_steps 1024, batch 4096, n_epochs 5, SubprocVecEnv 8–16 envs,
CPU-only (GPU adds nothing at this scale), `torch.set_num_threads(2)`.
3M steps ≈ 1–2 h on 8 cores. E3 fine-tune 5–10M.

Diagnostics to log every 250k steps (this is how local optima get caught):
each reward component separately; per-dim policy σ (torso σ collapse = exploration
death); torso action amplitude; measured torso roll RMS; eval net_fwd. Rescue ladder:
more seeds → ent_coef 0.005→0.02 → energy-term A/B (drop it entirely, 1–2 h, to test
Ben's "does the energy term suppress torso" worry) → vx_cmd or μ curriculum
(declared amendment). "RL never uses the torso and loses to the designed envelope"
is a REPORTABLE RESULT, not a failure — do not shape it away.

## 6. Evaluation (comparability contract — do not deviate)

Freeze policy → run the FROZEN sweep protocol: 24 s trials, μ ∈ {0.1,0.3,0.5,0.7}
exact ±5% jitter, 5 seeded repeats, same pass rule (survive ∧ heading>0.5 ∧
net_fwd>0.05), via `gait_sweep.run_trial` with the policy driving `data.ctrl`
(50 Hz hold between control steps). Report the same 12-col aggregates + rich metrics
(`ds_move_frac`, slip, clearance, single_frac) + **effective κ** (regress torso world
roll on hip-axis roll over the walk window) + straightness/heading (post the drift
finding, always report heading explicitly). Baselines for comparison: designed-gait
K=5 rows in `results/grid4_report/c3|c6/topupK5*.csv` and `finalists*.csv`.

## 7. Deliverables

1. `rl/grid4_rl_env.py`, `rl/train_grid4.py`, `rl/collect_expert.py`, `rl/bc_init.py`,
   `rl/eval_grid4_policy.py` — committed, no AI attribution, branch friction-experiments.
2. Per arm × seed: training curves (component-wise), eval table, demo mp4 (side+back,
   μ=0.1 and μ=0.7), effective-κ number.
3. E3: retention curve figure + before/after demos.
4. A session memo in `docs/` in the style of `grid4_analysis_session_2026-08-20.md`
   (corrections-first if anything gets retracted).

## 8. Compute etiquette

Check `docs/grid4_fleet_memo.md` for who is busy. Currently: Mac runs the c4 topup
(~4-5 d), rml3 runs c2 (~2 d). rml2 is free after c1 (use `SWEEP_NICE=19` — shared
with Isaac Lab). Smoke first: 50k steps, confirm reward components move and nothing
NaNs, before any multi-seed launch.
