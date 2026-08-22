# RL ice arm — experiment definition & frozen configuration (2026-08-20)

Companion to `rl_grid4/` (grid4_rl_env.py, train_grid4.py, eval_grid4_policy.py,
render_grid4_policy.py). Run outputs: `rl_grid4/runs/<tag>/` with `ckpts/` and
`videos/` subdirs, `diag.csv` and eval CSVs at the run root. The old `rl/`
directory is the legacy CPG stack — do not mix.
Working agreement: `pengu_mujoco/CLAUDE.md` (measurement only, Ben draws
conclusions, no species labels, no AI attribution, branch `friction-experiments`).

## Experiment

One scratch-training arm on the COM-1.31 body (total mass 2.2724 kg constant):

- mu ~ U(0.1, 0.4) per episode, NOT observable by the policy
- vx_cmd = 0.47 m/s, fixed
- >= 4 seeds x 3M steps after Gate 0 passes

Why 0.47: the c6 (kappa=2, COM-1.31) K=5 net_fwd_mean ceiling at mu=0.1 is
0.4689 (`results/grid4_report/c6/topupK5.csv`); the c3 (kappa=0, same body)
ceiling there is 0.1636. Within the designed-gait sweep family, tracking 0.47
at low mu therefore requires torso use — the commanded speed creates the
pressure and the reward stays torso-neutral. The sweep family does not bound
what direct-control RL can find: a non-torso solution that tracks 0.47 on ice
would be a reportable result, not a failure. Note kappa0 reaches 0.4901 at
mu=0.3, so the pressure binds at the low-mu end of the episode distribution.

E1 (0.49 wide-mu arm) and E3 (distill-retention) from `rl_design_memo.md` are
shelved pending this arm's results. The 1.05 body (same 2.2724 kg, via
`apply_com_variant`) is a later extension.

## Environment contract (frozen after Gate 0)

- Model: PENGU_MODEL=1.31, `models/pengu1_31/scene.xml`,
  `grid4_sweep.apply_com_variant(model, 1.31)`. Asserted at construction:
  COM slide +8.73 mm, total mass 2.2724 kg, mujoco 3.8.x. Verified 2026-08-20.
- Action: 5-dim [-1,1] @ 50 Hz -> position targets over the FULL ctrlrange,
  canonical order `gait_config.ACTUATORS` = [hip-L, hip-R, crank1-R, torso,
  crank1-L] (crank1-R drives joint crank2_R). First-order filter alpha=0.2.
  Position control: matches the XM430 servo interface and the actuation layer
  every designed-gait sweep used.
- No leg-phase structure of any kind: no oscillator, no clock input, no
  imposed left-right coupling. Alternation must be learned.
- Obs 36-dim (scaled to ~[-1,1]): projected gravity(3), base angvel(3), base
  linvel body-frame(3), driven joint pos(5), driven joint vel(5), actuator
  torque(5), passive slider pos+vel(4), foot contact flags(2), last action(5),
  vx_cmd(1). Excluded: clock/phase, mu, world yaw/position, root height.
  Body-frame math uses the com_wiper self-calibration (root=leftthighmotor;
  easytorso local +y points world-down at neutral).
- Episode: 10 s @ 50 Hz. Fall = root z<0.08 or |roll|>60deg or |pitch|>60deg;
  fall reward -5 and terminate.
- Reset: mu via `friction_utils.set_floor_friction` (floor priority hack);
  stand pose + jitter yaw +-5deg / pitch +-3deg / lateral +-1 cm; then SETTLE:
  hold stand targets until rocking decays (max|qvel|<0.3, 0.3-1.0 s), resample
  the jitter draw if the robot topples during the hold (staged-start analog;
  applies to training AND eval episodes).

## Reward (frozen)

```
r = 0.8*exp(-(vx-0.47)^2/0.02)      tracking kernel (sigma~0.14)
  + 4.0*max(0,vx) + 2.0*min(0,vx)   forward driver / backward penalty
  - 0.0005*sum_{legs,hips}|tau*qd|  energy, TORSO EXCLUDED
  + 1.0*clip(swing_rate, 0, 0.6)    stepping prior: swing-foot forward speed
  - 0.8*scrub                       stepping prior: stance-foot slip speed
  - 0.01*||a_t - a_{t-1}||^2        action rate (all 5 dims incl torso)
  - 5.0 on fall
```

vx is the root displacement per control step projected on the spawn-calibrated
heading axis. swing_rate/scrub definitions and weights are the validated ones
from `rl/pengu_env.py:250-284`.

Declared prior: with swing/scrub in, "stepping" itself is seeded, not
emergent. The stepping MECHANISM (torso weight-shift vs leg extension) gets no
term in either direction — torso roll is the measured variable. Deliberately
absent: alive bonus, upright/posture terms, bob penalty, single-support
reward (its mechanism IS weight transfer), cadence terms, yaw penalty (add as
a minimal amendment only if circling shows up, uniformly across arms).

## Domain randomization (training only; eval disables all of it)

- Tier 1 (on): init joint pos +-0.05 rad, small qvel noise; obs Gaussian noise
  (see `_obs`); action delay 1 control step (20 ms).
- Tier 2 (off, `--tier2`): actuator kp and joint damping +-10% per episode;
  root push 1-4 N for 0.1 s every 2-4 s.
- Never randomized: total mass, COM (design variables), forcerange.

## Capability vs preference knobs

- Capability (tunable during Gate 0 ONLY, then frozen, every change logged
  here): filter alpha, action mapping range, ent_coef, init pose, mirror
  symmetry augmentation (not implemented; first resort against degenerate
  gaits), tier-2 DR.
- Preference (never): anything touching torso / weight transfer. The
  swing/scrub weights freeze with the rest — they are declared, not tunable.

Reward change log (Ben authorized autonomous reward iteration 2026-08-20;
torso neutrality is NOT touched by any entry):
- v1 -> v2 (2026-08-20): progress 4*max(0,vx) -> 1*max(0,vx); fall -5 -> -10.
  Frame-strip + component analysis of gate0 v1 runs (with and without settle):
  the 250k->500k reward rise came from r_progress/r_track while ep_len FELL
  (15.4->14.4) -- the dash harvests progress+track+swing from step 1 (~70/ep)
  and survival is priced ~0, so the optimizer loses nothing by falling.
  v2 reprices: per-episode ladder stand(0) < dash(~+4) < step-in-place(~+100)
  < walk(~+750). Runs tagged with reward version (gate0_r2_s0, ...); v1 runs
  kept: gate0_s0_nosettle (v1, no settle), gate0_s0 (v1, settle, cut at 1.1M).

Knob change log:
- 2026-08-22 (REWARD r2 -> r3, declared amendment: hf high-frequency action
  penalty, w=0.5). The no-penalty e2x2 cells converged to ~5 Hz aerial
  mincing (hip-diff zero-crossing ~5 Hz, 42% airborne, 2% double support;
  commanded slew exceeds the XM430-W350 no-load speed 4.82 rad/s 56-81% of
  the time on hips/torso — the alpha=0.2 execution filter attenuates 5 Hz to
  37% but the policy repays with ~2.7x pre-filter amplitude). Ben ruled the
  morphology out (not executable on hardware) and chose a reward-side clamp:
  r_hf = -0.5 * ||a_t - act_filt_t||^2 (exactly the content the alpha filter
  rejects; all 5 dims incl torso, same symmetric-regularizer precedent as
  smooth). Calibration (per-step resid^2, mu=0.1, deterministic): a1p1-final
  0.811, a2p0-final 0.303, c6 teacher 0.133 -> at w=0.5 c6 pays 7% of its
  positive reward (10% gate: pass), a1p1 54%, a2p0 16%. Known softness: the
  16% figure for a2p0-style gaits; w tunable per-round via --rw hf=.. during
  the iterate loop, frozen once the recipe freezes. Execution layer, obs,
  eval protocol unchanged. Runs tagged r3 (e2x2hf/*).
- 2026-08-21 (mu curriculum, declared amendment per the rescue ladder):
  e2 stage A runs at fixed mu=0.4 (easy end of the arm's range); stage B
  runs the full U(0.1,0.4). Under the full range from scratch, stage A
  dash-locked (1M steps, ep_len 16-18, fall 1.00, sigma shrinking) — on
  low-mu episodes survival income is invisible again. Applied uniformly to
  all arm seeds.
- 2026-08-21 (curriculum c1, declared amendment per the rescue ladder):
  vx_cmd performance-gated ramp 0.12 -> 0.47 (+0.05 when recent mean ep vx
  >= 0.6*cmd and fall_rate <= 0.5, checked every 25k steps). Basis: r2a1e1
  both seeds converged to a "safe creep" (survive 24 s deterministic, net_fwd
  0.02-0.05 vs pass line 0.05; s0 pass 0/5, s1 pass 2/5 at mu=0.7). The
  tracking kernel (sigma 0.14, center 0.47) prices every speed below ~0.2 at
  ~zero, so from creep speed there is no kernel gradient — the ramp keeps the
  kernel reachable at every stage. Eval commands 0.47 fixed, unchanged.
- 2026-08-21 (exploration, Gate 0 period): log_std_init 0 -> -1.0 (e1).
  Probe: the robot is open-loop unstable under ANY uncorrelated action noise
  (zero-mean policy survives 0.7 s at sigma 0.85, 2.7 s even at sigma 0.15),
  so at PPO's default init sigma~1 no rollout ever survives and the dash is
  the only harvestable strategy — reward pricing (v2) alone could not fix it.
- 2026-08-21 (action mapping, Gate 0 period): cranks narrowed to -1.2+-0.6 rad
  (a1), reset settles at the working stance (stance-angle scan: -1.2 most
  topple-robust, 7/8 seeds). Hips/torso untouched.
- 2026-08-20 (init pose, Gate 0 period): added reset SETTLE + topple-resample.
  Finding: gate0_s0 (no settle) converged to a lunge local optimum (3M steps,
  ep_len flat ~28, 100% fall, vx saturated at cmd; eval tier 1 no-walk at
  mu=0.7). Zero-action probe: without settle 2/5 jitter draws toppled on their
  own within ~1.1 s (episodes doomed at t=0); with settle 10/10 stand 10 s.
  gate0_s0 run kept as the no-settle baseline record.

## Gate 0 — PASSED 2026-08-21 (config frozen)

Record: gate0_r2a1e1c2_w_s1 frozen eval at mu=0.7: pass 3/5 (net_fwd
0.047-0.246, mean 0.146; heading 0.82-0.96; survived 4/5; torso roll RMS
44-50 deg; eff_kappa 2.2-5.3). Tier 3 readout. gate0_r2a1e1c2_w_s0: pass 2/5
(net_fwd mean 0.043). Full run archive under rl_grid4/runs/ (nosettle / r2a0
/ r2a1 / c1 / c2-scratch dead ends kept for the record).

FROZEN training protocol (the validated recipe; applies to every arm seed,
see run_e2_arm.sh): stage A = from scratch, fixed vx_cmd 0.47, 3M steps
(reward v2, action a1, exploration e1, settle reset) -> stage B = warm-start
from A + curriculum c2, 3M steps. All capability knobs now frozen at the
values in the logs above; the r3 (swing weight) candidate was NOT needed and
is not applied.

## Gate 0 (original definition)

mu fixed 0.7 (+-5%), vx_cmd 0.47, 2 seeds x 3M steps. Pass = frozen-eval pass
rule (survive AND heading_align>0.5 AND net_fwd>0.05) for at least one seed.
If full-range crank mapping cannot learn locomotion, narrow the crank mapping
toward the measured effective region (about -1.5 +- 0.4 rad, the
`train_penguin_crank_fix.py` finding) and log it above. Lunge-trap signature
watched in diag.csv: persistently low ep_len with high per-step progress.

## Training / diagnostics

Validated hyperparams: SB3 PPO MlpPolicy [256,256], lr 3e-4, gamma 0.99,
lambda 0.95, clip 0.2, ent 0.005, target_kl 0.03, n_steps 1024, batch 4096,
n_epochs 5, SubprocVecEnv 8-16, CPU, torch threads 2.
`diag.csv` every 250k steps: per-component episode means, per-dim policy sigma
(torso sigma collapse = exploration death), torso roll RMS, single_frac,
ep_len, fall rate. Rescue ladder (in order, logged): more seeds -> ent 0.02 ->
energy-term A/B -> anything else goes back to Ben.

## Frozen evaluation & three-tier report

24 s trials, mu in {0.1, 0.2, 0.3, 0.4} +-5%, 5 seeded repeats, eval_mode env
(pose jitter only), measurement window from t=2 s. Metrics per trial: survive,
net_fwd (displacement on mean heading / window), heading_align, torso roll
RMS, root roll RMS, effective kappa (LS slope of torso world roll on hip-axis
roll), single_frac. Pass rule as above. Designed K=5 baselines exist at
mu=0.1 and 0.3 (c3/c6 topupK5.csv); 0.2/0.4 are RL-only characterization.

Per-seed tiers (tiers 1 and 2 are never merged):
1. no-walk — pass_rate < 0.6 at any mu; zero standing on torso claims
2. walk, torso silent — median torso roll RMS < 5 deg (cut reported, re-cuttable)
3. walk, torso active — median torso roll RMS >= 5 deg

## Commands

```
cd pengu_mujoco
python rl_grid4/train_grid4.py --mode gate0 --seed 0            # gate (3M)
python rl_grid4/train_grid4.py --mode e2 --seed 0               # ice arm (3M)
python rl_grid4/train_grid4.py --mode gate0 --seed 0 --smoke    # 50k check
python rl_grid4/eval_grid4_policy.py rl_grid4/runs/e2_s0/ckpts/final.zip
python rl_grid4/render_grid4_policy.py rl_grid4/runs/e2_s0/ckpts/final.zip --mu 0.1
```

Compute etiquette: check `docs/grid4_fleet_memo.md`; Mac currently also runs
the c4 topup — train with `nice -n 10`, 8 envs.
