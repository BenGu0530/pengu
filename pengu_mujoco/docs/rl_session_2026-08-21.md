# RL session 2026-08-20/21 — ice arm: recipe iteration, Gate 0, first arm seeds

Machine: Mac (alongside the c4 topup). Code: `rl_grid4/`. Companion spec:
`rl_e2_ice_memo.md` (frozen config + full amendment/knob log). Style per
working agreement: numbers only, corrections first, no verdicts.

## 0. Corrections

- **(added post-session, from machine D's rl_session_2026-08-21_machineD.md)**
  Round-6 claim "c2 from scratch: stand/rock, no path to locomotion" was
  premature: that run was cut at 1.25M. Machine D ran the same config to the
  full 3M and reached vx 0.224 (curriculum hit cmd 0.47 at 2.45M) — i.e.
  scratch+curriculum can produce locomotion given the full budget; the
  warm-start stage may not have been necessary. The two-stage recipe stays
  the recorded protocol for the arm data in §3, with this caveat.
- **Clarification re machine D's open item on the curriculum restart**: in
  the frozen recipe, stage B STARTING the ramp at cmd=0.12 on top of the
  warm-started policy is the designed behavior (this session, round 7), not
  an accidental resume; the `--init-from`-resumes-mid-curriculum case that
  machine D fixed with --cmd0 is a genuine bug for continuation runs.
  Whether the Gate-0 records stand as-is: Ben's call.
- **Machine D's convergence finding applies to §3**: at 6M steps vx
  oscillates between fast/fragile and slow/safe attractors; seed spread is
  the same magnitude as within-run swing. The §3 table (3M+3M) carries the
  same caveat: arm-level comparisons are not interpretable until stability
  is addressed.

- **`rl_design_memo.md` §1 has the c3/c6 ice numbers swapped.** Verified from
  `results/grid4_report/*/topupK5.csv`: at mu=0.1 (K=5 net_fwd_mean) c6
  (kappa=2, COM-1.31) = 0.4689 and c3 (kappa=0) = 0.1636 — the memo wrote
  "c6@mu0.1 = 0.163". The original E2 vx_cmd=0.16 therefore sat AT the
  kappa=0 ceiling; the arm now commands 0.47 (c6 ceiling, rounded), which the
  kappa=0 designed family does not reach at mu=0.1.
- The earlier training-diag `vx` column overstates cruise speed (episode-mean
  weighting favors short fast episodes); frozen-eval net_fwd is the
  comparable number.

## 1. Experiment as now defined (differs from rl_design_memo)

- One arm running: COM-1.31, total mass 2.2724 kg, mu ~ U(0.1,0.4) per
  episode (not observable), vx_cmd = 0.47, >=4 seeds. E1 and E3 shelved
  (Ben, 2026-08-20). 1.05 body later.
- Obs 36-dim (adds actuator torque, passive slider pos/vel, foot contacts to
  the 25-dim memo contract). Action: 5-dim position targets @50 Hz.
- Reward v2 + declared stepping priors (swing/scrub, Ben 2026-08-20). No
  term in any direction on torso / weight transfer, all stages (energy term
  excludes torso). Full reward table: rl_e2_ice_memo.md.
- Scaffolding (all logged as amendments): reset settle; crank mapping
  narrowed to -1.2+-0.6 rad (imports the crank_fix workspace finding);
  log_std_init -1.0; two-stage per seed = stage A scratch at fixed mu=0.4,
  cmd 0.47 (3M) -> stage B warm-start + performance-gated vx_cmd ramp
  0.12->0.47 (c2) under full U(0.1,0.4) (3M). Claim scope: stepping is
  seeded and staged; the STRATEGY (incl. torso use) carries no reward term.

## 2. Recipe iteration log (7 rounds, all dead ends archived in rl_grid4/runs/)

| round | config | outcome (diag) | diagnosis -> next |
|---|---|---|---|
| 1 | memo reward, no settle | lunge: ep_len 28, fall 1.0, vx=cmd | survival priced 0 -> reward v2 (progress 4->1, fall -5->-10) |
| 2 | v2, settle | same lunge | probe: open-loop unstable at ANY sigma (0.7 s @0.85; 2.7 s @0.15) -> a1 + e1 |
| 3 | + a1 crank band | same lunge | exploration sigma, not workspace -> e1 |
| 4 | + e1 (sigma0 0.37) | balance first; "safe creep": 24 s survival, net_fwd 0.02-0.05 (pass line 0.05) | kernel prices 0.05-0.3 at ~0 -> curriculum |
| 5 | c1 vx ramp (speed gate) | stand deadlock @1M: kernel tail pays 0.39/step at vx=0 | c2 survival gate (receding carrot) |
| 6 | c2 from scratch | stand/rock, no path to locomotion (r_swing 0.06) | warm-start from the creep policy |
| 7 | w1 = creep + c2 | **Gate 0 pass** | frozen recipe |

Gate 0 record (mu=0.7, 24 s, deterministic, 5 reps): gate0_r2a1e1c2_w_s1
pass 3/5, net_fwd 0.047-0.246 (mean 0.146), heading 0.82-0.96, torso roll
RMS 44-50 deg, eff_kappa 2.2-5.3. s0: pass 2/5, net_fwd mean 0.043.

## 3. Arm results — FINAL, all 4 seeds (frozen eval: 24 s, mu +-5%, 5 reps, deterministic)

net_fwd mean (pass/5) | median torso roll RMS deg | median eff_kappa, per mu.
Designed K=5 baselines (c3|c6 topupK5): mu0.1 kappa0=0.164 kappa2=0.469;
mu0.3 kappa0=0.490 kappa2=0.242.

| mu | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| 0.1 | 0.041 (1) 26.3 k2.3 | 0.044 (2) 30.0 k3.0 | **0.180 (4) 38.9 k2.7** | 0.053 (2) 52.7 k4.1 |
| 0.2 | 0.034 (1) 26.5 k2.4 | 0.018 (0) 33.2 k3.2 | 0.253 (5) 34.7 k2.2 | 0.083 (5) 59.9 k4.8 |
| 0.3 | 0.114 (5) 25.8 k2.4 | 0.025 (1) 38.2 k3.2 | **0.344 (5) 30.7 k1.1** | 0.170 (5) 57.3 k4.6 |
| 0.4 | 0.138 (5) 25.7 k1.7 | 0.017 (0) 42.6 k3.2 | 0.095 (4) 29.0 k2.2 | 0.092 (3) k4.2 |

Three-tier (strict all-mu pass>=0.6; tiers 1 and 2 never merged):
- seed 0: tier 1 (walks mu 0.3/0.4 at 5/5; fails 0.1/0.2). hip_corr -0.32..-0.41
  (alternating gait).
- seed 1: tier 1 (passes nowhere; best at mu 0.1).
- **seed 2: tier 3 — walk, torso ACTIVE** (median RMS of passing trials
  33.7 deg). At mu=0.1 its 0.180 exceeds the kappa0 designed-family ceiling
  0.164; at mu=0.3 its 0.344 exceeds the kappa2 baseline 0.242 and sits
  below kappa0's 0.490.
- seed 3: tier 1 by the mu=0.1 cell only (0.4 pass); passes 0.2/0.3/0.4.
  Largest torso use (RMS 53-60 deg, eff_kappa 4.1-4.8).

Readouts, stated without interpretation:
- 0/4 seeds landed in tier 2 (walk + torso silent). Every seed that walks
  anywhere does so with torso roll RMS >= 26 deg and eff_kappa 1.1-4.8.
- 1/4 seeds (s2) covers the full ice range; 3/4 walk at mu>=0.2 or 0.3.
- RL vs designed envelope: below it everywhere except s2@mu0.1 vs kappa0
  (0.180 > 0.164) and s2/s3@mu0.3 vs kappa2 (0.344/0.170 vs 0.242 — s2 only).
- Behavior note from demos: high-cadence short-stride stepping; swing income
  collects without much translation (candidate reward lever left to Ben).

## 3a. Best-checkpoint results (added 2026-08-22; supersedes final.zip rows in §3)

Selection-rule amendment: rml2's checkpoint sweep (docs/ckpt_sweep_rml2_memo)
showed final.zip samples the late-training oscillation at an arbitrary phase
(best-vs-final pass deltas +2..+10 of 12). Adopted procedure: select each
seed's checkpoint by frozen-eval pass count (3 reps), then CONFIRM on
independent trial seeds (--trial-seed-base 50000, 5 reps). Confirmation
numbers below are the reportable ones (net_fwd mean, pass/5):

| seed@ckpt | mu0.1 | mu0.2 | mu0.3 | mu0.4 | tier | eff_k med |
|---|---|---|---|---|---|---|
| s0@1.5M | 0.386 (5) | 0.364 (5) | 0.339 (5) | 0.088 (3) | 3 | 0.44 |
| s1@750k | 0.085 (5) | 0.111 (5) | 0.057 (4) | 0.059 (2) | 1 | 0.63 |
| s2@3.0M | 0.050 (2) | 0.251 (5) | 0.483 (5) | 0.085 (3) | 1 | 2.83 |
| s3@2.5M | 0.125 (5) | 0.120 (4) | 0.282 (5) | 0.241 (5) | 3 | 3.14 |

Readouts: tier 3 = 2/4 seeds (was 1/4 on final.zip); tier 2 still empty
(torso RMS 34-58 deg on every seed). s0@1.5M mu0.1 mean 0.386 = 2.4x the
kappa0 ceiling 0.164, single trials up to 0.574 (kappa2 ceiling 0.469);
s2@3.0M mu0.3 0.483 vs kappa0's 0.490 there. Effective-kappa is strongly
seed-dependent (0.44 / 0.63 / 2.83 / 3.14). Selection-vs-confirmation
shrinkage exists but is moderate (s0 pooled 0.356 -> 0.29). CSVs:
runs/e2/s*/stageB/eval_bestckpt_confirm.csv.

## 3c. Comparability bar (C4 measurement, 2026-08-22)

c6 champion replayed under the RL env's exact conditions (reset/settle/50 Hz
hold, full-range action bypass), 20 seeds, walk-window vx:

| mu | fall rate | vx mean | median | range | sweep figure |
|---|---|---|---|---|---|
| 0.1 | 0.15 | 0.392 | 0.391 | 0.341-0.450 | 0.4689 (K=5) |
| 0.3 | 0.30 | 0.084 | 0.098 | -0.131-0.240 | 0.242 |

Under identical measurement conditions the mu=0.1 bar is ~0.39. Side by
side with §3a: s0@1.5M confirmed 0.386 (5/5 pass, no falls) vs c6 0.392
(15% falls); at mu=0.3 s2@3.0M 0.483 vs c6 0.084 (c6 is ice-specialized).
Raw: rl_grid4/c6_under_rl_env.txt.

## 3b. runs/ layout (renamed 2026-08-21; old tag-soup names -> readable tree)

```
rl_grid4/runs/
  e2/s{0..3}/{stageA,stageB}/     # the arm; stageB holds eval_frozen.csv + demos
  gate0/s{0,1}/{stageA,stageB}/   # frozen-recipe gate rounds (mu=0.7)
  archive/                        # dead ends, readable names:
    gate0_v1_nosettle_s0, gate0_v1_settle_s0     (lunge, reward v1)
    gate0_v2_fullrange_s0, gate0_v2_crankband_s0 (lunge, v2 pricing)
    gate0_c1_standdeadlock_s0, gate0_c2_scratch_s0
    e2_fullmu_stageA_s0                          (dash-lock, pre mu-curriculum)
  gate0_r2nsa1e1c2_s{0,1,2}/      # Thomas (rml3) no-smooth ablation - NOT
                                  # renamed, his naming, different diag schema
```
Old->new: e2_r2a1e1_mu0.4_sN = e2/sN/stageA; e2_r2a1e1c2_w_sN = e2/sN/stageB;
gate0_r2a1e1_sN = gate0/sN/stageA; gate0_r2a1e1c2_w_sN = gate0/sN/stageB.
Video files now carry the run name (demo_e2_s2_stageB_mu0.1_final.mp4).
train_grid4 --name sets run dirs directly; run_e2_arm.sh updated.

## 4. Deliverables / archive

- Per seed: eval_frozen.csv, diag.csv (A and B stages), ckpts every 250k +
  final, demo mp4 at mu=0.1 (s2 also has the mu=0.1 headline demo; s0 also
  mu=0.4). Gate runs likewise.
- All under rl_grid4/runs/, committed. Dead-end rounds kept.
- Compute note: entire arm ran on the Mac alongside the c4 topup (nice 10).

## 5. Single-stage 2x2 probe (no-penalty, r2) — COMPLETE 2026-08-22, archived as the pre-clamp record

Protocol: single-stage from scratch (C7 adopted: no warm-start, no stage A/B),
mu ~ U(0.1,0.4), c2 vx-curriculum 0.12->0.47, 3M steps, seed 0, frozen eval
(final.zip, 5 reps). Cells: band (a1 frozen / a2 = crank 0.0+-1.9) x priors
(p1 = swing/scrub frozen / p0 = swing=0 scrub=0). Runs: rl_grid4/runs/e2x2/.

| cell | mu0.1 pass / net_fwd | mu0.2 | mu0.3 | mu0.4 | tier |
|---|---|---|---|---|---|
| a1p1 | 0.8 / 0.339 | 0.6 / 0.117 | 0.4 / 0.150 | 0.0 / 0.293 | 1 |
| a2p1 | 1.0 / 0.247 | 1.0 / 0.288 | 0.6 / 0.143 | 0.6 / 0.063 | 3 (torso med 27.1 deg) |
| a1p0 | 0.8 / 0.218 | 0.6 / 0.058 | 1.0 / 0.119 | 0.4 / 0.103 | 1 |
| a2p0 | 1.0 / 0.656 | 1.0 / 0.561 | 0.2 / 0.037 | 0.0 / 0.018 | 1 |

(best-ckpt sweep + confirmation NOT run on these: superseded by the r3 rerun
before that stage; final.zip numbers only. eval printed `torso_rms med=nan`
on some tier-1 rows — prints only over passing trials; not a data bug.)

Morphology probe on a1p1 final (mu=0.1, deterministic): vx 0.328, hip-diff
zero-crossing ~5 Hz, contact duty double 0.02 / single 0.56 / airborne 0.42;
commanded slew > 4.82 rad/s (XM430-W350 no-load) 56-81% of the time on
hips/torso. Same probe on the c6 replay: 1.77 Hz, double 0.20 / single 0.41 /
airborne 0.39 (longest airborne run 180 ms) — airborne fraction does NOT
separate the two; frequency and double-support do. Note: c6's CRANK velocity
exceeds 4.82 rad/s 60% of its walk window (max 8.4 rad/s) — if the crank
servo is W350-spec, the designed family shares this feasibility question
(hardware spec numbers needed to settle it).

Ben's ruling 2026-08-22: the 5 Hz + curved-feet morphology is a cheating
answer (robust but not executable / not the target style); rework as an
iterate-until-goal task. -> REWARD r3 (hf penalty, see rl_e2_ice_memo knob
log), full 2x2 rerun under r3 at rl_grid4/runs/e2x2hf/, goal: walks at
mu 0.1-0.4, stride <= ~2 Hz, no aerial mincing, torso free, speed as high as
possible. Iteration rounds logged here.

## 6. r3 iterate log

- Round 1 (2026-08-22, launched): e2x2hf 2x2, w_hf=0.5 (calibration: c6 tax
  7% of positive reward, a1p1-5Hz 54%, a2p0 16%). Watch: does a1p1-style
  high-freq reappear; sigma_torso for penalty-induced sigma collapse; r_hf
  trend in diag.
  - Round 1 cell a1p1 DONE (final eval: mu0.1 0.0/0.018, mu0.2 1.0/0.106,
    mu0.3 1.0/0.085, mu0.4 0.6/0.069, tier 1 by the mu0.1 fail). Morphology
    (deterministic, seed 0): the 5 Hz mincing is GONE (freq 2.5-2.65 Hz,
    airborne 2-6% vs 42% under r2, r_hf paid ~0.01/step) — but the mu0.2-0.4
    passes are HELD-LEAN walking: torso parked at -53 deg (roll_mean -53.2 ~=
    RMS 53.5), legs shuffle under the lean; mu0.1 is a 69%-double-support
    shuffle at vx 0.029. Mid-run note: the 2M ckpt walked mu0.1 at 0.114 with
    freq 3.8 Hz — final drifted away from it (stopping-rule effect again).
    Frames + video pushed to Ben (a1p1hf_final_mu0.2.mp4).
  - Round 1 mid-run finding (a2p1 @1.5M probe): mu0.1 vx 0.246 at freq 4.5 Hz,
    airborne 34%, falls at 4.9 s — the wide band BUYS BACK high frequency.
    Cause: r_hf is computed on the NORMALIZED action; with crank half 1.9 vs
    a1's 0.6 the same physical crank motion pays (0.6/1.9)^2 ~ 1/10 the tax.
    Round-2 candidate fix: price r_hf in physical units (resid * ctrl_half,
    fixed reference scale) so bands are taxed equally.
  - Round 1 cell a2p1 DONE (final eval: mu0.1 0.4/0.417, mu0.2 0.6/0.141,
    mu0.3 0.4/0.024, mu0.4 0.0/0.013, tier 1). Morphology (final, seed 0,
    deterministic): mu-dependent switch — mu0.1 walks 0.142 at 1.55 Hz,
    torso quiet (RMS 9.9, mean -5.1), double support 63%; mu0.2 bounces
    0.300 at 3.1 Hz, airborne 46%, torso RMS 31.5 mean 25.5. The band
    dilution (see mid-run note) lets the bouncy mode persist at mu>=0.2.
  - Round 1 cell a1p0 DONE (final eval: mu0.1 0.8/0.216, mu0.2 0.2/0.017,
    mu0.3 0.0/0.016, mu0.4 0.0/0.020, tier 1). Morphology (final, seed 0):
    mu0.1 vx 0.363 at 3.25 Hz, airborne 28%, torso HELD-LEAN (mean 35.1 ~=
    RMS 36.8); mu0.3 near-stand (vx 0.011). Low-mu specialist, lean cheat.
  - Round 1 cell a2p0 DONE (final eval: mu0.1 1.0/0.448, mu0.2 1.0/0.368,
    mu0.3 0.0/0.033, mu0.4 0.0/0.022, tier 1). Morphology (final, seed 0):
    mu0.1 vx 0.307 at 3.25 Hz, single support 76%, airborne 18%, torso QUIET
    (RMS 15.6, mean 6.6) — the most walk-like cell of round 1; mu0.2 vx 0.279
    at 2.9 Hz but torso parked at -43.6 (lean switch). Demo + strips sent.

  ROUND 1 SUMMARY (all finals, frozen eval pass / net_fwd):
  | cell | mu0.1 | mu0.2 | mu0.3 | mu0.4 | morphology note |
  |---|---|---|---|---|---|
  | a1p1 | 0.0/0.018 | 1.0/0.106 | 1.0/0.085 | 0.6/0.069 | 2.5-2.65 Hz, HELD-LEAN -53 at mu0.2 |
  | a2p1 | 0.4/0.417 | 0.6/0.141 | 0.4/0.024 | 0.0/0.013 | mu0.1 1.55 Hz torso-quiet walk / mu0.2 3.1 Hz bounce |
  | a1p0 | 0.8/0.216 | 0.2/0.017 | 0.0/0.016 | 0.0/0.020 | 3.25 Hz lean-bounce at mu0.1 |
  | a2p0 | 1.0/0.448 | 1.0/0.368 | 0.0/0.033 | 0.0/0.022 | mu0.1 3.25 Hz torso-quiet 76% single support |
  Cross-cell: 5 Hz mincing eliminated everywhere (freq 1.55-3.25 Hz, airborne
  <= 28% vs 42%); recurring cheats now (a) HELD-LEAN torso parked 35-53 deg
  as static counterweight, (b) mu-specialization (low-mu fast / high-mu
  stand), (c) final-vs-best drift persists. No cell passes all mu.

- Round 2 (2026-08-22, launched): REWARD r3b — hf residual scaled to
  band-independent units (x ctrl_half / a1-reference halves; a1 pricing
  unchanged, a2 cranks now pay the same physical tax, previously diluted
  ~10x). Smoke: identical 5 Hz physical crank wiggle pays identical
  -0.0878/step in both bands. Only a2 cells re-run (a1 cells are bit-for-bit
  the same recipe as round 1): runs/e2x2hf2/{a2p1,a2p0}. OPEN QUESTION for
  Ben (red-line adjacent, not acted on): is HELD-LEAN walking a cheat to
  penalize or a legitimate emergent weight-shift strategy? No torso-shaping
  change made either way.

- Round 2 (r3b) KILLED at 250k: learned suicide. Band-fair scaling taxed
  exploration NOISE itself at -0.94/step in the a2 band (scale^2 x sigma^2
  through the cranks) vs early income ~0.6-0.9/step, so dying (-10 ~= 11
  steps of tax) beat living: a2p1 diag ep_len 18, fall 1.00, r_hf -0.941
  (round-1 same point: ep_len 172). Partial run archived at runs/e2x2hf2/.
  Lesson logged: any per-step penalty must be checked against the suicide
  breakeven (penalty x expected ep_len vs fall cost) before launch.

- Round 3 (2026-08-22, launched): REWARD r3c — hf priced on the EXECUTED
  signal (act_filt vs a second alpha cascade; the servo tracks act_filt, so
  commanded noise the filter absorbs is physically harmless) and on
  HIPS+TORSO ONLY. Per-dim audit: cheat HF lives in torso/hips (a1p1 torso
  0.045/dim) while c6's HF lives in the cranks (0.231/dim, hips/torso
  0.003/0.010) — and the honest crank speed limit is an open hardware
  question (Ben: servo model/gearing per joint?), so cranks are unpriced
  for now. Under executed-signal pricing c6's crank would otherwise pay the
  MOST of all streams (casc resid^2 0.476 vs 5Hz-cheat 0.087) — consistent
  with the measured crank slew exceeding the W350 no-load limit 60-69% of
  its walk window; whether that rules the designed family infeasible is a
  hardware-facts question, deferred. w recalibrated 6.0 (3-dim): 5Hz-cheat
  tax 50% of pos reward, a2p0-cheat 25%, c6 9%, noise 0.085/step (suicide
  breakeven ~118 steps). Smoke: hip5Hz -0.044, crank5Hz 0.000, noise
  -0.078/step. Full 2x2 re-run (a1 pricing changed too): runs/e2x2hf3/.
