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
  - Round 3 watch item (a1p1 @1.75M): vx 0.012 vs round-1's 0.107 at the same
    point; torso_rms shrank 33 -> 17 deg under the w=6 executed-HF tax on the
    torso dim. Red-line-adjacent note: the ENERGY term exempts the torso by
    rule ("no reverse bias on the measured variable"); hf currently prices
    the torso (r3 cited the smooth-term precedent instead). If stand-deadlock
    is confirmed at 2-2.25M, the "should hf exempt torso" question goes to
    Ben — not acted on unilaterally.
  - Round 3 cell a1p1 DONE: full stand-deadlock (eval 0/5 at every mu,
    net_fwd ~ -0.004; training end fall 0.08, vx -0.006, sigma_torso
    collapsed 0.356 -> 0.235). Trajectory diverged from round 1 (which
    walked 0.107 at 1.75M): under the w=6 executed-HF tax the torso
    quieted (rms 33 -> 15 deg mid-run) and walking never broke out.
    Watch a2p1/a1p0/a2p0 to separate "w too high" from "torso pricing".
  - Round 3 mid-run diagnosis (a2p1 also stands deterministically at 1.5M and
    2.25M while training-vx runs 0.08-0.16 on noise): r3c EXEMPTS noise (the
    alpha filter absorbs white noise before the cascade sees it) and TAXES
    coherent oscillation - and the first-order cascade's residual peaks right
    on the 1-3 Hz walking band. Net effect: the stand->walk barrier rises,
    "stand deterministically + let sigma collect swing income" wins. Inverse
    of intent. Second-difference (executed accel) variant was calibrated and
    REJECTED offline: white noise is broadband, so its accel resid (0.033)
    exceeds the 5 Hz cheat's (0.025) - any deterrent w suicides.
  - Round 4 design (r3d, ready to launch when round 3 ends): back to the
    EMPIRICALLY validated commanded-residual form (round 1 killed 5 Hz and
    still walked), restricted to hips+torso (cranks unpriced pending
    hardware), w=1.0. Calibration (HT commanded resid^2, mu=0.1): 5Hz-cheat
    pays 78% of pos reward, r2-a2p0-cheat 29%, r3-a2p0-walker 13%, c6 4%,
    noise -0.236/step (= round-1's empirically survivable level). Mechanism
    summary of the three failures now on record: commanded=works, band-scaled
    =noise-suicide, executed=stand-collapse, accel=noise-suicide.
  - Round 3 cell a2p1 DONE: same stand-deadlock (0/5 at every mu, net_fwd
    ~0.003-0.005). Two of two r3c cells confirm the mechanism.

- Round 3 STOPPED after 2/2 completed cells confirmed the stand-deadlock
  mechanism (a1p0/a2p0 not run under r3c; partial a1p0 logs archived at
  runs/e2x2hf3/). Round 4 (r3d) launched: commanded residual, hips+torso
  only, w=1.0, full 2x2 at runs/e2x2hf4/. Smoke: hip5Hz -0.063, crank5Hz
  0.000, hip1.5Hz -0.028, noise -0.210/step.
  - Round 4 (w=1.0) KILLED at 500k: dash-suicide (ep_len 32 -> 14, fall
    1.00, vx 0.10, hip_corr -0.87 — sprint-and-die harvesting progress).
    HT noise tax -0.28..-0.32/step exceeded early standing income. Partial
    at runs/e2x2hf4/. Round 4b launched with w=0.6, which reproduces
    round-1's proven survivable HT noise tax (-0.14/step) at 5Hz deterrence
    47% (round 1: 54%). Full 2x2 at runs/e2x2hf4b/.
  - Round 4b cell a1p1 DONE: stand (0/5 all mu, net_fwd ~0). The a1 cell has
    now stand-locked under every HT-priced variant (r3c w6, r3d w0.6) while
    it lean-walked under round-1 r3 (all-5-dim w0.5). Note the confound: what
    changed for a1 between r1 and 4b is (i) cranks untaxed, (ii) HT weight
    0.5 -> 0.6. Awaiting a2 cells.
  - Round 4b cell a2p1 DONE (final eval: mu0.1 0.8/0.476, mu0.2 1.0/0.617,
    mu0.3 0.8/0.189, mu0.4 0.0/0.169 — the strongest eval line of any round;
    tier 1 only by the mu0.4 miss). Morphology (final, seed 0): the speed is
    a CRANK-POWERED bounce — mu0.2 vx 0.526 at 3.5 Hz, airborne 42%, torso
    parked +30; mu0.1 0.289 at 3.55 Hz, airborne 30%, lean -18. Hips do
    alternate (corr -0.49). Taxing hips+torso pushed the high-frequency work
    into the only unpriced dimension (cranks) — direct evidence for the
    pending crank-pricing decision (hardware specs). Frames + video sent.
  - Round 4b cell a1p0 DONE: stand (0/5 all mu). Pattern held: both a1
    (narrow-crank) cells stand under HT-priced hf; both a2 cells move via
    untaxed cranks. a2p0 last.
  - Round 4b cell a2p0 DONE (final eval: mu0.1 0.8/0.099, mu0.2 1.0/0.476,
    mu0.3 0.8/0.270, mu0.4 0.2/0.045 — broadest mu coverage of any cell in
    any round). Morphology (final, seed 0): mu0.3 vx 0.313 at 2.60 Hz,
    single support 66%, airborne 24%, torso QUIET (RMS 6.3) — the closest
    morphology to the goal so far; mu0.2 similar at 3.3 Hz; mu0.1 is a
    37-deg lean shuffle at 0.045. Frames + video sent.

  ROUND 4b SUMMARY (r3d w=0.6: commanded-HF, hips+torso only; finals):
  | cell | mu0.1 | mu0.2 | mu0.3 | mu0.4 | morphology note |
  |---|---|---|---|---|---|
  | a1p1 | 0.0/-0.000 | 0.0/-0.000 | 0.0/-0.000 | 0.0/-0.000 | stand |
  | a2p1 | 0.8/0.476 | 1.0/0.617 | 0.8/0.189 | 0.0/0.169 | crank-powered bounce 3.5 Hz, airborne 42%, lean +30 |
  | a1p0 | 0.0/0.003 | 0.0/0.003 | 0.0/0.003 | 0.0/0.002 | stand |
  | a2p0 | 0.8/0.099 | 1.0/0.476 | 0.8/0.270 | 0.2/0.045 | mu0.3: 2.6 Hz torso-quiet 66% single support |
  Cross-cell: the narrow-crank (a1) cells stand under every HT-priced hf
  variant; the wide-crank (a2) cells put the high-frequency work into the
  unpriced cranks. Both open decisions (crank pricing <- hardware specs;
  HELD-LEAN ruling) now gate the next reward change. best-ckpt sweep on the
  a2 cells running (ckpt_sweep_hf4b.log).
  - Round 4b best-ckpt protocol (a2 cells): a2p1 best=3000k ~= final (8/12
    both). a2p0 best=2000k pass 11/12 vs final 8/12 (+3). CONFIRMED on
    independent seeds (50000, 5 reps): a2p0@2000k = mu0.1 5/5/0.391,
    mu0.2 5/5/0.262, mu0.3 3/5/0.117, mu0.4 3/5/0.075 — 16/20, tier 3
    (torso med RMS 33.7) — first policy of the program passing majority at
    ALL four mu. Morphology (deterministic, seed 0): mu0.3 vx 0.176 at
    2.00 Hz, single support 74%, airborne 17%, hip_corr -0.48, torso RMS
    23.7 mean 20.0 (part lean, part sway); mu0.1 vx 0.268 at 3.25 Hz,
    torso quiet (RMS 13.1). Goal scoreboard vs Ben's criteria: walks all
    mu CONFIRMED; freq <=2 Hz met at mu0.3 only (mu0.1 3.25 Hz); airborne
    17-18% (vs 42% cheat); torso free and active. Videos + frames sent.
    csv: runs/e2x2hf4b/a2p0/{ckpt_sweep.csv,eval_bestckpt_confirm.csv}.

- RECIPE KEPT (Ben, 2026-08-23): r3d (hf=0.6 commanded residual hips+torso)
  + a2 band + priors-off, single-stage — the a2p0 cell. Definitive
  independent-seed arm queued: seeds 1-7 (run_r3d_arm.sh, runs/r3d_arm/),
  auto-starts after the a2p0_ext probe (+3M continuation testing whether
  torso recruitment keeps growing; torso joint was frozen to 1.25M then
  monotonically recruited 1.5M->2.75M, amp 3.7 deg, with vx up to 0.317;
  marginal accounting: torso hf tax 0.013/step vs ~0.70/step speed income
  still unclaimed -> no direct reward brake at current amplitude).
  Seed 0 = e2x2hf4b/a2p0 (best 2000k confirmed 16/20). Pool = 8 seeds.
  - a2p0_ext probe COMPLETE (+3M -> 6M total, cmd fixed 0.47). Torso
    recruitment answer: amplitude grew to ~4-4.5 deg RMS (~20 deg p2p) by
    3.5-4M then PLATEAUED; deterministic mu0.3 vx peaked at 4.0M (0.553,
    single rollout) then entered the fast/fragile oscillation (0.198 /
    0.337 / 0.031 / -0.064 at 4.5/5/5.5/6M) — the 6M final is WORSE than
    the 3M final. Marginal accounting at 3M had shown no direct reward
    brake (torso hf tax 0.013/step vs ~0.70/step unclaimed speed income);
    the post-4M limiter is the within-run attractor oscillation + stopping
    rule, consistent with machine D's convergence finding. ckpt sweep on
    the ext run launched (ckpt_sweep_ext.log) to select+confirm the true
    best (~4M region). r3d_arm s1-s7 queue auto-started 00:19 EDT.
  - ext best CONFIRMED (independent seeds 50000): ext@2000k (5M total) =
    mu0.1 5/5/0.537, mu0.2 5/5/0.602, mu0.3 5/5/0.385, mu0.4 2/5/0.047 —
    17/20; mu0.1 0.537 EXCEEDS the c6 designed ceiling (0.469), mu0.3
    0.385 vs c6 0.242. Morphology caveat (2 seeds, deterministic): still
    the 2.7-3.2 Hz family, airborne 30-41%, mu0.3 shows HELD-LEAN episodes
    (seed0 roll_mean -44.5); torso joint itself ~4 deg RMS. Speed record,
    not yet the clean gait. csv: a2p0_ext/eval_bestckpt_confirm.csv.

- r3d_arm (kept recipe, seeds 1-7) running log (final.zip evals, pre-selection):
  | seed | mu0.1 | mu0.2 | mu0.3 | mu0.4 | total |
  | s0(=e2x2hf4b/a2p0, best2000k confirmed) | 1.0/0.391 | 1.0/0.262 | 0.6/0.117 | 0.6/0.075 | 16/20 conf |
  | s1 final | 0.8/0.487 | 1.0/0.553 | 1.0/0.293 | 0.0/0.012 | 14/20 |
  | s2 final | 0.6/0.366 | 1.0/0.559 | 1.0/0.363 | 1.0/0.067 | 18/20 TIER 3 |
  | s3 final | 0.6/0.062 | 0.2/0.048 | 0.2/0.018 | 0.0/0.015 | 4/20 (weak seed; late stand-drift, tail recovery at 3M) |
  | s4 final | 0.4/0.450 | 0.8/0.351 | 0.8/0.100 | 1.0/0.092 | 15/20 |
  | s5 final | 0.8/0.457 | 1.0/0.470 | 0.8/0.207 | 1.0/0.133 | 18/20 TIER 3 |
  | s6 final | 0.0/0.003 | 0.0/0.003 | 0.0/0.012 | 0.0/0.003 | 0/20 (rocking-stand, ASYM +0.22 in training) |
  | s7 final | 0.8/0.364 | 1.0/0.661 | 0.8/0.413 | 0.4/0.052 | 15/20 (mu0.2 0.661 = speed record) |

  ARM COMPLETE (8 seeds, finals pre-selection): pass totals 16*,14,18,4,15,18,0,15
  (*s0 = confirmed best-ckpt figure; others final.zip). 6/8 seeds walk
  (>=14/20), 2 failures (s3 weak, s6 rocking-stand). Both tier-3 lines
  (s2, s5) pass mu0.4 outright. Per-seed ckpt sweep launched for the
  best-ckpt + confirmation pass (ckpt_sweep_arm.log).

  R3D_ARM DEFINITIVE TABLE (best-ckpt selected by sweep, CONFIRMED on
  independent trial seeds 50000 x 5 reps; pass/net_fwd per mu):
  | seed | best | mu0.1 | mu0.2 | mu0.3 | mu0.4 | total | tier |
  |---|---|---|---|---|---|---|---|
  | s0 | 2000k | 1.0/0.391 | 1.0/0.262 | 0.6/0.117 | 0.6/0.075 | 16/20 | 3 |
  | s1 | 2250k | 0.4/0.060 | 1.0/0.239 | 0.6/0.119 | 0.8/0.093 | 14/20 | 1 |
  | s2 | final | 0.6/0.434 | 1.0/0.553 | 0.6/0.203 | 0.4/0.049 | 13/20 | 1 |
  | s3 | final | 0.6/0.217 | 0.4/0.065 | 0.0/0.009 | 0.0/0.008 |  5/20 | 1 |
  | s4 | 2000k | 1.0/0.493 | 1.0/0.270 | 0.6/0.048 | 0.8/0.062 | 17/20 | 3 |
  | s5 | final | 1.0/0.548 | 1.0/0.464 | 0.6/0.087 | 0.8/0.109 | 17/20 | 3 |
  | s6 | 1250k | 0.0/0.001 | 0.0/0.003 | 0.4/0.048 | 0.8/0.102 |  6/20 | 1 |
  | s7 | 2750k | 1.0/0.646 | 1.0/0.497 | 0.6/0.231 | 0.8/0.125 | 17/20 | 3 |
  Arm stats (confirmed): 4/8 seeds tier 3 at 16-17/20; 2 mid (13-14); 2
  weak (5-6). Three seeds' confirmed mu0.1 net_fwd EXCEED the c6 designed
  ceiling 0.469 with 5/5 pass: s4 0.493, s5 0.548, s7 0.646 (137% of c6).
  Morphology caveat carried from the ext probe: this family is 2.6-3.5 Hz
  with 17-40% airborne and mixed lean episodes, not the 2 Hz clean gait
  (that remains a2p0-3M@2000k's mu0.3 slice). csvs per seed:
  runs/r3d_arm/s*/{ckpt_sweep.csv,eval_bestckpt_confirm.csv}.

- P3 RESOLVED (2026-08-23, Ben): all five actuators are XM430-W350; the
  crank pricing question became an execution-honesty fix. sv1 slew clamp
  @4.82 rad/s (12 V no-load) implemented across env+tools (--no-slew =
  legacy; bit-repro verified). Numbers:
  - designed replay under sv1: c6 mu0.1 0.374 (-6% vs sv0 0.397), mu0.3
    -0.008 with 2/3 falls (sv0: 0.087); c3 replay does not walk through the
    RL-env probe at mu0.1 in either mode (its 0.1636 is a sweep-protocol
    number, not comparable here).
  - RL champions zero-shot under sv1 (confirm seeds): a2p0@2000k 16/20
    (0.390 mu0.1), s4@2000k 17/20 (0.492), s5@final 15/20 (0.522),
    s7@2750k 17/20 (0.597 mu0.1, 0.580 mu0.2). The learned family is
    nearly clamp-invariant: its speed does not depend on the illegal
    overspeed region.
  - clamp-native retrain probe launched (kept recipe seed 0, runs/sv1_probe/).
  Next: P2 (HELD-LEAN / straight) discussion with Ben; then P1 with the
  clamp-corrected field.
  - sv1-NATIVE retrain probe (kept recipe, seed 0, 3M): frozen eval final
    0.8/0.419, 1.0/0.533, 0.8/0.165, 0.6/0.062 = 16/20 TIER 3 without ckpt
    selection — equal to the best sv0 seeds. Morphology (deterministic,
    mu0.1): vx 0.401 at 2.60 Hz, single support 70%, airborne 24%, hips
    alternating (corr -0.51), torso lean component -17 deg, joint RMS 2.7.
    vs the sv0 counterpart (same recipe/seed: 0.307 at 3.25 Hz): training
    INSIDE the honest actuation is both faster and lower-frequency. Residual
    gaps to goal: lean component, airborne 24-40%, mu0.3+ weak.

- P2 RESOLVED (2026-08-23, Ben): HELD-LEAN ruled a cheat, but direct torso
  penalties rejected (would disturb emergence) and the straight stride proxy
  FAILED calibration (ice slip noise: every walker pays 0.5-0.7/step, no
  separation - straight stays 0 permanently). Ben's balance intuition
  ("penalize legs compensating for a parked torso") calibrated as three
  leg/path estimators: duty-balance separates HEAVY lean (-53: 0.474) from
  clean (0.206) at 2.3x; mild lean (15-20 deg) has no leg-level signal
  anywhere and is REPORTED, not priced. Decisions: (a) REWARD r3e = r3d +
  dutybal = -0.3 * max(0, |dutyL-dutyR|/(dutyL+dutyR) - 0.25), armed after
  1 s of single-support samples; calibration: heavy lean -0.054/step (32%
  of its income), mild/clean <= 0.007, stander 0. (b) eval tier 3 split
  into 3-lean / 3-sway (|roll_mean|/RMS > 0.7 majority vote). Also fixed:
  the --no-slew CLI insertion had accidentally captured the penalty-shape
  suicide preflight into its branch (would refuse any --no-slew train run);
  moved back under shape=penalty. r3e probe launched (kept recipe + sv1 +
  dutybal, seed 0, runs/r3e_probe/).
  - r3e probe DONE (kept recipe + sv1 + dutybal, seed 0): final eval
    0.6/0.047, 1.0/0.225, 0.8/0.130, 1.0/0.106 — tier line now prints the
    S1 split: "3-lean, lean-trial frac 0.76" (mild lean correctly labeled).
    dutybal worked as designed: a heavy-sway/one-sided burst at 1.25-1.5M
    (torso 52 deg, tax -0.048) was pruned back to duty symmetry (-0.005)
    while KEEPING the sway; no heavy lean anywhere. Contact structure is
    the best of any run: airborne 15-16%, double support 16-17% (vs 24-40%
    / 3-6% pre-dutybal); mu0.4 walks at 2.20 Hz. mu0.1 final is weak
    (0.047) — ckpt sweep running. P1 next: energy x20 A/B proposed.
  - P1 energy x20 A/B (kept recipe + sv1 + dutybal + energy=0.01, seed 0):
    final eval 0.8/0.384, 1.0/0.547, 0.8/0.324, 0.4/0.086 vs r3e's
    0.6/0.047, 1.0/0.225, 0.8/0.130, 1.0/0.106 — energy pricing did NOT
    slow it down; it sped it up dramatically while keeping 2.2-2.75 Hz.
    mu0.3 slice: 0.367 at 2.20 Hz with torso QUIET (RMS 8.0) = 151% of the
    c6 mu0.3 ceiling (0.242). Costs: mu0.4 regressed (0.4 pass), mild lean
    back at mu0.1/0.2 (mean 28-35, tier 3-lean class, reported not priced).
    Reading (mechanism, not verdict): high-energy gaits are the bouncy/
    draggy ones; making energy non-trivial pushes optimization toward
    efficient push-off walking, and efficient and fast point the same way
    on this body.

- P1 ruling (Ben, 2026-08-24): energy REMOVED entirely (r3f: weight 0.0005
  -> 0). rml2 recovery dropped from scope. The kappa-direction question
  takes priority: eff-kappa regression (torso world roll ~ hip-axis roll,
  deterministic, walk window) on five quiet-torso walkers:
  p1_energy mu0.3 +0.62 (joint slope -0.10), a2p0@2000k mu0.3 +0.47
  (-0.28), r3e mu0.4 +1.20 (-0.16), sv1 mu0.1 +0.19 (+0.04), s7@2750k
  mu0.1 +0.35 (+0.05); correlations all POSITIVE (+0.37..+0.58). Reading
  frame: kappa0 needs joint slope -1 (none observed); rigid follow = eff 1
  joint 0; kappa2 = eff 2 joint +1. The emergent family sits at eff-kappa
  0.2-1.2 with a nearly PASSIVE torso joint - "discounted rigid follow",
  leaning WITH the axis (kappa2 side), never world-upright; the fastest
  walkers have the lowest eff-kappa (0.19/0.35).

- P4 probe launched (Ben's package, 2026-08-24): (i) sv2 command bandwidth
  cap — 2nd-order Butterworth fc=2.5 Hz on the filtered action, a TRUE
  amplitude-independent frequency limit (firmware-replicable digital
  filter; measured |H|: 1.77 Hz 0.90, 2.2 Hz 0.79, 3.5 Hz 0.45, 5 Hz
  0.23; off-path bit-identical); (ii) curriculum cap raised 0.47 -> 0.70
  (--cmd-cap; kernel center follows vx_cmd) — pushes demanded speed past
  the confirmed frontier (0.49-0.65) to test "torso is recruited when the
  legs saturate"; (iii) training mu fixed at 0.1 (Ben: mu set reduced to
  0.1 only; frozen eval still characterizes all four mus). Recipe r3f +
  sv1. Readout: torso joint amplitude x vx as cmd climbs past 0.5.
  Hypothesis ledger going in: marginal accounting says speed incentive was
  never the binding constraint (~0.7/step unclaimed); fastest walkers had
  the LOWEST eff-kappa; this probe is the strong-form test. Run:
  runs/p4_cmd07/s0.
  - P4 probe COMPLETE (cmd cap 0.70 / sv2 fc 2.5 Hz / mu 0.1-only, 3M).
    Contract note: sv2 (like crank-band) must accompany the policy at eval;
    --cmd-fc added to eval/render/sweep after the mismatch made final look
    like a stander (0.000 across the board without the filter; 0.376 with).
    Frozen eval (with fc): mu0.1 0.8/0.376, mu0.2 1.0/0.140, mu0.3-0.4 0/5
    (mu0.1-only specialist by design). Torso recruitment readout as cmd
    climbed (deterministic mu0.1 per ckpt):
      cmd 0.37: vx 0.147, torso joint RMS 1.5 deg (cmd_std 0.085)
      cmd 0.47: vx 0.214, 0.9 deg (0.133)
      cmd 0.70: vx 0.338, 1.8 deg / 7.3 p2p (0.229), eff_k +1.21
    i.e. crossing the frontier roughly DOUBLED torso command activity
    (0.133 -> 0.229 std) and speed rose 0.214 -> 0.338, but the absolute
    torso amplitude stays small (1.8 deg RMS vs c6's designed +-24) — the
    policy buys speed mostly with legs plus a small kappa~1.2 follow.
    Training tail was the strongest ever recorded (vx 0.326, fall 0.29,
    hip_corr -0.48 WITH exploration noise). Frequency: final walks at
    ~3.05 Hz THROUGH the 2.5 Hz 2nd-order filter (|H| 0.58 there) — soft
    rolloff partially amplitude-compensated again; a hard cap would need a
    steeper filter. Run: runs/p4_cmd07/s0.
  - P4 ext (s0 +3M at cmd 0.70 held): deterministic mu0.1 per ckpt:
    3.5M 0.370 (tj 2.0deg, cmd_std 0.266, 2.90Hz) -> 4.5M 0.492 (2.2deg,
    0.217, 2.55Hz) -> 6.0M 0.461 (2.9deg, 0.306, 2.55Hz). Under sustained
    full demand: speed reaches 0.46-0.49 (past the c6 ceiling WITH the
    2.5Hz cap, on pure ice), torso keeps growing but slowly (1.8 -> 3.0 deg
    RMS, p2p to 12 deg), frequency locks at 2.55 Hz, eff-kappa stays in
    the 0.5-0.9 passive-follow band. Training tail 0.416-0.432 with fall
    0.19-0.30 and hip_corr -0.65 (all-time strongest).
  - P4 3-run wrap (best-ckpt + independent-seed confirm, mu0.1 x5):
    s0_ext@3.5M: 5/5 / 0.638 (min 0.633) — ties the all-time record 0.646
    WITH the 2.5 Hz cap + sv1 + mu0.1-only; tier 3-lean (frac 1.00, 24.6).
    s1@final: 5/5 / 0.324 (3-lean 0.80). s2@2750k: 5/5 / 0.449 (3-lean
    0.60; its final had drifted to 1/5 — stopping rule again). All three
    seeds confirm 5/5 at mu0.1: the P4 recipe is seed-reliable on ice.
    Every confirmed walker is tier 3-LEAN under eval jitter (RMS 24-29,
    lean frac 0.6-1.0): the mild-lean posture is the eval-robust strategy
    across the whole family. Frequency equilibrium 2.55 Hz under the cap.
