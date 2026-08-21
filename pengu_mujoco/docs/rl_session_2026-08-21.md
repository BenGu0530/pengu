# RL session 2026-08-20/21 — ice arm: recipe iteration, Gate 0, first arm seeds

Machine: Mac (alongside the c4 topup). Code: `rl_grid4/`. Companion spec:
`rl_e2_ice_memo.md` (frozen config + full amendment/knob log). Style per
working agreement: numbers only, corrections first, no verdicts.

## 0. Corrections

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

## 3. Arm results so far (frozen eval: 24 s, mu +-5%, 5 reps, deterministic)

net_fwd mean (pass count/5) per mu; designed K=5 baselines at mu 0.1/0.3
from c3|c6 topupK5.csv for reference:

| mu | seed 0 | seed 1 | kappa0 | kappa2 |
|---|---|---|---|---|
| 0.1 | 0.041 (1) | 0.044 (2) | 0.164 | 0.469 |
| 0.2 | 0.034 (1) | 0.018 (0) | — | — |
| 0.3 | 0.114 (5) | 0.025 (1) | 0.490 | 0.242 |
| 0.4 | 0.138 (5) | 0.017 (0) | — | — |

- torso roll RMS: s0 ~26 deg at all mu; s1 30-43 deg. eff_kappa columns in
  each run's eval_frozen.csv.
- Seed variance is large (s0 walks mu 0.3/0.4 5/5; s1 passes nowhere but is
  best at mu 0.1). Strict all-mu tier rule puts both in tier 1; per-mu rows
  above are the informative readout.
- Both seeds sit well below the designed envelope everywhere measured.
- Behavior note (Ben, from demo): high-frequency stepping with clearance but
  near-stationary — swing income is collected without translation; the
  marginal translation income (1*vx + kernel tail) is small at creep speeds.
  Candidate lever (NOT applied; would unfreeze the reward): gate swing on
  forward body velocity or rebalance progress/swing. Decision: Ben.

## 4. Status / pending

- Running: seed 2 stage B; then seed 3 (A+B). Auto: frozen eval per seed.
- Pending after seeds land: seed x mu table + three-tier + eff-kappa
  summary, mu=0.1/0.3 baseline comparison, demo mp4 per seed, this memo's
  results section update.
- Committed artifacts: final.zip per completed run (not intermediate ckpts),
  diag.csv, eval CSVs, demos for the gate passer and e2 s0.
