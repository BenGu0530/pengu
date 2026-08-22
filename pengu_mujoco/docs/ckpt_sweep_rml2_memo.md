# Task memo — checkpoint-sweep evaluation (for rml2)

Context: machine D's convergence finding (rl_session_2026-08-21_machineD.md
§4) — late-training vx oscillates between a fast/fragile and a slow/safe
attractor, and single-seed spread matches within-run swing. If the
oscillation is large, `final.zip` may be far from each run's best policy and
part of the "instability" is a stopping-rule problem, which is much cheaper
to fix (select checkpoints by eval) than the optimization itself.

## The job (one command)

From `pengu_mujoco/` on rml2, after `git pull` on `friction-experiments`:

```
nice -n 19 python rl_grid4/eval_ckpt_sweep.py rl_grid4/runs/e2/s0/stageB \
  rl_grid4/runs/e2/s1/stageB rl_grid4/runs/e2/s2/stageB rl_grid4/runs/e2/s3/stageB \
  --repeats 3 >> rl_grid4/ckpt_sweep_rml2.log 2>&1
```

Env: any python with mujoco 3.8.x + stable_baselines3 + torch (CPU). The
script needs no display. `SWEEP_NICE=19` etiquette applies (shared with
Isaac Lab).

What it does: every saved ckpt (250k..3M + final) of each arm seed's stage B
-> frozen eval (24 s, mu {0.1,0.2,0.3,0.4} +-5%, 3 repeats) -> per-run
`ckpt_sweep.csv` + printed best-vs-final table. Selection metric is fixed
inside the script (total pass count, tie-break mean net_fwd) — do not
re-rank by another metric.

Cost estimate: 4 runs x 13 ckpts x 12 trials x 24 s sim ≈ 15k sim-seconds;
a few hours single-process. Optional second pass if capacity allows, same
command with `rl_grid4/runs/gate0/s0/stageB rl_grid4/runs/gate0/s1/stageB`
and `--mus 0.7`.

## Deliverables

1. `rl_grid4/runs/e2/s*/stageB/ckpt_sweep.csv` (git add -f — *.csv is
   gitignored) + the log.
2. Append the printed best-vs-final summary table to this memo under
   "Results", with machine + date. Numbers only, no verdicts — in
   particular do not conclude "instability solved/not solved"; that
   assessment is Ben's, with the delta table in front of him.
3. Push to `friction-experiments` (rebase first; Mac and rml3 are also
   pushing).

## Results

Machine rml2 (16 cores, shared with Isaac Lab, nice 19), 2026-08-22.
`eval_ckpt_sweep.py`, 13 ckpts per run (250k..3M + final), frozen eval
24 s, mu {0.1,0.2,0.3,0.4} +-5%, 3 repeats = 12 trials per ckpt.
Selection metric as fixed in the script: total pass count, tie-break mean
net_fwd.

| run | best ckpt | pass best | pass final | delta | net_fwd best | net_fwd final |
|---|---|---|---|---|---|---|
| e2/s0/stageB | 1500k | 12/12 | 7/12 | +5 | 0.3556 | 0.0730 |
| e2/s1/stageB |  750k | 11/12 | 1/12 | +10 | 0.0925 | 0.0225 |
| e2/s2/stageB | 3000k | 12/12 | 10/12 | +2 | 0.2378 | 0.2363 |
| e2/s3/stageB | 2500k | 12/12 | 7/12 | +5 | 0.2144 | 0.1140 |

Per-ckpt rows: `rl_grid4/runs/e2/s*/stageB/ckpt_sweep.csv`.
Log: `rl_grid4/ckpt_sweep_rml2.log`.

Note on s2: its best ckpt (3000k) is the last saved checkpoint, which is a
different file from `final.zip`; the two differ by the steps between the last
checkpoint save and the end of training.


## Phase 2 (C2 retrofit for the ablation arms) — for rml2

Same procedure, the 8 ablation arms + 3 no-smooth arms (single seed each):

```
nice -n 19 python rl_grid4/eval_ckpt_sweep.py \
  rl_grid4/runs/gate0_r2-track0a1e1c2_s0 rl_grid4/runs/gate0_r2-energy0a1e1c2_s0 \
  rl_grid4/runs/gate0_r2-back0a1e1c2_s0 rl_grid4/runs/gate0_r2-fall0a1e1c2_s0 \
  rl_grid4/runs/gate0_r2-swing0a1e1c2_s0 rl_grid4/runs/gate0_r2-scrub0a1e1c2_s0 \
  rl_grid4/runs/gate0_r2-smooth0a1e1c2_s0 rl_grid4/runs/gate0_r2-progress0a1e1c2_s0 \
  rl_grid4/runs/gate0_r2nsa1e1c2_s0 rl_grid4/runs/gate0_r2nsa1e1c2_s1 \
  rl_grid4/runs/gate0_r2nsa1e1c2_s2 --repeats 3 >> rl_grid4/ckpt_sweep_rml2_p2.log 2>&1
```

Then CONFIRM each arm's best ckpt on independent seeds:
`eval_grid4_policy.py <best> --repeats 5 --trial-seed-base 50000 --out
<run>/eval_bestckpt_confirm.csv`, and append a best-vs-final + confirmed
table under Results-2. Numbers only. After that, triage_arms.sh judgments
should be re-read against confirmed numbers (do not re-run triage on
final.zip values).

## Results-2

(to be filled by rml2)
