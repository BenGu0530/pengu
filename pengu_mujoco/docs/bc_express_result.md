# BC expressibility test — result (rml3, 2026-08-22)

Script: `rl_grid4/bc_express.py`, default settings (100 episodes, 20 epochs,
DART noise 0.05, seed 0). Clone at `rl_grid4/runs/bc_c6/bc_c6.zip`.

## What was measured

```
[collect] 96098 pairs from 100 episodes (49 fell)
[bc] epoch 1/20  mse 0.00886   ->  epoch 20/20  mse 0.00038      monotone
[bc] saved bc_c6.zip (final mse 0.00038)
[clone] mu=0.1: 24.0s fell=0.0 vx=0.000 torso_rms=0.4 hip_corr=+0.98
[clone] mu=0.3: 24.0s fell=0.0 vx=0.000 torso_rms=0.4 hip_corr=+0.98
frozen eval: mu 0.1/0.2/0.3 all pass=0.0, net_fwd -0.000..0.000, torso_rms ~0.3 deg
```

Action statistics of the clone over a 500-step deterministic rollout at mu=0.1
(action space is `[-1,1]^5`):

```
per-dim std    [0.0039 0.0034 0.0007 0.0008 0.0021]
per-dim range  [0.0192 0.0161 0.0047 0.0076 0.0141]
```

## Reading

The clone outputs a constant. Std is 0.0007–0.0039 on a unit-scaled action
space, i.e. three orders of magnitude below the teacher's swing.

`hip_corr = +0.98` is the other half of the same fact: the c6 teacher runs its
two hips in antiphase (`hip_corr` near −1), and the clone puts them at +0.98 —
both outputs sitting at their own means, hence almost perfectly correlated.

Low MSE with the wrong behaviour is consistent with regression to the
conditional mean: the mean of a sinusoid is its centre, so a constant output
scores well on MSE while producing no motion. The static pose is also stable,
which is why `fell=0.0` and the episodes run the full 24 s.

## What this does and does not answer

It does **not** answer the question the test was written for. From the script's
own docstring:

> the teacher is time-indexed and obs has no clock, but at steady state each
> joint's (pos, vel) pair determines the sinusoid phase, so a single frame is in
> principle sufficient; the printed clone-vs-teacher tracking error is the
> check. If it fails, frame-stacking is the declared fallback.

That check failed. The result is about the **observation**, not the network: a
memoryless policy on this 36-dim obs cannot recover the teacher's phase, so BC
has no well-posed target to fit. Whether the policy network plus the a2 crank
band can express c6 remains open.

## Two follow-ups, answering different questions

1. **Clock-augmented BC** — append the teacher's phase to the obs and re-run BC.
   If the clone then tracks, network capacity is not the limit and the issue is
   purely observability. This is the direct test of expressibility and costs
   about what the run above cost. It uses an oracle input, so it is a
   capability probe, not a candidate policy.
2. **Frame-stacking** — the fallback named in the docstring. Tests whether the
   frozen obs contract plus history is enough, which is the question that
   matters for a deployable policy.

## Separate observation worth checking

The teacher fell in **49 of 100 collection episodes**. c6 is the designed
family's mu=0.1 ceiling (net_fwd 0.4689 in the K=5 sweep), but that number was
measured under the sweep's conditions. Under this env's reset, pose jitter and
settle it falls half the time. If the 0.47 figure is being used as the bar the
RL arm is measured against, the two are not currently measured under the same
conditions.
