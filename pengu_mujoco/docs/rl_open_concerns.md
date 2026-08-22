# RL arm — open concerns, ranked (as of 2026-08-22)

One place for everything currently unresolved, with the evidence pointer and
what would settle each. Numbers only; the calls are Ben's.

Source memos: `rl_e2_ice_memo.md` (frozen config), `rl_session_2026-08-21.md`
(Mac), `rl_session_2026-08-21_machineD.md` (machine D), `bc_express_result.md`,
`ckpt_sweep_rml2_memo.md`.

---

## C1 — Training does not converge; single-run numbers are draws, not measurements

**Blocks**: every arm-to-arm comparison, including all 8 ablations.

Frozen baseline seed 0, full 6M, vx per 250k (recovered from
`logs/base6m_s0.log`, see C8):

```
0.25M -0.032   2.25M +0.081   4.25M +0.079
0.50M +0.050   2.50M +0.081   4.50M +0.103
0.75M +0.032   2.75M +0.068   4.75M +0.146
1.00M +0.046   3.00M +0.061   5.00M +0.101
1.25M +0.057   3.25M +0.025   5.25M +0.135
1.50M +0.060   3.50M +0.042   5.50M +0.179
1.75M +0.022   3.75M +0.016   5.75M +0.193
2.00M +0.069   4.00M +0.039   6.00M +0.098
```

It rises and falls repeatedly and drops by half in the final interval while
fall_rate goes 0.78 -> 0.28. Across the four 6M seeds, `vtail` (mean over the
last 1M) is 0.145 / 0.043 / 0.086 / 0.089 and one seed's own `vswing` is 0.095 —
**the between-seed spread and the within-seed swing are the same size**.

6M did not help: median `vtail` went 0.068 (3M) -> 0.088 (6M) with two of four
seeds still flagged unstable. More steps do not fix an oscillation.

Settles it: a stopping rule that is not "last checkpoint" (C2 is already
evidence this matters), or a change that makes the two attractors stop trading
places — the *shape* of the fall penalty rather than its weight, or PPO
`target_kl` (0.03) which may permit too large a step per update. Not more steps.

## C2 — `final.zip` is far from each run's best policy

Every saved checkpoint of the four e2 arm seeds, frozen eval, 12 trials each:

| run | best ckpt | pass best | pass final | delta | net_fwd best / final |
|---|---|---|---|---|---|
| e2/s0 | 1500k | 12/12 | 7/12 | +5 | 0.3556 / 0.0730 |
| e2/s1 | 750k | 11/12 | 1/12 | +10 | 0.0925 / 0.0225 |
| e2/s2 | 3000k | 12/12 | 10/12 | +2 | 0.2378 / 0.2363 |
| e2/s3 | 2500k | 12/12 | 7/12 | +5 | 0.2144 / 0.1140 |

s0's best checkpoint is **4.9x** the delivered policy's net_fwd. The gate0 sweep
on rml3 shows the same shape (s1: 1250k 0/3, 1500k 3/3, 1750k 3/3, 2000k 3/3).

Note s2's small delta is an artifact: its "best" is the 3000k checkpoint file,
which differs from `final.zip` only by the steps after the last save.

Settles it: whether reported arm results should use best-by-eval rather than
final. That changes what every number in the project means, so it is a protocol
decision, not a scripting one.

## C3 — Every from-scratch seed learns an in-phase gait

`hip_corr` (negative = alternating legs, positive = both together):

```
frozen from-scratch, 6M:   +0.58  -0.04  +0.59  +0.52
frozen from-scratch, 3M:   +0.31  +0.30  +0.26  +0.29
no-smooth (3 seeds):       +0.50  +0.20  +0.68
8 ablation arms:           +0.10 to +0.73
warm-started e2_w_s0:      -0.20
```

Nine from-scratch seeds across every arm are in-phase. The one alternating run
is also the only one passing 5/5 at mu 0.3 and 0.4 in frozen eval.

`no_swing` was the direct test of whether the stepping prior locks this in. It
does not: `hip_corr +0.20`, still in-phase. So the cause is elsewhere — action
space, observation, or the body.

In-phase means both legs moving together, which is a **categorical** difference
from walking, not a magnitude one. If the ceiling near vx 0.1 is a gait-class
problem, no reward weight moves it.

## C4 — The 0.47 bar and the RL arm are not measured under the same conditions

`vx_cmd = 0.47` comes from c6's K=5 sweep ceiling at mu=0.1 (net_fwd 0.4689).
That was measured under the sweep's conditions. Under the RL env's reset, pose
jitter and settle, **the same c6 teacher fell in 49 of 100 episodes**
(`bc_express` collection log).

Settles it: score the c6 teacher under the RL env's exact reset protocol and
report that number alongside 0.4689, so the bar and the arm are comparable.

## C5 — The BC expressibility question is still open

The clone fit well (MSE 0.00886 -> 0.00038) and then emitted a constant: per-dim
action std [0.0039 0.0034 0.0007 0.0008 0.0021] on a [-1,1] space, `hip_corr`
+0.98 against a teacher that runs its hips in antiphase, vx 0.000.

That is regression to the conditional mean, which means the test measured the
**observation**, not the network: a memoryless policy on this 36-dim obs cannot
recover the teacher's phase. The script's own docstring flagged this check and
named frame-stacking as the fallback.

Settles it: clock-augmented BC (oracle phase in the obs) answers "can the
network express c6" directly and costs one run. Frame-stacking answers the
deployable version of the question. Full detail in `bc_express_result.md`.

## C6 — Both Gate-0 passers ran with a silently restarted curriculum

`Curriculum.__init__` set `cmd = CMD0` unconditionally and `--init-from`
restores weights only, so every `--init-from --curriculum` resume restarted the
vx_cmd ramp at 0.12. That includes `gate0_r2a1e1c2_w_s0` and `w_s1`, the runs
the Gate 0 record rests on.

`vx_cmd` was also print-only, so their actual command schedule is unrecoverable.
Fixed going forward (`--cmd0`, and `vx_cmd` is now a diag column), but the
existing record cannot be reconstructed.

Settles it: a decision on whether those two runs still stand as the Gate 0
record, or whether Gate 0 should be re-run under the fixed code.

## C7 — Warm-start makes seeds non-independent, and is not needed

From-scratch reaches 0.224 at 3M against the warm-started 0.138 and 0.223. The
earlier "no path to locomotion" verdict came from a run cut at 1.25M — its own
trajectory shows it took off at 1.5M and climbed to 3M.

While warm-start is in the protocol, seeds inherit their starting point and
their torso roll (25-30 deg) from one shared creep policy rather than each
finding it, so "N seeds show X" has an effective N closer to the number of
independent stage-A runs.

Settles it: dropping warm-start from the protocol. The evidence for doing so is
already in hand.

## C8 — Process: an uncommitted file was overwritten during a sync

While resolving a path collision after a pull (the Mac had reorganised `runs/`
into `runs/archive/...` while this machine had 6M data at the old path), a
`git checkout HEAD --` on the archive directory overwrote the local
`diag.csv`, losing 22 of seed 0's 24 diag rows.

Recovered in full from `logs/base6m_s0.log`, which happens to carry the same
rows on stdout, and written to `runs/gate0_r2a1e1c2_s0/diag_from_log.csv` (the
truncated original kept as `diag.csv.partial`). Recovery was luck, not method.

Worth noting because run outputs live in gitignored paths and are only in the
repo via `git add -f`: a machine that reorganises `runs/` while another is
writing into the old layout can silently destroy data on the second machine.

---

## Suggested order

C1 first — while a run's vx is a draw rather than a measurement, the ablation
table, the tuning arms and any arm ranking are all uninterpretable. C2 is
cheap and may be a large part of C1. C3 and C5 are the two that could change
what the arm is capable of at all; C4 changes what it is being compared to.
