# RL session 2026-08-21 (machine D) — warm-start, reward tuning, and a measurement problem

Companion to `rl_e2_ice_memo.md` (frozen config) and `rl_session_2026-08-21.md`
(the Mac's arm results). Everything here ran on machine D and rml3.
Working agreement: measurement only, Ben draws conclusions.

**Headline: the biggest finding is not about any reward term. It is that the
current training setup does not converge, so arm-to-arm comparisons cannot yet
be made.** Details in §4.

---

## 0. Corrections to claims made earlier in this session

| claim | status |
|---|---|
| warm-start is necessary for c2 | **wrong** — from-scratch matches it, §1 |
| "c2 from scratch has no path to locomotion" (`rl_session` §2 round 6) | **wrong** — the run was cut at 1.25M, before it took off |
| scrub penalty is capping speed | **wrong** — scrub is flat 0.16–0.23 regardless of speed |
| 3M steps is enough | **wrong** — and 6M is not enough either, §4 |

---

## 1. Warm-start is not necessary

`gate0_r2a1e1c2_s0` (from scratch, 3M) vs the two warm-started Gate-0 runs:

```
gate0_r2a1e1c2_w_s0   warm    3M   vx 0.138
gate0_r2a1e1c2_w_s1   warm    3M   vx 0.223
gate0_r2a1e1c2_s0     scratch 3M   vx 0.224
```

The earlier "from scratch does not work" conclusion came from a run stopped at
1.25M. Its trajectory shows why:

```
1.00M 0.011   1.25M 0.028  <- old run cut here
1.50M 0.058   1.75M 0.071   2.00M 0.102
2.25M 0.123   2.50M 0.130   2.75M 0.166   3.00M 0.224
```

The curriculum did not reach vx_cmd 0.47 until 2.45M, so the policy had ~550k
steps at full command. The cut landed just before take-off.

**Consequence for the protocol**: warm-start can be dropped, which makes seeds
independent again. As it stands the seeds inherit their starting point (and
their torso roll, 25–30 deg) from a shared creep policy rather than each finding
it, so "N seeds show X" has an effective N closer to the number of independent
stage-A runs than to the number of seeds.

`gate0_r2a1e1c2_s0_cut1.25M` is kept as the record of the truncated run.

## 2. `--init-from` silently restarts the curriculum

`Curriculum.__init__` sets `self.cmd = CMD0` (0.12) unconditionally, and
`--init-from` restores policy and value weights only. Every
`--init-from --curriculum` resume so far — including both Gate-0 passers —
restarted the vx_cmd ramp from 0.12.

`vx_cmd` was also print-only, so the ramp history of an existing run cannot be
recovered. `gate0_r2a1e1c2_s0`'s command at its 1.25M cut is unknown; from the
gate conditions it probably never fired (speed gate needs vx ≥ 0.072, the run
peaked at 0.033; survival gate needs ep_len ≥ 400 with fall ≤ 0.3, its best was
396.6 at 0.29) but that is inference, not a record.

Fixed: `--cmd0` sets the starting command, and `vx_cmd` is now a diag column.

## 3. Diagnostics added

None of these touches the reward, the action space, or any frozen quantity.

**`torso_roll_mean_deg`** — RMS² = mean² + var, so RMS alone cannot separate a
±30 deg waddle (mean ≈ 0) from a steady 30 deg lean (mean ≈ RMS); they give the
same number. Ben's concern about "torso leaning one direction forever" was
unmeasurable with what was logged. Verified: a zero-action standing episode
gives |mean|/RMS = 0.992 (pure DC lean of −1.4 deg), a random-action episode
0.153.

**`torso_roll_rate_rms_dps`** — Ben asked whether a diff could replace the mean.
`diff(RMS)` cannot: a lean and a waddle both give a flat windowed RMS, so its
derivative is ~0 for both. `RMS(diff)` can, and is better than the mean because
it ignores a drifting baseline. Standing 1.2 deg/s vs random-action 80–97 deg/s.

**`stride_L_m` / `stride_R_m` / `stride_asym`** — Ben's design: touchdown-to-
touchdown distance per foot, asym = (L−R)/(L+R). Equal strides = walking
straight. This reads straightness off the **legs**, so unlike a yaw penalty it
cannot leak into the torso variable, and nothing optimises against it.

**On the heading term that prompted this — not added.** `vx` is projected on the
FIXED spawn heading axis (`fh2 = self.fh0[:2]`), and `r_track`, `r_progress` and
`swing_rate` all consume that `vx`, so course error is already priced at
cos(θ): 45 deg off costs 29% of the vx income, 75 deg costs 74%. Measured
`heading_align` is 0.73–0.92 across the walking runs, so the memo's trigger
("add only if circling shows up") is not met. A per-step yaw penalty would also
be unsafe: on the designed gaits roll and yaw are one motion (corr +0.960, 8 ms
lag on a 510 ms cycle), so penalising yaw penalises roll — the one variable that
must stay unrewarded.

## 4. **The measurement problem — read this before any arm comparison**

vx does not converge. It oscillates between a fast-and-falling and a
slow-and-safe policy, and the swing is as large as any effect being looked for.

Frozen baseline, 6M steps, final 1M of each run:

```
seed 0   5.75M vx 0.193 fall 0.78  ->  6.00M vx 0.098 fall 0.28
seed 1   4.75M vx 0.084            ->  6.00M vx 0.035     (declining)
seed 2   4.75M vx 0.029  5.00M 0.016  5.50M 0.084  6.00M 0.123  (still climbing)
```

Seed 0 halves its speed in a single 250k-step interval while its fall rate drops
by two thirds. That is not convergence, it is the optimiser moving between two
attractors of similar value.

Two consequences:

1. **A run's final vx is a random draw, not a measurement.** Seed 0's adjacent
   diag points differ by 2×.
2. **The `<climbing` test is insufficient** — it only looks at the last
   increment, and seed 0's last increment is negative, so it would pass while
   being obviously unsettled.

Replaced by a tail-window statistic, now in `summarize_tune.sh`:

```
vtail  = mean vx over the last 4 diag points (1M steps)
vswing = max - min over those 4 points;  > 0.05 = still swinging
```

Recomputed, **almost every run is flagged unstable**:

| run | steps | vx | vtail | vswing | |
|---|---|---|---|---|---|
| s0 | 6M | 0.098 | 0.145 | 0.095 | unstable |
| s1 | 6M | 0.035 | 0.043 | 0.027 | |
| s2 | 6M | 0.123 | 0.086 | 0.071 | unstable |
| ns_s0 | 3M | 0.112 | 0.082 | 0.060 | unstable |
| ns_s2 | 3M | 0.087 | 0.053 | 0.066 | unstable |

**The baseline's seed-to-seed spread (0.043 / 0.086 / 0.145) and a single seed's
internal swing (0.095) are the same size.** Under this setup an ablation arm at
n=1 — or even n=3 — cannot be separated from noise.

This is a stronger candidate for why the policy sits near 0.2 while the designed
family reaches 0.47–0.60 than any single reward weight: the two attractors are
priced similarly and PPO never settles in either.

Things worth trying that this points at, none of them a reward *weight*:
- the **shape** of the fall penalty (flat −10 now; scaling it with speed would
  change the relative value of the two attractors rather than shifting both)
- PPO `target_kl` (0.03) — may be permitting too large a policy change per update
- longer runs will not fix an oscillation, only a truncation

## 5. no-smooth ablation (3 seeds, rml3)

```
frozen (from scratch, 3M)   0.224  0.093  0.085  0.063    median 0.089
no-smooth        (3M)       0.112  0.087  0.014           median 0.087
```

No effect, as expected — `r_smooth` is 2.5% of positive income. But ns_s0 and
ns_s2 were still climbing (+0.060, +0.049) at 3M **and** are flagged unstable, so
this comparison does not stand on its own either.

## 6. Every from-scratch seed learns an in-phase gait

`hip_corr` (negative = alternating legs, positive = both together):

```
seven from-scratch seeds:  +0.20 to +0.68     all in-phase
warm-started e2_w_s0:      -0.20              alternating
```

`e2_w_s0` is also the only run that passes 5/5 at mu 0.3 and 0.4 in frozen eval.

In-phase means both legs moving together — hopping rather than walking — and it
is a **categorical** difference, not a magnitude one. If the ceiling near vx 0.1
is a gait-class problem rather than a pricing problem, no reward weight will move
it. The `no_swing` ablation is the direct test: if `hip_corr` changes sign
without the stepping prior, the prior is what locks in the in-phase gait.

## 7. Tooling added (`rl_grid4/`)

| file | purpose |
|---|---|
| `--rw key=val` in `train_grid4.py` | override any frozen reward weight; tag records it so a tuning arm can never be pooled with a frozen run. Verified the default path is bit-identical (max per-step reward difference 0.000e+00 over 68 steps) |
| `--no-smooth`, `--cmd0` | equivalent to `--rw smooth=0`; curriculum resume |
| `ablate_arms.txt` | 8 on/off ablations, each one term to zero |
| `tune_arms.txt` | 9 magnitude arms, marked do-not-run-before-ablations |
| `run_tune_queue.sh` | walks a list N at a time, evals after each, resumable |
| `summarize_tune.sh` | one table, frozen and tuning arms labelled separately |
| `triage_arms.sh` | which arms deserve more seeds; refuses to rank unconverged runs |

**Ablation vs tuning (Ben's correction).** The first arm list mixed them:
`progress=3.0` vs `progress=6.0` picks between two arbitrary magnitudes and
answers "which number is better", not "is this term earning its place". The
lists are now separate and the ablations run first.

**Bug caught during this**: `RW_DEFAULT["fall"]` was first written as 5.0 while
the frozen `FALL_PENALTY` is 10.0 (v2 raised it). That would have silently
halved the fall penalty on every run including the frozen ones. Now checked
against the source constants.

**Second bug**: the first attempt at the stride touchdown hook silently no-opped
— the `str.replace` pattern had the wrong indentation, and Python does not
complain. Caught because `n_stride` stayed 0 while contact flips were 31–35 per
episode.

## 8. Machine capacity (machine D, 16 cores / 30 GB)

Training is CPU-only (`device="cpu"` is hardcoded, and would not benefit from a
GPU: the policy is a [256,256] MLP, the bottleneck is mujoco stepping).

```
one job, 8 envs, alone:      4.64 cores, 3.94 GB PSS, 11 processes
three jobs concurrently:     3.7 cores each (20% queueing), 11.0 cores total
```

Isaac Lab holds ~1 core and the GPU. Practical limit **3 concurrent** at 8 envs;
memory binds slightly before CPU. Lowering `--n-envs` trades concurrency for
per-run speed at roughly constant total throughput.

## 9. In flight at time of writing

- machine D: frozen baseline seeds 0–2 done at 6M, seed 3 running
- rml3: 8-arm ablation queue, 3 concurrent, 6M each, started 14:51

## 10. Open items

1. **The convergence problem in §4 outranks everything else.** Arm comparisons
   are not interpretable until a run's vx is a measurement rather than a draw.
2. Warm-start can be removed from the protocol (§1).
3. `no_swing` result — does `hip_corr` flip? (§6)
4. Both Gate-0 passers ran with a silently restarted curriculum (§2); worth
   deciding whether they still count as the Gate-0 record.
5. Magnitude tuning (`tune_arms.txt`) only after the ablations, and only after §4.
