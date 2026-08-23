# Overnight reward search — journal

Started 2026-08-22 00:38 on machine D (rml2), running unattended.

## Brief

Goal unchanged: **torso use emerging** under a reward that says nothing about
the torso. Only existing reward weights may be changed, including to zero. **No
new reward terms.**

Method: each generation trains 3 candidates concurrently (3M steps, seed 0),
evaluates them, renders a demo at mu=0.1, and extracts frames at t = 3/6/9 s.
The next generation is chosen after **looking at the frames**, not only the
table.

## Why frames and not just numbers

Ben's observation, and the thing this search is built around: several local
optima are invisible in the summary. The one named explicitly — torso parked
over to one side with the legs compensating every step — produces
"fast, high torso roll RMS, survives", which is exactly what a good result also
produces. RMS cannot tell a ±30 deg swing from a held 30 deg lean, because
RMS² = mean² + var.

Columns added for this, and the flags the report raises:

| flag | test | what it catches |
|---|---|---|
| `HELD-LEAN` | \|roll_mean\|/roll_RMS > 0.7 | torso parked to one side, not swinging |
| `STATIC-TORSO` | roll RMS > 15 deg but roll rate < 30 deg/s | leaning, barely moving |
| `ASYM` | \|stride_asym\| > 0.25 | one leg taking much longer strides — compensating, or curving |
| `unstable` | vswing > 0.05 over the last 1M | vtail is a draw, not a value |

Reference points from the env, measured earlier: a zero-action standing episode
gives \|mean\|/RMS = 0.992 and roll rate 1.2 deg/s; a random-action episode
gives 0.153 and 80–97 deg/s.

Every generation also prints the per-step reward budget, so when a candidate
looks wrong the next question is answerable: **which term is it harvesting?**

## Standing context this search inherits

- Training does not converge; between-seed spread and within-seed swing are the
  same size (`rl_open_concerns.md` C1). A single generation's vtail is therefore
  weak evidence — read the flags and frames first.
- All nine from-scratch seeds so far learn an **in-phase** gait (hip_corr > 0);
  the one run that passes 5/5 at mu 0.3/0.4 is the alternating one. `no_swing`
  did not flip this, so the stepping prior is not the cause (C3).
- `final.zip` is far from each run's best checkpoint (C2). These generations
  report the final policy; a promising candidate should get a checkpoint sweep
  before being believed.

---

## gen01 — where does the speed ceiling come from?

Launched 00:38.

| candidate | change | why |
|---|---|---|
| `base` | none (`smooth=0.01`, the frozen value) | control for this harness |
| `kernel_wide` | `sigma2=0.20` (sigma 0.14 → 0.45) | the kernel prices everything below vx 0.2 at ~zero; widening it should give gradient across the whole range instead of only near 0.47 |
| `kernel_off` | `track=0.0` | in the ablation sweep this gave the fastest run of the project (vtail 0.296 vs a 0.088 baseline) at fall 0.71. Is that a real gait or a dash? |

Prediction to check against: if `kernel_off` is fast because it removed the
slow-and-safe attractor's price, its frames should show a lunge, and its
fall rate should stay high. If it is fast *and* the frames show stepping, the
kernel shape — not any weight — is the thing capping speed.

(results appended below by each generation)

### gen01 results (00:59)

| cand | ePass | eNfwd | eval roll | eval roll mean | eval rate | flags |
|---|---|---|---|---|---|---|
| `base` | **0.75** | **0.146** | 30.7 | — | 97 | — |
| `kernel_wide` | 0.05 | 0.029 | 22.1 | — | 92 | FAKE-TORSO, unstable |
| `kernel_off` | 0.00 | 0.009 | 41.3 | −47.3 | **1.5–3.0** | FAKE-TORSO, unstable |

**Both kernel changes made it worse. The frozen baseline won.**

`kernel_wide` (sigma2 0.20): the wide kernel pays at *any* speed. `r_track`
went 0.104 → 0.360 and `pos_sum` 0.354 → 0.597, a 68% richer budget, while
ePass fell 0.75 → 0.05. Widening it did not add gradient toward 0.47; it
removed the reason to go fast at all.

`kernel_off` (track=0): **this is the false positive Ben described, caught.**
Eval: `torso_roll_rms 41.3`, `eff_kappa 4.22` — both read as strong torso use,
above the designed κ=2 — with `net_fwd 0.0094` and **0/20 pass**. The frames at
t=3/6/9 s are the same pose three times.

The per-trial columns say it exactly:

```
torso_roll_rms 47.27   torso_roll_mean -47.26   rate 3.0 deg/s
torso_roll_rms 42.62   torso_roll_mean -42.62   rate 1.5 deg/s
```

|mean| = RMS to four figures, and the rate is at the **zero-action standing
reference (1.2 deg/s)**. The torso is parked at −47° and held there.

**Harness fix this forced.** The flags were computed from the training diag,
where kernel_off showed rate 138 deg/s and tripped nothing — training has
exploration noise, the deterministic eval collapses to the static lean. Flags
now read the eval rollout, `eval_grid4_policy.py` emits `torso_roll_mean_deg`
and `torso_roll_rate_rms_dps`, and a new `FAKE-TORSO` flag fires on high eval
roll RMS with net_fwd < 0.03. Without the frames this would have been recorded
as the best torso-use result of the night.

**Reading**: the kernel is not the lever. Neither widening nor removing it
helps, and removing it produces the parked-lean optimum.

### gen02 — the forward/backward pricing imbalance (launched 01:03)

`base`'s per-step budget:

```
track +0.104  progress +0.141  swing +0.108          pos_sum 0.354
back  -0.160  scrub    -0.128  smooth -0.022  fall -0.022  energy -0.008
```

**`r_back` (−0.160) is larger in magnitude than `r_progress` (+0.141).** The
weights are back 2.0 against progress 1.0, so backward motion is taxed at twice
the rate forward motion is paid, and a waddling gait has backward phases every
stride. Net reward is ≈ +0.014/step — barely above break-even.

The earlier ablation's `back0` also had the best eval of that whole set
(ePass 0.80).

| candidate | change |
|---|---|
| `back05` | `back=0.5` |
| `back0` | `back=0.0` |
| `prog3` | `progress=3.0` |

Prediction: if the backward tax is what caps speed, `back05`/`back0` gain
without the fall rate exploding. If instead they just let the robot rock
backward for free, expect ASYM or a drop in net_fwd despite a richer budget.

---

## Incident — 18.5 h lost (01:06 → 19:34)

I edited `overnight/run_gen.sh` at 01:06 to add the C2 protocol. gen02 had been
launched from that same file at 01:03 and was still executing it. Bash reads a
script incrementally by byte offset, so rewriting a running script sends it to a
wrong position; the parent shell died and its three training subshells took
SIGHUP with it. The logs stop cleanly at `[curriculum] vx_cmd start 0.12` with
no traceback, and dmesg shows no OOM — that is the signature. The machine idled
until 19:34.

Fix in place: generations now run from a frozen copy (`run_gen_v2.sh`,
`queue_v2.sh`) that is never touched while a queue is live. Changes go into a
new versioned copy.

## Re-framing after Ben's Resolutions (C2 adopted)

`rl_session_2026-08-21.md` §3a changes what this search should be optimising.
Under checkpoint-selection + independent-seed confirmation:

```
s0@1.5M  mu=0.1  0.386 (5/5, no falls)     kappa0 ceiling 0.164
                                            c6 under the RL env 0.392 (15% falls)
```

**The arm is already level with the designed c6 gait under matched conditions.**
The "caps at 0.2" premise was largely a `final.zip` artifact. Tuning reward
weights to raise a final-checkpoint number was optimising the wrong quantity.
The harness now selects by eval and confirms on independent seeds, and every
generation from here reports confirmation numbers.

## Finding — torso follow is ANTI-correlated with speed

From the four confirmed arm seeds, 79 surviving trials
(`runs/e2/s*/stageB/eval_bestckpt_confirm.csv`):

```
corr(net_fwd, eff_kappa)       -0.644
corr(net_fwd, hip_corr)        -0.485
corr(net_fwd, torso_roll_rms)  -0.295
corr(net_fwd, root_roll_rms)   -0.235
```

Binned by torso follow:

| eff_kappa | n | mean net_fwd |
|---|---|---|
| < 1.0 (torso barely follows) | 19 | **+0.341** |
| 1.0–2.0 | 13 | +0.236 |
| > 2.0 (strong follow) | 47 | **+0.129** |

**Not a seed artifact.** Within-seed correlations are −0.908 / −0.211 / −0.775 /
−0.547 (mean −0.610), same sign in all four. **Not a mu artifact** either:
across the 16 (seed, mu) cells the mean within-cell correlation is −0.417,
negative in 12 of 16, and mu itself is flat against both
(corr(mu, eff_kappa) +0.120, corr(mu, net_fwd) −0.055).

**Not passive following.** If a high eff_kappa merely meant the torso riding a
bigger rock, it would track root roll — `corr(root_roll, eff_kappa) = -0.082`,
essentially zero, while `corr(root_roll, torso_roll) = +0.866`. The follow
*coefficient* predicts slowness far better than the amount of rocking does.

`eff_kappa` and `hip_corr` are only weakly related (+0.288) and both survive
partialling:

```
corr(net_fwd, eff_kappa | hip_corr)  -0.602
corr(net_fwd, hip_corr  | eff_kappa) -0.409
```

so they are independent predictors. The 2×2:

| | hip_corr < 0.18 | hip_corr > 0.18 |
|---|---|---|
| **eff_k < 2.47** | **+0.332** (n=26) | +0.177 (n=13) |
| **eff_k > 2.47** | +0.120 (n=13) | +0.115 (n=27) |

The fast quadrant — torso not following, legs alternating — is 2.9× the other
three, which are indistinguishable from each other.

### Why this matters to the brief

The experiment's premise (`rl_e2_ice_memo.md`) is that tracking 0.47 on ice
*requires* torso use, because in the designed family c6 (κ=2) reaches 0.4689 at
mu=0.1 where c3 (κ=0) reaches 0.1636. In the RL arm the relationship runs the
other way: the policies that follow the roll least are the fast ones.

Stated as measurement, not conclusion. Caveats: eff_kappa is observed, not
controlled; there are 79 trials but only 4 independent policies; and this says
nothing about whether *commanding* torso use would help — only that among what
these policies found, more follow goes with less speed.

The clean test is an intervention, not another correlation: hold the torso
actuator fixed and retrain (a capability ablation, no reward change), or
compare against the arm's own low-eff_kappa checkpoints. Recorded, not run —
the queue is on the assigned C2 retrofit.

---

## Incident 2 — the C2 selection step silently did nothing (v2 harness)

`eval_ckpt_sweep.py` prints a **label**:

```
[runs/overnight/retro_a/no_track] best=2000k (pass 2, nf 0.0331) vs final (pass 0, nf 0.0052)
```

`run_gen_v2.sh` did `ck=$(grep -oE "best=[^ ]+" | cut -d= -f2)` and then `[ -f "$ck" ]`,
which can never succeed for the string `2000k`, so every generation fell back to
`final.zip`. All of `retro_a/*.selected_ckpt.txt` contained `.../ckpts/final.zip`.
**The C2 protocol was not in effect for retro_a/b/c or gen02.**

The sweep itself was fine — `ckpt_sweep.csv` holds the full per-checkpoint table —
so `fixup_c2.sh` re-read it, picked by `(n_pass, mean_net_fwd)`, and re-ran only the
confirm and render. No retraining. `run_gen_v3.sh` maps label -> path
(`2000k` -> `ckpt_2000000_steps.zip`) and now WARNS when it cannot resolve instead of
falling back silently.

Selected checkpoints, none of which was final:

```
no_track 2000k   no_progress 2500k   no_swing 2500k
no_scrub 2000k   no_smooth   1250k   no_energy 2000k
no_back  2750k   no_fall     1250k
back0    2750k   back05      2750k   prog3     1500k
```

Lesson taken: validate a harness on one case before queueing a batch behind it.

## Training is deterministic given (seed, config)

`retro_c/no_back` and `gen02/back0` are the same config (`back=0.0`, seed 0) launched
26 min apart as separate processes. Their `policy.pth` and `policy.optimizer.pth` are
**byte-identical**; `diag.csv` is byte-identical; the eval CSVs differ only in the
`ckpt` path column. Only the sb3 `data` member (metadata) differs.

Two consequences. The `unstable` flag (vswing) is measuring **within-run oscillation**,
not run-to-run scatter — C1's spread is between-seed plus within-run, with no third
source. And the duplicate cost ~25 min of compute: `back0` was already an ablation arm
under the name `no_back`, and I did not notice when writing gen02. `gen02/back0` is
excluded from all counts below.

## C2 retrofit — the 8 ablation arms + gen02 (confirmation numbers)

Selected by frozen eval (3 reps), confirmed on independent trial seeds
(`--trial-seed-base 50000`, 5 reps, mu 0.1/0.2/0.3/0.4, 20 trials each).

| arm | roll_mean | rate | ePass | eNfwd | flags |
|---|---|---|---|---|---|
| `no_back` | **-30.4** | 48 | 0.90 | **0.321** | HELD-LEAN, unstable |
| `no_progress` | -20.9 | 40 | 0.70 | 0.205 | HELD-LEAN, unstable |
| `no_swing` | -10.4 | 67 | **1.00** | 0.165 | unstable |
| `prog3` | +6.5 | 109 | 0.00 | 0.114 | unstable |
| `no_scrub` | +2.1 | 35 | 0.40 | 0.046 | unstable |
| `back05` | -1.9 | 34 | 0.25 | 0.044 | unstable, ASYM |
| `no_smooth` | -8.7 | 28 | 0.35 | 0.036 | STATIC-TORSO, unstable |
| `no_track` | +2.2 | 52 | 0.20 | 0.029 | FAKE-TORSO, unstable |
| `no_energy` | -15.8 | 18 | 0.00 | 0.007 | STATIC-TORSO, FAKE-TORSO |
| `no_fall` | +19.1 | 15 | 0.00 | 0.005 | STATIC-TORSO, FAKE-TORSO |

Across the 10 arms, `corr(|roll_mean|, net_fwd) = +0.650` and
`corr(roll_rate, net_fwd) = +0.371`. Arm-level, n=10, and the arms differ in reward,
so this is not a controlled comparison.

`no_back` per-trial, the arm with both the best speed and the deepest lean:

```
mu 0.1   -39.4   +6.3  -24.6   -3.8  -21.5     nfwd 0.344  pass 1.00
mu 0.2   -39.4  -30.7  -14.9  -25.8  -39.5          0.388       1.00
mu 0.3   -29.1  -38.9  -34.3  -39.4  -37.9          0.398       1.00
mu 0.4       -  -43.2      -  -43.1  -47.7          0.154       0.60
```

Negative in **17 of 18** scored trials, and the lean deepens monotonically with mu.
RMS 34.3 against mean -30.4 gives a swing of sqrt(34.3^2 - 30.4^2) = **15.9 deg about
a -30 deg offset** — a real oscillation riding on a large held lean, which is why
`STATIC-TORSO` (rate 48 dps) does not fire and only `HELD-LEAN` does. This is the
gait Ben described: *"the torso leaning one side and the leg is compensating all the
time to make it walk, from data it looks good as fast."* It is the top-scoring arm
of the night.

## Mechanism — nothing in the reward can tell a held lean from a swing

Reading the reward source rather than the results:

| term | weight | reads the torso? |
|---|---|---|
| `track`, `progress`, `back` | 0.8 / 1.0 / 2.0 | no — base vx only |
| `swing`, `scrub` | 1.0 / 0.8 | no — legs and feet |
| `fall` | 10.0 | **no** — `_tilt()` is the ROOT body (`grid4_rl_env.py:514`) |
| `energy` | 0.0005 | **no** — `f[LEG_IDX]*v[LEG_IDX]`, "torso EXCLUDED" (line 468) |
| `smooth` | 0.01 | yes, but taxes action CHANGE |
| `hf` | **0.6** | yes, `HF_IDX` = hips+torso, taxes commanded HF residual |

So the torso is an unpriced actuator in one direction only: **holding it at a lean
costs nothing, moving it is taxed twice, and no term pays for either.** The fall test
watches the root, so a torso parked at -45 deg world roll never terminates an episode.
A held lean does not merely tie with a swing under this reward — it strictly dominates.

That bounds the whole search. Within the brief ("tune existing weights, even to zero,
no new terms"), no weight can reward torso swinging, because no term measures it. The
only reachable move is to stop taxing it.

**And `hf` had never been measured.** It is absent from `ablate_arms.txt` (both the arm
list and its frozen-baseline header) and was absent from the budget table in
`report_gen.sh`, so it never appeared in any generation report I produced — while being
**-0.163/step on `no_back`, the second-largest penalty in the budget and 15x r_smooth**.
`report_gen.sh` now prints `hf` and a `net` column. Adding `net` immediately shows
`no_progress` at **-0.094/step** and `no_track` at -0.064/step: both survive ~390 steps
while accruing negative reward every step, because falling costs -10 once.

### gen03 — measuring the term that was never measured (launched 21:23)

| candidate | change |
|---|---|
| `no_hf` | `hf=0.0` |
| `hf02` | `hf=0.2` |
| `hf0_sm0` | `hf=0.0 smooth=0.0` (all torso-motion tax removed) |

Prediction, stated before the run: removing the tax should NOT by itself produce a
swing, because nothing rewards one. If the lean survives at `hf=0`, that is evidence
the lean is load-bearing for the gait rather than a pricing artifact — and the
question stops being a reward-weight question, which is a result for Ben to judge
rather than something to tune around.
