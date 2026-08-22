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
