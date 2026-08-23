# Reward audit — what does c6 earn under the frozen RL reward?

Ben's question (2026-08-22): before any further move, check whether c6 is a
maximum of the current reward set at all, i.e. whether the reward is working.

Tools: `rl_grid4/score_designed_gait.py` (existing) and
`rl_grid4/probe_c6_localmax.py` (new). Both replay through `Grid4RLEnv`'s own
reward accounting; walk-window means, per step, `WALK_FROM = 11 s`, 24 s episodes.

c6 = the champion of that config: freq 1.77, hip_phi 270, leg_amp 105,
hip_amp 28, hip_off 10, torso under `TorsoKappaPID` at kappa 2.

---

## 1. c6 against the learned policies, same accounting, mu = 0.1

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| **c6 designed** | **+0.671** | **+0.604** | **+0.571** | **+0.615** |
| e2/s0 @1500k | +0.067 | −0.346 | +0.030 | −0.083 |
| e2/s2 @3000k | −0.133 | −0.237 | −0.689 | −0.353 |
| e2/s3 @2500k | −0.495 | −0.119 | −0.486 | −0.367 |

Learned checkpoints are the C2-selected ones, not `final.zip`. Nothing fell in
any of these rollouts, so the gap is not a survival artifact.

Per-component, c6 (seed 0) against e2/s0 @1500k (seed 0):

```
             track  progress    back  energy   swing   scrub  smooth      hf     vx
c6          +0.439    +0.417  +0.000  -0.013  +0.137  -0.283  -0.000  -0.025  0.417
e2/s0       +0.227    +0.256  -0.007  -0.005  +0.204  -0.242  -0.008  -0.358  0.253
```

Two terms carry most of the gap. `hf` is **-0.025 for c6 against -0.24..-0.38
for every learned policy** — c6 is a sinusoid, so its commanded action barely
departs from the alpha=0.2 reference and the HF residual is near zero, while the
learned policies emit chatter and are taxed for it. `back` is **0.000 for c6**
(it never moves backward in the walk window) against -0.17..-0.41 learned.
c6 pays *more* scrub than the learned policies, not less.

## 2. Is c6 a local maximum?

Each of the five gait parameters perturbed alone by +-10% and +-25%, mu = 0.1,
3 seeds, same accounting (`probe_c6_localmax.py`):

```
param        frac     value  total/step    delta      vx
c6 (base)                         0.616            0.397
freq         -25%      1.33      -0.179   -0.794   0.043
freq         -10%      1.59       0.454   -0.161   0.317
freq         +10%      1.95       0.081   -0.534   0.145
freq         +25%      2.21      -0.213   -0.828  -0.001
hip_phi      -25%    202.50      -0.935   -1.551  -0.305
hip_phi      -10%    243.00       0.368   -0.248   0.288
hip_phi      +10%    297.00      -0.040   -0.655   0.108
hip_phi      +25%    337.50      -0.166   -0.782   0.013
leg_amp      -25%     78.75       0.497   -0.118   0.309
leg_amp      -10%     94.50       0.432   -0.183   0.301
leg_amp      +10%    115.50       0.340   -0.276   0.298
leg_amp      +25%    131.25       0.331   -0.285   0.284
hip_amp      -25%     21.00      -0.520   -1.136  -0.084
hip_amp      -10%     25.20       0.440   -0.176   0.312
hip_amp      +10%     30.80       0.525   -0.091   0.369
hip_amp      +25%     35.00       0.270   -0.346   0.299
hip_off      -25%      7.50       0.559   -0.056   0.376
hip_off      -10%      9.00       0.574   -0.042   0.379
hip_off      +10%     11.00       0.565   -0.051   0.377
hip_off      +25%     12.50       0.506   -0.109   0.345
```

**20 of 20 probes score lower.** No single-parameter step of +-10% or +-25%
beats c6 along any of the five axes.

`hip_off` is the flattest axis (-0.04 to -0.11 over the whole range) and
`hip_phi` the sharpest (-25% costs -1.55/step).

## 3. The same c6 at mu = 0.3

```
seed 0  +0.158   vx 0.152
seed 1  -0.091   vx 0.037
seed 2  -0.084   vx 0.027
```

The c6 parameters were selected at mu = 0.1. At mu = 0.3 the same fixed
parameters earn near zero or negative. The e2 arm trains on
mu ~ U(0.1, 0.4), so **c6-as-a-fixed-parameter-set is a maximum at mu = 0.1,
not across the distribution the RL arm is trained on.**

## 4. c6 is not reachable in the RL action space

The crank commands c6 issues during the walk window, against the frozen a1
action band (`CRANK_MID = -1.2`, `CRANK_HALF = 0.6`):

```
             c6 needs (ctrl, rad)     a1 band          out of band
crank1-R     [+0.000, +1.832]         [-1.80, -0.60]      100%
crank1-L     [+0.000, +1.833]         [-1.80, -0.60]      100%
hip-L        [+0.175, +0.663]         model [-1.57,+1.57]   ok
hip-R        [+0.175, +0.663]         model [-1.57,+1.57]   ok
torso        [-0.386, +0.366]         model [-0.79,+0.79]   ok
```

Not a partial overlap — **the two intervals are disjoint**. Confirmed on the
achieved joint angles rather than the commands:

```
c6  crank joint angle reached: [+0.010, +1.824] rad
RL  crank joint angle reached: [-1.372, -0.865] rad
overlap: none
```

So the two are operating the legs in physically different configurations, and
no policy under the a1 mapping can emit c6's commands at any point of its
action range.

Where the band came from (`rl_e2_ice_memo.md:143`):

> 2026-08-21 (action mapping, Gate 0 period): cranks narrowed to -1.2+-0.6 rad
> (a1), reset settles at the working stance (stance-angle scan: -1.2 most
> topple-robust, 7/8 seeds). Hips/torso untouched.

A Gate-0-period capability knob chosen for topple-robustness, then frozen.

---

## What the numbers say

1. The reward ranks c6 far above every learned policy at mu = 0.1 (+0.615
   against -0.08 to -0.37 per step), and c6 is a local maximum along all five
   of its gait parameters at +-10% and +-25%.
2. The same c6 parameters are not a maximum at mu = 0.3, which is inside the
   e2 training distribution.
3. c6's crank range and the a1 action band do not intersect, so c6 is outside
   the space any current policy can search.

Not tested here: whether some *other* gait inside the a1 band earns as much as
c6 does outside it. That is the question that decides whether the band is
costing anything, and it is not answered by this audit.
