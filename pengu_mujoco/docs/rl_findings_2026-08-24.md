# RL findings digest — 2026-08-24 (Mac -> machine D)

Status digest of the ice-arm RL line for machine D. Informational only; no
tasks assigned here. Numbers are measurements; rulings are Ben's, quoted as
such. Full detail: docs/rl_session_2026-08-21.md sections 5+, knob log in
docs/rl_e2_ice_memo.md, code-side version comments in grid4_rl_env.py.

## 1. Execution layer: sv0 -> sv1 (servo slew clamp)

All five actuators are XM430-W350. Applied position targets are now clamped
at the no-load speed, 46 rpm @ 12 V = 4.82 rad/s (models the servo's Profile
Velocity; the XML forcerange +-4.1 N*m is the same datasheet's 12 V stall
torque). `--no-slew` reproduces legacy sv0 bit-for-bit; run tags append
`sv1`; sv0/sv1 pools must not be mixed.

Measured impact:
- c6 designed replay: mu0.1 0.397 -> 0.374 (-6%); mu0.3 0.087 -> -0.008
  (2/3 falls). c6's cranks ran 60-69% of the walk window past the limit in
  sv0 (max 8.4 rad/s) — that overspeed was NOT load-bearing at mu0.1.
- Confirmed RL champions are nearly clamp-invariant zero-shot: a2p0@2000k
  16/20 -> 16/20; s7@2750k 17/20 -> 17/20 (mu0.1 0.646 -> 0.597).
- Training NATIVELY under sv1 (same recipe/seed as an sv0 counterpart) came
  out faster AND lower-frequency: mu0.1 0.401 @ 2.60 Hz vs 0.307 @ 3.25 Hz.

## 2. Reward line: r3d -> r3e -> r3f

- r3d (hf=0.6, commanded residual, hips+torso only) is the surviving hf
  variant. Mechanism ledger of the failures is in grid4_rl_env.py comments:
  band-scaled = noise suicide; executed-signal = stand collapse (taxes the
  1-3 Hz walking band, exempts noise); executed-accel = noise suicide.
- r3e adds `dutybal` = -0.3 * max(0, |dutyL-dutyR|/(dutyL+dutyR) - 0.25),
  armed after 1 s of single-support samples. Ruling (Ben): HELD-LEAN (torso
  parked 30-53 deg) is a cheat, but direct torso penalties are off-limits
  (disturb emergence). The `straight` stride proxy FAILED calibration (ice
  slip noise: every walker pays 0.5-0.7/step, no separation; it stays 0).
  Single-support duty imbalance separates HEAVY lean (0.474 at -53 deg) from
  clean walkers (0.206) at 2.3x; mild lean (15-20 deg) has no leg-level
  signal anywhere and is REPORTED, not priced. Probe outcome: a one-sided
  big-sway burst got pruned back to duty symmetry while KEEPING the sway;
  best contact structure of any run (airborne 15-16%, double support
  16-17%, mu0.4 walking at 2.20 Hz).
- r3f: energy weight 0.0005 -> 0 (removed, Ben 2026-08-24). For the record:
  an energy x20 A/B SPED THE WALKER UP (mu0.2 0.547, mu0.3 0.367 at 2.20 Hz
  with torso RMS 8 deg = 151% of the c6 mu0.3 ceiling) rather than slowing
  it; the removal is Ben's ruling, logged without interpretation.

## 3. Eval-side change: tier 3 split

Tier 3 now prints `3-lean` / `3-sway` (trial-level flag
|roll_mean|/roll_RMS > 0.7, majority over passing trials). Lean and sway are
never pooled in torso-activity claims. This matches machine D's HELD-LEAN /
STATIC-TORSO flag logic at the report level.

## 4. Headline numbers (all best-ckpt + independent-seed confirmed, sv0 era)

r3d arm, 8 seeds (runs/r3d_arm/, runs/e2x2hf4b/a2p0 = s0): 4/8 seeds tier-3
at 16-17/20; three seeds' confirmed mu0.1 net_fwd EXCEED the c6 designed
ceiling 0.469: s4 0.493, s5 0.548, s7 0.646 (137%). Failure modes: 2 weak/
stand seeds. Known trade-off: the fast family is 2.6-3.5 Hz with 17-40%
airborne; the cleanest morphology (2.00 Hz, 74% single support) is
a2p0-3M@2000k's mu0.3 slice at 0.176.

## 5. Kappa direction of the emergent sway (Ben's priority question)

eff-kappa regression (torso WORLD roll ~ hip-axis roll, deterministic walk
window) on five quiet-torso walkers:

| walker | eff_k | torso-joint slope | corr |
|---|---|---|---|
| p1_energy mu0.3 (8-deg sway) | +0.62 | -0.10 | +0.58 |
| a2p0@2000k mu0.3 (2 Hz) | +0.47 | -0.28 | +0.51 |
| r3e mu0.4 (2.2 Hz) | +1.20 | -0.16 | +0.56 |
| sv1 probe mu0.1 (0.40) | +0.19 | +0.04 | +0.37 |
| s7@2750k mu0.1 (0.65 record) | +0.35 | +0.05 | +0.37 |

Reading frame: kappa0 requires joint slope -1 (active world-upright; never
observed); rigid follow = eff 1, joint 0; kappa2 = eff 2, joint +1. The
emergent family sits at eff-kappa 0.2-1.2 with a nearly PASSIVE torso joint
("discounted rigid follow", leaning WITH the axis — kappa2 side, never
kappa0), and the fastest walkers have the LOWEST eff-kappa (0.19/0.35).
This runs in the same direction as machine D's corr(net_fwd, eff_kappa) =
-0.644 observation on the old two-stage arms.

## 6. Current recipe & artifact map

Recipe (as of r3f): single-stage e2, c2 curriculum, a2 band (0.0 +-1.9),
swing=0 scrub=0, hf=0.6, dutybal=0.3, energy=0, sv1 clamp. No arm has been
run under r3f yet. Runs: e2x2* (r2/r3 rounds), r3d_arm (8 seeds),
sv1_probe, r3e_probe, p1_energy; per-run ckpt_sweep.csv and
eval_bestckpt_confirm.csv where the protocol was applied.
