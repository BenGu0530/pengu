# GRID-5 — round-2 co-design sweep: frozen protocol

Status: **grid5-v2, relaunched 2026-08-26 (late night)**.
REVISION v2 (Ben, 2026-08-26): **the map is a PURE DETERMINISTIC sweep** — exact
nominal mu, no pose jitter, no RNG anywhere, K=1 ("everything fixed, no random
stuff at sweep — don't contaminate the sweep"). Every row = exact gait parameters
x exact nominal environment, twice-run bit-identical (verified). DR/robustness
testing moves ENTIRELY to the post-map champion stage ("add more DR after sweep is
done and check the champ if it is legit"), to be designed once the map completes;
grid5/topup_k.py (the v1 jittered merge) is guarded out and refuses to run.
The v1 jittered partial maps (~half day of c4/c5/c6 rows) are archived under
results/gait_sweep/old_jittered_v1/ on each machine, never merged.
Code: `grid5/` (self-contained duplicate of the frozen GRID-4 pipeline @427b701;
`physics/` and the root modules are the untouched GRID-4 backup — do not edit them).

## What round 2 changes vs GRID-4 (protocol a22f80b), and why

| area | GRID-4 | GRID-5 | evidence |
|---|---|---|---|
| configs | 6 (kappa{0,2} x COM{1.05,1.20,1.31}) | **10** (+ COM 1.10, 1.40 -> c7..c10) | Ben 2026-08-26; 1.20->1.31 order-of-magnitude cliff |
| freq | 1.00–2.00 @0.01 (101) | **1.21–2.00 (80)** | no top-20 champion below 1.21 in any (config,mu) (c6@0.7: 2/20 only) |
| hip_phi | 0–350 @10 (36) | **FULL 36 (v2.1)** — the earlier {150..190} trim was dropped | trim evidence came from the jittered + step-start GRID-4 protocol (proven to kill whole strata); Ben 2026-08-26: measure the full circle cleanly. freq trim stays (champion-physics rationale) |
| leg_amp | 85..125 (5) | **65..165 @10 (11, v2.2)** | high edge: top-20 saturated at 125 and geometry shows stroke headroom (135=79%, 165=97% of the 50 mm rail); low edge: c5/c6 passer density peaks at 85 and the low-amplitude/COT band starts there (65 = 9.3 mm, just above contact noise) |
| hip_amp | 12..28 (5) | **8..32 @4 (7, v2.2)** | high edge: top-20 saturated at 28; low edge 8 deg = ~13 mm swing, low-amplitude/COT band |
| hip_off | 10..50 (5) | **0..50 (6)** | high-COM champions sat on the off=10 edge; no 60+ (Ben: hardware-meaningless lean) |
| start | fixed hold 5s; hip_off STEP at transition start | **staged**: quiescence hold (max\|qvel\|<0.3, 2–10s) + **rest lean 5 deg** + hip_off **ramps** with alpha | 91.7% of c6 mu0.1 falls were pre-measurement; ramp+lean revives the off>=30 shelf (pass x5 on the c6 subgrid); rest lean >=5 deg -> 100% standing under full DR jitter at every COM incl. 1.40 (which cannot stand at lean 0 at all). Mirrors firmware READY (HIP_REST_DEG) |
| metrics | 12 cols | **+19 ext cols** (fall timing/phase, dual-criterion slip, cone/GRF, lateral, COT, torso-IMU) | slip/roll boundary, COT tracks, failure-mode decomposition (Ben 2026-08-26) |
| manifest | none | **manifest.json per CSV; consumers refuse on mismatch** | lessons_2026-08-25 §5h |
| DR | jittered K=1 map (mu ±5%, pose jitter) + topup K=5 | **NONE in the map (v2)** — deterministic rows; DR only at the post-map champion stage | Ben 2026-08-26: uncontaminated per-row numbers; champion legitimacy checked with DR afterwards |
| gates / execution layer | — | unchanged (pass = surv ∧ head>0.5 ∧ net>0.05; no slew/no cmd filter) | comparability with GRID-4 |

Rows per config: 80 x 36 x 11 x 7 x 6 x 4 mu = **5,322,240** (2.93x GRID-4).
(v2.1: hip_phi restored to the full circle; v2.2: leg_amp 65-165, hip_amp 8-32 —
both mid-run: deterministic v2 rows have no seed/index dependence, completed rows
stay valid and resume simply adds the new cells.)
All raw records are saved; pass tiers are recomputed post-hoc (surv-only / pass /
strict heading>=0.9 / clean-pass = pass ∧ slip_ratio2<=0.05).

## Evidence appendix: staged start vs GRID-4 map (2026-08-26 validation batch)

Six configs, 1944-cell subgrid (freq@0.2, phi@30, leg/hip/off 3x3x3), mu {0.1,0.3,0.7},
same seeds as the map, staged start WITHOUT rest lean (final protocol adds lean 5 deg,
which additionally removes the jitter standing falls — EXP2b: 100% standing at every COM):

| config | mu0.1 pass map->staged | mu0.3 | mu0.7 | mu0.1 surv map->staged | startfail@0.1 |
|---|---|---|---|---|---|
| c1 k0 1.05 | 10.2 -> 9.7  | 15.7 -> 16.6 | 6.3 -> 6.8 | 99.8 -> 99.8 | 0.0% |
| c2 k0 1.20 | 9.3 -> 12.9  | 9.1 -> 11.2  | 3.6 -> 4.4 | 65.5 -> 84.9 | 10.1% |
| c3 k0 1.31 | 3.0 -> 6.9   | 4.2 -> 5.0   | 1.0 -> 1.4 | 21.7 -> 37.6 | 60.5% |
| c4 k2 1.05 | 29.2 -> 25.3 | 28.9 -> 30.1 | 9.8 -> 10.9 | 93.5 -> 93.6 | 4.0% |
| c5 k2 1.20 | 19.8 -> 29.6 | 9.3 -> 16.2  | 2.0 -> 3.9 | 51.5 -> 76.0 | 10.5% |
| c6 k2 1.31 | 2.5 -> 12.9  | 1.4 -> 3.7   | 1.1 -> 1.3 | 15.2 -> 35.4 | 53.4% |

Gains grow monotonically with COM (c1 neutral -> c6 x5); the single small regression is
c4@0.1 (gentle start removes some slide-passers). High-mu pass barely moves — the
high-friction scarcity is real walking failure and is preserved as signal.

## Configs

c1..c6 as GRID-4 (kappa{0,2} x COM{1.05,1.20,1.31}); c7=k0@1.10, c8=k0@1.40,
c9=k2@1.10, c10=k2@1.40. All built from the single hardened base `models/pengu1_31`
(2.2724 kg) by the load-time mass-conserving COM slide; **the ratio is defined at the
hips-0 design neutral** (rest lean is a start-protocol pose only — calibration sets
lean=0, so c3/c6 slides match GRID-4: 1.31->+8.73mm). Baked reference models
`models/pengu1_10`, `models/pengu1_40` (robot.xml differs from 1_31 only in the
easytorso inertial pos; verified vs the slide to <1e-10 m; STLs byte-identical).
Phase A sweeps c1..c6 first; Phase B adds c7..c10 (Ben's ordering).
Hardware note: 1.40 needs the counterweight ~33 mm above the 1.31 position.

## Slip dual criterion (frozen 2026-08-26 by grid5/slip_calib_probe.py)

A contact is SLIPPING iff BOTH:
- cone:      |Ft| >= (1-0.05) * mu_trial * Fn        (penetration/patch-immune)
- kinematic: |v_tan| >= 1.0 * |omega_foot| * r_patch + 0.005 m/s

Calibration: static stand noise floor p99 = 0.0023 m/s (v0 = 2x); ROLLING regime
(mu0.9 slow gait) has v_tan p50=0.022/p99=0.095 m/s — the old kinematic-only integral
counted this rolling as slip — but cone util p99=0.82 so the cone leg rejects 100% of
it; SLIDING regime (mu0.05 aggressive) is 94.4% classified, cone pegged at 1.0.
Separation is total and insensitive across v0 in [0.002,0.02], c in [0.5,2].
`slip_dist` (old integral) is still recorded for continuity; `slip_dist2/slip_frac/
roll_dist` are the new decomposition; `cone_util_p50/p95` is the GRF readout.

## Extended columns (appended after the 12 GRID-4 columns)

t_start, t_fall, fall_phase("hold|trans|settle|walk" tally), slip_dist2, roll_dist,
slip_frac, cone_util_p50, cone_util_p95, fn_peak, fn_mean [BW], lat_disp,
lat_vel_rms (heading-frame), e_pos [J, positive mechanical work], cot_net, cot_path,
power_mean, imu_roll_mean, imu_roll_rms, imu_pitch_rms [deg; torso attitude relative
to the per-trial rest pose, z-x-y intrinsic; roll = gravity-method lateral lean].
Aggregation over K: nan-mean (fall_phase: count tally). COT selection uses cot_net.

## Selection mechanism (per (config,mu); frozen before analysis)

1. eligibility = pass (GRID-4 rule; strict/clean tiers reported alongside);
2. three INDEPENDENT champion tracks, top-20 each:
   T-speed (net_fwd_mean desc) | T-cot (cot_net asc) | T-slip (slip_ratio2 asc),
   the latter two with floor net_fwd >= 50% of that cell's T-speed #1;
3. union -> champion DR stage (jittered repeats; exact design fixed post-map —
   `grid5/topup_k.py` v1 merge is guarded out until then);
4. confirmation on independent seeds — reportable numbers (post-map design);
5. champion neighborhood fine scan (hip_off ±5 @1 deg, freq ±0.02 @0.005, others ±1
   grid step); champions ranked by neighborhood mean, annotated spike/plateau.

## Verification already done (2026-08-26)

- grid5 vs pristine GRID-4 code, flags off: **bit-identical** (40 trials x 16 metrics,
  0 mismatches). Map-vs-local diffs are cross-platform FP (known, pass-level stats).
- EXTENDED_METRICS on: legacy columns still bit-identical.
- staged smoke: t_start/fall_phase correct, T_HOLD not leaked, ext columns populated.
- end-to-end: initcsv -> manifest -> run -> resume(12/12 skip) -> topup merge (K=3).
- manifest refusal fires on K / slip-constant mismatch.
- COM calibration at hips-0 neutral: c6 slide +8.73mm == GRID-4 == baked models.

## Ops (fleet)

    bash grid5/run_sweep.sh cN [n_shards]     # initcsv+manifest+shards, resume-safe
    bash grid5/resume_config.sh cN            # pull -> snapshot -> resume -> watchdog
    bash grid5/sweep_watchdog.sh install cN   # @reboot + 10-min cron (grid5-tagged,
                                              #  does not disturb grid4 crontab lines)
    CONFIG=cN python grid5/topup_k.py <map.csv> <select.csv|->

Machines (Ben): Phase A on Mac x2, rml3 x2, C, E; rml2 stays on the RL track.
Throughput anchor: GRID-4 ran ~2-3 machine-days per 1.82M rows; grid5 is 1.18x that
per config. Ship-back: gzip CSV + manifest, `git add -f`, confirm branch with Ben.
