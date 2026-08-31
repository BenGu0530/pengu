# GRID-5 planning session memo — 2026-08-26

Scope: everything planned and built today for the round-2 sweep. The frozen protocol
itself lives in `docs/grid5_design.md`; this memo records the path to it — what was
measured, what Ben decided, what was implemented and verified, and what remains.

---

## 1. Starting point

GRID-4 was complete (6 configs x 1,818,000 rows, protocol a22f80b; c4 full-passer
topupK5 shipped). Ben called for a second round with a better design, with concerns
named up front: slip vs rolling needs a real mathematical boundary (penetration,
contact happens while rolling); GRF as a slip readout; lateral distance/velocity;
the IMU-on-torso attitude problem (Euler roll mixes once the torso pitches); a
properly designed selection mechanism (not post-hoc top-20); COT scanned everywhere
so the lowest-COT and lowest-slip champions are selected independently, not picked
from the speed champions.

## 2. What was measured today (evidence behind every design change)

**GRID-4 math audit.** Seeding, COM bisection, hinge-axis tilt math, anti-windup,
per-trial reset all verified correct. Found: `hip_off` is applied WITHOUT the alpha
blend (`gait_config.py:127`) — a 0->50 deg step into a kp=50 servo at the first
transition instant — while the `gait_sweep.py:52` comment claims it is "eased in,
NOT a teleport" (comment wrong). Also: the sim kappa-PID runs at 1 kHz vs 50 Hz on
firmware (gain equivalence unverified, open item); rerunning c1@mu0.7 fallers on
this machine flips 20% to survivors (cross-platform FP on borderline cells — the
known reason pass-level stats are the robust layer).

**Robust-region decomposition (the mu-0.7 question).** Survival itself falls with
mu for every config (c1 99.9->74.4%, c4 95.5->73.1% from mu0.1 to 0.7) — not a gate
artifact alone. Among surviving non-passers the heading gate is a minor factor
(0.1–3.8%); the mass is net_fwd failure (stall/retrograde; 34–51% of survivors have
negative net). Paired cells (same gait, mu0.1 vs 0.7): 14–26% of the whole grid
survives on ice and falls with grip, signature = large hip_amp; c4's such cells had
median net +0.059 at mu0.1 — i.e. low-mu passers that slide. Among ALL passers at
mu0.1 the median slip_ratio is 0.9–2.5 (slid 1–2.5x the stepped path); at mu0.7 it
drops to 0.11–0.30. Low-mu volume is largely slide-walking; high-mu passers are
mostly true steppers.

**Fall-time probes (the biggest single finding).** Replaying c6@mu0.1 fallers with
the map's exact seeds: 41.7% fall during HOLD (before any gait), 47.5% during the
transition — 91.7% of c6's falls happen before the measurement window; only 8.3%
fall while walking. Variants: hip_off ramp alone revives 18%; ramp + no jitter
revives 46%; the hip_off step alone kills 71% even without jitter. c1@mu0.7 is the
opposite: 76% of falls happen in settle/walk — high-friction falls are real.

**Jitter calibration (per-axis, 5 COM points, standing only).** The killer axis is
pitch alone: yaw +-10 deg and lateral +-2 cm are 100% harmless at COM 1.05–1.31.
Pitch cliffs: 1.05/1.10 at +-4, 1.20 at +-3, 1.31 degrades from +-1 (58% at the
protocol's +-3). COM 1.40 cannot stand at pitch=0 AT ALL (deterministic fall;
yaw/lat-only = 0% survival) — the same physics as the hardware "tips backward at
hips 0" that firmware fixed with HIP_REST_DEG.

**Stand rest lean scan.** Forward lean >= +5 deg gives 100% standing survival under
the FULL nominal jitter at both 1.31 and 1.40 (vs 73%/33% at lean 0). Jitter does
not need to be reduced; the start pose needed the firmware's rest lean.

**Staged-start validation, all six configs** (1944-cell subgrid x mu{0.1,0.3,0.7},
same seeds, staged without rest lean): gains grow monotonically with COM — c1
neutral, c3 mu0.1 pass 3.0->6.9%, c5 19.8->29.6%, c6 2.5->12.9% (x5); single small
regression c4@0.1 (29.2->25.3, gentle start removes some slide-passers); high-mu
pass barely moves (real walking failure preserved). Full table in
`docs/grid5_design.md` appendix.

**Penetration / contact structure.** Champion replays: median penetration 0.12–0.42
mm, impact tail p90 4 mm / max 6.4 mm at mu0.1; GRF impact spikes to 3.3x BW; a
loaded foot usually has ONE contact point (median 1, max 2–3) — the mesh-hull foot
rolls as discrete facet hops, so the real robot's smooth-curve (roly-poly) support
does not exist in sim. Per Ben: geometry stays matched to the hardware; penetration
is a known engine property, documented, not "fixed".

**hip_phi dead band.** Pooled dead core 150–190 deg (<0.4% of passers each, no real
champion anywhere; the single c6@high-mu top-20 seat at 180 comes from a 29–39-cell
population). Band width varies strongly by config (c1 dead 50–220; kappa-2 configs
only 150–190) — only the common core is safe to cut.

**Axis edge saturation.** Per-mu top-20 sit ON the leg_amp=125 and hip_amp=28 edges
(20/20 in several cells); hip_off saturates BOTH ends, opposite by COM (low COM at
off=50, high COM at off=10). freq <=1.20 produced no champions anywhere (c6@0.7:
2/20 only).

**Slip dual-criterion calibration** (`grid5/slip_calib_probe.py`): static stand
noise floor p99 = 0.0023 m/s; ROLLING regime (mu0.9 slow gait) shows v_tan p50
0.022 / p99 0.095 m/s — the old kinematic-only integral counted this rolling as
slip — but its cone utilization p99 = 0.82, so the cone leg rejects 100% of it;
SLIDING regime (mu0.05): cone pegged at 1.0, 94.4% classified. Separation is total
and insensitive across v0 in [0.002,0.02], c in [0.5,2].

**IMU frame validation** (`grid5/imu_frame_probe.py`): synthetic round-trip exact
(6e-15 deg over 125 poses, yaw+-150/pitch+-60/roll+-70); gravity-vector roll ==
correctly-decomposed Euler roll IDENTICALLY (same matrix row) — the hardware Euler
mixing is a BNO055 device property, so firmware should compute
`roll = atan2(g_x, -g_z)` from the raw gravity vector. On the c6 champion
trajectory (yaw RMS 47.6, pitch RMS 30.6 deg), gravity-roll matches the hinge-axis
roll (what the kappa-PID nulls) with residual RMS 0.23 deg / max 0.69 deg after the
hinge-sign correction — this one formula is the whole "extra layer of math" mocap
needs. Demo video: `results/grid5_probes/imu_frame_demo.mp4`.

## 3. Decisions (Ben, 2026-08-26)

1. Physics and execution layer IDENTICAL to GRID-4 (no sv1 slew, no cmd filter, no
   torso clamp) — direct A/B comparability. Foot/contact geometry stays matched to
   hardware.
2. Slip = DUAL criterion (cone AND kinematic-with-deadband); constants frozen at
   eps=0.05, v0=0.005, c=1.0. Old slip integral kept as a continuity column.
3. COT recorded as cot_net AND cot_path; selection uses cot_net. E = positive
   mechanical work (servos do not regenerate).
4. Champion tracks: T-speed / T-cot / T-slip selected INDEPENDENTLY per (config,mu),
   with a relative speed floor (net_fwd >= 50% of that cell's speed champion).
5. Pass gate unchanged; ALL raw records saved; tiers (surv-only / pass / strict
   heading>=0.9 / clean-pass slip<=0.05) recomputed post-hoc — no rushed threshold
   decisions.
6. COM ladder extended: {1.05, 1.10, 1.20, 1.31, 1.40} x kappa{0,2} = 10 configs
   (c7=k0@1.10, c8=k0@1.40, c9=k2@1.10, c10=k2@1.40). Phase A sweeps c1–c6 first,
   Phase B adds c7–c10. 1.40 hardware note: counterweight ~33 mm above the 1.31
   position (travel to be confirmed).
7. Axes: freq 1.21–2.00 (80); hip_phi minus {150..190} (31); leg_amp +{135} (6);
   hip_amp +{32} (6); hip_off {0..50} (6, no 60+ — that lean is hardware-
   meaningless); mu unchanged. = 2,142,720 rows/config (1.18x GRID-4). Old low-freq
   data is NOT copied into grid5 (protocol changed; GRID-4 remains the reference
   for that band).
8. Start protocol: quiescence hold (max|qvel|<0.3, 2–10 s) + rest lean 5 deg for
   ALL initial poses + hip_off ramps with the transition alpha (from the rest lean,
   downward for off=0 — negative command excursion accepted). DR jitter amplitudes
   UNCHANGED (rest lean makes standing 100% at every COM).
9. IMU: no CAD needed for orientation — torso is rigid, so torso attitude IS the
   IMU attitude up to a fixed mount rotation; record torso z-x-y Euler + pitch, and
   gravity-roll is the firmware/mocap form. CAD only if a mounted-rotated IMU or a
   simulated accelerometer is ever needed.

## 4. Built and verified today

- `models/pengu1_10/`, `models/pengu1_40/` — baked from pengu1_31, only the
  easytorso inertial pos differs (-67.82 / +41.53 mm); total mass 2.2724 kg
  bit-identical; ratios 1.1000/1.4000; cross-checked against the load-time slide to
  <1e-10 m; registered in `_COM_MODELS` (grid5 copy).
- `grid5/` — self-contained duplicate of the frozen GRID-4 pipeline (427b701),
  edited only here; `physics/` + root modules untouched as the GRID-4 backup:
  `gait_config.py` (RAMP_HIP_OFFSET), `gait_sweep.py` (STAGED_START +
  EXTENDED_METRICS, +19 columns), `grid5_sweep.py` (10 configs, new axes, staged
  start, per-CSV manifest.json with refuse-on-mismatch), `topup_k.py` (K=1->5 merge
  incl. extended columns), `run_sweep.sh` / `resume_config.sh` / `sweep_watchdog.sh`
  (grid5-tagged cron, does not disturb grid4 lines), `slip_calib_probe.py`,
  `imu_frame_probe.py`.
- Verification: grid5 vs pristine GRID-4 code with flags off is BIT-IDENTICAL
  (40 trials x 16 metrics, 0 mismatches; extended-on leaves legacy columns
  bit-identical); staged smoke correct (t_start/fall_phase, no T_HOLD leak); full
  chain smoke (initcsv -> manifest -> run -> resume -> topup merge); manifest
  refusal fires on K and slip-constant mismatch; COM calibration pinned to the
  hips-0 design neutral (c6 slide +8.73 mm == GRID-4 == baked models).

## 5. Timeline (throughput anchor: GRID-4 ~2–3 machine-days / 1.82M rows)

| when | what | where |
|---|---|---|
| 08-26 | this session: design, evidence, code, verification | rml2 |
| 08-27/28 | commit (branch TBC with Ben), write `grid5_select.py`, launch Phase A | fleet |
| ~08-29 – 09-04 | Phase A: c1–c6 (2.14M rows each, ~2.5–3.5 d/config/machine) | Mac x2, rml3 x2, C, E |
| ~09-04 – 09-08 | Phase B: c7–c10 on first-free machines; Phase A topup K=5 + 3-track selection + neighborhood scans interleaved | fleet |
| ~09-08 – 09-10 | Phase B topup/selection; 10-config report (4 robust-region tiers + diagonal); champions -> hardware sketch candidates | rml2 + idle |
| throughout | RL track untouched | rml2/D |

## 6. Open items

- Commit + branch confirmation (Ben), then fleet launch order.
- `grid5_select.py` (three-track selection / +50000-seed confirmation / neighborhood
  fine scan) — needed before analysis, not before launch.
- Gravity-roll firmware patch for the three champ sketches (validated here; to be
  written before the next mocap session).
- 1.40 counterweight rail travel check on the physical robot.
- Sim kappa-PID rate (1 kHz) vs firmware (50 Hz) gain-equivalence probe.
- EXP raw outputs of today's probes live in the session scratch only; the numbers
  are recorded here and in `docs/grid5_design.md`.

---

## Addendum (2026-08-26, late night): protocol revision grid5-v2 — deterministic map

After reviewing what "DR at K=1" actually means per row (each row = exact gait
params x ONE random draw of mu±5% + pose jitter; no pure run anywhere), Ben ruled:
**"everything fixed, no random stuff at sweep — don't contaminate the sweep."**

- The map is now a PURE DETERMINISTIC sweep: exact nominal mu, no pose jitter, no
  RNG, K=1. Verified bit-identical across two full smoke runs.
- DR/robustness testing moves entirely to the post-map champion stage ("add more
  DR after sweep is done and check the champ if it is legit"), design fixed once
  the map completes. `grid5/topup_k.py` (v1 jittered merge) is guarded out.
- Manifest protocol bumped to grid5-v2; v1 artifacts are refused by every consumer.
- All three running machines (rml2 c6, rml3 c5, naomio c4) were killed, their
  ~half-day v1 partials archived to `results/gait_sweep/old_jittered_v1/`, and
  relaunched under v2 the same night. mac (c3->c8) and laptop (c1) deploy
  unchanged — a fresh pull is already v2.
- Post-map test candidates recorded: champion DR repeats (form TBD), deterministic
  nominal-mu reference eval, independent-seed confirmation.

## Addendum (2026-08-31): the naomio OOM — first real out-of-memory in the project

Symptom: naomio at 30.5/31.8 GB used, swap full, kernel oom_kill eating shards
(30 -> 22 alive); each shard RSS 1.34 GB. Cause: the inherited `_load_done`
resume loads EVERY completed row as a Python tuple into a per-shard set —
~400 B/row x 3M rows x 30 independent processes (no COW sharing; they are nohup
python, not forks). Cost scales with map fill, so it looked fine at launch and
detonated at ~3M rows. Not a leak (steady while running, bigger on every
restart), not page cache, not zombies.

Fix: bitmap resume — every deterministic-grid row has a unique integer index
(cell_index*len(MUS)+mu_index), done state = 1 bit/row. Measured on the live
4.6M-row c6 CSV: 22 s load, 0.65 MB bitmap, 83 MB process peak RSS (16x less
per shard). All three machines rolled onto the fix the same hour; naomio shard
total 29.3 GB -> 4.9 GB. Rule recorded: per-shard resume state must not scale
with map size. Side effect to clean at ship time: ~141k duplicate/torn rows in
naomio's c4 CSV from the oom-kill/watchdog double-write window (bitmap resume
dedups; the ship-time integrity battery removes them).
