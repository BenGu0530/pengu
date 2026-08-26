# Pengu GRID-4 project status — 2026-08-25

Scope: sweeps, analysis, hardware. (RL phase runs in its own track and reports
separately — deliberately excluded here.)

---

## 1. Headline: the co-design table is COMPLETE

All six configurations swept, verified (1,818,000 rows, 0 malformed, unique
6-tuples each), committed:

| | COM 1.05 | COM 1.20 | COM 1.31 |
|---|---|---|---|
| **Gait 1 (κ=0)** | c1 ✅ | c2 ✅ | c3 ✅ |
| **Gait 2 (κ=2)** | c4 ✅ | c5 ✅ | c6 ✅ |

Protocol (frozen `a22f80b`): base = hardened `models/pengu1_31` (2.2724 kg),
mass-conserving COM slides (−86.05 / −31.37 / +8.73 mm), μ axis {0.1,0.3,0.5,0.7}
±5% rel., K=1 map + seeded staged-K topup, pose jitter, no mass jitter,
pass = survive ∧ heading>0.5 ∧ net_fwd>0.05 (slip recorded, not gated).

## 2. Six-config map (K=1, fresh rerun today — first time with c2)

Pass>0 % by μ (full table in `results/grid4_report/REPORT.md`):

| μ | c1 | c2 | c3 | c4 | c5 | c6 |
|---|---|---|---|---|---|---|
| 0.1 | 12.6 | 10.6 | 4.2 | **31.7** | 23.6 | 5.6 |
| 0.3 | 20.1 | 10.1 | 4.1 | **31.7** | 10.6 | 2.1 |
| 0.5 | 12.2 | 7.1 | 1.9 | 12.8 | 2.5 | 0.9 |
| 0.7 | 8.3 | 4.1 | 1.1 | 9.9 | 2.0 | 0.8 |

With c2 in, both single-variable ladders are complete for the first time:
- κ0 COM ladder (c1→c2→c3): volume falls monotonically with COM at every μ.
- κ2 COM ladder (c4→c5→c6): same direction, steeper.
- Matched-COM κ0-vs-κ2: κ2 leads at μ=0.1 at every COM; κ0 takes 1.20/1.31 at μ≥0.3
  (c4 is the exception — it also leads at 0.3/0.7 at COM 1.05).

Key corrections already on record (machine-D analysis session,
`docs/grid4_analysis_session_2026-08-20.md`): the μ=0.7 "collapse" was a
selection artifact (per-μ re-selection: every config walks at 0.7, pass 0.85–1.00);
"κ2 dominant" is only supportable as "κ2 leads on low friction"; low-μ passing is
partly shuffling (ds_move_frac) and net_fwd loss at high μ is partly circling
(roll–yaw coupling r≈0.96), not falling.

## 3. K=5 topups (paper-grade numbers)

- **c4: ALL 391,285 passers upgraded to true K=5** (completed today, shipped as
  `sweep_grid4_c4_topupK5.csv.gz`). First config with fully DR-hardened passer set.
- c1/c3/c5/c6: 130-cell selections (robust top-50 @μ0.1 + per-μ speed top-20)
  upgraded; c1 additionally has off=30 and freq≈1.4 special selections.
- Headline K=5 lessons: single-cell K=1 champions inflate (c6@0.1: 0.602→0.163
  mean; c5@0.3 champion pass 0.2), c4 barely shrinks (90% keep pass≥0.8),
  c3@μ0.3's 0.660 survives as 0.490 with pass 1.0 — the fastest verified gait.
- Remaining: full-passer topups for c1/c2/c3/c5/c6 (`physics/topup_all.sh cN`,
  ~4–43k..390k rows each; idle machines can take them per `grid4_topup_memo.md`).

## 4. Champion gaits → hardware (3 Arduino sketches, all K=5-verified picks)

| folder | config | body | gait (freq/phi/leg/hip/off) | K5 net @μ0.1 |
|---|---|---|---|---|
| `Arduino/pengu_champ` | c6 κ2 | stock 1.31 | 1.67/340/95/24/20 | 0.376 |
| `Arduino/pengu_champ_k0` | c3 κ0 | stock 1.31 | 1.61/330/115/28/10 | 0.164 |
| `Arduino/pengu_champ_k0_105` | c1 κ0 | counterweight −86 mm | 1.80/270/125/28/30 | 0.125 |

Firmware features accumulated this week: extended-position mode (fixes silent
rejection of negative goals), READY→absolute-zero convention with computed
extended-zero home, hip rest lean 10° (robot tips backward at 0), staged start
(4s ramp/6s settle/4s blend), κ-PID torso with `S_TILT` sign check (`s`),
hand-rock stream (`t`), motor health report (`p`: ping/pos/torque/hwErr),
IMU re-home + hardcoded −2° roll reference and ±10° torso clamp (k0_105 only so far).
Also available: c1 alternates at freq≈1.4 (1.33/350/125/28/20 best; 1.50/0/125/28/30
straightest) with rendered demos in `results/grid4_report/c1/demos/`.

## 5. Hardware day 1 (μ≈0.1 surface, c6 sketch) — mocap results

Data: `HardwareData/mu01_c6_0825/` (OptiTrack, 1000 Hz, 5 rigid bodies × 4 markers,
4 usable takes; force plates untouched). Analysis figures in its `analysis/`.

| metric | hardware (best take) | sim (c6 champion) |
|---|---|---|
| gait frequency | **1.61–1.72 Hz across all takes** | 1.67 Hz commanded |
| net speed | 0.075–0.122 m/s | 0.376 m/s |
| torso roll RMS | ~4.4° (only clean take) | 20–25° expected for κ2 |
| foot clearance | 2.3–3.0 cm | ~1.1 cm |
| straightness | 0.07–0.41 | ~1.0 |

Readings: (a) CPG timing transfers essentially losslessly (frequency validated);
(b) speed shrinks 3–5× — consistent with XM430 velocity saturation (commanded crank
peak ≈500°/s vs ~276°/s no-load) plus (c) **the κ=2 torso lean is NOT happening on
hardware** (4.4° vs 20–25°) — matches the live observation that the torso motor
misbehaved; with torso effectively locked, the sim analogue is κ0@1.31 (0.164 m/s),
much closer to what was measured. Curving matches the sim roll–yaw coupling finding.

Known analysis caveats + approved next step (plan `harmonic-foraging-fox`):
1. per-frame walking/idle/carried segmentation (takes include non-walking time);
2. heading-frame yaw→pitch→roll decomposition (Euler roll and pitch mix once the
   torso pitches 20° forward — affects BOTH the mocap metric and the on-robot
   κ-PID, which currently feeds on BNO055 Euler roll);
3. firmware fix: gravity-vector-based roll (pitch-immune) for all three sketches.

## 6. Fleet / infra

- Machines: Mac (free — c4 topup finished), rml2/D (RL track), rml3/F (free after
  c2), B (retired), C, E (idle). Watchdog (`sweep_watchdog.sh`) has auto-revive,
  reboot survival, 6h snapshots, `.done`-aware skip.
- One-line tooling: `run_sweep.sh cN`, `resume_config.sh cN`, `topup_all.sh cN`,
  `grid4_report.py`, `grid4_finalists.py`; onboarding doc `SETUP_SWEEP.md`.
- Engine pinned mujoco 3.8.x in the launcher; platform FP noise accepted by Ben's
  ruling (2026-08-19) — pass-level stats are the robust layer.

## 7. Open decisions (Ben)

1. HEAD_MIN=0.5 admits 75° course error — add a strict heading≥0.9 reporting tier?
2. hip_off 10° grid aliases 65%-per-degree structure — annotate champions
   (spike vs plateau) or fine-scan around finalists?
3. μ=0.3 / μ=0.5 per-μ top-20 re-selection still pending (2 of 4 columns done).
4. Full-passer topups for the other five configs — assign to idle machines?
5. Six-config finalists/demos rerun (`grid4_finalists.py`) now that c2 landed —
   and integrate topupK5 values into the report's speed curves.
6. Hardware: fix/verify torso motor (use `p`), re-capture mocap; the roll-fix
   firmware (plan Part B) should go in before the next capture session.

## 8. Immediate queue

1. Execute approved plan: mocap re-analysis (segmentation + heading-frame pose)
   + gravity-roll firmware for all three sketches.
2. Idle machines pick up remaining full topups.
3. Finalists + demos rerun for six configs; report speed curves switch to K=5 values.
