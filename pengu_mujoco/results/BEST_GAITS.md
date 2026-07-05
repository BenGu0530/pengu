# Best gaits registry (penguV3, 25° fwd pitch, floor μ=0.7)

Model: penguV3 | COM/leg = 1.05 (penguin) | hip_off=30° (~25° fwd pitch, eased in)
Sim: T_HOLD=5s, T_TRANSITION=4s, measure window after SETTLE=11s, SIM=24s.
Metrics: net_fwd_speed = net forward displacement / walk time (real progress);
straightness = net disp / path len; single_frac = single-support fraction;
μ_req p95 = stance-gated 95th-pct friction demand.

---

## ⭐ fine2_best_netfwd  — "so beautiful"  (marked 2026-07-02)
The best gait by REAL forward progress + grace. Found in fine2 (wide-phase re-sweep);
fine1 had missed it (only swept hip_phi 250-300).

| param      | value |
|------------|-------|
| freq       | 1.59 Hz |
| hip_phi    | 180°  |
| leg_amp    | 110°  |
| hip_amp    | 20°   |
| torso_amp  | 20°   |
| torso_phi  | 0°    |
| hip_off    | 30° (~25° fwd pitch) |

Results:
- net_fwd_speed = 0.226 m/s   (path_speed 0.436)
- straightness  = 0.521
- single_frac   = 0.999  (near-perfect L/R alternation)
- μ_req p95      = 0.55  (peak 0.70)
- GRACE: lat_sep 0.086 m | COM_regularity 0.70 | weight_transfer +0.94

Reproduce: physics/gait_report.py with G = freq1.59/leg110/hip20/torso20/hip_phi180/torso_phi0/hip_off30.
Video: results/gait_sweep/fine2_best_netfwd.mp4

Why it wins: path_speed's #1 (freq1.96/hip_phi210) actually had net_fwd = -0.12 (net
backward / curling). net_fwd_speed + wide-phase sweep together surfaced hip_phi=180.

---

## ⭐ fine3c_penguin_f1.27 — recommended NATURAL penguin gait  (marked 2026-07-05)
From the completed 3.97M-cell fine3c low-freq sweep. Same gait family as the 1.59 gait
(hip_phi~180-210, torso_phi=0) but at the true penguin frequency, with the lowest
friction demand of any gait found. See results/SWEEP_ANALYSIS.md for full analysis.

| param      | value |
|------------|-------|
| freq       | 1.27 Hz (penguin) |
| hip_phi    | 210°  |
| leg_amp    | 115°  |
| hip_amp    | 22°   |
| torso_amp  | 20°   |
| torso_phi  | 0°    |
| hip_off    | 30° (~25° fwd pitch) |

Results:
- net_fwd_speed = 0.215 m/s   (path_speed 0.392)
- straightness  = 0.549
- single_frac   = 1.000  (perfect L/R alternation)
- μ_req p95      = 0.469  (peak 0.70)  <- lowest of all gaits; best for slippery floor
- GRACE: lat_sep 0.139 m | COM_regularity 0.82 | weight_transfer +0.57

Reproduce: physics/gait_report.py with G = freq1.27/leg115/hip22/torso20/hip_phi210/torso_phi0/hip_off30.
Video: results/gait_sweep/fine3c_penguin_f1.27.mp4

Note: forward walking in this family switches on abruptly at freq 1.24 Hz (below that it
rocks in place); the penguin plateau is 1.24-1.31 Hz. A second high-freq plateau exists
at 1.42-1.50 Hz (net_fwd up to 0.228).
