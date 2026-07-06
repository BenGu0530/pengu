# MEMO — Torso strategy × friction: first decisive results & a design correction

**Date:** 2026-07-05 · **Author:** Fable (advisory) · **Status:** first data, needs firming

## TL;DR
1. **Paper direction CONFIRMED (on 2 models):** at matched walking speed, the penguin
   **over_stance** torso strategy needs **less floor friction** than **upright** — and
   walks on lower μ. Positive sign on both v2 and v3.
2. **Design correction (important):** you **cannot** build the upright / over_swing
   conditions by flipping the torso on a fixed over-stance gait — the gait then stops
   walking, so μ_req becomes meaningless. Each mode **must be independently re-optimized
   to matched speed** (which the v2 `friction_study.py` already does). The updated
   `fable_prompt.md` step "inject marked gait into friction_study" is only valid in the
   *re-optimize-per-mode* sense, NOT "fix gait B and flip the torso."
3. **Two metric traps found & fixed** — see §3.
4. **Open flag:** the v2 effect is huge (2.4×), the v3 effect is modest (1.2×). Resolve
   before quoting a number.

## 1. What was run
| script | model | what |
|---|---|---|
| `physics/killtest_torso_modes.py` | v3 | gait A/B held FIXED, torso flipped (upright/stance/swing) × μ ladder → `results/friction_study/killtest_v3_AB.csv` |
| `physics/matched_speed_v3.py` | v3 | each mode re-optimized (leg/hip/freq) to V=0.08 m/s, then μ ladder → `results/friction_study/matched_speed_v3.csv` |
| `physics/friction_study.py` (pre-existing, Jun 24) | v2 | same matched-speed design, CMA → `results/friction_study/penguin_configs.csv` |

All μ_req use the SAME stance-gate (Fn>4N, via `gait_sweep.run_trial`) so numbers are comparable across v3 scripts; v2 uses Fn>1N (see §3).

## 2. Results — direction is consistent across all sources
**Matched-speed friction demand (μ_req_p95) and lowest clean-walk μ:**
| source | model | upright μ_req | over_stance μ_req | ratio | min_μ upright | min_μ over_stance |
|---|---|---|---|---|---|---|
| `penguin_configs.csv` | v2 | 0.997 | **0.419** | **2.4×** | 0.4 | **0.2** |
| `matched_speed_v3.csv` | v3 | 0.625 | **0.523** | 1.2× | 0.7 | **0.4** |

→ over_stance always demands less friction and walks on lower μ than upright.
On v2 it clears **acrylic (0.30)**-class surfaces; upright cannot. Only over_stance shows
`torso_stance_corr ≈ +0.5` (torso genuinely over the stance foot); upright/over_swing ≈ 0.

**over_swing is a degenerate control** across all three runs (v2 corr≈0, low leg amp; v3
can't reach matched speed, erratic single_frac). Read as: *torso-over-swing is not a
viable walking strategy on this robot* — itself a finding, and it means the clean paper
contrast is **upright vs over_stance** (over_swing = "we tried; it doesn't produce a gait").

## 3. Metric traps (found in the kill test, now handled)
- **μ_req_p95 is meaningless without matched speed.** A near-stationary gait has tiny
  tangential force → low μ_req that looks "good." In the fixed-gait kill test, upright
  A showed μ_req 0.44 < over_stance 0.53 *only because upright barely moved* (net_fwd
  0.025 vs 0.226). Always condition μ_req on the robot actually walking at a set speed.
- **`min_mu_to_walk` as "lowest μ with net_fwd>thresh" is gameable by low-μ skidding.**
  On ice the feet slip and the body drifts forward while `single_frac` collapses (0.4)
  and the path is non-straight — passing a weak forward threshold without walking. Fixed
  by requiring `single_frac>0.6` AND taking the lowest μ **contiguous from the top** of
  the ladder (rejects the skid revival at μ=0.06). (Caveat: contiguous-from-top returns
  `None` if a mode fails at the very top rung, e.g. v3 over_swing — needs a small tweak.)
- **Cross-tool μ gate still split:** v2 `friction_study` uses Fn>1N, the sweep/kill/v3
  tools use Fn>4N. Do not mix v2 and v3 μ numbers without noting this.

## 4. The v2 ≫ v3 magnitude gap (must resolve before a paper number)
v2 shows 2.4×, v3 shows 1.2×. Candidate causes, to disentangle:
- **Under-optimization on v3:** `matched_speed_v3.py` used maxfev=140, single CMA seed;
  the upright optimum was scrappy (`single_frac` 0.65). Re-run with larger budget +
  multiple seeds before trusting the magnitude.
- **UPDATE (Mac, maxfev=400, seed=1, commit 31723eb):** bigger budget SHRANK the gap
  (1.20×→1.10×) — the 140 over_stance was slightly underspeeding (err −0.0005), and an
  underspeeding gait fakes a LOW mu_req (small Ft). So the margin is budget-sensitive
  and 140 overestimated it. Qualitative min_mu story got STRONGER at 400: over_stance
  cleanly walks down to μ=0.06 (ice) vs upright 0.5. μ_req is a measured quantity at
  the optimum, not the objective → non-monotone in budget; single seed cannot separate
  1.10× from noise. `matched_speed_v3.py` now takes `<maxfev> <seeds>` and emits
  per-seed CSVs + `_agg.csv` (mean±std) + a speed-gated cross-seed verdict
  (|speed_err|≤0.01 for BOTH modes, else seed excluded — the smoke test showed an
  underspeed upright faking mu_req 0.419 and reversing the verdict).
- **Real model difference:** v2 = crank-slider closed-loop, native −30° pitch; v3 =
  upright re-export with dynamic hip_off=30° pitch. Different contact/dynamics could
  genuinely change the effect size.
- **Which model is the paper's?** (Ben's call.) v3 is where the gaits + 3.97M sweep live;
  v2 is where the strong friction effect + the existing friction_study live.

## 5. Recommended next steps (in order)
1. **Firm up v3 matched-speed:** maxfev ≥ 400, 3–5 CMA seeds → mean±std μ_req per mode.
   Confirm the direction and pin the magnitude. (Addresses the single-seed rigor gap.)
2. **Multi-speed:** repeat at V ∈ {0.05, 0.08, 0.12} → μ_req-vs-speed curves per mode
   (reviewers will ask; one speed is too thin).
3. **Decide v2 vs v3** for the paper (Ben), then run the full matched-speed × μ-ladder ×
   (later) COM-variant grid on that model only.
4. **Robustness / artifact check:** the sweep's sharp 1.24 Hz onset & the min_μ behavior
   are single deterministic rollouts — spot-check vs timestep and ±spawn-pitch.
5. **COM variants (issue #6):** do NOT re-run a full 3.97M sweep per COM; local CMA around
   the known family + a freq×phase 2D slice suffices (saves ~10 days).

## 6. Files
- New: `physics/killtest_torso_modes.py`, `physics/matched_speed_v3.py`
- Data: `results/friction_study/{killtest_v3_AB,matched_speed_v3,penguin_configs}.csv`
- Context: `docs/fable_review_2026-07-04.md`, `docs/fable_strategy_notes.md`
