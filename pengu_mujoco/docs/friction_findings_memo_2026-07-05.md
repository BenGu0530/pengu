# MEMO — Torso strategy × friction: first decisive results & a design correction

**Date:** 2026-07-05 · **Author:** Fable (advisory) · **Status:** first data, needs firming

## TL;DR
1. **Paper direction holds on v2 only.** At matched walking speed the penguin
   **over_stance** strategy needs **less floor friction** than **upright** on **v2 (2.4×,
   strong)**. On **v3 it does NOT survive multi-seed CMA** — 5-seed sweep: over_stance
   wins only **2/5**, gap mean **0.94×** (slightly *reversed*). The v3 "advantage" was a
   single-seed artifact of upright's multi-modality. See §4 UPDATE 2.
2. **Design correction (important):** you **cannot** build the upright / over_swing
   conditions by flipping the torso on a fixed over-stance gait — the gait then stops
   walking, so μ_req becomes meaningless. Each mode **must be independently re-optimized
   to matched speed** (which the v2 `friction_study.py` already does). The updated
   `fable_prompt.md` step "inject marked gait into friction_study" is only valid in the
   *re-optimize-per-mode* sense, NOT "fix gait B and flip the torso."
3. **Two metric traps found & fixed** — see §3.
4. **RESOLVED (§4 UPDATE 2):** the v3 1.2×/1.1× was a single-seed artifact of upright's
   multi-modality; the 5-seed sweep gives mean **0.94× ± 0.19**, over_stance wins **2/5**.
   Do NOT quote a v3 μ_req number. v2's 2.4× stands (still to be seed-checked).

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
- **UPDATE 2 — RESOLVED (Mac, maxfev=400, seeds 1–5):** the 5-seed sweep KILLS the v3
  μ_req claim. All 5 seeds matched speed (0 excluded); speed-gated verdict:

  | seed | 1 | 2 | 3 | 4 | 5 | mean±std |
  |---|---|---|---|---|---|---|
  | upright μ_req | 0.631 | 0.397 | 0.469 | 0.448 | 0.620 | **0.513 ± 0.106** |
  | over_stance μ_req | 0.573 | 0.534 | 0.540 | 0.551 | 0.527 | **0.545 ± 0.018** |
  | gap (up/over) | 1.10 | 0.74 | 0.87 | 0.81 | 1.18 | **0.94× ± 0.19** |

  over_stance wins **2/5**; mean gap **0.94×** (on average over_stance needs MORE friction).
  **Mechanism:** upright is *multi-modal* at matched speed. CMA finds a low-friction
  high-freq upright gait (freq 1.7–2.0, single_frac≈1.0, μ_req 0.40–0.47) in 3/5 seeds and
  the high-friction low-freq basin (μ_req 0.62–0.63) in only 2/5. Seed 1 and the 140
  baseline both happened to land in the high-friction basin — that, not physics, produced
  the 1.10–1.20× "advantage." over_stance is a tight single point (std 0.018); upright is a
  basket spanning 0.40–0.63.
  **Worse for the claim:** best-per-mode across seeds is upright **0.397** (s2) vs
  over_stance **0.527** (s5) — so a "μ_req-into-the-objective" constrained-matched-speed
  redesign would likely **REVERSE** the claim on v3, not rescue it: upright has access to
  lower-friction matched-speed gaits than over_stance does. (Do not assume the redesign
  saves v3 — it may falsify it.)
  **min_mu fallback is weaker than TL;DR §2 implies:** over_stance walks on ≤μ in 4/5 seeds
  (over 0.06|0.2|0.4|0.4|0.3 vs up 0.5|0.4|0.5|0.3|0.7), but the headline "0.06 vs 0.5" was
  seed 1 only — over_stance's median min_mu is ~0.3, and seed 4 reverses it. Real tendency
  but coarse (one deterministic rollout per ladder rung).
  **over_swing** confirmed degenerate across all 5 seeds (never reaches matched speed, err
  −0.006…−0.032, min_mu all None).
  Data: `results/friction_study/matched_speed_v3_mf400_s{1..5}.csv` + `_agg.csv`.
- **Real model difference:** v2 = crank-slider closed-loop, native −30° pitch; v3 =
  upright re-export with dynamic hip_off=30° pitch. Different contact/dynamics could
  genuinely change the effect size.
- **Which model is the paper's?** (Ben's call.) v3 is where the gaits + 3.97M sweep live;
  v2 is where the strong friction effect + the existing friction_study live.

## 5. Recommended next steps (in order)
1. ~~**Firm up v3 matched-speed:** maxfev ≥ 400, 3–5 CMA seeds → mean±std μ_req per mode.~~
   **DONE (§4 UPDATE 2) — negative:** 5-seed mean 0.94×, over_stance wins 2/5. v3 μ_req does
   not support the claim. **Next on v3:** either (a) seed-check v2's 2.4× to anchor the paper
   there, or (b) rebuild the v3 objective as constrained-matched-speed (min μ_req s.t.
   speed=V) — but expect it may reverse v3 (upright's best 0.397 < over_stance's 0.527).
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

---

## 7. Evening update (2026-07-05): seed variance verdict + reframe + COM probe

**7a. μ_req_p95 verdict REVERSED by seed variance.** Local seeds 6-10 (mf400, all
matched-speed): over_stance beats upright on μ_req in only **2/6 seeds** (incl. Mac
seed 1); gap mean 0.92× ± 0.25. The earlier 1.10×/1.20× was optimizer noise — upright
finds very different families per seed (freq 1.36-1.99) incl. low-μ_req operating
points (0.39-0.41). **μ_req_p95 is dead as the headline metric on v3.**
**BUT min_mu_to_walk is seed-robust:** over_stance ≤ upright in **6/6 seeds** (4 strictly
lower, incl. two runs reaching 0.06/0.25 vs upright 0.3-0.7). Peak *demand* vs actual
*slip robustness* are different things; the paper cares about the latter, and the
robust readout favors over_stance. Data: `matched_speed_v3_mf400_s{6..10}.csv`.

**7b. Reframe (Ben, aligned with the plan):** friction is the READOUT, not the subject.
Manipulated variables = gait (torso phasing) × mass distribution. The seed-variance
work was readout calibration: the big matrices use **min_mu (clean-walk) + foot
roll/pitch + COM**, with μ_req demoted to diagnostic.

**7c. Mass axis → Onshape (Ben).** In-memory mass transfer (probe + prototype harness
`physics/com_variant_study.py`) showed: baseline penguin COM = **36.7%** of standing
height (0.188/0.512 m, 1.772 kg total); naive transfer to 42% breaks gait B outright,
≥47% cannot even stand with the penguin stand pose → real CAD variants are the
defensible route. Onshape targets: human 54-57%, suggested mid ≈ 46%. The harness
(stand calibration + per-mode matched-speed + min_mu ladder) becomes the runner for
the exported variants: swap `make_variant` for loading the variant XML.

**7d. Gait axis DONE (3 families × 3 seeds × mf400, all speed-matched err≈0):**
`physics/gait_family_torso_study.py`, data `gait_family_mf400_*`. Per-family torso-phase
calibration matters — the true phi_stance differs by family (**0° / 225° / 270°** for
hip_phi 110 / 210 / 270), confirming torso_phi=0 ≠ "over stance" outside the 210 family
(matched_speed_v3's over_swing was mis-phased).

min_mu (clean-walk), over_stance vs upright:
| family | upright | over_stance | over_stance better? |
|---|---|---|---|
| 110 | 0.5/0.4/0.5 | 0.3/0.4/0.15 | yes 3/3 |
| 210 | 0.5/0.5/0.3 | 0.3/0.25/0.3 | yes 3/3 |
| 270 | 0.3/0.3/0.25 | 0.5/0.7/0.7 | **NO — reversed 0/3** |

→ **the min_mu advantage is NOT gait-family universal** — real in 110/210, reversed in
270 (there over_stance needs high freq 1.48-1.65 to match speed while upright finds a
low-freq low-μ gait). Weakens a clean "penguin gait needs less friction" claim.

**The one robust thing across ALL families/seeds:** `torso_stance_corr` for over_stance =
**0.68-0.96 (mean ~0.83, small std)** — over_stance genuinely realizes torso-over-stance
everywhere; over_swing is erratic (0.02-0.58). So the seed/family-robust result is the
KINEMATIC signature (penguin posture), while the friction benefit is a gait-dependent
by-product. This reframes the likely paper spine: *the penguin torso strategy is a
robust posture, and where it helps friction depends on the gait regime* — not *penguin
gait universally needs less friction*.

**Cumulative status:** μ_req_p95 dead on v3 (n=10, two independent replications, 0.9×);
min_mu directionally real but family-dependent + coarse; torso_stance_corr robust; the
strong friction effect still lives only in v2 (single seed, unverified — top priority).
