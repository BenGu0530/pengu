# GRID-2 design: the min_mu landscape sweep (v3, penguin mass)

**Date:** 2026-07-05 · **Status:** DESIGN — awaiting Ben's GO
**Decisions this encodes:** v2 is retired (fundamental model problems — its 2.4× is a
dead number, not worth seed-verifying). Paper model = penguV3. Readout = min_mu
(clean-walk) + torso_stance_corr + foot roll/pitch; μ_req_p95 demoted to diagnostic.
hip_off (forward pitch) is promoted to a swept dimension.

## Why a grid, not CMA
The mf400 campaigns proved upright is *multi-modal at matched speed*: CMA seeds land in
different basins (μ_req 0.40 vs 0.63), so every CMA number is a lottery ticket. A grid
is deterministic and landscape-complete — there are no seeds to verify. The open-loop
analog of seed-verification is **perturbation robustness** (Stage C). This is the
"nonlinear locomotion cannot skip the sweep" principle: fine3c's 0.01 Hz resolution was
what exposed the 1.24 Hz bifurcation; coarser grids produce false maps.

## Hard lessons folded in
1. **freq must extend to 2.0 Hz** (fine3c capped at 1.5): today's low-friction upright
   gaits live at 1.7–2.0 Hz — fine3c's band choice inadvertently favored the penguin
   band and never gave upright its space. Keep 0.01 Hz steps.
2. **upright = torso_amp 0 slice of the SAME grid**, not a separate "mode". Torso
   strategies are classified POST-HOC from measured torso_stance_corr (family study
   showed phi_stance = 0/225/270 for hip_phi 110/210/270 — phi labels lie, corr doesn't).
3. **Speed fairness via binning, not matched-speed optimization**: measure everything,
   compare torso strategies WITHIN net_fwd bins (0.04–0.08, 0.08–0.12, 0.12–0.20 m/s).
   Bonus: μ-vs-speed curve families per strategy (a reviewer ask).
4. **min_mu = clean-walk, contiguous from μ=0.7 down** (survived AND net_fwd>0.0115 AND
   single_frac>0.6) — the anti-skid definition from the kill test.

## Stage A — walkability grid (1 rollout/cell @ μ=0.7)
| dim | values | n |
|---|---|---|
| freq | 1.00–2.00 step 0.01 | 101 |
| hip_phi | 0–350 step 10° | 36 |
| leg_amp | 95, 105, 115 | 3 |
| hip_amp | 16, 20, 24 | 3 |
| torso_amp | 0, 10, 20 | 3 |
| torso_phi | 0–315 step 45° (collapsed to 1 value when torso_amp=0) | 8→1 |
| hip_off | 10, 20, 30, 40 | 4 |

Cells = 101·36·3·3·(1 + 2·8)·4 = **2,224,368** ≈ 1.5 days on 16 shards (fine3c
throughput ~1.5M rollouts/day). Amps stay coarse deliberately: fine3c showed amp
marginals are smooth; the nonlinearity lives in freq × phases (where we stay fine).
Records the standard run_trial metrics (incl. net_fwd, single_frac, mu_req_p95@0.7).

## Stage B — min_mu ladder on survivors (~8–12% of cells)
For every Stage-A clean walker: μ ladder 0.5→0.06 (0.7 already known), stop at first
failure (contiguous rule). Per rung also record **torso_stance_corr** and **foot
roll/pitch amplitude** (instrumented rollout, code exists in gait_family_torso_study).
Est. 150–250k cells × ≤8 rungs (early-stop cuts the average to ~4) ≈ **1–1.5 days**.

## Stage C — robustness spot-check (~0.5 day)
Top-N cells per (speed bin × torso strategy [corr>0.5 / |corr|<0.2 / corr<−0.2]):
K=5 rollouts with spawn-pitch ±1°, ±0.5° jitter → survival rate, min_mu spread.
Doubles as the 1.24 Hz bifurcation artifact check (strategy-notes item).

## Analysis deliverables
1. **min_mu landscape**: min_mu vs (torso_amp, corr-classified strategy, hip_off),
   within speed bins → the paper's design law for the penguin-mass condition.
2. μ-vs-speed curve family per torso strategy.
3. foot roll-vs-pitch single-peak check on slippery vs non-slippery rungs (plan metric).
4. The Onshape COM variants (when exported) re-run the SAME grid per variant →
   the full gait × mass matrix with identical methodology.

## Infra
- Stage A: extend `gait_sweep.py` AXES + a cell filter (skip torso_phi>0 when
  torso_amp==0), keeping initcsv/resume/shard semantics; launch via `run_grid.sh`
  (flock, pidfiles, @reboot-safe). Sharding is machine-agnostic: Mac can take a
  disjoint SHARD_ID range, CSVs merge.
- Stage B/C: new sharded script reusing run_trial + the instrumented corr/foot rollout.
- All outputs under results/gait_sweep/grid2_* ; registries updated by hand as usual.

## Cost summary
~3–3.5 days machine time total (comparable to fine3c), zero CMA/seed ambiguity,
directly reusable per COM variant. Ben: reply GO (or edit dims) and Stage A launches.
