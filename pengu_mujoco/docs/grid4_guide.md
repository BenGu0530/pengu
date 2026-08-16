# GRID-4: the gait × friction co-design sweep (paper Section III)

**Supersedes** `grid3_mac_memo.md` / `RESUME_k0dr.md` and all GRID-3 sweeps (k0/k1p5/k2
deterministic, k0dr/k2dr wide-band DR). Those results stay committed as reference but do
not feed the paper. This spec matches the Overleaf draft (Sim Methods, 2026-08-15).

## The 6 configurations (paper Table `tab:configs`)

| config | torso strategy | COM ratio | env |
|---|---|---|---|
| **c1** | Gait 1 (κ=0) | 1.05 | `CONFIG=c1` |
| **c2** | Gait 1 (κ=0) | 1.20 | `CONFIG=c2` |
| **c3** | Gait 1 (κ=0) | 1.31 | `CONFIG=c3` |
| **c4** | Gait 2 (κ=2) | 1.05 | `CONFIG=c4` |
| **c5** | Gait 2 (κ=2) | 1.20 | `CONFIG=c5` |
| **c6** | Gait 2 (κ=2) | 1.31 | `CONFIG=c6` |

**Base model for ALL configs = hardened `models/pengu1_31`** (native total mass
**2.2724 kg** — matches the physical build; no scaling). COM variants are
**mass-conserving**: `easytorso` inertial COM slid along the counterweight axis
(world-up at neutral stand) by load-time bisection; total/per-link masses, geometry,
actuation, contact untouched. Measured base ratio 1.286; slides as solved by the
script at load (fleet-verified startup lines): 1.31→+8.73 mm, 1.20→−31.37 mm,
1.05→−86.05 mm.
Rung labels 1.05/1.20/1.31 are Ben's model naming. penguV3 (1.77 kg) and the
`grid3` k0dr/k2dr data are fully superseded.

## Sweep definition (per config)

6 axes — 5 gait + friction:

| axis | values | n |
|---|---|---|
| freq | 1.00–2.00 Hz @ 0.01 | 101 |
| hip_phi | 0–350° @ 10° | 36 |
| leg_amp | 85, 95, 105, 115, 125 | 5 |
| hip_amp | 12, 16, 20, 24, 28 | 5 |
| hip_off | 10, 20, 30, 40, 50 | 5 |
| **mu** | **0.1, 0.3, 0.5, 0.7** | 4 |

= 454,500 gait cells × 4 μ levels = **1,818,000 rows/config**.

Per (cell, μ): **K=5 trials**, each with
- μ jittered **relative ±5%** (μ × U(0.95, 1.05)),
- initial pose jitter: yaw ±5°, pitch ±3°, lateral ±1 cm,
- **NO mass jitter** (removed 2026-08-15 — it blurred the COM design axis),
- seeded per (cell-index, μ-index, repeat) → any machine reproduces exactly.

Pass = survived ∧ heading_align>0.5 ∧ net_fwd>0.05 m/s. **Slip is RECORDED
(`slip_mean`), not a pass gate** — low-μ gaits that advance while partially slipping
stay in and are ranked on performance (Ben, 2026-08-15).
Row = 6 axis cols + `pass_rate, surv_rate, net_fwd_mean, net_fwd_min, slip_mean,
head_mean` (12 cols). 9.09M trials/config, 54.5M total.

**No early-skip across μ levels**: single-μ pass/fail is non-monotone
(dr_filter finding), and the paper reports feasible-gait counts per level — every
(cell, μ) is measured. Full factorial, no silent truncation.

## Running

One line on any clone (CPU-only; auto-venv if no mujoco python):

```bash
bash physics/run_sweep.sh c1        # …c2..c6; optional 2nd arg = shard count
```

- Script: `physics/grid4_sweep.py` (env `CONFIG=c1..c6`).
- Output `results/gait_sweep/sweep_grid4_c1_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv`
- Resume is by 6-axis tuple; `initcsv` writes the header once; shards append lock-free.
- Two machines on DIFFERENT configs, never the same CSV. Merge = concat + dedupe on 6 axis cols.
- ~20–24 h/config on a 10-shard machine; 6 configs ≈ 3 days on 2 machines.

## Ship-back + integrity (before every commit)

```bash
CSV=results/gait_sweep/sweep_grid4_cN_....csv
awk 'NF' "$CSV" > t && mv t "$CSV"          # strip stray blank lines
# rows = 1,818,000 (+1 header); NF==12 for all; unique 6-tuples = 1,818,000
gzip -kf "$CSV" && git add -f "$CSV.gz" && git commit -m "GRID-4 CN complete" && git push
```
(No AI attribution in commits. Working branch: `friction-experiments` — confirm before push.)

## Downstream (order matters)

1. Sweep C1–C6 → per-config best net_fwd per COM point.
2. RL (6 policies = 3 COM × {generalist μ~U(0.1,0.7), specialist μ~U(0.1,0.4)}):
   `vx_cmd` per COM point = best net_fwd from that point's sweep → RL trains AFTER
   that config's sweep lands.
3. Analysis for paper: pass-rate-vs-μ profiles, feasible-gait counts per level,
   best-at-μ=0.1 comparison, effective-κ regression for RL gaits.
