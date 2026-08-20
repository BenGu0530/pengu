# GRID-4 analysis package — index

Generated 2026-08-20 on machine D. **5 of 6 configs complete** (c1, c3, c4, c5, c6).
**c2 is excluded** — partial at 264,949 / 1,818,000 rows, still running on rml3.
Rerun the whole package when c2 lands: see "How to regenerate" below.

```
config   kappa   COM ratio
c1       0       1.05
c3       0       1.31
c4       2       1.05
c5       2       1.20
c6       2       1.31
c2       0       1.20      <- MISSING (partial)
```

> Note: with c2 absent the kappa=0 arm has only 1.05 and 1.31, so the kappa=0 vs
> kappa=2 comparison is not matched at COM 1.20. c5 has no kappa=0 counterpart yet.

## Layout

```
REPORT.md            map-level tables (pass % / mean pass / robust volume / best speed) per config x mu
INDEX.md             this file
cross/               cross-config figures
  volume_vs_mu.png     robust-region volume vs mu (log scale)
  passfrac_vs_mu.png   share of cells with pass>0 vs mu
  speed_vs_mu.png      best net_fwd among passers vs mu
  overlap_mu01.png     Jaccard overlap of passing cell sets at mu=0.1
  roll_to_speed.png    measured torso roll RMS vs net speed (mu=0.1 finalists)
  ds_move_mu01.png     shuffle-vs-stepping distribution
cN/                  per-config
  heatmap.png          nbhd-mean pass over freq x hip_phi, one panel per mu
  top_gaits.csv        top 50 gaits ranked at mu=0.1
  finalists.csv        top 20 gaits re-run at ALL 4 mu, nominal, 21 metrics
  demos/               #1 gait rendered at every mu (side+back, 1280x480, 30fps, 24s)
    demo_mu01.mp4  demo_mu03.mp4  demo_mu05.mp4  demo_mu07.mp4
```

## Finalist transfer across mu (top-20 mean, nominal conditions)

**Two different gates — do not confuse them.** The map and the finalist eval do not
measure the same thing:

```
MAP   pass  = survived AND heading_align>0.5 AND net_fwd>0.05
FINAL valid = survived AND n_steps[L]>=2 AND n_steps[R]>=2 AND ...
```

`valid` is a STEPPING gate (at least 2 discrete steps per leg), NOT the pass criterion.
A robot that slides forward without stepping scores pass=1, valid=0.

### PASS proxy (survived AND net_fwd>0.05) — comparable to the map

`heading_align` is not carried in finalists.csv, so this is pass minus the heading term.

| config | kappa, COM | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|---|
| c1 | 0, 1.05 | 1.00 | 1.00 | 0.45 | 0.00 |
| c3 | 0, 1.31 | 1.00 | 0.40 | 0.00 | 0.00 |
| c4 | 2, 1.05 | 1.00 | 0.50 | 0.00 | 0.00 |
| c5 | 2, 1.20 | 1.00 | 0.40 | 0.00 | 0.00 |
| c6 | 2, 1.31 | 1.00 | 0.10 | 0.00 | 0.00 |

All five pass at mu=0.1. c1 holds furthest up the mu axis.

### valid = STEPPING gate, with ds_move_frac (shuffle fraction)

| config | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|
| c1 | 0.00 / ds 0.766 | 0.10 / ds 0.566 | 1.00 / ds 0.432 | 1.00 / ds 0.409 |
| c3 | 1.00 / ds 0.591 | 1.00 / ds 0.119 | 0.00 | 0.00 |
| c4 | 1.00 / ds 0.323 | 0.55 / ds 0.085 | 0.00 | 0.00 |
| c5 | 1.00 / ds 0.240 | 0.80 | 0.00 | 0.00 |
| c6 | 1.00 / ds 0.126 | 0.30 | 0.00 | 0.00 |

c1 at mu=0.1 makes the best forward progress of any config while taking fewer than two
steps per leg, with the highest shuffle fraction (0.766) — it advances without stepping.

## Region thickness — width, not just peak

With K=1 the map is binary (pass_rate is 0 or 1), so width comes from neighborhood
structure. thickness = robust cells (nbhd>=0.8) / raw passing cells. High = broad
plateau; low = thin scattered spikes.

| config | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|
| c1 | 0.613 | 0.698 | 0.436 | 0.305 |
| c3 | 0.156 | 0.080 | 0.028 | 0.027 |
| c4 | 0.677 | 0.704 | 0.381 | 0.316 |
| c5 | 0.565 | 0.392 | 0.079 | 0.065 |
| c6 | 0.146 | 0.013 | 0.009 | 0.013 |

Peak speed and thickness rank almost oppositely at mu=0.1:

| config | best net_fwd | raw pass | robust | thickness |
|---|---|---|---|---|
| c6 | **0.602** (fastest) | 25,376 | 3,697 | **0.146** (thinnest) |
| c5 | 0.574 | 107,243 | 60,625 | 0.565 |
| c4 | 0.370 | 144,043 | 97,468 | 0.677 |
| c3 | 0.171 | 18,952 | 2,948 | 0.156 |
| c1 | **0.170** (slowest) | 57,327 | 35,135 | **0.613** |

## Measured torso roll RMS [deg] / PID saturation fraction

| config | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|
| c1 (k=0) | 4.59 / 0.000 | 4.32 / 0.000 | 4.53 / 0.000 | 5.06 / 0.009 |
| c3 (k=0) | 2.62 / 0.000 | 3.38 / 0.000 | 27.60 / 0.105 | - |
| c4 (k=2) | 31.02 / 0.000 | 28.17 / 0.005 | 18.15 / 0.089 | 22.58 / 0.077 |
| c5 (k=2) | 27.62 / 0.000 | 7.74 / 0.001 | **91.79 / 0.439** | 40.00 / 0.113 |
| c6 (k=2) | 22.38 / 0.000 | 21.25 / 0.021 | 43.96 / 0.081 | 60.06 / 0.177 |

kappa=0 configs hold the torso near 0 roll (2.6-5.1 deg) as the Gait-1 definition intends;
kappa=2 configs sit at 18-31 deg at low mu. Saturation is ~0 at mu=0.1 for every config.
c5 at mu=0.5 reaches sat_frac 0.439 with 91.8 deg roll RMS — the +/-4.1 Nm torso motor
failing to hold, which CLAUDE.md flags as a co-design result rather than a bug.

## How to regenerate

```bash
conda deactivate 2>/dev/null; cd pengu_mujoco
.sweep_venv/bin/python physics/grid4_report.py              # map tables + heatmaps + top_gaits
.sweep_venv/bin/python physics/grid4_finalists.py --no-demo # finalists.csv + cross figures
MUJOCO_GL=egl .sweep_venv/bin/python physics/grid4_demos.py # demos at every mu
```

Configs are auto-detected; partial ones are skipped. `grid4_demos.py` accepts config
names to render a subset, e.g. `python physics/grid4_demos.py c2`.
