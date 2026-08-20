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

`valid` = fraction of the 20 finalists that pass at that mu. Finalists are chosen at mu=0.1.

| config | kappa, COM | mu=0.1 | mu=0.3 | mu=0.5 | mu=0.7 |
|---|---|---|---|---|---|
| c1 | 0, 1.05 | 0.00 | 0.10 | **1.00** | **1.00** |
| c3 | 0, 1.31 | 1.00 | 1.00 | 0.00 | 0.00 |
| c4 | 2, 1.05 | 1.00 | 0.55 | 0.00 | 0.00 |
| c5 | 2, 1.20 | 1.00 | 0.80 | 0.00 | 0.00 |
| c6 | 2, 1.31 | 1.00 | 0.30 | 0.00 | 0.00 |

c1 runs opposite to the other four: its mu=0.1-selected finalists fail at mu=0.1
nominal but pass at mu=0.5 and 0.7. Reported as measured, not interpreted. Note the
map is K=1 WITH pose jitter while finalists are nominal (exact mu, no jitter), so the
two are not the same test.

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
