# GRID-4 analysis task — run on any FINISHED machine (suggested: F / rml3)

The Mac is reserved for finishing c1; the analysis package runs elsewhere.
Everything is scripted; this is ~30 min of compute + one push.

## What it produces

```
results/grid4_report/
  REPORT.md                     summary tables
  cross/   volume_vs_mu.png  passfrac_vs_mu.png  speed_vs_mu.png  overlap_mu01.png
           roll_to_speed.png (measured torso roll RMS vs net speed, mu=0.1 finalists)
           ds_move_mu01.png  (shuffle-vs-stepping distribution)
  cN/      heatmap.png  top_gaits.csv  finalists.csv  demo_mu01.mp4     (one per config)
```

Configs are auto-detected: every complete `sweep_grid4_c*_...csv(.gz)` in
`results/gait_sweep/` is included; partial ones are skipped. Rerunning later
(after c1/c2 land) regenerates everything with the new configs added.

## Steps

```bash
cd ~/Documents/ben/pengu/pengu_mujoco          # adjust to this machine's repo path
git pull
# one-time: rendering deps into the sweep venv
.sweep_venv/bin/python -m pip install -q imageio imageio-ffmpeg

# 1) map-level report (fast, ~2 min)
.sweep_venv/bin/python physics/grid4_report.py

# 2) finalist rich-eval + per-config demos (~25 min)
MUJOCO_GL=egl .sweep_venv/bin/python physics/grid4_finalists.py
#   if EGL errors (no GPU/driver): sudo apt install -y libosmesa6
#   then: MUJOCO_GL=osmesa .sweep_venv/bin/python physics/grid4_finalists.py

# 3) ship the package back (results/ is gitignored -> add -f)
git add -f results/grid4_report
git commit -m "GRID-4 analysis package (report + finalists + demos)"
git push
```

Notes
- Step 2 needs ~2 cores only; fine to run even if this machine is also sweeping.
- Demos are mu=0.1, nominal conditions, #1 gait from each config's top_gaits.csv.
- If a config's `top_gaits.csv` is missing, step 1 wasn't run or the config is
  incomplete — check `results/gait_sweep/` for its csv.gz.
- Rerun cadence: whenever a new config's data lands, repeat steps 1-3.
