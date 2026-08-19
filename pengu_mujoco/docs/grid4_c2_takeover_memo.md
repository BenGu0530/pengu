# c2 takeover memo — machine E picks up c2 from machine B's snapshot

c2 (κ=0, COM 1.20) is the last slow config: machine B (Ryzen laptop) runs it at
~7.5k rows/h. Machine E (32-core, finished c5 in ~2 days) takes it over from B's
pushed snapshot. Resume is by axis-tuple and the randomization is seeded per
(cell, mu, repeat), so E reproduces exactly the rows B would have produced —
a mid-sweep handover loses nothing.

## 1. On machine B — STOP (after its snapshot is pushed)

```bash
cd ~/pengu/pengu_mujoco
touch results/gait_sweep/WATCHDOG_OFF          # if a watchdog was installed
pkill -f 'grid4_sweep[.]py'
# confirm the snapshot made it to git BEFORE stopping for good:
git log --oneline -2                           # expect a "c2 ... snapshot" commit pushed
```
B is then free (and its WSL anchor can be dropped).

## 2. On machine E — take over

```bash
cd ~/pengu/pengu_mujoco                        # adjust to E's repo path
git pull
CSV=results/gait_sweep/sweep_grid4_c2_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
gunzip -kf "$CSV.gz"                           # unpack B's snapshot as the live CSV
head -1 "$CSV"                                 # MUST be the freq,hip_phi,... header
wc -l "$CSV"                                   # note the starting row count
bash physics/run_sweep.sh c2                   # resumes: skips every row already present
bash physics/sweep_watchdog.sh install c2
```
Pre-flight: startup line in `results/gait_sweep/c2_run.log` must show
`kappa=0.0 com=1.2000 slide=-31.37mm mass=2.2724kg K=1`, and the launcher must
print `resuming from <N> done rows` with N = B's snapshot count (not 0, not -1).

## 3. Ship-back at 1,818,000 rows (standard)

```bash
touch results/gait_sweep/WATCHDOG_OFF
CSV=results/gait_sweep/sweep_grid4_c2_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
awk 'NF' "$CSV" > t && mv t "$CSV"
wc -l "$CSV"                                   # expect 1818001
gzip -kf "$CSV" && git add -f "$CSV.gz"
git pull --rebase && git commit -m "GRID-4 c2 complete" && git push
```

## Rules

- **B must be stopped before/while E runs** — two machines producing the same
  config wastes compute and, if both later push, the second push must be merged
  by concat + dedupe on the 6 axis columns (avoid by just stopping B).
- Do not edit any sweep parameter; c2 must stay comparable with c1..c6.
