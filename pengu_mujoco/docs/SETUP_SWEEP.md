# Run a Pengu GRID-4 sweep on your laptop (from `git clone`)

## TL;DR — copy-paste and go (replace `c5` with YOUR config from Ben)

**Linux:**
```bash
sudo apt install -y git python3-venv
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
bash physics/run_sweep.sh c5
sudo systemctl mask sleep.target suspend.target hibernate.target   # no sleep
```

**macOS:**
```bash
git clone https://github.com/robomechanics/pengu.git && cd pengu/pengu_mujoco
git checkout friction-experiments
bash physics/run_sweep.sh c5
caffeinate -dimsu &                                                # no sleep
```

**Windows** (admin PowerShell first, reboot once, then in the Ubuntu terminal):
```powershell
wsl --install
powershell -c "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden"
```
```bash
sudo apt install -y git python3-venv
git clone https://github.com/robomechanics/pengu.git && cd ~/pengu/pengu_mujoco   # inside ~, NOT /mnt/c
git checkout friction-experiments
bash physics/run_sweep.sh c5
```
…and set Windows power/sleep/lid-close (on AC) to **Never**.

Then do the 2-minute verification in §3. Details and troubleshooting below.

---

We are sweeping 6 robot configurations (gait × center-of-mass) over ~1.8M
gait/friction combinations each, in MuJoCo. It's **CPU-only** (no GPU), runs
unattended for a few days, and is safe to interrupt — progress resumes
automatically. Your laptop = one worker in a small fleet.

**Before you start, ask Ben which config you're running** (one of `c1`…`c6`).
Everything below uses `c5` as the example — substitute yours.

---

## 1. Get the code

You need access to the private repo `robomechanics/pengu` on GitHub (ask Ben).

```bash
git clone https://github.com/robomechanics/pengu.git
cd pengu/pengu_mujoco
git checkout friction-experiments
```

### Windows users: one extra step first

The sweep does **not** run on native Windows (MuJoCo's DLL fails to load — we
tried). Install WSL2 first, then do everything inside the Ubuntu terminal:

```powershell
wsl --install        # admin PowerShell, then reboot once
```

Two Windows-specific rules:
- Clone the repo inside WSL's own filesystem (`~/pengu`), **not** `/mnt/c/...`
  (10× slower disk writes).
- WSL shuts down ~45 s after you close its last terminal — killing the sweep
  silently. Arm this **anchor** once, and re-arm after every reboot:
  ```powershell
  powershell -c "Start-Process wsl.exe -ArgumentList '-d','Ubuntu','-e','sleep','infinity' -WindowStyle Hidden"
  ```

### Linux prerequisite

```bash
sudo apt install -y git python3-venv     # Ubuntu 24.04: also python3.12-venv if venv creation fails
```

macOS needs nothing extra.

---

## 2. Launch (one line)

```bash
bash physics/run_sweep.sh c5
```

That's it. The script finds (or builds) a Python environment with MuJoCo,
creates the output CSV, and starts `cores − 2` worker processes. First run
takes a few minutes to pip-install; after that, workers start instantly.

Optional: `bash physics/run_sweep.sh c5 8` forces 8 workers (use fewer if you
need the laptop for other things — any number works).

---

## 3. Verify it's actually working (2 minutes — please do all four)

**① Right code version** (compare with Ben):
```bash
git rev-parse --short HEAD
```

**② Right config.** This prints the sweep summary without running anything:
```bash
CONFIG=c5 .sweep_venv/bin/python physics/grid4_sweep.py count
# want: config=c5 kappa=2.0 com=1.2  cells=454500  rows=1818000  K=1
```

**③ Right model.** Look at the top of the log:
```bash
head -5 results/gait_sweep/c5_run.log
```
The startup line must show `K=1`, `mass=2.2724kg`, and the COM slide for your
config:

| config | kappa | com | slide |
|---|---|---|---|
| c1 / c4 | 0 / 2 | `com=1.0500` | `-86.05mm` |
| c2 / c5 | 0 / 2 | `com=1.2000` | `-31.37mm` |
| c3 / c6 | 0 / 2 | `com=1.3100` | `+8.73mm` |

**④ It's writing.** The header line + a growing row count:
```bash
CSV=results/gait_sweep/sweep_grid4_c5_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
head -1 "$CSV"      # must start with: freq,hip_phi,leg_amp,...  (a header, not numbers)
wc -l "$CSV"        # run twice a minute apart; the number must grow
```

---

## 4. Keep the machine awake

The sweep dies if the machine sleeps. Plug into AC power and:

- **Windows**: Settings → Power → set *everything* (sleep, hibernate, lid
  close on AC) to **Never**. Plus the WSL anchor from step 1.
- **macOS**: run `caffeinate -dimsu &` in a terminal and leave it.
- **Linux**: `sudo systemctl mask sleep.target suspend.target hibernate.target`

Closing the laptop lid is the #1 way these runs die. Leave it open on AC.

---

## 5. Daily check-in (30 seconds)

```bash
cd pengu/pengu_mujoco     # or wherever you cloned
pgrep -f grid4_sweep.py | wc -l      # ≈ your worker count. 0 = it died, see §6
wc -l results/gait_sweep/sweep_grid4_c5_*.csv    # growing = healthy
```

Progress target: **1,818,000 rows** (+1 header line). Typical laptops do
100–350 rows/hour *per worker*, so expect several days total.

---

## 6. Stopping / restarting — always safe

Interrupting is harmless: every finished row is saved immediately, and a
relaunch skips existing rows and continues. After a reboot, a crash, or if
`pgrep` shows 0 workers:

```bash
cd pengu/pengu_mujoco
git pull                      # pick up any fleet-wide fixes
bash physics/run_sweep.sh c5  # resumes where it left off
```
The log should say `resuming from <N> done rows` with a large N — if it says
`0` or `-1` while your CSV clearly has data, STOP and ping Ben (that's a known
header issue with a 2-line fix; don't relaunch repeatedly, it duplicates work).

---

## 7. When it finishes (row count reaches 1,818,000)

Run this block, then tell Ben:

```bash
CSV=results/gait_sweep/sweep_grid4_c5_freq_hip_phi_leg_amp_hip_amp_hip_off_mu.csv
awk 'NF' "$CSV" > t && mv t "$CSV"        # drop stray blank lines
wc -l "$CSV"                              # expect exactly 1818001
gzip -kf "$CSV"
git add -f "$CSV.gz"
git commit -m "GRID-4 c5 complete"
git push origin friction-experiments
```

---

## Rules of the fleet (short version)

1. **Never edit sweep parameters** (`physics/grid4_sweep.py` values, μ levels,
   K, jitter). One changed number silently makes your data incomparable with
   everyone else's.
2. **One machine, one config.** Don't run a config someone else is running.
3. Your laptop is otherwise yours — the workers run at normal priority and
   tolerate you using the machine; just don't let it sleep.

## Troubleshooting

| symptom | fix |
|---|---|
| `ModuleNotFoundError` at startup | `git pull` (old launcher missed a dep), relaunch |
| venv build fails with `ensurepip` (Ubuntu 24.04) | `sudo apt install python3.12-venv`, relaunch |
| workers vanish on Windows when terminal closed | re-arm the WSL anchor (step 1), relaunch |
| `resuming from 0/-1 done rows` but CSV has data | header lost — ping Ben, don't relaunch |
| want to use a specific Python | `GRID3_PY=/absolute/path/to/python bash physics/run_sweep.sh c5` (absolute path only) |

Questions → Ben. Deeper detail → `docs/grid4_guide.md` (the science),
`docs/grid4_fleet_memo.md` (fleet status), the per-machine memos in `docs/`.
