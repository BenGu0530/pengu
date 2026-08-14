# Pengu Model Inventory — Onshape → URDF/MuJoCo

**Date:** 2026-08-14
**Author:** Ben Gu (BenGu0530)
**Scope:** All Pengu CAD models pulled from Onshape and converted to simulation formats.

---

## TL;DR

Three active models — **1.05, 1.20, 1.31** — are **the same robot with a swept counterweight**.
They differ only in torso payload mass (250 g steps) and the resulting COM height.
Geometry is otherwise identical.

**penguV2 is legacy. Do not use it for new work.** It is kept on disk for history only.

---

## Active models

| Model | Dir | Total mass | COM (x, y, z) | Counterweight | Role |
|---|---|---|---|---|---|
| **1.05** (= V3) | `models/pengu1_05/` | 1.7724 kg | `0.00000, −0.00525, **+0.00811**` | 75 mm / 64.07 cm³ | Lightest, lowest COM. Baseline for friction/slip work. |
| **1.20** | `models/pengu1_20/` | 2.0224 kg | `0.00000, −0.00723, **+0.03230**` | 60 mm / 51.25 cm³ | Mid rung. |
| **1.31** | `models/pengu1_31/` | 2.2724 kg | `0.00000, −0.00878, **+0.05149**` | 60 mm / 51.25 cm³ | Heaviest, highest COM. |

COM figures are at the **reference configuration** (`mj_forward` at initial `qpos`, no settling).

### The ladder is exact

Total mass steps by **exactly 0.250 kg** per rung:

```
1.7724  →  2.0224  →  2.2724       (Δ = 0.2500 kg each)
```

COM height climbs monotonically with it: **8 mm → 32 mm → 51 mm**. The added mass sits
high in the torso, so this sweep is effectively a **COM-height ladder** — useful as a
co-design axis (stability vs. payload).

### Mechanism behind the sweep

A single cylindrical counterweight, `assets/part_1__2.stl`, Ø33 mm:

- **1.05** — 75 mm long, 64.07 cm³, mount origin at `y = −0.25175`
- **1.20 / 1.31** — 60 mm long, 51.25 cm³, mount origin at `y = −0.23675`

**1.20 and 1.31 have byte-identical STLs across all 14 meshes.** They differ *only* in the
torso link's `<mass>` and `<inertia>` — i.e. same part, different assigned density:

| | torso mass | torso COM (xyz) |
|---|---|---|
| 1.05 | 0.622085 kg | `1.99e-09, −0.141397, 0.021594` |
| 1.20 | 0.872085 kg | `1.42e-09, −0.159272, 0.021503` |
| 1.31 | 1.122080 kg | `1.10e-09, −0.169850, 0.021453` |

So part of the sweep is geometric (1.05's longer slug) and part is a mass override
(1.20 → 1.31). Worth knowing if you re-derive inertia from CAD density — the 1.20/1.31
split will not reproduce itself from geometry alone.

---

## penguV2 — LEGACY, ignore it

`penguV2/` stays on disk as history. **Do not build on it.** Two concrete reasons:

1. **COM is on the wrong side.** V2 sits at **y = +0.00660**, while all three 1.x models sit
   at −0.005 to −0.009 — opposite side of the origin. It does not stand in a proper
   initial position; the COM placement is effectively arbitrary.
2. **It is only half-actuated.** V2 exposes **5 actuators**; the 1.x models expose **9**.

```
penguV2   (5): hip-L, hip-R, crank1-R, torso, crank1-L
1.x       (9): hip-L, hip-R, slider-R, crank2_R, crank1-R,
               torso, slider-L, crank1-L, crank2-L
```

The sliders and `crank2_*` joints are unactuated in V2, so any controller written against
it will not transfer. It also carries the 75 mm counterweight like 1.05 but weighs 2.17 kg
vs 1.05's 1.77 kg — roughly 400 g lives somewhere unaccounted for.

V2 has no URDF export and no `robot.pkl`; it was not regenerated in this pass.

---

## Directory layout

```
pengu/
  models/
    pengu1_05/        robot.xml scene.xml robot.urdf robot.pkl assets/   (= V3)
                      config.json (mujoco)  config_urdf.json (urdf)
    pengu1_20/        same layout
    pengu1_31/        same layout
  penguV2/            robot.xml scene.xml assets/          ← LEGACY, do not use
```

Both formats live in **one directory per model, sharing a single `assets/`**. This works
because MuJoCo's `<compiler meshdir="assets"/>` and the URDF's `package://assets/...` both
resolve relative to the model dir. It halves on-disk size (11 MB total vs 21 MB when the
two formats were kept in separate dirs with duplicated STLs).

Each `assets/` holds 14 STLs (~3.4 MB). The MuJoCo and URDF exports of a model were
converted from the **same `robot.pkl`**, so they are guaranteed consistent with each other.

`penguV2/` deliberately stays at the repo root rather than moving under `models/` —
seven scripts in `pengu_mujoco/` hardcode `penguV2/scene.xml`, and moving it would break
them for no benefit.

## Onshape sources

| Model | Document URL |
|---|---|
| 1.05 | `https://cad.onshape.com/documents/af9a366decc160f668eeab8a/w/240dda95e8c00a87d53a39cb/e/3a00180f1f1236d2a817e06b` |
| 1.20 | `https://cad.onshape.com/documents/327456a96965c95eb8955d8e/w/b78e2a8bbbac7eed78ed3d08/e/b4782a5dd527cb884d54a5cc` |
| 1.31 | `https://cad.onshape.com/documents/92ba184c0b15865297849245/w/789dbcff3171ede58dbe8577/e/33c14d568273b6f514d751c2` |
| V2 | `https://cad.onshape.com/documents/9471e6447c2451e765475198/w/06224f5761f3788c9391fbc0/e/b86bfedcbde02be2dd2ef9ae` |

---

## Toolchain

Everything runs in the **`pengu_sim`** conda env (Python 3.11).

```bash
conda activate pengu_sim
onshape-to-robot models/pengu1_05     # retrieve + convert (uses config.json)
```

Installed: `onshape-to-robot` 1.8.2, `yourdfpy` 0.0.60, `trimesh` 5.0.0,
alongside the existing `mujoco` 3.6.0 / `numpy` 2.4.3 (numpy was **not** upgraded).

`config.json` in each dir carries the Onshape URL and `output_format`
(`mujoco` | `urdf` | `sdf`) — one format per run.

**Credentials:** Onshape API keys live in `ben_pengu/.env` (mode `600`), one level
**above** the repo root, so they are outside the git tree and cannot be committed.
`python-dotenv` finds them by walking up from cwd.

### Regenerating both formats from one pull

Since both formats now share one directory, re-export the second format in place by
pointing at the alternate config. `--convert` reuses the existing `robot.pkl`, so this
costs no extra Onshape API calls:

```bash
cd models/pengu1_05
cp config.json .config.bak && cp config_urdf.json config.json
onshape-to-robot --convert .          # writes robot.urdf alongside robot.xml
mv .config.bak config.json
```

Caveat: `--convert` only writes STLs on the run that *generates* them. If you ever copy a
`robot.pkl` into a fresh empty dir, copy `assets/` with it or the export comes out
mesh-less.

---

## Verification status

All four models load in MuJoCo and step 300 frames with no NaN:

```
model        mj   urdf  stl     mass     COMz   act  dof
penguV2      OK    --    14   2.1722  +0.05823   5   15
pengu1_05    OK   18L    14   1.7724  +0.00811   9   15
pengu1_20    OK   18L    14   2.0224  +0.03230   9   15
pengu1_31    OK   18L    14   2.2724  +0.05149   9   15
```

URDFs: 18 links / 17 joints (7 revolute, 2 prismatic, 8 fixed). Every `filename=` mesh
reference resolves to a real file. Rendered through `yourdfpy` in front/side/iso —
all three 1.x models show a correctly assembled biped, feet flat on the ground plane,
crank-slider linkage intact. Bounding box `X=0.284  Y=0.185  Z=0.481 m`.

## Known caveats

1. **Missing joint limits.** `hip-L`, `hip-R`, `crank1-L/R`, `crank2-L/R`, `torso` have
   **no limits defined in the CAD** and export unbounded. Only the sliders are bounded
   (`[−0.05, 0.0]`). Fix via mate limits in Onshape if the sim needs them respected.
2. **URDF meshes use `package://assets/...`.** Fine under ROS and with a `yourdfpy`
   filename handler; PyBullet / Isaac generally want plain relative paths. Not yet converted.
3. **Duplicate part names.** Three parts are all named "Part 1" in CAD; the exporter
   auto-renames to `part_1`, `part_1__2`, `part_1__3`. `part_1__2` is the counterweight.
4. **Joint/body naming is inconsistent.** The URDF link carrying the payload inertia and
   the MuJoCo body named `torso` are *not* the same body — the MuJoCo `torso` body is a
   small 0.02495 kg part in every model. Compare whole-body COM, not the `torso` body.

## Related

Simulation/controller code for this robot lives on branch **`friction-experiments`**
(renamed 2026-08-14 from `fable/friction-experiments`; 38 commits, same SHA `dc2930ac`),
under `pengu_mujoco/`. Pairs with **1.05 / V3**.
