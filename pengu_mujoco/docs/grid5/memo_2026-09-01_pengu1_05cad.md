# Memo 2026-09-01 — pengu1_05cad: real CAD export at the lowered-counterweight (1.05) build

Source: https://cad.onshape.com/documents/a7ec639d7406c686b0eb487f/w/598e1ad77f4f0b911527a205/e/df7cae32df1f78da8e921944
Exported with onshape-to-robot (conda env pengu_sim, credentials via ../.env) into
`models/pengu1_05cad/` (raw export: 9 actuators, unhardened — analysis only for now).
Geometry eyeballed in a render: full robot, counterweight block visibly at the LOW
position between the hip pods.

## Is it 1.05?

Measured at the hips-0 neutral stand (base z = 0.18, same convention as the ladder):

| model | total mass | COM ratio | easytorso mass |
|---|---|---|---|
| **pengu1_05cad (this export)** | **2.1724 kg** | **1.0440** | 1.02208 |
| slide-tuned 1.05 (base 1.31, ipos slid -86.05 mm) | 2.2724 kg | 1.0500 | 1.12208 |
| pengu1_31 (base) | 2.2724 kg | 1.2861 | 1.12208 |

Close to 1.05 but not exact (1.0440), and — the headline discrepancy —

## The CAD build is 100.0 g LIGHTER than the sweep model

2.1724 vs 2.2724 kg, all of it in easytorso (1.022 vs 1.122). The GRID-4/GRID-5
protocol treats the COM ladder as MASS-CONSERVING (same 2.2724 kg robot, weights
slid); this export breaks that assumption — either the CAD has one 100 g plate
fewer than the 1.31 assembly, or the physical 1.05 build really dropped a plate.
To confirm against the CAD parts list (export saw: 100g, 10g, "10g holder short",
"holder back"). Until resolved, this model is NOT a drop-in for c1/c4 replays.

## CAD-true vs the against-physics slide tune (the requested comparison)

The sweep's 1.05 variant slides easytorso's inertial COM by -86.05 mm and keeps
the 1.31 inertia tensor — deliberately unphysical (mass moves, spread doesn't).
The real CAD pays the physics:

| quantity | CAD-true | slide-tuned | CAD / tuned |
|---|---|---|---|
| easytorso principal inertia I1 | 8.72e-3 | 5.13e-3 | **1.70x** |
| I2 | 7.17e-3 | 5.11e-3 | **1.40x** |
| I3 (long axis) | 1.85e-3 | 0.31e-3 | **5.9x** |
| easytorso inertial pos y | -88.8 mm | -84.9 mm | 3.9 mm |
| whole-robot COM fore-aft y | -8.67 mm | -5.11 mm | 3.6 mm |

Reading: physically splitting the mass (structure high, weights low) spreads the
torso mass about its own COM, so every principal inertia grows — 1.4-1.7x about
the bending axes and 5.9x about the long axis. The slide tune under-states torso
rotational inertia by exactly these factors; torso roll dynamics (what the
kappa-PID drives, and the waddle's natural frequency) are the quantities most
exposed. Part of the gap (~10%) also comes from the 100 g mass difference itself.

## Follow-ups (not started)

1. Ben to confirm whether the -100 g is intentional (physical build) or a CAD
   omission; if intentional, the "mass-conserving ladder" language needs a caveat.
2. If c1/c4 hardware comparisons are planned, a hardened 5-actuator version of
   this CAD (penguV3-convention) would replace the slide variant for replays;
   a c1/c4 sensitivity check (slide vs CAD-true inertia on the same champion
   gaits) would quantify what the map numbers miss.
