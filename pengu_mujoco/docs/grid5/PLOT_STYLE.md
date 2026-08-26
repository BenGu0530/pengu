# GRID-5 figure style — frozen visual contract

Decided by Ben 2026-08-26. Single source of truth in code:
`grid5/analysis/style5.py`. This page is the human-readable copy; if they ever
disagree, `style5.py` is wrong and must be fixed to match this page.

**This supersedes the round-1 per-config colour assignments**
(`physics/grid4_support_figs.py:24`). Round-1 figures in
`results/grid4_report/` are historical artifacts and are not restyled; every
new figure — including re-plots of GRID-4 data — uses this contract.

## The hard rule

> **colour + linestyle = gait (κ)**, **marker shape = COM ratio.**

Colour and linestyle deliberately encode the *same* variable: a
black-and-white print drops colour, the linestyle survives. Marker shape
carries the second axis for the same reason. Nothing may be encoded in colour
alone.

## Gait (κ) → colour + linestyle

| gait | κ | colour | linestyle | markers |
|---|---|---|---|---|
| Gait 1 | 0 | `#1f77b4` blue | `-` solid | filled |
| Gait 2 | 2 | `#d62728` red | `--` dashed | hollow |
| (reserved) | 1 | `#2ca02c` green | `-.` dashdot | filled |
| (reserved) Gait 3, no torso | — | `#7f7f7f` gray | `:` dotted | filled |

(Ben 2026-08-26: blue solid / red dashed — a red solid line reads too heavy.)

Hollow-vs-filled markers add a third redundant gait channel (kept from the
round-1 diagonal figures).

## COM ratio → marker shape

| COM | 1.05 | 1.10 | 1.20 | 1.31 | 1.40 | (reserved) 1.60 |
|---|---|---|---|---|---|---|
| marker | `o` | `^` | `s` | `D` | `v` | `P` |

## Crowding: 5 same-colour lines per gait

All lines of a gait share the exact colour; marker shape alone separates COM
(Ben rejected a per-COM shade ramp, 2026-08-26). To keep overlapping points
readable the markers are large (`ms=10`, `mew=1.7`) and κ=2 markers are
hollow (see-through where they stack).

## Mandatory mechanics

- Every figure exits through `style5.finish(fig, path, K=, tier=, stat=, ...)`,
  which stamps the footer (**K**, tier, mean-vs-best-of-best, caveats,
  repo commit) and writes a greyscale twin into a sibling `bw/` directory.
  Never call `plt.savefig` directly from a figure script.
- Legends via `style5.legend_two(ax)`: one gait legend (colour+linestyle),
  one COM legend (gray markers) — never a 10-entry per-config legend. Where a
  single line is referenced, its label is `style5.label_for(cfg)` →
  `c4 (κ=2, COM 1.05)`; never a bare `c4`, never a species label.
- Missing data = a gap plus a label ("no survivors", "PARTIAL"), never an
  implicit zero. Partial configs name the axis values actually present.
- Heatmaps: `viridis` (monotonic luminance). Diverging quantities
  (`net_fwd_mean` crosses 0, trap T7) get a diverging map **plus a drawn zero
  contour**, because any diverging palette turns both ends dark in greyscale.
- `matplotlib.use("Agg")`, `dpi=130`, `tight_layout()` (all inside style5).

## Acceptance check for any new figure

1. Open the `bw/` twin: every line must still be identifiable (linestyle +
   marker + shade, no colour).
2. Footer states K, tier, and mean vs best-of-best.
3. Colours/linestyles/markers come from `style5` — no local colour tables.
