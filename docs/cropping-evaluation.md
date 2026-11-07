# Cropping evaluation — stages 0-3

Run before investing in a reader, on the principle that measuring the half that
exists beats building the half that doesn't.

## Method

One page per form makes direct testing impossible, so the second scan is
manufactured: warp each page by a homography we chose, degrade it, and push it
back through normalize -> identify -> register -> crop. Registration returns
everything to canonical space, so boxes from a perturbed run are directly
comparable to boxes from the clean run.

**Coverage is the headline metric, not IoU.** A crop that loses part of the
reference ink has failed however good its overlap; a crop with extra whitespace
has not. IoU is reported only as a tightness tie-breaker.

## The finding that mattered

The aggregate looked acceptable (mean coverage 0.95 at moderate degradation)
but the minimum was **0.191** — some crop had missed almost entirely. Breaking
it down per field inverted the expected answer:

- blank fields: **1.000 coverage, perfectly stable**
- filled fields: unstable, `bp.systolic` collapsing to **0.230**

The only thing filled fields do differently is pass through ink refinement.
Isolating it on the base card, 18 perturbed crops at moderate degradation:

| | mean coverage | min coverage | crops below 0.95 |
|---|---|---|---|
| refinement off | 0.990 | 0.973 | 0 / 18 |
| refinement on (free) | 0.860 | **0.230** | **8 / 18** |
| refinement on (grow-only) | 0.990 | 0.973 | 0 / 18 |

**Refinement was making cropping worse.** Blur, noise and JPEG shift the Otsu
threshold; components fragment; the ink-derived box collapses onto part of the
number. The feature built to prevent clipping was itself clipping, and only on
degraded scans — which is to say, on real ones.

## The fix

Refinement is now **grow-only**: the ink-derived box is unioned with the nominal
box. The nominal box is a human's statement of where the field is — ink may
extend it, ink may not overrule it.

Coverage of the nominal box is then 1.0 by construction, and overflow is still
caught: the clipped box `[2038, 540, 2226, 660]` still refines to `[2015, ...]`,
picking up the leading `1` at x=2016.

Cost: an over-drawn box no longer gets trimmed. Under a coverage metric that is
free, and it buys back all the stability.

## Property traded away

Refinement used to converge on the same rectangle regardless of the starting
box, which was an appealing property and is now gone by design. The replacement
guarantees, both tested, are that refinement never shrinks the nominal box, and
that either starting point ends up covering the true ink.

## Final results, 8 trials per level, 4 forms, 8 fields

| level | form ID | registration | mean cov | min cov | mean IoU | centre offset |
|---|---|---|---|---|---|---|
| clean | 32/32 | 32/32 | 0.995 | 0.973 | 0.967 | 1.6 px |
| mild | 32/32 | 32/32 | 0.997 | 0.972 | 0.968 | 1.5 px |
| moderate | 32/32 | 32/32 | 0.997 | 0.973 | 0.967 | 1.6 px |
| heavy | 25/32 | 25/32 | 0.996 | 0.973 | 0.984 | 0.7 px |

Before the grow-only fix, moderate showed min coverage 0.191 and heavy 0.068.

**Minimum coverage is now flat at ~0.973 across every level, including heavy.**
Degradation has no measurable effect on crop quality any more. The residual
2.7% is the pipeline's own round-trip cost — the evaluation re-normalises and
re-registers an already-canonical page, resampling it twice — not damage from
the degradation. Chasing it is not worth it against boxes that are padded
anyway.

Degradation levels: mild = 1.5° rotation, 3px blur, JPEG 75. moderate = 3°,
5px blur, JPEG 55. heavy = 6°, 10% scale, 90px shift, 7px blur, sigma-14 noise,
JPEG 35 — well past anything a flatbed would produce.

## Failure behaviour at heavy degradation

Form identification drops to 25/32. Checked separately over 32 heavy trials:

- correct: 26
- **refused (returned UNKNOWN): 6**
- **misidentified: 0**

Every failure was an abstention, not a wrong answer. The margin gate calibrated
in `calibrate_margin.py` holds up under stress, which is the property that
matters: an unrecognised page goes to a human, it does not get cropped against
the wrong template.

## What this does not cover

Every number here comes from **synthetic** perturbation of four pages. It shows
the pipeline is stable against the transformations we simulated; it says
nothing about artefacts we did not think to simulate — crooked stickers, staple
holes, coffee, a form revision. Only real scans can answer that, and the corpus
has one page per form.

## How much of the page was actually scored

Three different scopes, worth not conflating:

| what | scope |
|---|---|
| form identification | whole page — ORB keypoints come from everywhere |
| registration | whole page — 144-point grid, and the heatmap below |
| **crop coverage** | **0.915% of the page** — the 8 BP boxes only |

The 8 boxes total 79,638 px against a 8,699,840 px sheet: **about 1/109**. Every
coverage number in this document describes under one percent of the paper.

### Registration error across the full sheet

14x20 grid over the whole page, 16 runs at moderate degradation:

| | error |
|---|---|
| page centre | 0.15 px |
| overall mean | 0.50 px |
| four corners (mean) | 1.15 px |
| worst cell | 1.51 px |

Error grows radially from the centre — roughly **10x from centre to corner** —
which is the expected behaviour of a single global homography fitted by least
squares. But the absolute magnitude stays tiny: 1.51 px at 300 dpi is 0.128 mm,
about **1.4% of a digit's height**.

Two consequences:

1. **The unmeasured 99% is geometrically sound.** A field added in a page corner
   would sit at ~1.5 px registration error instead of ~0.15 px, and that is still
   negligible against any field's dimensions.
2. **The BP boxes sit in the best part of the page.** Four of the five cluster in
   the upper-middle, where error is lowest. The 0.973 coverage figure was
   measured in the most favourable region available.

### What still generalises poorly

Registration generalises across the sheet; **refinement does not necessarily**.
Every refinement constant — `CHAIN_GAP_FRAC`, `SEED_OVERLAP_FRAC`,
`MIN_HEIGHT_FRAC` — was tuned against handwritten digits on ruled lines. Their
behaviour on checkboxes, circled `Y N` tokens, lab-grid cells or anatomical
diagram marks is entirely unmeasured, and there is no reason to assume it
transfers.

## The insert was never actually tested (and was broken)

Found 2026-08-20, after the sweeps above.

Every evaluation up to this point perturbed the **whole page**, which moves the
sticker together with the form. So the one thing inserts exist for — the sticker
being glued on somewhere different — had never been exercised.

When it finally was, by lifting the sticker off and re-pasting it elsewhere:

| sticker moved | inliers | % on sticker | card's own `bp` field |
|---|---|---|---|
| not moved | 5000 | 89% | similarity **1.000** |
| +60, −40 px | 3080 | **100%** | **0.018** |
| −90, +110 px | 2918 | **100%** | **−0.067** |
| +120, −80 px & −6° | 2452 | **100%** | **0.107** |

**The card was being dragged around by a piece of paper stuck to it.** The base
card's template contains the sticker, the sticker is the densest printed content
on the sheet, so RANSAC found its largest consistent set there and warped the
whole page to put the sticker back where the template had it — pulling the
card's own fields off their values.

A second symptom of the same root cause: a base card with **no sticker at all**
scored a 2.66x form-ID margin, under the 6.5x gate, and came back `UNKNOWN`.

### Fix

`mask_inserts()` blanks declared insert regions out of the template before both
form identification and page registration. Inserts now record where they were
cut from (`source_region` in the form YAML).

| | before | after |
|---|---|---|
| card's `bp` with sticker moved | 0.018 | **1.000** |
| sticker's own field | frozen in place | **tracks the sticker** |
| form-ID margin, stickered page | 278x | 28x |
| form-ID margin, unstickered page | 2.66x (**UNKNOWN**) | **24.5x** |

Inliers drop from ~5000 to ~500 — still 20x the 25 required.

Guarded by `tests/test_insert_independence.py`, which moves the sticker four
ways and asserts both halves: the card's field must hold still, and the
insert's field must not.

### The lesson

This is the third finding to trace back to one root cause: **the base card's
template is a filled page with a sticker on it.** The same fact blocked template
subtraction for ink refinement, and forced the tuned constants in `refine.py`.
Obtaining a genuinely blank base OPD card would dissolve all three at once, and
is now the single highest-value item on the list that isn't "get more pages".

## Can the sticker be anywhere?

Structurally, yes: locating an insert is a full-page feature match with **no
positional prior**. `source_region` is fixed, but it only says where the sticker
sits in the *template* (which is one fixed image), not where it may appear on a
new page.

Measured envelope, sticker lifted off and re-glued:

| moved by | insert located | card's own `bp` |
|---|---|---|
| up to +1100, +1350 px | yes | 1.000 |
| rotated 10, 20, 45, **90 degrees** | yes | 1.000 |
| scaled 0.75x, 0.9x, 1.1x | yes | 1.000 |
| hanging off the right or bottom edge | yes | 1.000 |
| over the header | yes | 1.000 |

Nothing in that sweep failed. ORB is rotation- and scale-invariant by
construction, so this is expected rather than lucky. Field-position error stays
under 1 px for pure translation and rises to 18-30 px under large rotation,
because `project()` takes the axis-aligned bounds of a rotated rectangle and
refinement then grows into the extra area.

### The one case that did fail

A sticker moved **on top of** a base-form field. The field is then physically
covered, so cropping its coordinates returns the sticker's content:

```
sticker COVERS the card's BP   similarity 0.672   reads "unreadable"   flags: -
```

No flag. It failed safe only because the reader happened to choke on whatever
was there; had that area held digits it would have produced a confident wrong
blood pressure with nothing to signal it.

Fixed: inserts are now located **before** base fields are cropped, and any field
overlapping an insert's footprint by >=20% is flagged `occluded_by:<insert>`,
which forces review. Verified to fire on the covering case (100%) and not on the
other four.
