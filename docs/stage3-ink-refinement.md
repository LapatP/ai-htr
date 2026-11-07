# Ink-aware crop refinement + loud failures

Status: **BUILT** 2026-08-20 as `src/aihtr/refine.py`, but **not by the method
described below.** Read the deviation note first.

## Deviation: no template subtraction

Template subtraction proved unusable. `base-opd-card-v1`'s template IS the one
filled page in the corpus, so differencing it cancels the very handwriting we
are hunting — and it is the only page with real BP ink, so the technique could
not be tested where it matters.

Implemented instead: strip printed rules morphologically, connected-component
what remains, and chain outward from the nominal box. No blank template needed.

Measured on the real card, replaying the actual bug:

| box fed in | refined to | true ink extent |
|---|---|---|
| `[2038, 540, 2226, 660]` (clipped the `1`) | `[2015, 545, 2219, 651]` | `[2016, 546, 2218, 656]` |
| `[2008, 538, 2226, 664]` (correct) | `[2015, 545, 2219, 651]` | same |

It converges on the ink regardless of the seed rectangle. Filled/blank
classification is 8/8 across the corpus.

**What the deviation costs.** Rule-stripping removes horizontal rules but not
printed *glyphs*. On the blank MS form the printed `/` separator seeded the
field through a one-pixel overlap and reported ink that was never written.
Fixed with `SEED_OVERLAP_FRAC` (a component must overlap the box by 25% of
itself, not merely graze it) and `MIN_HEIGHT_FRAC`. Template subtraction would
have separated print from pen cleanly and needed neither. **This remains the
argument for obtaining a blank base OPD card.**

## The problem

Stage 3 as built is purely coordinate-based. The box is a fixed rectangle in
canonical space; nothing looks at the handwriting. Registration aligns *printed*
features only, so when handwriting overflows its field the crop clips a digit
**and registration still reports a clean sub-pixel fit**. The failure is silent.

The base card already demonstrates it: `143/90` is written large and spills
above the ruled line.

Three failure modes, none of which currently announce themselves:

| Mode | Result |
|---|---|
| Overflow | digit clipped |
| Written well off the line | empty crop |
| Neighbour bleed | adjacent field's ink intrudes |

Current mitigation is generous padding (`PAD_X=0.55`, `PAD_Y=0.45` in
`apply_boxes.py`) plus scoring on coverage rather than IoU. That is a guess,
not a measurement.

## The fix

Registration is accurate to ~0.3 px, which makes **template subtraction** viable.
Align the blank template to the page, difference them, and the printed furniture
cancels — what survives is only what somebody wrote.

Then per field:

1. Crop the generous context box from the residual
2. Connected-component the ink
3. Keep components that touch or overlap the nominal tight box
4. Their union bounding box becomes the real crop

The stored box stops being an assumption and becomes a seed. Overflow is handled
by construction because ink is measured, not assumed.

### Blocker for this

Three of four templates (nutrition, ms-form, neuro) are blank pages, so
subtraction works on them today. **base-opd-card-v1 is not** — its template is
the one page with `143/90` already on it, so subtracting it would cancel the
handwriting we are hunting.

Two ways out, second preferred:

- inpaint the handwriting out of that template by hand
- obtain a **blank base OPD card** and rebuild the template — it is the standard
  hospital outpatient form, so a clean copy should be easy to get, and it would
  improve registration too

Note the SiData+ dataset ships exactly this as
`compiled-template/assets/canonical-movable.png` (a derived blank background).
If gated access lands, that resolves it directly.

### Ruled out: colour separation

Measured 2026-08-20 on `base-opd-card-v1_001.jpg`. Mean saturation is **0.0/255**
across handwriting, printed Thai and blank paper alike — it is a pure black-and-
white scan. Pen-vs-print cannot be separated by hue on this corpus. May become
available on future colour scans; do not assume it.

## Make failures loud (build this even if refinement slips)

- ink touching the context-box edge → flag `possibly_clipped`, route to review
- no ink anywhere in the box → report `field_empty`

`field_empty` and `unreadable` must never collapse into one output. A BP that was
never recorded and a BP that was misread are clinically different facts.

## When to reach for a learned detector

Only if measured placement variance is large. These forms are filled by trained
staff in fixed fields, so coordinates + ink refinement should cover the great
majority. Test it once the boxes exist: crop with the dumb rectangle, then count
how often ink escapes. Let that number decide, not intuition.
