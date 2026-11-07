"""Ink-aware crop refinement, and the flags that make failures loud.

Why this exists
---------------
Stage 3 crops a fixed rectangle. When handwriting overflows its field the crop
clips a digit **and nothing complains** — registration still reports sub-pixel
error, the reader confidently transcribes what it was handed. This module
measures where the ink actually is and grows the box to fit it.

Deviation from docs/stage3-ink-refinement.md
--------------------------------------------
That note proposed template subtraction: align the blank template, difference
it, keep the residual. Sound in principle, unusable here — the base OPD card's
template IS the one filled page we have, so subtracting it would cancel the
handwriting. It is also the only page in the corpus with real BP ink, so the
technique could not be tested where it matters.

Instead: strip the printed rules morphologically, connected-component what is
left, and chain outward from the nominal box. This needs no blank template and
works on every page today. Template subtraction stays a worthwhile upgrade for
whenever a genuinely blank base card turns up — it would separate print from
pen more cleanly than rule-stripping can.

What it does NOT do
-------------------
It never invents ink and never changes a transcription. It moves a rectangle
and raises flags. Deciding what a digit says remains stage 4's job, and
deciding whether to trust it remains validate.py's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# A component is chained into the field if it sits within this fraction of the
# box height of the ink already accepted. Measured on the real card: digits
# inside `143/90` are at most 21 px apart against a 120 px box height (0.18),
# while the printed `BP` label sits 40 px away (0.33). 0.25 separates them.
CHAIN_GAP_FRAC = 0.25

# A chained component must be at least this tall relative to the tallest ink
# already accepted. Printed labels on these forms are markedly shorter than the
# handwritten values beside them (`BP` is 27 px against ~90 px of pen), so this
# stops a label being absorbed even when it sits close.
CHAIN_MIN_HEIGHT_FRAC = 0.45

# A component only seeds the field if this fraction of it lies inside the
# declared box. Without it a printed glyph grazing the box edge by one pixel
# seeds the whole field: on the blank MS form the printed `/` separator did
# exactly that, and the field reported ink that was never written.
SEED_OVERLAP_FRAC = 0.25

# Ink shorter than this fraction of the field height is rule debris, not a
# written value.
MIN_HEIGHT_FRAC = 0.18

# Ink closer than this to the context edge means there may be more beyond it.
EDGE_TOUCH_PX = 2

# Below this fraction of dark pixels the field is treated as unfilled.
EMPTY_INK_FRAC = 0.004


@dataclass
class Refinement:
    box: list[int]                  # refined tight box, page coordinates
    original: list[int]
    flags: list[str] = field(default_factory=list)
    n_components: int = 0
    ink_fraction: float = 0.0

    @property
    def grew(self) -> bool:
        return self.box != self.original

    @property
    def growth(self) -> tuple[int, int, int, int]:
        """Per-side growth in px: (left, top, right, bottom). Positive = wider."""
        o, b = self.original, self.box
        return (o[0] - b[0], o[1] - b[1], b[2] - o[2], b[3] - o[3])

    def __str__(self) -> str:
        s = f"{self.box}"
        if self.grew:
            l, t, r, btm = self.growth
            s += f"  grew L{l:+d} T{t:+d} R{r:+d} B{btm:+d}"
        if self.flags:
            s += "  [" + ", ".join(self.flags) + "]"
        return s


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink mask. Otsu, guarded against near-uniform crops.

    A blank field is almost uniform paper, and Otsu on a uniform image splits
    noise into a meaningless half-and-half mask. Checking contrast first is what
    stops an empty field reporting a page full of ink.
    """
    if gray.size == 0 or gray.std() < 8:
        return np.zeros_like(gray, np.uint8)
    _, m = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return m.astype(np.uint8)


def _strip_rules(mask: np.ndarray) -> np.ndarray:
    """Remove the printed ruled lines, dotted or solid.

    Dotted rules are not continuous, so a plain opening misses them. Close along
    the horizontal first to fuse the dots into a line, then open with a long
    thin kernel to isolate structures that run across the field, then subtract.
    """
    h, w = mask.shape[:2]
    if w < 20 or h < 6:
        return mask
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1)))
    rules = cv2.morphologyEx(joined, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 3), 1)))
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    return cv2.subtract(mask, rules)


def _seeds(comp, box) -> bool:
    """Does enough of `comp` lie inside `box` to call it part of this field?"""
    iw = min(comp[2], box[2]) - max(comp[0], box[0])
    ih = min(comp[3], box[3]) - max(comp[1], box[1])
    if iw <= 0 or ih <= 0:
        return False
    return (iw >= SEED_OVERLAP_FRAC * (comp[2] - comp[0])
            and ih >= SEED_OVERLAP_FRAC * (comp[3] - comp[1]))


def _boxes_touch(a, b, gap: float) -> bool:
    """Do two boxes come within `gap`, allowing for vertical overlap?"""
    hgap = max(a[0] - b[2], b[0] - a[2])
    vgap = max(a[1] - b[3], b[1] - a[3])
    return hgap <= gap and vgap <= gap


def refine_field(
    context: np.ndarray,
    context_box: list[int],
    tight_box: list[int],
    chain_gap_frac: float = CHAIN_GAP_FRAC,
    grow_only: bool = True,
) -> Refinement:
    """Grow `tight_box` to the ink that actually belongs to this field.

    `context` is the context crop; both boxes are in page coordinates.

    `grow_only` unions the result with the nominal box, so refinement can only
    ever expand. This is not a detail — it was measured. Letting the ink fully
    determine the box made cropping *less* reliable on a degraded scan: blur,
    noise and JPEG shift the Otsu threshold, components fragment, and the box
    can collapse onto part of the number. Across 18 perturbed crops of the base
    card, free refinement dropped minimum coverage to 0.230 with 8 crops below
    0.95, while no refinement at all held 0.973. Growing only keeps coverage of
    the nominal box at 1.0 by construction and still catches overflow — and
    since crops are scored on coverage, the extra whitespace costs nothing.
    """
    cx, cy = context_box[0], context_box[1]
    local = [tight_box[0] - cx, tight_box[1] - cy, tight_box[2] - cx, tight_box[3] - cy]

    mask = _strip_rules(_binarize(context))
    ink_frac = float(mask.mean()) if mask.size else 0.0

    if ink_frac < EMPTY_INK_FRAC:
        return Refinement(list(tight_box), list(tight_box), ["field_empty"], 0, ink_frac)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    boxes = []
    min_area = max(6, int(0.0004 * mask.size))
    min_h = MIN_HEIGHT_FRAC * max(1, local[3] - local[1])
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a >= min_area and h >= min_h:
            boxes.append([x, y, x + w, y + h])

    # Seed: components with meaningful overlap, not merely grazing the edge.
    kept = [b for b in boxes if _seeds(b, local)]
    if not kept:
        return Refinement(list(tight_box), list(tight_box), ["field_empty"], 0, ink_frac)

    # Chain outward. This is what recovers a digit sitting just outside the box:
    # it is not touching the rectangle, but it is touching its neighbour.
    gap = chain_gap_frac * max(1, local[3] - local[1])
    min_h = CHAIN_MIN_HEIGHT_FRAC * max(b[3] - b[1] for b in kept)
    pool = [b for b in boxes if b not in kept]
    changed = True
    while changed:
        changed = False
        union = [min(b[0] for b in kept), min(b[1] for b in kept),
                 max(b[2] for b in kept), max(b[3] for b in kept)]
        for b in list(pool):
            if (b[3] - b[1]) >= min_h and _boxes_touch(b, union, gap):
                kept.append(b)
                pool.remove(b)
                changed = True

    ref = [min(b[0] for b in kept), min(b[1] for b in kept),
           max(b[2] for b in kept), max(b[3] for b in kept)]
    if grow_only:
        # Union with what was declared. The nominal box is a human's statement
        # of where the field is; ink evidence may extend it but never overrule it.
        ref = [min(ref[0], local[0]), min(ref[1], local[1]),
               max(ref[2], local[2]), max(ref[3], local[3])]
    # Never escape the context crop — beyond it we have no evidence at all.
    ref = [max(0, ref[0]), max(0, ref[1]),
           min(context.shape[1], ref[2]), min(context.shape[0], ref[3])]

    flags = []
    # Ink running to the context edge means the context box itself may be too
    # small, so the true extent is unknown. That is a review case, not a fix.
    if (ref[0] <= EDGE_TOUCH_PX or ref[1] <= EDGE_TOUCH_PX
            or ref[2] >= context.shape[1] - EDGE_TOUCH_PX
            or ref[3] >= context.shape[0] - EDGE_TOUCH_PX):
        flags.append("possibly_clipped")

    page_box = [ref[0] + cx, ref[1] + cy, ref[2] + cx, ref[3] + cy]
    if page_box != list(tight_box):
        flags.append("box_adjusted")

    return Refinement(page_box, list(tight_box), flags, len(kept), ink_frac)
