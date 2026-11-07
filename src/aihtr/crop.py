"""Stage 3 — cut the field crops out of a registered page.

Two crops per field, following the SiData+ dataset's own convention:

* **tight**   — just the value. What the reader model is asked to transcribe.
* **context** — the value plus the printed label around it (`BP`, `mmHg`).
                Useful for debugging, for a human reviewing a flagged case, and
                for giving a reader model something to disambiguate against.

Inserts are the interesting part. Base-form fields sit at fixed canonical
coordinates, so cropping them is a slice. An insert — the triage sticker — is
glued on by hand and lands somewhere different on every page, so its fields are
stored in *its own* coordinate space and have to be projected onto the page at
runtime by locating the sticker first.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from .refine import Refinement, refine_field
from .register import register
from .schema import BBox, Field, FormSpec


@dataclass
class FieldCrop:
    field_id: str
    owner: str                 # form_id, or the insert's id
    tight: np.ndarray
    context: np.ndarray
    tight_box: list[int]       # page coordinates, after any insert projection
    context_box: list[int]
    projected: bool            # True if it came via an insert homography
    refinement: Refinement | None = None
    extra_flags: list[str] = dc_field(default_factory=list)

    @property
    def flags(self) -> list[str]:
        return (self.refinement.flags if self.refinement else []) + self.extra_flags

    @property
    def needs_review(self) -> bool:
        return any(f.startswith(("possibly_clipped", "field_empty", "occluded_by"))
                   for f in self.flags)


def _slice(page: np.ndarray, box) -> np.ndarray:
    h, w = page.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return np.full((1, 1), 255, np.uint8)
    return page[y1:y2, x1:x2]


def project(box: BBox, H: np.ndarray) -> list[int]:
    """Map an axis-aligned box through a homography.

    The result of a homography on a rectangle is a quadrilateral, not a
    rectangle, so we take the axis-aligned bounding box of the four corners.
    That over-covers slightly under rotation — which is the right direction to
    err, since crops are scored on coverage rather than IoU.
    """
    pts = np.float32([[box.x1, box.y1], [box.x2, box.y1],
                      [box.x2, box.y2], [box.x1, box.y2]]).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [int(np.floor(out[:, 0].min())), int(np.floor(out[:, 1].min())),
            int(np.ceil(out[:, 0].max())),  int(np.ceil(out[:, 1].max()))]


def mask_inserts(template: np.ndarray, spec: FormSpec, fill: int = 255) -> np.ndarray:
    """Blank out insert regions in a template before page registration.

    Without this the pipeline silently breaks whenever a sticker is glued on
    differently from the page the template was cut from.

    The base card's template contains the triage sticker, and the sticker is by
    far the densest printed thing on the sheet — 89% of the matched inliers land
    on it. So when a new page has the sticker 60 px to the left, RANSAC finds
    its largest consistent set on the sticker and warps the *whole page* to put
    the sticker back where the template has it. Measured: the card's own `bp`
    field dropped from 1.000 similarity to 0.018. The card was dragged around by
    a piece of paper stuck to it.

    Masking leaves ~500 inliers instead of ~5000, which is still an order of
    magnitude above the 25 needed, and the card's fields then hold still while
    the insert's fields correctly track the insert.

    A genuinely blank base card template would make this unnecessary, since the
    sticker's features would have nothing to match against in the first place.
    """
    if not spec.inserts:
        return template
    out = template.copy()
    h, w = out.shape[:2]
    for ins in spec.inserts:
        r = ins.source_region
        if r is None:
            continue
        out[max(0, r.y1):min(h, r.y2), max(0, r.x1):min(w, r.x2)] = fill
    return out


def _context_of(f: Field) -> BBox:
    """Fall back to a padded tight box when no context box was drawn."""
    if f.context_bbox:
        return f.context_bbox
    return f.bbox.pad(int(0.4 * max(*f.bbox.wh)))


def _build(page, field_id, owner, tight, ctx, projected, do_refine) -> FieldCrop:
    """Cut one field, optionally letting the ink correct the rectangle.

    Refinement runs against the *context* crop, not the tight one — the whole
    point is to see ink that falls outside the nominal box, which by definition
    is not in the tight crop.
    """
    context_img = _slice(page, ctx)
    ref = None
    if do_refine:
        ref = refine_field(context_img, ctx, tight)
        tight = ref.box
    return FieldCrop(field_id, owner, _slice(page, tight), context_img,
                     list(tight), list(ctx), projected, ref)


OCCLUSION_FRAC = 0.20   # this much of a field under an insert counts as covered


def _overlap_frac(box, region) -> float:
    iw = max(0, min(box[2], region[2]) - max(box[0], region[0]))
    ih = max(0, min(box[3], region[3]) - max(box[1], region[1]))
    return iw * ih / max(1, (box[2] - box[0]) * (box[3] - box[1]))


def crop_fields(
    page: np.ndarray,
    spec: FormSpec,
    insert_templates: dict[str, np.ndarray] | None = None,
    do_refine: bool = False,
) -> list[FieldCrop]:
    """Crop every resolved field on `page`, which must already be registered
    into canonical space.

    With `do_refine`, each box is grown to the ink actually present and flagged
    if that ink runs to the edge of its context (so the true extent is unknown)
    or if the field looks unfilled.
    """
    out: list[FieldCrop] = []
    insert_templates = insert_templates or {}

    # Locate every insert FIRST. A field sitting under a sticker is physically
    # covered: cropping its coordinates returns the sticker's content, and
    # without this the only symptom is that the reader struggles. Measured on a
    # page with the sticker moved over the card's BP: similarity 0.672, reading
    # unreadable, and not one flag raised. It failed safe by luck.
    located: list[tuple[str, list[int], object]] = []
    for ins in spec.inserts:
        tmpl = insert_templates.get(ins.id)
        if tmpl is None:
            continue
        res = register(tmpl, page)          # where is it on THIS page?
        if not res.ok:
            continue
        th, tw = tmpl.shape[:2]
        footprint = project(BBox(0, 0, tw, th), res.homography)
        located.append((ins.id, footprint, res))

    for f in spec.fields:
        if not f.resolved:
            continue
        tb = [f.bbox.x1, f.bbox.y1, f.bbox.x2, f.bbox.y2]
        cb = _context_of(f)
        fc = _build(page, f.id, spec.form_id, tb,
                    [cb.x1, cb.y1, cb.x2, cb.y2], False, do_refine)
        for ins_id, footprint, _ in located:
            frac = _overlap_frac(fc.tight_box, footprint)
            if frac >= OCCLUSION_FRAC:
                fc.extra_flags.append(f"occluded_by:{ins_id}({frac:.0%})")
        out.append(fc)

    for ins in spec.inserts:
        match = next((l for l in located if l[0] == ins.id), None)
        if match is None:
            continue
        _, _, res = match
        for f in ins.fields:
            if not f.resolved:
                continue
            out.append(_build(page, f.id, ins.id, project(f.bbox, res.homography),
                              project(_context_of(f), res.homography), True, do_refine))
    return out
