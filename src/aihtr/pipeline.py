"""End to end: a scanned page in, validated readings out.

    normalize -> identify -> register -> crop+refine -> read -> validate

Every stage can decline. An unidentified form, a failed registration, an empty
field, an unreadable glyph and an impossible value all produce a result that
says so, and none of them produce a number that looks fine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .crop import FieldCrop, crop_fields, mask_inserts
from .formid import FormIdentifier
from .normalize import CANONICAL_SIZE, normalize_page
from .read import BPReading, read_pattern_a, read_pattern_split
from .register import register, warp
from .schema import FormSpec
from .validate import Validation, Verdict, cross_check, validate_bp


@dataclass
class FieldResult:
    group: str
    reading: BPReading
    validation: Validation | None
    crops: list[FieldCrop] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        if not self.reading.readable:
            return True
        if self.validation and self.validation.needs_review:
            return True
        return any(c.needs_review for c in self.crops)


@dataclass
class PageResult:
    source: Path
    form_id: str | None
    reason: str = ""
    fields: dict[str, FieldResult] = field(default_factory=dict)
    page_check: Validation | None = None

    @property
    def ok(self) -> bool:
        return self.form_id is not None

    @property
    def needs_review(self) -> bool:
        if not self.ok:
            return True
        if self.page_check and self.page_check.verdict is not Verdict.PLAUSIBLE:
            return True
        return any(f.needs_review for f in self.fields.values())


def _role(field_id: str) -> tuple[str, str]:
    """Split a field id into (group, role).

    Patterns B and C store systolic and diastolic as separate fields; this is
    what pairs them back up so a single BP reading comes out the far end.
    """
    for suffix, role in ((".systolic", "systolic"), (".diastolic", "diastolic"),
                         ("_sbp", "systolic"), ("_dbp", "diastolic")):
        if field_id.endswith(suffix):
            return field_id[: -len(suffix)], role
    return field_id, "single"


def process_page(
    path: str | Path,
    forms: dict[str, FormSpec],
    templates: dict[str, np.ndarray],
) -> PageResult:
    src = Path(path)
    page = normalize_page(src).image

    # Identify by the CARD too, not by what is stuck on it. With the sticker
    # left in the template, a base card that simply has no triage sheet scores
    # a 2.66x margin — under the 6.5x gate — and comes back UNKNOWN. Masking
    # the insert region drops the margin on a stickered page from 278x to 28x
    # (still far clear of the gate) and lifts an unstickered page to 24.5x.
    ident = FormIdentifier({
        k: mask_inserts(v, forms[k]) for k, v in templates.items() if k in forms
    })
    res = ident.identify(page)
    if res.form_id is None:
        return PageResult(src, None, f"form not identified ({res.reason})")

    spec = forms[res.form_id]
    # Register against the CARD, not against whatever is stuck to it. See
    # mask_inserts() -- without this a sticker glued 60 px off drags the card's
    # own fields with it.
    reg = register(page, mask_inserts(templates[res.form_id], spec))
    if not reg.ok:
        return PageResult(src, res.form_id, f"registration failed ({reg})")
    page = warp(page, reg.homography, CANONICAL_SIZE)

    crops = crop_fields(page, spec, templates, do_refine=True)

    grouped: dict[str, dict[str, FieldCrop]] = {}
    for fc in crops:
        group, role = _role(fc.field_id)
        # Qualify by owner. The base card's top-level `bp` and the triage
        # sticker's `bp.systolic` both reduce to the group "bp", and merging
        # them silently loses the sticker's reading entirely.
        key = group if fc.owner == res.form_id else f"{fc.owner}/{group}"
        grouped.setdefault(key, {})[role] = fc

    out = PageResult(src, res.form_id)
    for group, parts in grouped.items():
        used = [parts[k] for k in ("single", "systolic", "diastolic") if k in parts]

        # An empty field is a fact, not a failure. Check BEFORE reading —
        # running the classifier over blank paper only manufactures glyphs out
        # of speckle, which is exactly the invented-value failure we are trying
        # to design out.
        if any("field_empty" in c.flags for c in used):
            out.fields[group] = FieldResult(
                group, BPReading(flags=["field_empty"]), None, used)
            continue

        if "single" in parts:
            reading = read_pattern_a(parts["single"].tight)
        else:
            s, d = parts.get("systolic"), parts.get("diastolic")
            if s is None or d is None:
                continue
            reading = read_pattern_split(s.tight, d.tight)

        val = validate_bp(reading.systolic, reading.diastolic) if reading.readable else None
        out.fields[group] = FieldResult(group, reading, val, used)

    readable = {
        g: (f.reading.systolic, f.reading.diastolic)
        for g, f in out.fields.items()
        if f.reading.readable
    }
    # Count the BP sources the page actually carries, not the ones that read.
    # Gating on len(readable) >= 2 was the bug: when one source came back
    # field_empty or unreadable the check never ran at all, so the page kept a
    # clean sheet precisely when half its redundancy had vanished.
    if len(out.fields) >= 2:
        out.page_check = cross_check(readable, expected=len(out.fields))
    return out
