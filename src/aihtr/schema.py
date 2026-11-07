"""Form definition schema.

The whole design rests on one idea: **a form is data, not code.** Adding a new
form should mean writing a YAML file, never editing Python.

Coordinate convention
---------------------
Every ``bbox`` is expressed in *canonical template space* -- the pixel
coordinates of that form's clean template image, NOT of the scan you feed in.
The registration stage warps an input page into canonical space first, so by
the time we crop, fixed coordinates are valid again.

Two boxes per field, mirroring the sidataplus dataset's own layout:
  * ``bbox``         -- tight: just the handwritten value.
  * ``context_bbox`` -- wide: includes the printed label ("BP", "mmHg").
                        Useful for debugging and for giving a reader model
                        disambiguating context.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Iterator

import yaml

# --- The seven field archetypes observed across the hospital's form family. ------
# Only the first is implemented in v1; the rest are declared so the schema does
# not need breaking changes later.
FIELD_TYPES = {
    "line_value",    # 1. handwritten value on a ruled line       [v1]
    "checkbox",      # 2. tick inside a printed box/circle
    "choice_circle", # 3. ring drawn AROUND one of N printed tokens
    "grid_cell",     # 4. cell in a lab/investigation table
    "free_text",     # 5. unstructured region
    "diagram",       # 6. mark placed on an anatomical figure  (deferred)
    "identity",      # 7. HN / name / date
}

# How a blood-pressure value is physically encoded on the page. This is the
# distinction that makes "BP everywhere" a real test of the design.
BP_PATTERNS = {
    "A",  # one blank; the writer supplies the slash:  BP [143/90] mmHg
    "B",  # two blanks flanking a PRINTED slash:       BP [143]/[90] mmHg
    "C",  # two separately labelled blanks, no slash:  SBP [143]  DBP [90]
}

READERS = {
    "bp_pair_single",   # pattern A -> one crop, split the string ourselves
    "bp_pair_split",    # pattern B -> two crops, one integer each
    "bp_labelled_split",# pattern C -> two crops, order carries the meaning
    "numeric",
    "text",
}


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def wh(self) -> tuple[int, int]:
        return self.x2 - self.x1, self.y2 - self.y1

    def pad(self, px: int) -> "BBox":
        """Grow the box. Crops are scored on *coverage*, not IoU -- a crop that
        clips a digit is a total failure, extra whitespace costs nothing."""
        return BBox(self.x1 - px, self.y1 - px, self.x2 + px, self.y2 + px)

    @staticmethod
    def parse(v) -> "BBox | None":
        if v is None:
            return None
        if len(v) != 4:
            raise ValueError(f"bbox needs 4 ints, got {v!r}")
        return BBox(*(int(n) for n in v))


@dataclass
class Field:
    id: str
    type: str
    label: str = ""
    reader: str | None = None
    bbox: BBox | None = None          # None = box not yet drawn
    context_bbox: BBox | None = None
    unit: str | None = None
    bp_pattern: str | None = None
    options: list[str] = dc_field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """True once someone has actually drawn this field's box."""
        return self.bbox is not None

    def validate(self) -> list[str]:
        errs = []
        if self.type not in FIELD_TYPES:
            errs.append(f"{self.id}: unknown type {self.type!r}")
        if self.reader and self.reader not in READERS:
            errs.append(f"{self.id}: unknown reader {self.reader!r}")
        if self.bp_pattern and self.bp_pattern not in BP_PATTERNS:
            errs.append(f"{self.id}: unknown bp_pattern {self.bp_pattern!r}")
        return errs


@dataclass
class Insert:
    """A separate sheet laid on top of the page -- the triage sticker, an MS
    form. Its position VARIES per page, so unlike base-form fields its
    coordinates cannot be fixed: it must be located before its fields resolve.
    """
    id: str
    template: str
    fields: list[Field] = dc_field(default_factory=list)
    locate: str = "feature_match"
    # Where this insert sat on the page the template was cut from. Needed so
    # page registration can mask it out -- see mask_inserts() in crop.py.
    source_region: BBox | None = None


@dataclass
class FormSpec:
    form_id: str
    name: str = ""
    template: str | None = None
    print_run: str | None = None
    extends: str | None = None
    page_size_px: tuple[int, int] | None = None
    fields: list[Field] = dc_field(default_factory=list)
    inserts: list[Insert] = dc_field(default_factory=list)
    source_path: Path | None = None

    def all_fields(self) -> Iterator[tuple[str, Field]]:
        """Yield (owner, field) over the form and every insert on it."""
        for f in self.fields:
            yield self.form_id, f
        for ins in self.inserts:
            for f in ins.fields:
                yield ins.id, f

    def validate(self) -> list[str]:
        errs = []
        seen = set()
        for owner, f in self.all_fields():
            key = (owner, f.id)
            if key in seen:
                errs.append(f"duplicate field id {f.id!r} in {owner}")
            seen.add(key)
            errs.extend(f.validate())
        return errs


def _parse_field(d: dict) -> Field:
    return Field(
        id=d["id"],
        type=d["type"],
        label=d.get("label", ""),
        reader=d.get("reader"),
        bbox=BBox.parse(d.get("bbox")),
        context_bbox=BBox.parse(d.get("context_bbox")),
        unit=d.get("unit"),
        bp_pattern=d.get("bp_pattern"),
        options=d.get("options", []) or [],
    )


def load_form(path: str | Path) -> FormSpec:
    path = Path(path)
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    size = d.get("page_size_px")
    spec = FormSpec(
        form_id=d["form_id"],
        name=d.get("name", ""),
        template=d.get("template"),
        print_run=str(d["print_run"]) if d.get("print_run") else None,
        extends=d.get("extends"),
        page_size_px=tuple(size) if size else None,
        fields=[_parse_field(f) for f in d.get("fields", []) or []],
        inserts=[
            Insert(
                id=i["id"],
                template=i["template"],
                locate=i.get("locate", "feature_match"),
                source_region=BBox.parse(i.get("source_region")),
                fields=[_parse_field(f) for f in i.get("fields", []) or []],
            )
            for i in d.get("inserts", []) or []
        ],
        source_path=path,
    )
    if errs := spec.validate():
        raise ValueError(f"{path.name}: " + "; ".join(errs))
    return spec


def load_all(directory: str | Path = "configs/forms") -> dict[str, FormSpec]:
    forms = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        spec = load_form(p)
        forms[spec.form_id] = spec
    return forms


def status(directory: str | Path = "configs/forms") -> str:
    """Report which fields still need boxes drawn on a template."""
    forms = load_all(directory)
    lines, done, total = [], 0, 0
    for spec in forms.values():
        lines.append(f"\n{spec.form_id}  ({spec.name})")
        tmpl = Path(spec.template) if spec.template else None
        have = tmpl.exists() if tmpl else False
        lines.append(f"  template: {spec.template}  [{'present' if have else 'MISSING'}]")
        for owner, f in spec.all_fields():
            total += 1
            done += f.resolved
            mark = "ok  " if f.resolved else "TODO"
            where = "" if owner == spec.form_id else f"  (insert: {owner})"
            lines.append(f"  [{mark}] {f.id:<24} {f.type:<12} {f.bp_pattern or '':<2}{where}")
    lines.append(f"\n{done}/{total} fields have boxes drawn.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(status())
