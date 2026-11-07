"""Generate the self-contained box-drawing tool.

Embeds each template as a half-scale JPEG data URI so the output is one HTML
file you can double-click — no server, no file:// loading problems.

Half scale keeps the payload near 1.5 MB. Precision is unaffected in practice:
one screen pixel is two canonical pixels, and boxes are padded generously
anyway because crops are scored on coverage rather than IoU.

Usage:  PYTHONPATH=src python scripts/make_box_tool.py
Output: docs/box-tool.html
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema  # noqa: E402

TEMPLATES = ROOT / "data" / "templates"
OUT = ROOT / "docs" / "box-tool.html"
DOWNSCALE = 2
JPEG_Q = 88

# The tool is only useful if it says WHERE to look. Written from the paper's
# point of view, not the schema's.
HINTS = {
    ("base-opd-card-v1", "bp"):
        "Top-right vitals line. The handwritten <b>143/90</b> sitting between the "
        "printed <b>BP</b> and <b>mmHg</b>. Note it is written large and spills "
        "above the rule — include the overflow.",
    ("base-opd-card-v1", "__region__.triage-card-v1"):
        "The whole pasted triage screening sheet — trace its outer edge. This one "
        "is not a field: it becomes the insert template, and the two boxes below "
        "get rebased into its coordinates.",
    ("base-opd-card-v1", "bp.systolic"):
        "Inside the triage sheet, on the <b>BP</b> line: the <b>143</b> before the "
        "printed slash.",
    ("base-opd-card-v1", "bp.diastolic"):
        "Inside the triage sheet: the <b>90</b> after the printed slash, before "
        "<b>mmHg</b>.",
    ("nutrition-0190-v1", "vitals.bp"):
        "On the <b>Vital Signs &amp; Body weight</b> line: the blank after <b>BP</b> "
        "and before <b>mmHg</b>. Blank on this page — box the space a value would go.",
    ("ms-form-v2_4", "screening.bp.systolic"):
        "In the <b>Screening</b> box, second row: the first blank in "
        "<b>BP ...... / ...... mmHg</b>.",
    ("ms-form-v2_4", "screening.bp.diastolic"):
        "Same line, the second blank — after the printed slash.",
    ("neuro-v1", "risk.home_sbp"):
        "Bottom of <b>Risk factors screening</b>: the ruled blank after <b>Home SBP</b>.",
    ("neuro-v1", "risk.home_dbp"):
        "Same line: the ruled blank after <b>DBP</b>.",
}

# Drawn first on the base card, because the two triage boxes are read relative
# to it.
REGIONS = {
    "base-opd-card-v1": [("__region__.triage-card-v1", "region", None)],
}


def encode(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // DOWNSCALE, h // DOWNSCALE), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    if not ok:
        raise RuntimeError(f"failed to encode {path}")
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def main() -> int:
    specs = schema.load_all(ROOT / "configs" / "forms")
    forms, missing = [], []

    for fid, spec in sorted(specs.items()):
        png = TEMPLATES / f"{fid}.png"
        if not png.exists():
            missing.append(fid)
            continue

        fields = []
        for rid, kind, pat in REGIONS.get(fid, []):
            fields.append({"id": rid, "pattern": kind,
                           "hint": HINTS.get((fid, rid), "")})
        for owner, f in spec.all_fields():
            fields.append({
                "id": f.id,
                "pattern": f.bp_pattern or "",
                "hint": HINTS.get((fid, f.id), f.label or f.id),
            })

        forms.append({"form_id": fid, "png": encode(png), "fields": fields})

    if missing:
        print(f"skipped (no template): {', '.join(missing)}")

    data = "const DATA = " + json.dumps(
        {"scale": DOWNSCALE, "forms": forms}, separators=(",", ":")) + ";"

    tpl = (ROOT / "scripts" / "box_tool.template.html").read_text(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(tpl.replace("/*__DATA__*/", data), encoding="utf-8")

    total = sum(len(f["fields"]) for f in forms)
    print(f"\n{OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1048576:.2f} MB)")
    print(f"{len(forms)} forms, {total} boxes to draw:\n")
    for f in forms:
        print(f"  {f['form_id']}")
        for fl in f["fields"]:
            kind = "region" if fl["id"].startswith("__region__") else f"pattern {fl['pattern']}"
            print(f"      {fl['id']:<32} {kind}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
