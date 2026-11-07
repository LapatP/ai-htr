"""Contact sheet showing what tight vs context cropping actually produces.

One row per field:
    left  — the context crop, with the tight box outlined inside it
    right — the tight crop on its own, at the same scale

With --refine, the nominal box is drawn in grey and the ink-refined box in
blue, so you can see where the rectangle was wrong and by how much.

Usage:  PYTHONPATH=src python scripts/preview_crops.py [--refine]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                              # noqa: E402
from aihtr.crop import crop_fields                    # noqa: E402
from aihtr.normalize import normalize_page            # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"
OUT = ROOT / "data" / "interim"

ROW_H = 150          # px per row in the sheet
SHEET_W = 1500
PAGES = {
    "base-opd-card-v1":  "base-opd-card-v1_001.jpg",
    "nutrition-0190-v1": "nutrition-0190-v1_001.jpg",
    "ms-form-v2_4":      "ms-form-v2_4_p1_001.jpg",
    "neuro-v1":          "neuro-v1_001.jpg",
}
ACCENT, INK, GREY = (163, 95, 46), (38, 33, 23), (160, 160, 160)
NOMINAL = (190, 190, 190)


def fit(img, h):
    """Scale to a target height, capped so tiny crops don't explode."""
    ih, iw = img.shape[:2]
    s = min(h / max(ih, 1), 6.0)
    return cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))),
                      interpolation=cv2.INTER_CUBIC), s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()
    specs = schema.load_all(ROOT / "configs" / "forms")
    inserts = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
               for p in TEMPLATES.glob("*.png")}
    rows = []

    for form_id, page_file in PAGES.items():
        spec = specs.get(form_id)
        src = RAW / page_file
        if not spec or not src.exists():
            continue
        page = normalize_page(src).image
        for fc in crop_fields(page, spec, inserts, do_refine=args.refine):
            rows.append((form_id, fc))

    if not rows:
        print("nothing to preview")
        return 1

    band = ROW_H + 34
    sheet = np.full((band * len(rows) + 40, SHEET_W, 3), 252, np.uint8)

    for i, (form_id, fc) in enumerate(rows):
        y = 34 + i * band
        ctx = cv2.cvtColor(fc.context, cv2.COLOR_GRAY2BGR)
        cx0, cy0 = fc.context_box[0], fc.context_box[1]
        # the nominal box first, in grey, so any correction is visible
        if fc.refinement and fc.refinement.grew:
            n = fc.refinement.original
            cv2.rectangle(ctx, (n[0] - cx0, n[1] - cy0),
                          (n[2] - cx0, n[3] - cy0), NOMINAL, 1)
        b = fc.tight_box
        cv2.rectangle(ctx, (b[0] - cx0, b[1] - cy0), (b[2] - cx0, b[3] - cy0), ACCENT, 2)

        ctx_s, _ = fit(ctx, ROW_H)
        tgt_s, _ = fit(cv2.cvtColor(fc.tight, cv2.COLOR_GRAY2BGR), ROW_H)

        label = f"{form_id} / {fc.field_id}"
        if fc.projected:
            label += "  [projected via insert]"
        if fc.flags:
            label += "   flags: " + ", ".join(fc.flags)
        cv2.putText(sheet, label, (14, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.46, INK, 1, cv2.LINE_AA)

        x = 14
        for img, cap in ((ctx_s, "context"), (tgt_s, "tight")):
            hh, ww = img.shape[:2]
            if x + ww > SHEET_W - 14:
                break
            sheet[y:y + hh, x:x + ww] = img
            cv2.rectangle(sheet, (x - 1, y - 1), (x + ww, y + hh), GREY, 1)
            cv2.putText(sheet, cap, (x, y + hh + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, GREY, 1, cv2.LINE_AA)
            x += ww + 46

        w0 = fc.tight_box[2] - fc.tight_box[0]
        h0 = fc.tight_box[3] - fc.tight_box[1]
        cv2.putText(sheet, f"{w0}x{h0}px", (SHEET_W - 150, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1, cv2.LINE_AA)

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / "crops_contact_sheet.png"
    cv2.imwrite(str(dst), sheet)
    print(f"{len(rows)} fields cropped -> {dst.relative_to(ROOT)}")
    for form_id, fc in rows:
        print(f"  {fc.field_id:<26} tight {fc.tight.shape[1]}x{fc.tight.shape[0]}"
              f"   context {fc.context.shape[1]}x{fc.context.shape[0]}"
              f"{'   (projected)' if fc.projected else ''}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
