"""Generate synthetic OPD pages with known ground truth.

Renders plausible BP values into the configured fields, perturbs and degrades
the page like a bad re-scan, and writes `truth.csv` / `truth.json` alongside so
you can check what the cropper got against what was actually put there.

WHAT THIS DOES AND DOES NOT TEST
--------------------------------
It tests the **plumbing and the robustness**: does form ID hold up, does
registration recover the warp, does the crop land on the value, does refinement
catch an overflowing digit, does the insert get located when the sticker moves.

It does **not** test the domain gap. These pages are rendered from the very
templates the pipeline matches against, so form ID and registration are being
graded on their own answer sheet. Real scans have artefacts nobody thought to
simulate. Treat a good score here as "the machinery works", never as "it works
on real cards".

The handwriting is Hershey script vector fonts, which do not look much like a
person's pen. Good enough to exercise segmentation and cropping; not a
substitute for real handwriting when evaluating a reader.

Usage:
  python scripts/make_synthetic.py                       # 5 per form
  python scripts/make_synthetic.py --per-form 20 --seed 7
  python scripts/make_synthetic.py --out data/synthetic --move-sticker
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                  # noqa: E402
from aihtr.normalize import CANONICAL_SIZE                # noqa: E402
from aihtr.register import warp                           # noqa: E402
from aihtr.synth import random_homography                 # noqa: E402

W, H = CANONICAL_SIZE
TEMPLATES = ROOT / "data" / "templates"

PEN_FONTS = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

LEVELS = {
    "clean":    dict(rot=0.0, scale=0.00, trans=0,  persp=0.0,    blur=0, noise=0,  jpeg=95),
    "mild":     dict(rot=1.5, scale=0.03, trans=25, persp=4e-5,   blur=3, noise=4,  jpeg=80),
    "moderate": dict(rot=3.0, scale=0.06, trans=50, persp=9e-5,   blur=5, noise=8,  jpeg=60),
    "heavy":    dict(rot=5.0, scale=0.09, trans=80, persp=1.4e-4, blur=5, noise=12, jpeg=40),
}


def plausible_bp(rng) -> tuple[int, int]:
    """A BP that a real clinic would actually record."""
    systolic = int(rng.integers(92, 186))
    diastolic = int(rng.integers(54, min(systolic - 18, 112)))
    return systolic, diastolic


def _fit_scale(text, font, target_h, thickness):
    """Font scale that renders `text` at roughly `target_h` pixels tall."""
    lo, hi = 0.2, 12.0
    for _ in range(24):
        mid = (lo + hi) / 2
        (_, h), _ = cv2.getTextSize(text, font, mid, thickness)
        if h < target_h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def render_value(img, box, text, rng, overflow: float = 1.0) -> None:
    """Write `text` into `box`, in a hand-ish font, with a little jitter.

    `overflow` > 1 deliberately renders larger than the box so the value spills
    out of it — that is the case ink refinement exists to catch, and the real
    card does it (`143/90` is written well above its rule).
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    font = int(rng.choice(PEN_FONTS))
    thickness = int(rng.integers(2, 5))
    target_h = bh * float(rng.uniform(0.62, 0.86)) * overflow
    scale = _fit_scale(text, font, target_h, thickness)
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    # sit it on the field, jittered, and let it run over the edges if it wants
    ox = x1 + int((bw - tw) * float(rng.uniform(-0.05, 0.45)))
    oy = y1 + int((bh + th) * float(rng.uniform(0.42, 0.95)))
    ink = int(rng.integers(15, 70))
    cv2.putText(img, text, (ox, oy), font, scale, ink, thickness, cv2.LINE_AA)


def clear_field(img, box, pad: int = 6) -> None:
    """Blank a field before writing into it.

    The base card's template already carries `143/90` in both BP positions, so
    without this the synthetic value lands on top of the real one. It does erase
    the dotted rule underneath, which is a visible artefact of synthesis - not
    something a real card would show.
    """
    x1, y1, x2, y2 = box
    img[max(0, y1 - pad):min(img.shape[0], y2 + pad),
        max(0, x1 - pad):min(img.shape[1], x2 + pad)] = 255


def degrade(img, p, rng):
    if p["blur"] >= 3:
        img = cv2.GaussianBlur(img, (p["blur"], p["blur"]), 0)
    if p["noise"]:
        img = np.clip(img.astype(np.int16) + rng.normal(0, p["noise"], img.shape),
                      0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else img


def move_sticker(page, region, rng):
    """Lift an insert off and glue it back on somewhere else. Returns the (dx, dy)."""
    x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
    patch = page[y1:y2, x1:x2].copy()
    sh, sw = patch.shape
    dx = int(rng.integers(-140, 200))
    dy = int(rng.integers(-160, 220))
    ang = float(rng.uniform(-5, 5))
    page[y1:y2, x1:x2] = 255
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
    M[0, 2] += x1 + dx
    M[1, 2] += y1 + dy
    canvas = np.full_like(page, 255)
    cv2.warpAffine(patch, M, (W, H), dst=canvas, flags=cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return np.minimum(page, canvas), (dx, dy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-form", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--overflow-rate", type=float, default=0.25,
                    help="fraction of fields written larger than their box")
    ap.add_argument("--move-sticker", action="store_true",
                    help="also reposition inserts, to exercise the insert path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    forms = schema.load_all(ROOT / "configs" / "forms")
    args.out.mkdir(parents=True, exist_ok=True)

    truth, n = [], 0
    print(f"{'file':<46} {'form':<20} {'level':<9} values")
    print("-" * 96)

    for form_id, spec in sorted(forms.items()):
        tpl_path = TEMPLATES / f"{form_id}.png"
        if not tpl_path.exists():
            print(f"  (skipping {form_id}: no template)")
            continue
        template = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)

        for i in range(args.per_form):
            page = template.copy()
            level = list(LEVELS)[i % len(LEVELS)]
            shift = (0, 0)

            # insert first: moving it changes where its fields land
            if args.move_sticker:
                for ins in spec.inserts:
                    if ins.source_region is not None:
                        page, shift = move_sticker(page, ins.source_region, rng)

            systolic, diastolic = plausible_bp(rng)
            placed = []

            def place(field, origin=(0, 0)):
                if not field.resolved:
                    return
                b = [field.bbox.x1 + origin[0], field.bbox.y1 + origin[1],
                     field.bbox.x2 + origin[0], field.bbox.y2 + origin[1]]
                if field.bp_pattern == "A":
                    text = f"{systolic}/{diastolic}"
                elif field.id.endswith((".systolic", "_sbp")):
                    text = str(systolic)
                else:
                    text = str(diastolic)
                over = 1.35 if rng.random() < args.overflow_rate else 1.0
                clear_field(page, b)
                render_value(page, b, text, rng, over)
                placed.append((field.id, text, b, over > 1.0))

            for f in spec.fields:
                place(f)
            for ins in spec.inserts:
                r = ins.source_region
                base = (r.x1 + shift[0], r.y1 + shift[1]) if r else (0, 0)
                for f in ins.fields:
                    place(f, base)

            p = LEVELS[level]
            if p["rot"] or p["trans"]:
                Ht = random_homography(CANONICAL_SIZE, rng, rot=p["rot"],
                                       scale=p["scale"], trans=p["trans"], persp=p["persp"])
                page = warp(page, Ht, CANONICAL_SIZE)
            page = degrade(page, p, rng)

            n += 1
            name = f"synth_{n:04d}_{form_id}_{level}.jpg"
            cv2.imwrite(str(args.out / name), page, [cv2.IMWRITE_JPEG_QUALITY, 92])
            for fid, text, b, over in placed:
                truth.append({"file": name, "form": form_id, "level": level,
                              "field": fid, "value": text,
                              "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
                              "overflowed": int(over)})
            vals = "  ".join(f"{fid}={t}" for fid, t, _, _ in placed)
            print(f"{name:<46} {form_id:<20} {level:<9} {vals[:40]}")

    if truth:
        with (args.out / "truth.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(truth[0]))
            w.writeheader()
            w.writerows(truth)
        (args.out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")

    print("-" * 96)
    print(f"{n} page(s), {len(truth)} field(s) with known values -> {args.out}")
    print(f"ground truth: truth.csv / truth.json  (boxes are canonical 2480x3508)")
    print("\nNOTE: rendered from the same templates the pipeline matches against, so")
    print("form ID and registration are graded on their own answer sheet. Good scores")
    print("here mean the machinery works, not that it works on real cards.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
