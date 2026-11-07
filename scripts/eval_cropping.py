"""End-to-end evaluation of stages 0-3, under progressively worse scans.

The question this answers: **if the same card were scanned again, badly, would
the crop still land on the value?** With one page per form that cannot be tested
directly, so we manufacture the second scan — warp by a homography we chose,
degrade it, and push it back through the whole pipeline.

Reference boxes come from running the pipeline on the clean page. Because
registration warps everything back into canonical space, the boxes from a
perturbed run are directly comparable to them.

The headline metric is **coverage**, not IoU: a crop that misses part of the
reference ink has failed no matter how good its IoU is, while a crop with extra
whitespace costs nothing. IoU is reported only as a tie-breaker on tightness.

Usage:  PYTHONPATH=src python scripts/eval_cropping.py [--trials 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                       # noqa: E402
from aihtr.crop import crop_fields                             # noqa: E402
from aihtr.formid import FormIdentifier                        # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_image, normalize_page  # noqa: E402
from aihtr.register import register, warp                      # noqa: E402
from aihtr.synth import random_homography                      # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"

PAGES = {
    "base-opd-card-v1":  "base-opd-card-v1_001.jpg",
    "nutrition-0190-v1": "nutrition-0190-v1_001.jpg",
    "ms-form-v2_4":      "ms-form-v2_4_p1_001.jpg",
    "neuro-v1":          "neuro-v1_001.jpg",
}

# Progressively nastier re-scans. `rot` degrees, `trans` px, `persp` keystone,
# `blur` kernel, `noise` sigma, `jpeg` quality.
LEVELS = {
    "clean":    dict(rot=0.0, scale=0.00, trans=0,  persp=0.0,     blur=0, noise=0,  jpeg=0),
    "mild":     dict(rot=1.5, scale=0.03, trans=25, persp=4e-5,    blur=3, noise=4,  jpeg=75),
    "moderate": dict(rot=3.0, scale=0.06, trans=50, persp=9e-5,    blur=5, noise=8,  jpeg=55),
    "heavy":    dict(rot=6.0, scale=0.10, trans=90, persp=1.6e-4,  blur=7, noise=14, jpeg=35),
}


def degrade(img, rng, p):
    if p["blur"] >= 3:
        img = cv2.GaussianBlur(img, (p["blur"], p["blur"]), 0)
    if p["noise"]:
        img = np.clip(img.astype(np.int16) + rng.normal(0, p["noise"], img.shape),
                      0, 255).astype(np.uint8)
    if p["jpeg"]:
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return img


def coverage(ref, test) -> float:
    """Fraction of the reference box the test box actually contains."""
    iw = max(0, min(ref[2], test[2]) - max(ref[0], test[0]))
    ih = max(0, min(ref[3], test[3]) - max(ref[1], test[1]))
    area = max(1, (ref[2] - ref[0]) * (ref[3] - ref[1]))
    return iw * ih / area


def iou(a, b) -> float:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def boxes_for(page, spec, tm):
    return {fc.field_id: (fc.tight_box, fc.flags)
            for fc in crop_fields(page, spec, tm, do_refine=True)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    forms = schema.load_all(ROOT / "configs" / "forms")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    if not tm:
        print("no templates; run scripts/build_templates.py")
        return 1
    ident = FormIdentifier({k: v for k, v in tm.items() if k in forms})
    rng = np.random.default_rng(args.seed)

    # Reference: the clean run.
    refs, clean_pages = {}, {}
    for fid, pf in PAGES.items():
        p = normalize_page(RAW / pf).image
        clean_pages[fid] = p
        refs[fid] = boxes_for(p, forms[fid], tm)

    n_fields = sum(len(v) for v in refs.values())
    print(f"{n_fields} fields across {len(PAGES)} forms, {args.trials} trials per level\n")
    print(f"{'level':<10} {'formID':>8} {'reg':>6} {'fields':>8} "
          f"{'cover<1':>8} {'meanCov':>8} {'minCov':>7} {'meanIoU':>8} {'ctr px':>7}")
    print("-" * 82)

    for level, p in LEVELS.items():
        id_ok = reg_ok = trials = 0
        covs, ious, offs, misses, lost = [], [], [], 0, 0

        for fid, pf in PAGES.items():
            for _ in range(args.trials):
                trials += 1
                base = clean_pages[fid]
                if p["rot"] or p["trans"]:
                    H = random_homography(CANONICAL_SIZE, rng, rot=p["rot"],
                                          scale=p["scale"], trans=p["trans"],
                                          persp=p["persp"])
                    img = warp(base, H, CANONICAL_SIZE)
                else:
                    img = base.copy()
                img = degrade(img, rng, p)

                page = normalize_image(img).image
                res = ident.identify(page)
                if res.form_id != fid:
                    continue
                id_ok += 1

                reg = register(page, tm[fid])
                if not reg.ok:
                    continue
                reg_ok += 1
                page = warp(page, reg.homography, CANONICAL_SIZE)

                got = boxes_for(page, forms[fid], tm)
                for field_id, (ref_box, _) in refs[fid].items():
                    if field_id not in got:
                        lost += 1
                        continue
                    tb = got[field_id][0]
                    c = coverage(ref_box, tb)
                    covs.append(c)
                    ious.append(iou(ref_box, tb))
                    offs.append(abs((tb[0]+tb[2])/2 - (ref_box[0]+ref_box[2])/2)
                                + abs((tb[1]+tb[3])/2 - (ref_box[1]+ref_box[3])/2))
                    misses += c < 0.999

        if not covs:
            print(f"{level:<10} {'ALL FAILED':>8}")
            continue
        print(f"{level:<10} {id_ok}/{trials:<6} {reg_ok}/{trials:<4} "
              f"{len(covs):>8} {misses:>8} {np.mean(covs):>8.3f} {min(covs):>7.3f} "
              f"{np.mean(ious):>8.3f} {np.mean(offs):>7.1f}")
        if lost:
            print(f"{'':<10} {lost} field(s) not produced at all")

    print("\ncoverage = fraction of the clean-run box the perturbed run still contains.")
    print("1.000 means nothing was clipped. This is the metric that matters;")
    print("IoU is only a tightness tie-breaker.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
