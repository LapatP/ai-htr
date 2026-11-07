"""A fast regression version of scripts/eval_cropping.py.

The full sweep runs four degradation levels with many trials and takes minutes;
this pins the property at one moderate level with one trial per form so the
suite stays quick. If cropping starts clipping under a plausible re-scan, this
fails.

Coverage, not IoU, is the assertion. A crop that loses part of the reference ink
has failed regardless of overlap score; a crop with extra whitespace has not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                                  # noqa: E402
from aihtr.crop import crop_fields                                        # noqa: E402
from aihtr.formid import FormIdentifier                                   # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_image, normalize_page  # noqa: E402
from aihtr.register import register, warp                                 # noqa: E402
from aihtr.synth import random_homography                                 # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"
FORM, PAGE = "base-opd-card-v1", "base-opd-card-v1_001.jpg"


def _coverage(ref, test):
    iw = max(0, min(ref[2], test[2]) - max(ref[0], test[0]))
    ih = max(0, min(ref[3], test[3]) - max(ref[1], test[1]))
    return iw * ih / max(1, (ref[2] - ref[0]) * (ref[3] - ref[1]))


@pytest.fixture(scope="module")
def env():
    if not (RAW / PAGE).exists() or not (TEMPLATES / f"{FORM}.png").exists():
        pytest.skip("corpus or templates not built")
    forms = schema.load_all(ROOT / "configs" / "forms")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    return forms, tm


def _boxes(page, spec, tm):
    return {fc.field_id: fc.tight_box
            for fc in crop_fields(page, spec, tm, do_refine=True)}


def test_crops_survive_a_moderately_bad_rescan(env):
    forms, tm = env
    spec = forms[FORM]
    clean = normalize_page(RAW / PAGE).image
    ref = _boxes(clean, spec, tm)
    assert ref, "no fields cropped on the clean page"

    rng = np.random.default_rng(3)
    H = random_homography(CANONICAL_SIZE, rng, rot=3.0, scale=0.06, trans=50, persp=9e-5)
    img = cv2.GaussianBlur(warp(clean, H, CANONICAL_SIZE), (5, 5), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 55])
    img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    page = normalize_image(img).image
    ident = FormIdentifier({k: v for k, v in tm.items() if k in forms})
    assert ident.identify(page).form_id == FORM, "form lost under moderate degradation"

    reg = register(page, tm[FORM])
    assert reg.ok, f"registration failed: {reg}"
    page = warp(page, reg.homography, CANONICAL_SIZE)

    got = _boxes(page, spec, tm)
    for fid, ref_box in ref.items():
        assert fid in got, f"{fid} was not produced after degradation"
        c = _coverage(ref_box, got[fid])
        assert c > 0.80, f"{fid}: only {c:.1%} of the reference box was covered"


def test_insert_fields_still_resolve_after_perturbation(env):
    """The sticker is located at runtime by registering its template into the
    page. If that breaks under rotation, its fields silently disappear."""
    forms, tm = env
    clean = normalize_page(RAW / PAGE).image
    rng = np.random.default_rng(9)
    H = random_homography(CANONICAL_SIZE, rng, rot=3.0, scale=0.05, trans=40, persp=8e-5)
    page = normalize_image(warp(clean, H, CANONICAL_SIZE)).image
    reg = register(page, tm[FORM])
    assert reg.ok
    page = warp(page, reg.homography, CANONICAL_SIZE)

    crops = crop_fields(page, forms[FORM], tm, do_refine=True)
    projected = [c for c in crops if c.projected]
    assert len(projected) == 2, f"expected 2 insert fields, got {len(projected)}"
