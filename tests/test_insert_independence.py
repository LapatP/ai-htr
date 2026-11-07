"""The insert must be able to move without dragging the card with it.

This is the whole reason inserts exist as a concept, and it went untested for a
long time: every earlier evaluation perturbed the *whole page*, which moves the
sticker together with the form and therefore never exercised the case.

When it was finally tested, it failed. The base card's template contains the
triage sticker, the sticker carries 89% of the matched inliers, and RANSAC
consequently aligned the page to the sticker — so moving the sticker 60 px
dragged the card's own `bp` field off its value (similarity 1.000 -> 0.018).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                   # noqa: E402
from aihtr.crop import crop_fields, mask_inserts           # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_page  # noqa: E402
from aihtr.register import register, warp                  # noqa: E402

W, H = CANONICAL_SIZE
RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"
FORM = "base-opd-card-v1"
REGION = [252, 962, 1342, 2118]
MOVES = [(0, 0, 0), (60, -40, 0), (-90, 110, 0), (120, -80, -6)]


@pytest.fixture(scope="module")
def env():
    if not (RAW / f"{FORM}_001.jpg").exists() or not (TEMPLATES / f"{FORM}.png").exists():
        pytest.skip("corpus or templates not built")
    forms = schema.load_all(ROOT / "configs" / "forms")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    clean = normalize_page(RAW / f"{FORM}_001.jpg").image
    return forms, tm, clean


def _repaste(clean, dx, dy, ang):
    """Lift the sticker off and glue it back on somewhere else."""
    sticker = clean[REGION[1]:REGION[3], REGION[0]:REGION[2]].copy()
    sh, sw = sticker.shape
    page = clean.copy()
    page[REGION[1]:REGION[3], REGION[0]:REGION[2]] = 255
    M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
    M[0, 2] += REGION[0] + dx
    M[1, 2] += REGION[1] + dy
    canvas = np.full_like(page, 255)
    cv2.warpAffine(sticker, M, (W, H), dst=canvas, flags=cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return np.minimum(page, canvas)


def _run(forms, tm, page):
    spec = forms[FORM]
    reg = register(page, mask_inserts(tm[FORM], spec))
    assert reg.ok, f"registration failed: {reg}"
    warped = warp(page, reg.homography, CANONICAL_SIZE)
    return {c.field_id: c for c in crop_fields(warped, spec, tm, do_refine=True)}, warped


def test_insert_source_region_is_recorded(env):
    """mask_inserts() cannot work without it, and its absence fails silently."""
    forms, _, _ = env
    for ins in forms[FORM].inserts:
        assert ins.source_region is not None, f"{ins.id} has no source_region"


def test_masking_actually_removes_the_insert_area(env):
    forms, tm, _ = env
    masked = mask_inserts(tm[FORM], forms[FORM])
    patch = masked[REGION[1]:REGION[3], REGION[0]:REGION[2]]
    assert patch.min() == 255, "insert region was not blanked"
    assert not np.array_equal(masked, tm[FORM])


@pytest.mark.parametrize("dx,dy,ang", MOVES)
def test_moving_the_sticker_leaves_the_cards_own_field_alone(env, dx, dy, ang):
    forms, tm, clean = env
    ref = clean[538:664, 2008:2226]
    crops, warped = _run(forms, tm, _repaste(clean, dx, dy, ang))
    b = crops["bp"].tight_box
    got = cv2.resize(warped[b[1]:b[3], b[0]:b[2]], (ref.shape[1], ref.shape[0]))
    sim = float(cv2.matchTemplate(got, ref, cv2.TM_CCOEFF_NORMED)[0, 0])
    assert sim > 0.9, f"card's bp field moved with the sticker (similarity {sim:.3f})"


@pytest.mark.parametrize("dx,dy,ang", MOVES[1:])
def test_the_inserts_own_fields_follow_the_insert(env, dx, dy, ang):
    """The complement: the card holds still, the sticker's fields must not."""
    forms, tm, clean = env
    base, _ = _run(forms, tm, _repaste(clean, 0, 0, 0))
    moved, _ = _run(forms, tm, _repaste(clean, dx, dy, ang))
    assert "bp.systolic" in moved, "insert was not located after being moved"
    shift = abs(moved["bp.systolic"].tight_box[0] - base["bp.systolic"].tight_box[0]) \
        + abs(moved["bp.systolic"].tight_box[1] - base["bp.systolic"].tight_box[1])
    assert shift > 20, f"insert field did not track the insert (moved {shift} px)"


def test_a_card_with_no_sticker_at_all_is_still_identified(env):
    """Not every OPD card gets a triage sheet. With the sticker left in the
    template such a page scored a 2.66x margin — under the 6.5x gate — and came
    back UNKNOWN. Masking lifts it to ~24x."""
    from aihtr.formid import FormIdentifier
    forms, tm, clean = env
    bare = clean.copy()
    bare[REGION[1]:REGION[3], REGION[0]:REGION[2]] = 255

    unmasked = FormIdentifier({k: v for k, v in tm.items() if k in forms})
    assert unmasked.identify(bare).form_id is None, "premise changed: this used to fail"

    masked = FormIdentifier({k: mask_inserts(v, forms[k])
                             for k, v in tm.items() if k in forms})
    assert masked.identify(bare).form_id == FORM


def test_masking_keeps_a_stickered_page_identifiable_too(env):
    """The fix must not cost accuracy on ordinary pages."""
    from aihtr.formid import FormIdentifier
    forms, tm, clean = env
    masked = FormIdentifier({k: mask_inserts(v, forms[k])
                             for k, v in tm.items() if k in forms})
    assert masked.identify(clean).form_id == FORM


@pytest.mark.parametrize("dx,dy,expect_occluded", [
    (0, 0, False),            # sticker where it belongs
    (1448, -662, True),       # sticker moved on top of the card's BP field
])
def test_a_field_covered_by_the_sticker_is_flagged(env, dx, dy, expect_occluded):
    """A field under an insert is physically covered — cropping its coordinates
    returns the sticker's content. Before this check the only symptom was the
    reader struggling (similarity 0.672, 'unreadable', zero flags). It failed
    safe by luck; had the covering area held digits it would have produced a
    confident wrong BP."""
    forms, tm, clean = env
    crops, _ = _run(forms, tm, _repaste(clean, dx, dy, 0))
    occluded = any(f.startswith("occluded_by") for f in crops["bp"].flags)
    assert occluded is expect_occluded, f"flags were {crops['bp'].flags}"
    if expect_occluded:
        assert crops["bp"].needs_review
