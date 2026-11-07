"""Ink-refinement tests.

Anchored on the real bug: my first `bp` box was [2038, 540, 2226, 660], which
clipped the leading `1` of `143/90` and produced `43/90` with every upstream
diagnostic reporting success. Measured true ink extent: [2016, 546, 2218, 656].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.crop import _slice                     # noqa: E402
from aihtr.normalize import normalize_page        # noqa: E402
from aihtr.refine import refine_field             # noqa: E402

RAW = ROOT / "data" / "raw"
BASE = RAW / "base-opd-card-v1_001.jpg"

CLIPPED = [2038, 540, 2226, 660]     # the bug
CORRECT = [2008, 538, 2226, 664]
CONTEXT = [1889, 482, 2345, 720]
TRUE_INK = [2016, 546, 2218, 656]    # measured by connected components


@pytest.fixture(scope="module")
def page():
    if not BASE.exists():
        pytest.skip("corpus not present")
    return normalize_page(BASE).image


def _refine(page, tight, context=CONTEXT):
    return refine_field(_slice(page, context), context, tight)


# --- the headline behaviour -------------------------------------------------

def test_recovers_the_clipped_leading_digit(page):
    r = _refine(page, CLIPPED)
    assert r.box[0] <= TRUE_INK[0], "left edge must reach the '1' at x=2016"
    assert r.box[0] < CLIPPED[0], "box should have grown leftwards"


def test_refined_box_matches_the_measured_ink_extent(page):
    r = _refine(page, CLIPPED)
    for i, (got, want) in enumerate(zip(r.box, TRUE_INK)):
        assert abs(got - want) <= 8, f"edge {i}: {got} vs measured {want}"


def test_both_starting_boxes_end_up_containing_the_true_ink(page):
    """Refinement used to be fully ink-driven, so a clipped box and a correct
    one converged on the same rectangle. That property was deliberately traded
    away: it made cropping unstable on a degraded scan, where the box could
    collapse onto part of the number (min coverage 0.230 across 18 perturbed
    crops, against 0.973 with no refinement at all).

    Refinement is now grow-only, so the result depends on the nominal box. What
    must still hold is that either starting point ends up covering the ink."""
    for start in (CLIPPED, CORRECT):
        b = _refine(page, start).box
        assert b[0] <= TRUE_INK[0] and b[1] <= TRUE_INK[1]
        assert b[2] >= TRUE_INK[2] and b[3] >= TRUE_INK[3]


def test_refinement_never_shrinks_the_nominal_box(page):
    """The nominal box is a human's statement of where the field is. Ink may
    extend it; ink may not overrule it. This is what keeps coverage at 1.0 by
    construction under a bad scan."""
    for start in (CLIPPED, CORRECT, [1950, 500, 2300, 700]):
        b = _refine(page, start).box
        assert b[0] <= start[0] and b[1] <= start[1]
        assert b[2] >= start[2] and b[3] >= start[3]


def test_reports_that_it_adjusted_the_box(page):
    assert "box_adjusted" in _refine(page, CLIPPED).flags


# --- flags ------------------------------------------------------------------

def test_possibly_clipped_fires_when_context_is_too_small(page):
    """Ink running to the edge of its context means the true extent is unknown.
    That is a review case — refinement must not pretend it found the end."""
    tiny = [2040, 560, 2180, 640]
    r = refine_field(_slice(page, tiny), tiny, [2060, 570, 2160, 630])
    assert "possibly_clipped" in r.flags


def test_no_clip_flag_when_ink_sits_clear_of_the_edges(page):
    assert "possibly_clipped" not in _refine(page, CLIPPED).flags


def test_empty_field_is_reported_not_guessed(page):
    """An unfilled field must say so. `field_empty` and `unreadable` are
    clinically different facts and must not collapse together."""
    npage = normalize_page(RAW / "nutrition-0190-v1_001.jpg").image
    ctx, tight = [939, 1439, 1154, 1538], [995, 1462, 1098, 1515]
    r = refine_field(_slice(npage, ctx), ctx, tight)
    assert "field_empty" in r.flags
    assert r.box == tight, "an empty field's box must be left alone"


# --- robustness -------------------------------------------------------------

def test_uniform_input_is_empty_not_a_page_full_of_ink():
    """Otsu on near-uniform paper splits noise half-and-half. The contrast guard
    is what stops a blank field reporting itself as solid ink."""
    blank = np.full((80, 200), 244, np.uint8)
    r = refine_field(blank, [0, 0, 200, 80], [40, 20, 160, 60])
    assert "field_empty" in r.flags
    assert r.ink_fraction < 0.01


def test_refined_box_never_escapes_its_context(page):
    r = _refine(page, CLIPPED)
    assert r.box[0] >= CONTEXT[0] and r.box[1] >= CONTEXT[1]
    assert r.box[2] <= CONTEXT[2] and r.box[3] <= CONTEXT[3]


def test_degenerate_input_does_not_crash():
    r = refine_field(np.zeros((0, 0), np.uint8), [0, 0, 0, 0], [0, 0, 0, 0])
    assert "field_empty" in r.flags


def test_growth_is_reported_per_side(page):
    r = _refine(page, CLIPPED)
    left, top, right, bottom = r.growth
    assert left > 0, "the clipped side must report positive growth"
