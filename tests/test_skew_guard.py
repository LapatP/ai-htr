"""Deskew must decline when the printed rules disagree.

Found while explaining registration. `estimate_skew` takes the median angle of
the long printed rules, which is exact under pure rotation (0/36 errors over
+/-6 degrees) and catastrophic under perspective: keystone stops the rules being
parallel, and the median reported -8.5 degrees on a page tilted -2.5.

That is not cosmetic. normalize_page then rotates by the bogus angle, leaving
the page MORE crooked, and registration lost so many inliers it failed the
minimum: 63 inliers without normalisation, 19 with it.

The guard: trust the median only when the lines agree (inter-quartile spread
<= MAX_SKEW_IQR) and there are enough of them. Declining is safe because a
homography absorbs rotation anyway.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.normalize import (CANONICAL_SIZE, MAX_SKEW_IQR, MIN_SKEW_LINES,  # noqa: E402
                             estimate_skew, normalize_page, rotate)
from aihtr.register import register                                        # noqa: E402

W, H = CANONICAL_SIZE
RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"
# nutrition has plenty of long printed rules, so the guard should not fire on it
FORM, PAGE = "nutrition-0190-v1", "nutrition-0190-v1_001.jpg"


@pytest.fixture(scope="module")
def page():
    if not (RAW / PAGE).exists():
        pytest.skip("corpus not built")
    return normalize_page(RAW / PAGE).image


def _add_perspective(img, angle, persp):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpPerspective(img, np.vstack([M, [persp, persp * 0.2, 1.0]]),
                               (W, H), borderValue=255)


@pytest.mark.parametrize("angle", [-6, -4, -2, -1, 1, 2, 4, 6])
def test_pure_rotation_is_still_measured(page, angle):
    """The guard must not over-trigger: parallel rules are trustworthy."""
    expected = estimate_skew(page) - angle
    assert abs(estimate_skew(rotate(page, angle)) - expected) < 1.0


@pytest.mark.parametrize("angle,persp", [(-2.5, 1.0e-4), (1.8, 1.4e-4), (-0.7, 0.8e-4)])
def test_perspective_is_declined_not_guessed(page, angle, persp):
    """Under keystone the median is meaningless. 0.0 means 'we did not rotate'."""
    assert estimate_skew(_add_perspective(page, angle, persp)) == 0.0


def test_too_few_lines_declines(page):
    """A crop with almost no printed rules cannot support an estimate."""
    assert estimate_skew(page[:200, :400]) == 0.0


def test_guard_constants_are_sane():
    assert MIN_SKEW_LINES >= 10
    assert 0.5 <= MAX_SKEW_IQR <= 3.0


def test_registration_copes_without_any_deskew(page):
    """Why declining is safe: ORB is rotation-invariant and the homography
    absorbs rotation, so a page that was never straightened still registers."""
    tmpl = cv2.imread(str(TEMPLATES / f"{FORM}.png"), cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        pytest.skip("template not built")
    for angle in (2, 6, 10):
        r = register(rotate(page, angle), tmpl)
        assert r.ok, f"{angle} deg: {r}"
        assert r.reproj_error < 2.0
