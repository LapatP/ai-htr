"""Stage 3 tests.

The insert path is what needs guarding. Base-form fields are a slice; insert
fields are projected through a homography found at runtime, and a mistake there
produces a crop of the wrong part of the page while every upstream diagnostic
still reports success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                    # noqa: E402
from aihtr.crop import crop_fields, project                 # noqa: E402
from aihtr.normalize import normalize_page                  # noqa: E402

TEMPLATES = ROOT / "data" / "templates"
RAW = ROOT / "data" / "raw"


def test_project_through_identity_is_a_no_op():
    b = schema.BBox(10, 20, 30, 50)
    assert project(b, np.eye(3, dtype=np.float64)) == [10, 20, 30, 50]


def test_project_applies_a_pure_translation():
    b = schema.BBox(10, 20, 30, 50)
    H = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]], float)
    assert project(b, H) == [110, 220, 130, 250]


def test_project_over_covers_under_rotation():
    """A rotated rectangle is a quadrilateral; we take its axis-aligned bounds.
    Over-covering is the right direction to err -- crops are scored on coverage."""
    b = schema.BBox(0, 0, 100, 100)
    a = np.radians(10)
    H = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    out = project(b, H)
    assert (out[2] - out[0]) > 100 and (out[3] - out[1]) > 100


def _page_and_spec(form_id, raw_name):
    src = RAW / raw_name
    if not src.exists() or not (TEMPLATES / f"{form_id}.png").exists():
        pytest.skip("corpus or templates not built")
    spec = schema.load_all(ROOT / "configs" / "forms").get(form_id)
    if spec is None:
        pytest.skip("config missing")
    return normalize_page(src).image, spec


def test_every_declared_field_yields_a_non_degenerate_crop():
    page, spec = _page_and_spec("base-opd-card-v1", "base-opd-card-v1_001.jpg")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    crops = crop_fields(page, spec, tm)
    if not crops:
        pytest.skip("no boxes drawn yet")
    for fc in crops:
        assert fc.tight.size > 1, fc.field_id
        assert fc.context.size >= fc.tight.size, fc.field_id


def test_insert_fields_are_marked_projected_and_land_on_the_page():
    page, spec = _page_and_spec("base-opd-card-v1", "base-opd-card-v1_001.jpg")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    crops = {fc.field_id: fc for fc in crop_fields(page, spec, tm)}
    if "bp.systolic" not in crops:
        pytest.skip("triage boxes not drawn yet")
    h, w = page.shape[:2]
    for fid in ("bp.systolic", "bp.diastolic"):
        fc = crops[fid]
        assert fc.projected
        x1, y1, x2, y2 = fc.tight_box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_cropped_bp_fields_actually_contain_ink():
    """Guards the failure that slipped past me: a box on the wrong line still
    produces a plausible-looking crop, so assert the filled fields have dark
    pixels and are not blank paper."""
    page, spec = _page_and_spec("base-opd-card-v1", "base-opd-card-v1_001.jpg")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    crops = {fc.field_id: fc for fc in crop_fields(page, spec, tm)}
    if not crops:
        pytest.skip("no boxes drawn yet")
    for fid in ("bp", "bp.systolic", "bp.diastolic"):
        if fid not in crops:
            continue
        frac = (crops[fid].tight < 120).mean()
        assert frac > 0.04, f"{fid}: only {frac:.1%} dark pixels — box may be misplaced"
