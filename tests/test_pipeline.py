"""End-to-end pipeline tests.

The property under test is that **no wrong value escapes silently**. The
baseline reader is weak; that is acceptable only because every error it makes is
caught downstream and routed to review. If a test here ever shows a wrong
reading passing as trustworthy, the safety design is broken — which matters far
more than the reader's hit rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                          # noqa: E402
from aihtr.pipeline import _role, process_page    # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"


@pytest.fixture(scope="module")
def env():
    if not list(TEMPLATES.glob("*.png")) or not RAW.exists():
        pytest.skip("corpus or templates not built")
    forms = schema.load_all(ROOT / "configs" / "forms")
    tm = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in TEMPLATES.glob("*.png")}
    return forms, tm


# --- field grouping ---------------------------------------------------------

@pytest.mark.parametrize("fid,group,role", [
    ("bp", "bp", "single"),
    ("bp.systolic", "bp", "systolic"),
    ("bp.diastolic", "bp", "diastolic"),
    ("screening.bp.systolic", "screening.bp", "systolic"),
    ("risk.home_sbp", "risk.home", "systolic"),
    ("risk.home_dbp", "risk.home", "diastolic"),
])
def test_field_ids_split_into_group_and_role(fid, group, role):
    assert _role(fid) == (group, role)


def test_base_card_yields_two_independent_bp_groups(env):
    """The top-level `bp` and the sticker's `bp.systolic` both reduce to the
    group "bp". Without qualifying by owner they merge and the sticker reading
    is silently lost — which is exactly what happened first time."""
    forms, tm = env
    r = process_page(RAW / "base-opd-card-v1_001.jpg", forms, tm)
    assert r.form_id == "base-opd-card-v1"
    assert len(r.fields) == 2, f"expected 2 BP groups, got {list(r.fields)}"
    assert any("triage" in g for g in r.fields)


# --- the safety property ----------------------------------------------------

def test_no_wrong_reading_is_ever_accepted_without_review(env):
    """Ground truth on this page is 143/90 in both places. Anything else that
    comes back must be flagged — passing silently is the failure mode."""
    forms, tm = env
    r = process_page(RAW / "base-opd-card-v1_001.jpg", forms, tm)
    for group, fr in r.fields.items():
        if fr.reading.readable and (fr.reading.systolic, fr.reading.diastolic) != (143, 90):
            assert fr.needs_review, f"{group} read {fr.reading.text} and was NOT flagged"


def test_blank_forms_report_empty_rather_than_inventing_a_reading(env):
    """Running a classifier over blank paper manufactures glyphs from speckle.
    The empty check must come first."""
    forms, tm = env
    for page in ("nutrition-0190-v1_001.jpg", "ms-form-v2_4_p1_001.jpg", "neuro-v1_001.jpg"):
        r = process_page(RAW / page, forms, tm)
        for group, fr in r.fields.items():
            assert not fr.reading.readable, f"{page}/{group} invented {fr.reading.text}"
            assert "field_empty" in fr.reading.flags


def test_cross_check_fires_when_the_two_page_readings_disagree(env):
    forms, tm = env
    r = process_page(RAW / "base-opd-card-v1_001.jpg", forms, tm)
    readable = [f for f in r.fields.values() if f.reading.readable]
    if len(readable) >= 2:
        vals = {(f.reading.systolic, f.reading.diastolic) for f in readable}
        if len(vals) > 1:
            assert r.page_check is not None
            assert r.page_check.verdict.value == "impossible"


def test_an_unreadable_page_still_returns_a_result_not_an_exception(env):
    forms, tm = env
    r = process_page(RAW / "chest-v1_001.jpg", forms, tm)
    assert r.form_id is None          # no template for the chest form
    assert r.needs_review and r.reason
