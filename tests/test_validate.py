"""Clinical plausibility tests.

The headline case is `43/90` — the exact value my first crop produced when it
clipped the leading digit of `143/90`. Every upstream diagnostic reported
success; this check is the only thing in the pipeline that catches it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aihtr.validate import Verdict, cross_check, validate_bp  # noqa: E402


# --- the invariant that matters --------------------------------------------

def test_the_clipped_digit_case_is_rejected():
    """143/90 misread as 43/90 after a clipped leading digit."""
    v = validate_bp(43, 90)
    assert v.verdict is Verdict.IMPOSSIBLE
    assert any("must exceed diastolic" in r for r in v.reasons)


def test_the_clipped_digit_case_hints_at_the_real_cause():
    v = validate_bp(43, 90)
    assert any("143/90" in h and "re-crop" in h for h in v.hints)


def test_equal_values_are_impossible_not_merely_odd():
    assert validate_bp(90, 90).verdict is Verdict.IMPOSSIBLE


@pytest.mark.parametrize("s,d", [(143, 90), (120, 80), (98, 62), (165, 95), (185, 110)])
def test_ordinary_readings_pass(s, d):
    assert validate_bp(s, d).verdict is not Verdict.IMPOSSIBLE


# --- digit-count damage -----------------------------------------------------

def test_single_digit_is_rejected_as_clipped():
    v = validate_bp(4, 90)
    assert v.verdict is Verdict.IMPOSSIBLE
    assert any("clipped" in r for r in v.reasons)


def test_four_digits_are_rejected_as_merged():
    v = validate_bp(1430, 90)
    assert v.verdict is Verdict.IMPOSSIBLE
    assert any("merged" in r for r in v.reasons)


# --- impossible vs merely unusual ------------------------------------------

def test_extremes_are_flagged_but_not_rejected():
    """A hypertensive crisis is real. Flag it; never throw it away."""
    v = validate_bp(230, 125)
    assert v.verdict is Verdict.IMPLAUSIBLE
    assert v.needs_review


def test_narrow_pulse_pressure_is_flagged():
    v = validate_bp(100, 88)
    assert v.verdict is Verdict.IMPLAUSIBLE
    assert any("narrow" in r for r in v.reasons)


def test_wide_pulse_pressure_is_flagged():
    v = validate_bp(200, 60)
    assert any("wide" in r for r in v.reasons)


# --- severity weighting -----------------------------------------------------

def test_readings_near_a_decision_threshold_are_surfaced():
    """142/91 is plausible, but one misread digit flips a hypertension
    diagnosis — so it still deserves eyes."""
    v = validate_bp(142, 91)
    assert v.verdict is Verdict.PLAUSIBLE
    assert "hypertension diagnosis" in v.near_thresholds
    assert v.needs_review


def test_a_reading_far_from_any_threshold_needs_no_review():
    v = validate_bp(118, 72)
    assert v.verdict is Verdict.PLAUSIBLE
    assert not v.needs_review


# --- never silently correct -------------------------------------------------

def test_validation_never_mutates_the_reading():
    """Hints describe; they must not repair. A guardrail that quietly rewrites
    values is worse than no guardrail."""
    v = validate_bp(43, 90)
    assert v.hints and all(isinstance(h, str) for h in v.hints)
    # nothing in the API returns a corrected value
    assert not hasattr(v, "corrected")
    assert not hasattr(v, "systolic")


# --- cross-page redundancy --------------------------------------------------

def test_agreeing_readings_on_one_page_pass():
    v = cross_check({"bp": (143, 90), "triage.bp": (143, 90)})
    assert v.verdict is Verdict.PLAUSIBLE


def test_disagreeing_readings_on_one_page_are_rejected():
    """The base card and the triage sticker record the same visit. If they
    disagree, one of them is wrong — detectable with no reference data."""
    v = cross_check({"bp": (143, 90), "triage.bp": (148, 90)})
    assert v.verdict is Verdict.IMPOSSIBLE
    assert any("disagree" in r for r in v.reasons)


def test_single_reading_cannot_be_cross_checked():
    assert cross_check({"bp": (143, 90)}).verdict is Verdict.PLAUSIBLE


def test_a_missing_second_source_is_not_treated_as_agreement():
    """The disarmed-redundancy bug: a page carrying two BP sources where one
    read and the other came back empty used to score exactly like a page where
    both agreed. A wrong number was caught; a missing one was not."""
    v = cross_check({"bp": (143, 90)}, expected=2)
    assert v.verdict is not Verdict.PLAUSIBLE
    assert any("1 of 2 BP sources" in r for r in v.reasons)


def test_both_sources_present_still_cross_checks_normally():
    agree = cross_check({"bp": (143, 90), "triage.bp": (143, 90)}, expected=2)
    assert agree.verdict is Verdict.PLAUSIBLE
    disagree = cross_check({"bp": (143, 90), "triage.bp": (148, 90)}, expected=2)
    assert disagree.verdict is Verdict.IMPOSSIBLE
