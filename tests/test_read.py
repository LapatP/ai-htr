"""Stage 4 contract tests.

These assert *safety properties*, not accuracy. Accuracy cannot be tested — the
corpus contains exactly one filled BP field, so any hit rate computed from it
would be a number about a single sample.

What can be tested is that the reader never invents: that it abstains rather
than guessing a separator, refuses glyphs it cannot match, keeps systolic and
diastolic as separate entities, and reports the confidence of its *worst*
glyph rather than its average.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.digits import SYMBOLS, Glyph, classify, read_glyphs, segment  # noqa: E402
from aihtr.read import read_pattern_a, read_pattern_split                # noqa: E402


def _render(text: str, h: int = 60) -> np.ndarray:
    """A clean synthetic crop — the easiest possible input."""
    import cv2
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
    img = np.full((h, w + 30), 245, np.uint8)
    cv2.putText(img, text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 20, 3, cv2.LINE_AA)
    return img


# --- closed set: the reader cannot emit anything it was not given -----------

def test_classifier_only_ever_returns_a_known_symbol():
    rng = np.random.default_rng(0)
    for _ in range(25):
        noise = (rng.random((30, 20)) > 0.5).astype(np.uint8)
        sym, score, margin = classify(noise)
        assert sym in SYMBOLS or sym == "?"


def test_empty_mask_is_refused_not_guessed():
    sym, score, _ = classify(np.zeros((20, 20), np.uint8))
    assert sym == "?" and score == 0.0


# --- abstention over guessing ----------------------------------------------

def test_pattern_a_refuses_when_no_separator_is_found():
    """Splitting on 'the biggest gap' would be a guess, and a guess here yields
    a plausible wrong BP. It must decline instead."""
    r = read_pattern_a(_render("14390"))
    assert not r.readable
    assert any("separator_not_found" in f for f in r.flags)


def test_pattern_a_on_blank_paper_reports_no_ink():
    r = read_pattern_a(np.full((60, 200), 246, np.uint8))
    assert not r.readable
    assert "no_ink" in r.flags


def test_split_reader_on_blank_paper_reports_no_ink():
    blank = np.full((50, 90), 246, np.uint8)
    r = read_pattern_split(blank, blank)
    assert not r.readable and "no_ink" in r.flags


# --- entity separation ------------------------------------------------------

def test_systolic_and_diastolic_are_separate_integers_never_a_string():
    r = read_pattern_a(_render("143/90"))
    assert not isinstance(r.systolic, str) and not isinstance(r.diastolic, str)
    if r.readable:
        assert isinstance(r.systolic, int) and isinstance(r.diastolic, int)


def test_split_reader_flags_are_attributed_to_the_right_component():
    blank = np.full((50, 90), 246, np.uint8)
    r = read_pattern_split(_render("143"), blank)
    assert all(f.startswith(("systolic:", "diastolic:")) or f == "no_ink" for f in r.flags)


# --- confidence semantics ---------------------------------------------------

def test_confidence_is_the_weakest_glyph_not_the_average():
    """One confident digit must not be able to mask an unreadable neighbour."""
    from aihtr.read import _confidence
    glyphs = [Glyph("1", 0.95, 0.3, [0, 0, 1, 1]), Glyph("4", 0.31, 0.01, [0, 0, 1, 1])]
    assert _confidence(glyphs) == pytest.approx(0.31)


def test_a_reading_is_never_marked_readable_without_both_values():
    from aihtr.read import BPReading
    assert not BPReading(systolic=143).readable
    assert not BPReading(diastolic=90).readable
    assert BPReading(systolic=143, diastolic=90).readable


# --- segmentation -----------------------------------------------------------

def test_segmentation_returns_boxes_left_to_right():
    boxes = segment(_render("143/90"))
    assert boxes == sorted(boxes, key=lambda b: b[0])


def test_segmentation_of_blank_paper_is_empty():
    assert segment(np.full((50, 200), 246, np.uint8)) == []


def test_reading_glyphs_of_a_degenerate_crop_does_not_crash():
    assert read_glyphs(np.zeros((0, 0), np.uint8)) == []
