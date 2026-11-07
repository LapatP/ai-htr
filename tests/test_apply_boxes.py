"""Tests for the box-application maths.

`rebase` is the subtle one: boxes drawn on the base card have to become
sticker-relative, because the sticker is glued on by hand and its position
varies from page to page. Getting the sign wrong here would put every triage
crop in the wrong place, and registration would still report a clean fit.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("apply_boxes", ROOT / "scripts" / "apply_boxes.py")
apply_boxes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(apply_boxes)


def test_rebase_makes_a_box_relative_to_the_sticker_origin():
    # sticker sits at (270, 1000) on the base card
    assert apply_boxes.rebase([430, 1560, 560, 1620], (270, 1000)) == [160, 560, 290, 620]


def test_rebase_of_the_origin_itself_is_the_zero_corner():
    assert apply_boxes.rebase([270, 1000, 400, 1100], (270, 1000)) == [0, 0, 130, 100]


def test_rebase_preserves_size():
    box, origin = [430, 1560, 560, 1620], (270, 1000)
    out = apply_boxes.rebase(box, origin)
    assert (out[2] - out[0], out[3] - out[1]) == (box[2] - box[0], box[3] - box[1])


def test_grow_expands_and_never_shrinks():
    box = [1000, 1000, 1200, 1100]
    out = apply_boxes.grow(box, 2480, 3508)
    assert out[0] < box[0] and out[1] < box[1]
    assert out[2] > box[2] and out[3] > box[3]


def test_grow_clamps_to_the_page_rather_than_going_negative():
    out = apply_boxes.grow([5, 5, 60, 40], 2480, 3508)
    assert out[0] == 0 and out[1] == 0

    out = apply_boxes.grow([2400, 3450, 2470, 3500], 2480, 3508)
    assert out[2] == 2480 and out[3] == 3508


def test_boxes_are_written_flow_style():
    """Block-style lists let ruamel relocate neighbouring comments into the
    middle of the value. Flow style keeps the value one node."""
    s = apply_boxes.seq([1, 2, 3, 4])
    assert list(s) == [1, 2, 3, 4]
    assert s.fa.flow_style() is True


@pytest.mark.parametrize("vals", [(1.0, 2.9, 3.2, 4.8), ("10", "20", "30", "40")])
def test_seq_coerces_to_int(vals):
    assert all(isinstance(v, int) for v in apply_boxes.seq(vals))
