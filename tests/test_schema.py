import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aihtr import schema  # noqa: E402

FORMS = Path(__file__).resolve().parents[1] / "configs" / "forms"


def test_all_configs_load_and_validate():
    forms = schema.load_all(FORMS)
    assert len(forms) == 4


def test_every_bp_pattern_is_covered():
    """The point of 'BP everywhere': all three encodings must be exercised."""
    patterns = {
        f.bp_pattern
        for spec in schema.load_all(FORMS).values()
        for _, f in spec.all_fields()
        if f.bp_pattern
    }
    assert patterns == {"A", "B", "C"}


def test_five_bp_sites_declared():
    n = sum(
        1
        for spec in schema.load_all(FORMS).values()
        for _, f in spec.all_fields()
        if f.bp_pattern
    )
    assert n == 8  # 5 sites -> 8 fields (B and C each split into two)


def test_bbox_padding_grows_the_box():
    b = schema.BBox(10, 10, 20, 20).pad(5)
    assert (b.x1, b.y1, b.x2, b.y2) == (5, 5, 25, 25)


def test_every_field_has_a_box():
    """All 8 boxes were drawn 2026-08-20. A field losing its box is a regression,
    and a silent one — stage 3 just skips unresolved fields."""
    unresolved = [
        f.id
        for spec in schema.load_all(FORMS).values()
        for _, f in spec.all_fields()
        if not f.resolved
    ]
    assert unresolved == []


def test_every_field_also_has_a_context_box():
    missing = [
        f.id
        for spec in schema.load_all(FORMS).values()
        for _, f in spec.all_fields()
        if f.resolved and f.context_bbox is None
    ]
    assert missing == []
