"""Smoke tests for the batch entrypoint.

Not about accuracy — about the CLI being usable on a folder of scans without
surprises: it must write a manifest, write crops, and report unrecognised pages
rather than guessing at them.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.__main__ import find_images, main   # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"


@pytest.fixture(scope="module")
def _corpus():
    if not list(RAW.glob("*.jpg")) or not list(TEMPLATES.glob("*.png")):
        pytest.skip("corpus or templates not built")


def test_find_images_picks_up_the_corpus(_corpus):
    assert len(find_images(RAW, recursive=False)) >= 12


def test_find_images_ignores_non_images(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "a.png").write_bytes(b"")
    assert [p.name for p in find_images(tmp_path, False)] == ["a.png"]


def test_batch_writes_a_manifest_and_crops(_corpus, tmp_path):
    rc = main([str(RAW), "--out", str(tmp_path), "--limit", "3"])
    assert rc == 0
    man = tmp_path / "manifest.csv"
    assert man.exists()
    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    assert rows, "manifest is empty"
    for col in ("page", "form", "field", "x1", "y1", "x2", "y2", "flags", "needs_review"):
        assert col in rows[0]
    assert list((tmp_path / "crops").rglob("*.tight.png")), "no tight crops written"
    assert list((tmp_path / "crops").rglob("*.context.png")), "no context crops written"


def test_read_flag_adds_reading_and_verdict_columns(_corpus, tmp_path):
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--read"])
    rows = list(csv.DictReader((tmp_path / "manifest.csv").open(encoding="utf-8")))
    assert rows and "reading" in rows[0] and "verdict" in rows[0]


def test_no_crops_flag_writes_manifest_only(_corpus, tmp_path):
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--no-crops"])
    assert (tmp_path / "manifest.csv").exists()
    assert not (tmp_path / "crops").exists()


def test_unrecognised_pages_are_reported_not_cropped(_corpus, tmp_path, capsys):
    """A form with no template must be listed, never cropped against the wrong one."""
    main([str(RAW / "chest-v1_001.jpg"), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "UNKNOWN" in out and "unrecognised" in out
    assert not (tmp_path / "manifest.csv").exists()


def test_empty_folder_exits_nonzero(tmp_path):
    assert main([str(tmp_path), "--out", str(tmp_path / "o")]) == 1


def test_run_writes_a_self_contained_html_report(_corpus, tmp_path):
    """The report is what a person actually opens — a CSV cannot show whether a
    box landed on the right value, but the context crops can."""
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--read"])
    rp = tmp_path / "report.html"
    assert rp.exists()
    h = rp.read_text(encoding="utf-8")
    assert "data:image/png;base64," in h, "crops must be embedded, not linked"
    assert "http://" not in h.replace("http://www.w3.org", ""), "must not link to local files"
    assert 'class="field' in h and "base-opd-card-v1" in h


def test_report_lists_unrecognised_pages_rather_than_hiding_them(_corpus, tmp_path):
    main([str(RAW / "chest-v1_001.jpg"), "--out", str(tmp_path)])
    h = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Unrecognised" in h and "chest-v1_001.jpg" in h


def test_no_report_flag_suppresses_it(_corpus, tmp_path):
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--no-report"])
    assert not (tmp_path / "report.html").exists()


def test_stale_crops_from_a_previous_run_are_reported(_corpus, tmp_path, capsys):
    """report.html and manifest.csv are rewritten wholesale each run, but
    crops/ is additive — a folder from an earlier run survives and no longer
    corresponds to anything in the report. It must be surfaced, not left to be
    discovered."""
    main([str(RAW), "--out", str(tmp_path)])
    first = {d.name for d in (tmp_path / "crops").iterdir() if d.is_dir()}
    assert len(first) >= 2, "need >1 page cropped to create a stale one"

    capsys.readouterr()
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "left over from an earlier run" in out
    assert "--clean" in out


def test_clean_flag_removes_previous_crops(_corpus, tmp_path, capsys):
    main([str(RAW), "--out", str(tmp_path), "--limit", "4"])
    capsys.readouterr()
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--clean"])
    out = capsys.readouterr().out
    assert "cleaned previous crops" in out
    assert "left over from an earlier run" not in out
    dirs = {d.name for d in (tmp_path / "crops").iterdir() if d.is_dir()}
    assert dirs == {"base-opd-card-v1_001"}


def test_report_mentions_stale_folders(_corpus, tmp_path):
    main([str(RAW), "--out", str(tmp_path)])
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path)])
    h = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "left over from a previous run" in h


def test_report_shows_the_raw_input_page(_corpus, tmp_path):
    """A reading is checked against the card, not against a CSV. The source page
    must be embedded beside its crops."""
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--read"])
    h = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'class="source"' in h and "raw input" in h
    assert 'img class="src"' in h
    assert "data:image/jpeg;base64," in h, "page thumbnails should be JPEG, not PNG"


def test_manifest_carries_systolic_and_diastolic_separately(_corpus, tmp_path):
    """The contract is two integers; `143/90` is only a rendering of them. A
    consumer joining on this should never have to split a string apart."""
    import csv as _csv
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--read"])
    rows = list(_csv.DictReader((tmp_path / "manifest.csv").open(encoding="utf-8")))
    assert "systolic" in rows[0] and "diastolic" in rows[0]
    for r in rows:
        if r.get("reading") and "/" in r["reading"]:
            assert r["systolic"] and r["diastolic"]
            assert r["reading"] == f"{r['systolic']}/{r['diastolic']}"


def test_report_labels_systolic_and_diastolic(_corpus, tmp_path):
    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path), "--read"])
    h = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<small>systolic</small>" in h and "<small>diastolic</small>" in h


# --- missing templates are reported, not silently absorbed ------------------

def test_missing_insert_template_is_reported(tmp_path, capsys, _corpus):
    """A missing INSERT template used to be invisible: crop_fields() cannot
    locate the sticker, drops its fields, and the run looks clean while the
    page's second BP source is never even looked for."""
    from aihtr.__main__ import load_env

    only_forms = tmp_path / "templates"
    only_forms.mkdir()
    for p in TEMPLATES.glob("*.png"):
        if p.stem == "triage-card-v1":
            continue
        (only_forms / p.name).write_bytes(p.read_bytes())

    _, _, missing, missing_inserts = load_env(ROOT / "configs" / "forms", only_forms)
    assert not missing
    assert any("triage-card-v1" in m for m in missing_inserts)

    main([str(RAW / "base-opd-card-v1_001.jpg"), "--out", str(tmp_path / "out"),
          "--templates", str(only_forms)])
    out = capsys.readouterr().out
    assert "insert template missing" in out and "triage-card-v1" in out


def test_insert_template_is_present_once_templates_are_built(_corpus):
    """build_templates.py cuts it from the parent form's source_region, so a
    fresh checkout that builds templates gets the sticker too."""
    from aihtr.__main__ import load_env

    _, _, _, missing_inserts = load_env(ROOT / "configs" / "forms", TEMPLATES)
    assert missing_inserts == []
