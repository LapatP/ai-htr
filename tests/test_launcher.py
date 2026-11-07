"""The drag-and-drop launcher must stay CRLF and ASCII.

cmd.exe parses batch files by seeking on byte offsets, assuming CRLF. An LF-only
.cmd desyncs and silently swallows the first characters of lines — `rem` becomes
`m`, `setlocal` becomes `tlocal` — producing a screenful of "not recognized as an
internal or external command" that points nowhere near the real cause.

Worse, it is content-length dependent: this file worked when first written and
broke after a one-line edit shifted the offsets. Hence a test rather than a
comment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "Process dropped scans.cmd"


@pytest.fixture(scope="module")
def raw() -> bytes:
    if not LAUNCHER.exists():
        pytest.skip("launcher not present")
    return LAUNCHER.read_bytes()


def test_launcher_is_not_empty(raw):
    assert len(raw) > 500


def test_every_line_ends_crlf(raw):
    bare_lf = raw.count(b"\n") - raw.count(b"\r\n")
    assert bare_lf == 0, f"{bare_lf} LF-only line ending(s) — cmd.exe will corrupt this"


def test_launcher_is_plain_ascii(raw):
    """The console codepage is not UTF-8; non-ASCII prints as mojibake."""
    offenders = [(i, b) for i, b in enumerate(raw) if b > 127]
    assert not offenders, f"non-ASCII byte(s) at {offenders[:5]}"


def test_launcher_still_calls_the_package(raw):
    text = raw.decode("ascii")
    assert "-m aihtr" in text
    assert r"data\inbox" in text
    assert "report.html" in text
