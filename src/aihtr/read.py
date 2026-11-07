"""Stage 4 — turn crops into two integers.

One reader per BP encoding pattern:

* **A** — one crop holding `143/90`; the writer supplied the slash, so we must
  find it and split there.
* **B** — two crops flanking a *printed* slash; each is a bare integer.
* **C** — two separately labelled blanks (`Home SBP` / `DBP`); mechanically the
  same as B, but the field ids carry which is which.

The output contract is two integers, never a string. `143/90` mangled into
`14/390` is a one-character slip and clinical nonsense, so systolic and
diastolic are separate entities from the moment they leave this module.

Nothing here guesses. If the separator cannot be found, or a glyph does not
match any template well enough, the reading comes back unreadable with a reason
attached. An unreadable field routed to a human is a nuisance; a confidently
invented blood pressure is a patient-safety event.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .digits import Glyph, read_glyphs


@dataclass
class BPReading:
    systolic: int | None = None
    diastolic: int | None = None
    confidence: float = 0.0          # weakest glyph in the reading
    glyphs: list[Glyph] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.systolic is not None and self.diastolic is not None

    @property
    def text(self) -> str:
        return f"{self.systolic}/{self.diastolic}" if self.readable else "unreadable"

    def __str__(self) -> str:
        s = f"{self.text:<12} conf {self.confidence:.2f}"
        if self.flags:
            s += "  [" + ", ".join(self.flags) + "]"
        return s


def _digits_to_int(glyphs: list[Glyph]) -> tuple[int | None, list[str]]:
    """Concatenate glyphs into an integer, refusing anything that isn't clean."""
    flags: list[str] = []
    if not glyphs:
        return None, ["no_glyphs"]
    if any(g.symbol == "/" for g in glyphs):
        return None, ["unexpected_separator"]
    if any(g.symbol == "?" for g in glyphs):
        return None, ["unrecognised_glyph"]

    weak = [g for g in glyphs if not g.confident]
    if weak:
        flags.append(f"low_confidence_glyph({','.join(g.symbol for g in weak)})")

    # 2-3 digits is what a BP component looks like. Anything else is a
    # segmentation fault, and validate.py will reject it anyway — but naming it
    # here tells us *where* it broke.
    if not 2 <= len(glyphs) <= 3:
        flags.append(f"unexpected_digit_count({len(glyphs)})")

    return int("".join(g.symbol for g in glyphs)), flags


def _confidence(glyphs: list[Glyph]) -> float:
    """Weakest link. A reading is only as trustworthy as its worst glyph —
    averaging would let one confident digit mask an unreadable neighbour."""
    return min((g.score for g in glyphs), default=0.0)


def read_pattern_a(crop: np.ndarray) -> BPReading:
    """One crop, writer-supplied slash: `143/90`."""
    glyphs = read_glyphs(crop)
    if not glyphs:
        return BPReading(flags=["no_ink"])

    slashes = [i for i, g in enumerate(glyphs) if g.symbol == "/"]
    if len(slashes) != 1:
        # Deliberately no fallback. Splitting on "the biggest gap" would be a
        # guess, and a guess here silently produces a plausible wrong BP.
        return BPReading(
            confidence=_confidence(glyphs), glyphs=glyphs,
            flags=[f"separator_not_found({len(slashes)} candidates)"],
        )

    i = slashes[0]
    sys_g, dia_g = glyphs[:i], glyphs[i + 1:]
    s, f1 = _digits_to_int(sys_g)
    d, f2 = _digits_to_int(dia_g)
    return BPReading(s, d, _confidence(glyphs), glyphs, f1 + f2)


def read_pattern_split(sys_crop: np.ndarray, dia_crop: np.ndarray) -> BPReading:
    """Patterns B and C: two crops, one integer each, printed separator or none."""
    gs, gd = read_glyphs(sys_crop), read_glyphs(dia_crop)
    if not gs and not gd:
        return BPReading(flags=["no_ink"])

    s, f1 = _digits_to_int(gs)
    d, f2 = _digits_to_int(gd)
    return BPReading(s, d, _confidence(gs + gd), gs + gd,
                     [f"systolic:{x}" for x in f1] + [f"diastolic:{x}" for x in f2])


# Dispatch by the `reader` name declared in configs/forms/*.yaml.
READERS = {
    "bp_pair_single": "A",
    "bp_pair_split": "B",
    "bp_labelled_split": "C",
}
