"""Digit segmentation and closed-set classification.

Design constraint, taken from the project's clinical-safety position: the reader
must not be able to invent a value. This classifier chooses among eleven fixed
symbols (`0`-`9` and `/`) by template correlation. It cannot emit a character it
was not given, and when nothing matches well it says so instead of guessing.

That is the whole argument against pointing a document VLM at this. A VLM asked
to read a smudged `143` can return a fluent, plausible, wrong `148` with no
signal that anything went awry. Here, a weak match is visible as a low score.

The templates are rendered from OpenCV's built-in Hershey fonts — several
faces and stroke weights, including the two script faces, which approximate pen
strokes without needing any font files on disk. This is a **baseline**, not a
finished reader: with one filled page in the whole corpus there is nothing to
train a learned model on, and no honest way to measure one.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SYMBOLS = "0123456789/"
CANVAS = 32          # templates and queries are normalised to this square
INK_SPAN = 24        # longest side of the glyph inside the canvas

# Hershey faces used to synthesise templates. The two SCRIPT faces matter most —
# they are the closest thing OpenCV ships to handwriting.
_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
]
_THICKNESS = (2, 3, 4)

# Below this correlation the best match is not trustworthy: report the symbol
# but flag the reading rather than letting it through silently.
MIN_SCORE = 0.55
# The runner-up must be this much worse, else the glyph is genuinely ambiguous
# (the classic 3/8, 1/7, 5/6 confusions).
MIN_MARGIN = 0.04


@dataclass
class Glyph:
    symbol: str
    score: float          # correlation with the winning template, 0..1
    margin: float         # winner minus runner-up
    box: list[int]        # in the crop's own coordinates

    @property
    def confident(self) -> bool:
        return self.score >= MIN_SCORE and self.margin >= MIN_MARGIN


def _normalise(mask: np.ndarray) -> np.ndarray | None:
    """Crop to the ink, scale the long side to INK_SPAN, centre in a square.

    Aspect ratio is preserved deliberately. A `1` is naturally narrow and a `0`
    naturally round; squashing both into a square would throw away the single
    most discriminative feature either has.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    glyph = mask[y1:y2, x1:x2].astype(np.float32)
    h, w = glyph.shape
    s = INK_SPAN / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    glyph = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)

    out = np.zeros((CANVAS, CANVAS), np.float32)
    ox, oy = (CANVAS - nw) // 2, (CANVAS - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = glyph
    return cv2.GaussianBlur(out, (3, 3), 0)


def _build_templates() -> dict[str, list[np.ndarray]]:
    """Render every symbol in every face and weight, normalised like a query."""
    out: dict[str, list[np.ndarray]] = {s: [] for s in SYMBOLS}
    for sym in SYMBOLS:
        for font in _FONTS:
            for th in _THICKNESS:
                (w, h), base = cv2.getTextSize(sym, font, 2.0, th)
                canvas = np.zeros((h + base + 20, w + 20), np.uint8)
                cv2.putText(canvas, sym, (10, h + 10), font, 2.0, 255, th, cv2.LINE_AA)
                n = _normalise(canvas > 40)
                if n is not None:
                    out[sym].append(n)
    return out


_TEMPLATES = _build_templates()


def classify(mask: np.ndarray) -> tuple[str, float, float]:
    """Best symbol for one glyph mask, with its score and margin over runner-up."""
    q = _normalise(mask)
    if q is None:
        return "?", 0.0, 0.0
    qn = q - q.mean()
    qd = np.sqrt((qn * qn).sum()) or 1.0

    best: dict[str, float] = {}
    for sym, templates in _TEMPLATES.items():
        top = 0.0
        for t in templates:
            tn = t - t.mean()
            td = np.sqrt((tn * tn).sum()) or 1.0
            top = max(top, float((qn * tn).sum() / (qd * td)))
        best[sym] = top

    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    (sym, score), (_, second) = ranked[0], ranked[1]
    return sym, max(0.0, score), max(0.0, score - second)


def segment(crop: np.ndarray, min_height_frac: float = 0.30) -> list[list[int]]:
    """Split a tight crop into left-to-right glyph boxes.

    Components shorter than `min_height_frac` of the tallest are dropped — on
    these forms that is rule debris and speckle, never a digit.
    """
    if crop.size == 0 or crop.std() < 8:
        return []
    _, m = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    m = m.astype(np.uint8)

    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in stats[1:] if a >= 8]
    if not boxes:
        return []

    tallest = max(b[3] - b[1] for b in boxes)
    boxes = [b for b in boxes if (b[3] - b[1]) >= min_height_frac * tallest]
    return sorted(boxes, key=lambda b: b[0])


def read_glyphs(crop: np.ndarray) -> list[Glyph]:
    """Segment a crop and classify every glyph in it, left to right."""
    if crop.size == 0:
        return []
    _, m = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    m = m.astype(np.uint8)

    out = []
    for b in segment(crop):
        sym, score, margin = classify(m[b[1]:b[3], b[0]:b[2]])
        out.append(Glyph(sym, score, margin, b))
    return out
