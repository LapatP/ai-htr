"""Generate crop-stress BP variants using only genuine handwritten ink.

ALPHABET LIMITATION
-------------------
The real page supplies only the digits ``0, 1, 2, 3, 4, 6, 9``.  Digits
``5, 7, 8`` are never generated and therefore are NEVER tested by scores from
this dataset.  Values requiring an unavailable digit are redrawn, not faked
with a font or substituted.  The 14 hardcoded samples are the visually verified
set recorded in ``diagnostics/glyph_bank.png``; this script does not relabel or
regenerate that contact sheet.

The generator harvests real pen components from the canonical base OPD card,
inpaints the existing BP values, composes a clinically plausible replacement
with per-glyph spacing/baseline/rotation variation, and writes the same value
to the base-card and triage-sticker BP fields.  ``--disagree-rate`` deliberately
makes the sticker disagree on some pages.  The base value can be displaced
inside its context, the sticker can be moved independently, and the whole page
is finally warped and degraded like a re-scan.

Generated pages and ``truth.csv`` / ``truth.json`` stay under
``data/synthetic/`` by default.  No patient scan directory is consulted.

Usage:
  python scripts/make_offset_variants.py --count 20 --seed 7 --move-sticker
  python scripts/make_offset_variants.py --disagree-rate 0.25
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr import schema                                  # noqa: E402
from aihtr.normalize import CANONICAL_SIZE                # noqa: E402
from aihtr.register import warp                           # noqa: E402
from aihtr.synth import random_homography                 # noqa: E402

W, H = CANONICAL_SIZE
FORM_ID = "base-opd-card-v1"
BASE_FIELD_ID = "bp"
INSERT_ID = "triage-card-v1"
AVAILABLE_DIGITS = frozenset("0123469")
MAX_HOMOGRAPHY_ATTEMPTS = 100
MAX_STICKER_ATTEMPTS = 500
GLYPH_ROTATION_DEG = 2.0
STICKER_ROTATION_DEG = 5.0

@dataclass
class GlyphSample:
    symbol: str
    source: str
    values: np.ndarray
    mask: np.ndarray
    origin: tuple[int, int]
    erase_group: str | None = None


# Exact boxes are the manually reviewed subset of segment() candidates.  Upper
# T/P/R duplicates whose strokes merge into solid/dotted rules are deliberately
# omitted; the R-row 2 has its own traced separation below.  This avoids a
# plausible-looking but silently mislabelled glyph bank.
GLYPH_SPECS = [
    # sticker BP 143 / 90
    ("1", "sticker BP 143", (627, 1549, 642, 1592), "sticker_systolic", "largest"),
    ("4", "sticker BP 143", (647, 1540, 669, 1585), "sticker_systolic", "largest"),
    ("3", "sticker BP 143", (665, 1539, 687, 1585), "sticker_systolic", "largest"),
    ("9", "sticker BP 90", (759, 1545, 786, 1596), "sticker_diastolic", "largest"),
    ("0", "sticker BP 90", (789, 1540, 824, 1580), "sticker_diastolic", "largest"),
    # sticker O2 Sat 96
    ("9", "sticker O2 96", (1197, 1536, 1228, 1580), None, "largest"),
    ("6", "sticker O2 96", (1230, 1539, 1248, 1594), None, "largest"),
    # physically separate base-card BP 143/90
    ("1", "base BP 143/90", (2015, 568, 2025, 657), "base", "largest"),
    ("4", "base BP 143/90", (2044, 554, 2091, 657), "base", "largest"),
    ("3", "base BP 143/90", (2083, 575, 2111, 641), "base", "largest"),
    ("/", "base BP 143/90", (2118, 545, 2143, 645), "base", "largest"),
    ("9", "base BP 143/90", (2139, 567, 2174, 649), "base", "largest"),
    ("0", "base BP 143/90", (2179, 566, 2219, 631), "base", "largest"),
    # the R-row 2 is separated from its touching 0 by the reviewed traced cut
    ("2", "sticker R 20", (1197, 1475, 1230, 1510), None, "r2_trace"),
]

# Duplicated from make_synthetic.py because scripts/ is not an importable
# package under every invocation style.  These values intentionally stay in
# lockstep with the original synthetic generator.
LEVELS = {
    "clean":    dict(rot=0.0, scale=0.00, trans=0,  persp=0.0,    blur=0, noise=0,  jpeg=95),
    "mild":     dict(rot=1.5, scale=0.03, trans=25, persp=4e-5,   blur=3, noise=4,  jpeg=80),
    "moderate": dict(rot=3.0, scale=0.06, trans=50, persp=9e-5,   blur=5, noise=8,  jpeg=60),
    "heavy":    dict(rot=5.0, scale=0.09, trans=80, persp=1.4e-4, blur=5, noise=12, jpeg=40),
}


def degrade(img, p, rng):
    """Apply the same blur, sensor noise, and JPEG damage as make_synthetic."""
    if p["blur"] >= 3:
        img = cv2.GaussianBlur(img, (p["blur"], p["blur"]), 0)
    if p["noise"]:
        img = np.clip(img.astype(np.int16) + rng.normal(0, p["noise"], img.shape),
                      0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else img


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite, non-negative number")
    return value


def _rate(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number from 0 to 1")
    return value


def _crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    return img[box[1]:box[3], box[0]:box[2]]


def _component_sample(template: np.ndarray, symbol: str, source: str,
                      box: tuple[int, int, int, int], erase_group: str | None,
                      allowed: np.ndarray | None = None) -> GlyphSample:
    """Keep one reviewed Otsu component plus a one-pixel real-grey fringe."""
    crop = _crop(template, box)
    _, binary = cv2.threshold(crop, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = binary.astype(np.uint8)
    if allowed is not None:
        binary &= allowed.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        raise RuntimeError(f"no ink component for curated glyph {source}={symbol}")
    areas = stats[1:, cv2.CC_STAT_AREA]
    target = (labels == (1 + int(np.argmax(areas)))).astype(np.uint8)
    fringe = cv2.dilate(target, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = ((target != 0) | ((fringe != 0) & (crop < 200))).astype(np.uint8) * 255

    ys, xs = np.nonzero(mask)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    mask = mask[y1:y2, x1:x2]
    source_pixels = crop[y1:y2, x1:x2]
    values = np.full_like(source_pixels, 255)
    values[mask != 0] = source_pixels[mask != 0]
    return GlyphSample(symbol, source, values, mask,
                       (box[0] + x1, box[1] + y1), erase_group)


def _harvest_glyphs(template: np.ndarray) -> tuple[dict[str, list[GlyphSample]],
                                                   list[GlyphSample]]:
    samples = []
    for symbol, source, box, erase_group, mode in GLYPH_SPECS:
        allowed = None
        if mode == "r2_trace":
            rh, rw = box[3] - box[1], box[2] - box[0]
            gy, gx = np.mgrid[0:rh, 0:rw]
            absolute_x, absolute_y = gx + box[0], gy + box[1]
            allowed = np.where(absolute_y < 1502, absolute_x <= 1225,
                               absolute_x <= 1229).astype(np.uint8)
        samples.append(_component_sample(
            template, symbol, source, tuple(box), erase_group, allowed))

    bank: dict[str, list[GlyphSample]] = {}
    for sample in samples:
        bank.setdefault(sample.symbol, []).append(sample)
    expected = set(AVAILABLE_DIGITS) | {"/"}
    if set(bank) != expected:
        raise RuntimeError(f"real glyph alphabet changed: got {sorted(bank)}, expected {sorted(expected)}")
    return bank, samples


def _sample_mask(samples: list[GlyphSample], group: str) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    found = 0
    for sample in samples:
        if sample.erase_group != group:
            continue
        x, y = sample.origin
        h, w = sample.mask.shape
        mask[y:y + h, x:x + w] = np.maximum(mask[y:y + h, x:x + w], sample.mask)
        found += 1
    if not found:
        raise RuntimeError(f"no original ink samples for erase group {group}")
    return mask


def _erase_field(page: np.ndarray, box, ink_mask: np.ndarray,
                 restore_rule_dots: bool = False) -> np.ndarray:
    """Widely inpaint pen halos and optionally rebuild a strong dotted rule.

    A 15x15 ellipse/radius-5 Telea pass removes the faint 5x5-mask ghosts.
    The base field needs only that wider mask.  The sticker's darker rule can
    otherwise bleed into Telea as triangular blobs, so its whole local rule
    strip is inpainted and reviewed dot-shaped cores are restored darken-only.
    """
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    local = ink_mask[y1:y2, x1:x2]
    wide = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    erase = np.zeros_like(page, np.uint8)
    erase[y1:y2, x1:x2] = wide

    strip = None
    if restore_rule_dots:
        region = page[y1:y2, x1:x2]
        lower = region.shape[0] // 2
        rule_y = lower + int(np.argmax((region[lower:] < 180).sum(axis=1)))
        sx1, sx2 = max(0, x1 - 8), min(W, x2 + 8)
        sy1, sy2 = max(0, y1 + rule_y - 5), min(H, y1 + rule_y + 7)
        erase[sy1:sy2, sx1:sx2] = 255
        strip = (sx1, sy1, sx2, sy2)

    out = cv2.inpaint(page, erase, 5, cv2.INPAINT_TELEA)
    if strip is not None:
        sx1, sy1, sx2, sy2 = strip
        original = page[sy1:sy2, sx1:sx2]
        core = (original < 110).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        dots = np.zeros_like(core)
        for i in range(1, n):
            x, y, w, h, area = map(int, stats[i])
            if (5 <= area <= 60 and 3 <= w <= 10 and 3 <= h <= 10
                    and y >= 2 and x > 0 and x + w < core.shape[1]):
                dots[labels == i] = 255
        restore = cv2.dilate(
            dots, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) != 0
        target = out[sy1:sy2, sx1:sx2]
        target[restore] = np.minimum(target[restore], original[restore])
    return out


def _bbox(mask: np.ndarray, origin: tuple[int, int] = (0, 0)) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("ink mask is empty")
    return [origin[0] + int(xs.min()), origin[1] + int(ys.min()),
            origin[0] + int(xs.max()) + 1, origin[1] + int(ys.max()) + 1]


def _tighten(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = _bbox(mask)
    x1, y1, x2, y2 = box
    return values[y1:y2, x1:x2], mask[y1:y2, x1:x2]


def _rotate_patch(values: np.ndarray, mask: np.ndarray, origin: tuple[int, int],
                  angle: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Rotate a masked ink patch around its centre without clipping it."""
    ph, pw = values.shape
    matrix = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, 1.0)
    corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
    turned = cv2.transform(corners, matrix).reshape(-1, 2)
    turned[:, 0] += origin[0]
    turned[:, 1] += origin[1]
    ox, oy = int(np.floor(turned[:, 0].min())), int(np.floor(turned[:, 1].min()))
    x2, y2 = int(np.ceil(turned[:, 0].max())), int(np.ceil(turned[:, 1].max()))
    local_matrix = matrix.copy()
    local_matrix[0, 2] += origin[0] - ox
    local_matrix[1, 2] += origin[1] - oy
    size = (x2 - ox, y2 - oy)
    rotated_values = cv2.warpAffine(
        values, local_matrix, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    rotated_mask = cv2.warpAffine(
        mask, local_matrix, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rotated_mask[rotated_values == 255] = 0
    rotated_values[rotated_mask == 0] = 255
    return rotated_values, rotated_mask, (ox, oy)


def _resize_patch(values: np.ndarray, mask: np.ndarray,
                  scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = values.shape
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    values = cv2.resize(values, (nw, nh), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask[values == 255] = 0
    values[mask == 0] = 255
    return _tighten(values, mask)


def _compose_text(bank: dict[str, list[GlyphSample]], text: str, box,
                  rng) -> tuple[np.ndarray, np.ndarray]:
    """Compose real glyphs with human-like spacing, jitter, and rotation."""
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    target_h = max(12, int(round(bh * float(rng.uniform(0.68, 0.78)))))
    prepared: list[tuple[np.ndarray, np.ndarray]] = []
    for symbol in text:
        if symbol not in bank:
            raise RuntimeError(f"no real glyph for {symbol!r}")
        sample = bank[symbol][int(rng.integers(0, len(bank[symbol])))]
        scale = target_h / sample.values.shape[0] * float(rng.uniform(0.94, 1.06))
        values, mask = _resize_patch(sample.values, sample.mask, scale)
        values, mask, _ = _rotate_patch(
            values, mask, (0, 0), float(rng.uniform(-GLYPH_ROTATION_DEG,
                                                    GLYPH_ROTATION_DEG)))
        prepared.append(_tighten(values, mask))

    gaps = [max(2, int(round(target_h * float(rng.uniform(0.045, 0.09)))))
            for _ in range(max(0, len(prepared) - 1))]
    total_w = sum(v.shape[1] for v, _ in prepared) + sum(gaps)
    max_width = max(1, int(round(0.96 * bw)))
    if total_w > max_width:
        shrink = max_width / total_w
        prepared = [_resize_patch(v, m, shrink) for v, m in prepared]
        gaps = [max(1, int(round(g * shrink))) for g in gaps]

    max_h = max(v.shape[0] for v, _ in prepared)
    jitter = max(1, int(round(0.035 * target_h)))
    baseline_jitter = [int(rng.integers(-jitter, jitter + 1)) for _ in prepared]
    canvas_w = sum(v.shape[1] for v, _ in prepared) + sum(gaps)
    canvas_h = max_h + 2 * jitter
    values_out = np.full((canvas_h, canvas_w), 255, np.uint8)
    mask_out = np.zeros((canvas_h, canvas_w), np.uint8)
    x = 0
    for i, (values, mask) in enumerate(prepared):
        h, w = values.shape
        y = jitter + max_h - h + baseline_jitter[i]
        y = min(max(0, y), canvas_h - h)
        values_out[y:y + h, x:x + w] = np.minimum(
            values_out[y:y + h, x:x + w], values)
        mask_out[y:y + h, x:x + w] = np.maximum(
            mask_out[y:y + h, x:x + w], mask)
        x += w + (gaps[i] if i < len(gaps) else 0)
    return _tighten(values_out, mask_out)


def _nominal_origin(box, patch: np.ndarray) -> tuple[int, int]:
    ph, pw = patch.shape
    x = box.x1 + (box.x2 - box.x1 - pw) // 2
    bottom_margin = max(2, int(round(0.04 * (box.y2 - box.y1))))
    y = box.y2 - bottom_margin - ph
    return x, y


def _composite(page: np.ndarray, values: np.ndarray,
               origin: tuple[int, int]) -> np.ndarray:
    x, y = origin
    h, w = values.shape
    if x < 0 or y < 0 or x + w > page.shape[1] or y + h > page.shape[0]:
        raise RuntimeError("ink patch escaped its canvas")
    layer = np.full_like(page, 255)
    layer[y:y + h, x:x + w] = values
    return np.minimum(page, layer)


def _place_offset(page: np.ndarray, values: np.ndarray, mask: np.ndarray,
                  origin: tuple[int, int], context, requested_dx: int,
                  requested_dy: int) -> tuple[np.ndarray, list[int], int, int, bool]:
    base = _bbox(mask, origin)
    min_dx, max_dx = context.x1 - base[0], context.x2 - base[2]
    min_dy, max_dy = context.y1 - base[1], context.y2 - base[3]
    if min_dx > max_dx or min_dy > max_dy:
        raise RuntimeError("composed handwriting cannot fit inside context_bbox")
    dx = min(max(requested_dx, min_dx), max_dx)
    dy = min(max(requested_dy, min_dy), max_dy)
    shifted = origin[0] + dx, origin[1] + dy
    page = _composite(page, values, shifted)
    return page, _bbox(mask, shifted), int(dx), int(dy), \
        bool(dx != requested_dx or dy != requested_dy)


def _plausible_bp(rng, exclude: tuple[int, int] | None = None) -> tuple[int, int]:
    """Redraw until both clinically plausible numbers use the real alphabet."""
    for _ in range(10000):
        systolic = int(rng.integers(92, 187))
        upper = min(112, systolic - 18)
        if upper < 54:
            continue
        diastolic = int(rng.integers(54, upper + 1))
        pair = systolic, diastolic
        if (pair != exclude
                and set(str(systolic) + str(diastolic)) <= AVAILABLE_DIGITS):
            return pair
    raise RuntimeError("could not draw a BP value from the real glyph alphabet")


def _box_corners(box: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    return np.float32([[box[0], box[1]], [box[2], box[1]],
                       [box[2], box[3]], [box[0], box[3]]]).reshape(-1, 1, 2)


def _transform_bbox(box: list[int] | tuple[int, int, int, int],
                    transform: np.ndarray) -> list[int]:
    if transform.shape == (2, 3):
        points = cv2.transform(_box_corners(box), transform).reshape(-1, 2)
    else:
        points = cv2.perspectiveTransform(_box_corners(box), transform).reshape(-1, 2)
    return [int(np.floor(points[:, 0].min())), int(np.floor(points[:, 1].min())),
            int(np.ceil(points[:, 0].max())), int(np.ceil(points[:, 1].max()))]


def _intersects(a: list[int], b: list[int]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _sticker_transform(region, forbidden, max_offset: int, rng,
                       move: bool) -> tuple[np.ndarray, int, int, float, list[int]]:
    sw, sh = region.x2 - region.x1, region.y2 - region.y1
    if not move:
        matrix = np.array([[1.0, 0.0, region.x1],
                           [0.0, 1.0, region.y1]], dtype=float)
        return matrix, 0, 0, 0.0, [region.x1, region.y1, region.x2, region.y2]

    forbidden_box = [forbidden.x1, forbidden.y1, forbidden.x2, forbidden.y2]
    for _ in range(MAX_STICKER_ATTEMPTS):
        angle = float(rng.uniform(-STICKER_ROTATION_DEG, STICKER_ROTATION_DEG))
        base = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), angle, 1.0)
        base[0, 2] += region.x1
        base[1, 2] += region.y1
        footprint0 = cv2.transform(_box_corners((0, 0, sw, sh)), base).reshape(-1, 2)
        lo_dx = max(-max_offset, int(math.ceil(-float(footprint0[:, 0].min()))))
        hi_dx = min(max_offset, int(math.floor(W - float(footprint0[:, 0].max()))))
        lo_dy = max(-max_offset, int(math.ceil(-float(footprint0[:, 1].min()))))
        hi_dy = min(max_offset, int(math.floor(H - float(footprint0[:, 1].max()))))
        if lo_dx > hi_dx or lo_dy > hi_dy:
            continue
        dx = int(rng.integers(lo_dx, hi_dx + 1))
        dy = int(rng.integers(lo_dy, hi_dy + 1))
        matrix = base.copy()
        matrix[0, 2] += dx
        matrix[1, 2] += dy
        footprint = _transform_bbox((0, 0, sw, sh), matrix)
        if not _intersects(footprint, forbidden_box):
            return matrix, dx, dy, angle, footprint
    raise RuntimeError("could not place the triage sticker safely on the page")


def _warp_sticker(page: np.ndarray, region, matrix: np.ndarray) -> np.ndarray:
    """Lift, blank, rotate, and darken-only glue the sticker like make_synthetic."""
    patch = page[region.y1:region.y2, region.x1:region.x2].copy()
    page = page.copy()
    page[region.y1:region.y2, region.x1:region.x2] = 255
    canvas = np.full_like(page, 255)
    cv2.warpAffine(patch, matrix, (W, H), dst=canvas, flags=cv2.INTER_CUBIC,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return np.minimum(page, canvas)


def _warped_ink_bbox(values: np.ndarray, mask: np.ndarray,
                      local_origin: tuple[int, int], matrix: np.ndarray) -> list[int]:
    """Raster-transform one insert-local ink layer and return its exact support."""
    adjusted = matrix.copy()
    adjusted[:, 2] += matrix[:, :2] @ np.asarray(local_origin, dtype=float)
    moved_values = cv2.warpAffine(
        values, adjusted, (W, H), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    moved_mask = cv2.warpAffine(
        mask, adjusted, (W, H), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    moved_mask[moved_values == 255] = 0
    return _bbox(moved_mask)


def _visible_homography(boxes: list[list[int]], sticker_footprint: list[int],
                        params: dict, rng):
    """Draw a page warp that keeps every truth target and the insert visible."""
    required = boxes + [sticker_footprint]
    for attempt in range(1, MAX_HOMOGRAPHY_ATTEMPTS + 1):
        homography = random_homography(
            CANONICAL_SIZE, rng, rot=params["rot"], scale=params["scale"],
            trans=params["trans"], persp=params["persp"])
        projected = [_transform_bbox(box, homography) for box in required]
        if all(b[0] >= 0 and b[1] >= 0 and b[2] <= W and b[3] <= H
               for b in projected):
            return homography, projected[:-1], attempt
    raise RuntimeError(
        f"could not keep all targets visible after {MAX_HOMOGRAPHY_ATTEMPTS} page warps")


def _bbox_outside(inner: list[int], outer: list[int]) -> int:
    return int(inner[0] < outer[0] or inner[1] < outer[1]
               or inner[2] > outer[2] or inner[3] > outer[3])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=_positive_int, default=8,
                    help="number of page variants to generate (default: 8)")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "synthetic" / "offset")
    ap.add_argument("--level", choices=[*LEVELS, "all"], default="all",
                    help="degradation level, or cycle all levels (default: all)")
    ap.add_argument("--max-offset-frac", type=_nonnegative_float, default=0.35,
                    help="maximum base-ink dx/width and dy/height (default: 0.35)")
    ap.add_argument("--ink-rotation", type=_nonnegative_float, default=4.0,
                    help="maximum whole base-value rotation in degrees (default: 4.0)")
    ap.add_argument("--move-sticker", action="store_true",
                    help="also lift, rotate, and reposition the triage insert")
    ap.add_argument("--max-sticker-offset", type=_nonnegative_int, default=200,
                    help="maximum absolute sticker dx and dy in pixels (default: 200)")
    ap.add_argument("--disagree-rate", type=_rate, default=0.15,
                    help="fraction with a deliberately different sticker BP (default: 0.15)")
    args = ap.parse_args()

    forms = schema.load_all(ROOT / "configs" / "forms")
    spec = forms.get(FORM_ID)
    if spec is None:
        raise RuntimeError(f"form not configured: {FORM_ID}")
    base_field = next((f for f in spec.fields if f.id == BASE_FIELD_ID), None)
    insert = next((i for i in spec.inserts if i.id == INSERT_ID), None)
    if (base_field is None or not base_field.resolved or base_field.context_bbox is None
            or insert is None or insert.source_region is None):
        raise RuntimeError("base BP field and triage insert must be fully configured")
    insert_fields = {f.id: f for f in insert.fields}
    systolic_field = insert_fields.get("bp.systolic")
    diastolic_field = insert_fields.get("bp.diastolic")
    if (systolic_field is None or diastolic_field is None
            or not systolic_field.resolved or not diastolic_field.resolved):
        raise RuntimeError("triage systolic/diastolic fields must be configured")

    template_path = ROOT / (spec.template or f"data/templates/{FORM_ID}.png")
    template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"could not read template: {template_path}")
    if template.shape != (H, W):
        raise ValueError(f"template must be {W}x{H}, got {template.shape[1]}x{template.shape[0]}")

    bank, samples = _harvest_glyphs(template)
    print("glyph bank: 0=2, 1=2, 2=1, 3=2, 4=2, 6=1, 9=3, /=1 (verified)")

    source_region = insert.source_region
    sys_abs = schema.BBox(
        source_region.x1 + systolic_field.bbox.x1,
        source_region.y1 + systolic_field.bbox.y1,
        source_region.x1 + systolic_field.bbox.x2,
        source_region.y1 + systolic_field.bbox.y2)
    dia_abs = schema.BBox(
        source_region.x1 + diastolic_field.bbox.x1,
        source_region.y1 + diastolic_field.bbox.y1,
        source_region.x1 + diastolic_field.bbox.x2,
        source_region.y1 + diastolic_field.bbox.y2)

    clean_page = _erase_field(template, base_field.bbox, _sample_mask(samples, "base"))
    clean_page = _erase_field(
        clean_page, sys_abs, _sample_mask(samples, "sticker_systolic"), True)
    clean_page = _erase_field(
        clean_page, dia_abs, _sample_mask(samples, "sticker_diastolic"), True)
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    level_cycle = list(LEVELS) if args.level == "all" else [args.level]
    bw, bh = base_field.bbox.wh
    max_dx = int(round(args.max_offset_frac * bw))
    max_dy = int(round(args.max_offset_frac * bh))
    truth: list[dict] = []

    print(f"{'file':<48} {'level':<9} {'base/sticker':<19} {'ink dxy':>11} "
          f"{'sticker dxy':>13} {'disagree':>8}")
    print("-" * 120)

    for i in range(args.count):
        level = level_cycle[i % len(level_cycle)]
        base_pair = _plausible_bp(rng)
        disagreed = bool(rng.random() < args.disagree_rate)
        sticker_pair = _plausible_bp(rng, exclude=base_pair) if disagreed else base_pair

        base_text = f"{base_pair[0]}/{base_pair[1]}"
        base_values, base_mask = _compose_text(bank, base_text, base_field.bbox, rng)
        base_origin = _nominal_origin(base_field.bbox, base_values)
        requested_dx = int(rng.integers(-max_dx, max_dx + 1)) if max_dx else 0
        requested_dy = int(rng.integers(-max_dy, max_dy + 1)) if max_dy else 0
        base_angle = float(rng.uniform(-args.ink_rotation, args.ink_rotation)) \
            if args.ink_rotation else 0.0
        base_values, base_mask, base_origin = _rotate_patch(
            base_values, base_mask, base_origin, base_angle)
        page, base_bbox, dx, dy, clamped = _place_offset(
            clean_page, base_values, base_mask, base_origin,
            base_field.context_bbox, requested_dx, requested_dy)

        sys_values, sys_mask = _compose_text(
            bank, str(sticker_pair[0]), systolic_field.bbox, rng)
        dia_values, dia_mask = _compose_text(
            bank, str(sticker_pair[1]), diastolic_field.bbox, rng)
        sys_origin = _nominal_origin(systolic_field.bbox, sys_values)
        dia_origin = _nominal_origin(diastolic_field.bbox, dia_values)
        page = _composite(page, sys_values,
                          (source_region.x1 + sys_origin[0], source_region.y1 + sys_origin[1]))
        page = _composite(page, dia_values,
                          (source_region.x1 + dia_origin[0], source_region.y1 + dia_origin[1]))

        sticker_matrix, sticker_dx, sticker_dy, sticker_angle, footprint = \
            _sticker_transform(source_region, base_field.context_bbox,
                               args.max_sticker_offset, rng, args.move_sticker)
        sys_bbox = _warped_ink_bbox(sys_values, sys_mask, sys_origin, sticker_matrix)
        dia_bbox = _warped_ink_bbox(dia_values, dia_mask, dia_origin, sticker_matrix)
        if args.move_sticker:
            page = _warp_sticker(page, source_region, sticker_matrix)

        params = LEVELS[level]
        true_boxes = [base_bbox, sys_bbox, dia_bbox]
        homography, warped_boxes, homography_attempts = _visible_homography(
            true_boxes, footprint, params, rng)
        page = warp(page, homography, CANONICAL_SIZE)
        page = degrade(page, params, rng)

        name = f"offset_{i + 1:04d}_{FORM_ID}_{level}.jpg"
        if not cv2.imwrite(str(args.out / name), page,
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"could not write {args.out / name}")

        base_nominal = [base_field.bbox.x1, base_field.bbox.y1,
                        base_field.bbox.x2, base_field.bbox.y2]
        sys_nominal = _transform_bbox(
            [systolic_field.bbox.x1, systolic_field.bbox.y1,
             systolic_field.bbox.x2, systolic_field.bbox.y2], sticker_matrix)
        dia_nominal = _transform_bbox(
            [diastolic_field.bbox.x1, diastolic_field.bbox.y1,
             diastolic_field.bbox.x2, diastolic_field.bbox.y2], sticker_matrix)
        ink_offset_px = float(math.hypot(dx, dy))
        sticker_offset_px = float(math.hypot(sticker_dx, sticker_dy))
        common = {
            "base_value": base_text,
            "sticker_value": f"{sticker_pair[0]}/{sticker_pair[1]}",
            "intended_disagreement": int(disagreed),
            "dx": dx, "dy": dy, "ink_rotation_deg": round(base_angle, 6),
            "requested_dx": requested_dx, "requested_dy": requested_dy,
            "clamped": int(clamped),
            "ink_offset_px": round(ink_offset_px, 3),
            "sticker_dx": sticker_dx, "sticker_dy": sticker_dy,
            "sticker_rotation_deg": round(sticker_angle, 6),
            "sticker_offset_px": round(sticker_offset_px, 3),
            "homography_attempts": homography_attempts,
            "glyph_alphabet": "0123469",
        }
        row_specs = [
            (BASE_FIELD_ID, base_text, base_bbox, warped_boxes[0], base_nominal,
             ink_offset_px, "ink"),
            ("bp.systolic", str(sticker_pair[0]), sys_bbox, warped_boxes[1], sys_nominal,
             sticker_offset_px, "sticker"),
            ("bp.diastolic", str(sticker_pair[1]), dia_bbox, warped_boxes[2], dia_nominal,
             sticker_offset_px, "sticker"),
        ]
        for field_id, value, true_box, warped_box, nominal_box, displacement, kind in row_specs:
            truth.append({
                "file": name, "form": FORM_ID, "level": level,
                "field": field_id, "value": value,
                "x1": true_box[0], "y1": true_box[1],
                "x2": true_box[2], "y2": true_box[3],
                "overflowed": _bbox_outside(true_box, nominal_box),
                "warped_x1": warped_box[0], "warped_y1": warped_box[1],
                "warped_x2": warped_box[2], "warped_y2": warped_box[3],
                "displacement_px": round(displacement, 3),
                "displacement_kind": kind,
                **common,
            })

        print(f"{name:<48} {level:<9} {base_text}/{common['sticker_value']:<19} "
              f"{dx:>+5},{dy:<+5} {sticker_dx:>+6},{sticker_dy:<+6} {str(disagreed):>8}")

    with (args.out / "truth.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(truth[0]))
        writer.writeheader()
        writer.writerows(truth)
    (args.out / "truth.json").write_text(
        json.dumps(truth, indent=2), encoding="utf-8")

    print("-" * 120)
    print(f"{args.count} page(s), {len(truth)} field(s) with known values -> {args.out}")
    print("ground truth: truth.csv / truth.json")
    print("  x1..y2 are actual ink bounds after base offset / sticker affine")
    print("  warped_x1..warped_y2 transform those bounds through the page homography")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
