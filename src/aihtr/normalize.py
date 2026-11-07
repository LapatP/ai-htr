"""Stage 0 — page normalization. Form-agnostic.

Takes whatever came off a phone or a flatbed and produces a clean, upright,
fixed-size grayscale page in *canonical space*.

Why deskew with Hough lines specifically: this form family is covered in long
printed horizontal rules. That is an unusually strong, reliable signal — far
better here than the usual trick of fitting a minAreaRect around the text mask,
which gets dragged around by handwriting and stray marks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Coordinate space every bbox in configs/forms/*.yaml refers to: A4 at 300 DPI.
# Deliberately larger than our 103 DPI sources -- see CLAUDE.md "Resolution
# ceiling". Upsampling now is harmless; pinning canonical space to the current
# low resolution would make future good scans permanently lossy.
CANONICAL_SIZE = (2480, 3508)  # (width, height)


@dataclass
class NormalizedPage:
    image: np.ndarray       # grayscale, CANONICAL_SIZE
    skew_deg: float         # rotation removed
    source: Path
    scale: float            # source px -> canonical px
    pad: tuple[int, int]    # (left, top) letterbox offset, canonical px


def load_gray(path: str | Path) -> np.ndarray:
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


# A skew estimate is only trusted when the printed rules agree with each other.
#
# Measured: under pure rotation the detected line angles have an inter-quartile
# spread of 0.06-0.47 degrees - they are parallel, so their median is the page
# tilt. Add perspective and the rules stop being parallel (keystone makes them
# converge), the spread jumps to 3.5-7.3 degrees, and the median becomes
# meaningless: it reported -8.5 degrees on a page tilted -2.5.
#
# That is not a cosmetic error. normalize_page would then rotate the page by
# that bogus angle, leaving it MORE crooked, and registration lost so many
# inliers it dropped below the minimum and failed outright (63 inliers without
# normalisation, 19 with it).
#
# So: when the lines disagree, decline. Declining is safe because a homography
# already absorbs rotation and ORB is rotation-invariant - deskew is a
# convenience for the later stages, never a prerequisite.
MIN_SKEW_LINES = 12
MAX_SKEW_IQR = 1.5


def estimate_skew(gray: np.ndarray, max_angle: float = 15.0) -> float:
    """Median angle of the long horizontal printed rules, in degrees.

    Only lines spanning >=30% of the page width vote, so short handwriting
    strokes and table borders cannot dominate. Returns 0.0 when nothing
    qualifies, or when the lines that did qualify disagree too much to trust -
    see MAX_SKEW_IQR.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = int(0.30 * gray.shape[1])
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 720, threshold=150,
        minLineLength=min_len, maxLineGap=20,
    )
    if lines is None:
        return 0.0

    angles = []
    # OpenCV 4 returns (N,1,4), OpenCV 5 returns (N,4). Normalise both.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1)))
        if abs(ang) <= max_angle:      # keep near-horizontal only
            angles.append(ang)
    if len(angles) < MIN_SKEW_LINES:
        return 0.0

    a = np.asarray(angles)
    q75, q25 = np.percentile(a, [75, 25])
    if (q75 - q25) > MAX_SKEW_IQR:
        return 0.0          # not parallel => perspective => median is meaningless

    return float(np.median(a))


def rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate about the centre, filling new corners with paper-white."""
    if abs(angle_deg) < 0.01:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        gray, m, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def enhance(gray: np.ndarray) -> np.ndarray:
    """Even out lighting so ORB sees the same features on a desk photo as on a
    flatbed scan. CLAHE (local histogram equalisation) matters far more here
    than denoising -- our pages differ mostly in illumination, not noise."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def letterbox(gray: np.ndarray, size: tuple[int, int] = CANONICAL_SIZE):
    """Resize into `size` preserving aspect ratio, padding with white.

    Padding rather than stretching: our sources range from 0.68 to 0.72 aspect
    against A4's 0.707. Stretching would inject anisotropic scaling that the
    homography then has to undo -- avoidable error, so we avoid it.
    """
    tw, th = size
    h, w = gray.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)

    out = np.full((th, tw), 255, dtype=np.uint8)
    left, top = (tw - nw) // 2, (th - nh) // 2
    out[top:top + nh, left:left + nw] = resized
    return out, scale, (left, top)


def normalize_image(
    gray: np.ndarray,
    size: tuple[int, int] = CANONICAL_SIZE,
    do_deskew: bool = True,
    do_enhance: bool = True,
    source: Path | None = None,
) -> NormalizedPage:
    """Stage 0 on an in-memory image.

    Split out from normalize_page so evaluation can perturb a page and feed it
    straight back through, without a round trip to disk re-encoding it.
    """
    skew = estimate_skew(gray) if do_deskew else 0.0
    if do_deskew:
        gray = rotate(gray, skew)
    if do_enhance:
        gray = enhance(gray)
    out, scale, pad = letterbox(gray, size)
    return NormalizedPage(image=out, skew_deg=skew, source=source or Path("<memory>"),
                          scale=scale, pad=pad)


def normalize_page(
    path: str | Path,
    size: tuple[int, int] = CANONICAL_SIZE,
    do_deskew: bool = True,
    do_enhance: bool = True,
) -> NormalizedPage:
    """Stage 0 end to end: load -> deskew -> enhance -> canonical size."""
    gray = load_gray(path)
    skew = estimate_skew(gray) if do_deskew else 0.0
    if do_deskew:
        gray = rotate(gray, skew)
    if do_enhance:
        gray = enhance(gray)
    out, scale, pad = letterbox(gray, size)
    return NormalizedPage(
        image=out, skew_deg=skew, source=Path(path), scale=scale, pad=pad
    )
