"""Synthetic perturbation — our substitute for having more than one scan.

With one page per form, any evaluation that compares a page to a template built
from that same page is measuring nothing. These helpers manufacture a plausible
"second scan": a known homography plus the degradations a phone camera adds.

Because we chose the homography, ground truth is exact.
"""
from __future__ import annotations

import cv2
import numpy as np


def random_homography(size, rng, rot=3.0, scale=0.05, trans=40, persp=8e-5):
    """Small rotation, scale, shift and mild keystone — a plausible re-scan."""
    w, h = size
    cx, cy = w / 2, h / 2

    ang = rng.uniform(-rot, rot)
    s = 1.0 + rng.uniform(-scale, scale)
    tx, ty = rng.uniform(-trans, trans, 2)

    a = np.radians(ang)
    rs = np.array([
        [s * np.cos(a), -s * np.sin(a), 0.0],
        [s * np.sin(a),  s * np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])
    to_c = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    fr_c = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], float)
    H = fr_c @ rs @ to_c

    H[2, 0] += rng.uniform(-persp, persp)
    H[2, 1] += rng.uniform(-persp, persp)
    return H


def degrade(img, rng):
    """Blur + sensor noise + JPEG artefacts."""
    k = int(rng.choice([3, 5]))
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 70))])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else img


def grid(size, n=12):
    """Interior sample points for measuring recovery error."""
    w, h = size
    xs = np.linspace(0.08 * w, 0.92 * w, n)
    ys = np.linspace(0.08 * h, 0.92 * h, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)
