"""Validate stage 2 by synthetic perturbation.

The problem: we have exactly one page per form, so a template is built from the
very page we would test against. Registering a page onto itself proves nothing.

The fix: take the template, warp it by a homography we CHOSE, so ground truth is
known exactly. Then ask the pipeline to recover it. Error is measured
end-to-end -- push a grid of points through the true warp, pull them back
through the recovered one, and see how far they land from where they started.

Perfect recovery = 0 px. What matters is whether error stays well under the
size of a BP field (~60 x 25 px at 103 DPI, ~175 x 73 px in canonical space).

Usage:  PYTHONPATH=src python scripts/eval_registration.py [--trials 5] [--degrade]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.normalize import CANONICAL_SIZE  # noqa: E402
from aihtr.register import register, warp    # noqa: E402

TEMPLATES = ROOT / "data" / "templates"
PX_TO_MM = 25.4 / 300.0    # canonical space is 300 DPI


def random_homography(size, rng, rot=3.0, scale=0.05, trans=40, persp=8e-5):
    """A plausible re-scan: small rotation, scale, shift, and mild perspective."""
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
    # rotate/scale about the centre, then translate
    to_c = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    fr_c = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]], float)
    H = fr_c @ rs @ to_c

    # mild keystone, as if photographed off-axis
    H[2, 0] += rng.uniform(-persp, persp)
    H[2, 1] += rng.uniform(-persp, persp)
    return H


def degrade(img, rng):
    """Blur, sensor noise, and JPEG artefacts -- what a phone photo adds."""
    k = int(rng.choice([3, 5]))
    img = cv2.GaussianBlur(img, (k, k), 0)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 70))])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else img


def grid(size, n=12):
    """Sample points across the page interior (margins excluded)."""
    w, h = size
    xs = np.linspace(0.08 * w, 0.92 * w, n)
    ys = np.linspace(0.08 * h, 0.92 * h, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--degrade", action="store_true", help="add blur/noise/JPEG")
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    templates = sorted(TEMPLATES.glob("*.png"))
    if not templates:
        print("no templates -- run scripts/build_templates.py first")
        return 1

    pts = grid(CANONICAL_SIZE)
    print(f"canonical {CANONICAL_SIZE[0]}x{CANONICAL_SIZE[1]} @300dpi   "
          f"degrade={args.degrade}   trials={args.trials}\n")
    print(f"{'form':<24} {'ok':>5} {'inliers':>8} {'mean px':>8} {'p95 px':>7} "
          f"{'max px':>7} {'mean mm':>8}")
    print("-" * 76)

    all_err, failures = [], 0
    for tp in templates:
        tmpl = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
        errs, inliers, oks = [], [], 0

        for _ in range(args.trials):
            H_true = random_homography(CANONICAL_SIZE, rng)
            pert = warp(tmpl, H_true, CANONICAL_SIZE)
            if args.degrade:
                pert = degrade(pert, rng)

            res = register(pert, tmpl)          # perturbed -> template
            if not res.ok:
                failures += 1
                continue
            oks += 1
            inliers.append(res.n_inliers)

            fwd = cv2.perspectiveTransform(pts, H_true)     # where they landed
            back = cv2.perspectiveTransform(fwd, res.homography)
            errs.append(np.linalg.norm(back - pts, axis=2).ravel())

        if not errs:
            print(f"{tp.stem:<24} {oks:>2}/{args.trials}  ALL FAILED")
            continue

        e = np.concatenate(errs)
        all_err.append(e)
        print(f"{tp.stem:<24} {oks:>2}/{args.trials} {np.mean(inliers):>8.0f} "
              f"{e.mean():>8.2f} {np.percentile(e, 95):>7.2f} {e.max():>7.2f} "
              f"{e.mean() * PX_TO_MM:>8.3f}")

    if all_err:
        e = np.concatenate(all_err)
        print("-" * 76)
        print(f"{'OVERALL':<24} {'':>5} {'':>8} {e.mean():>8.2f} "
              f"{np.percentile(e, 95):>7.2f} {e.max():>7.2f} "
              f"{e.mean() * PX_TO_MM:>8.3f}")
        print(f"\nregistration failures: {failures}")
        print(f"a BP field is roughly 175 x 73 px in canonical space -- "
              f"error must stay far below that.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
