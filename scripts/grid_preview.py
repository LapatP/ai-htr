"""Render a region of a template with a labelled canonical-coordinate grid.

A reading aid, not part of the pipeline. Lets you (or Claude) read box
coordinates straight off an image instead of guessing and iterating.

Usage:
  python scripts/grid_preview.py FORM                        # whole page
  python scripts/grid_preview.py FORM 1650 450 2350 750      # a region
  python scripts/grid_preview.py FORM 1650 450 2350 750 --step 25 --zoom 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "data" / "templates"
OUT = ROOT / "data" / "interim"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("form")
    ap.add_argument("box", nargs="*", type=int, help="x0 y0 x1 y1 in canonical px")
    ap.add_argument("--step", type=int, default=0, help="grid spacing, canonical px")
    ap.add_argument("--zoom", type=float, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = TEMPLATES / f"{args.form}.png"
    img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"no template: {src}")
        return 1
    H, W = img.shape[:2]

    x0, y0, x1, y1 = args.box if len(args.box) == 4 else (0, 0, W, H)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    step = args.step or (200 if max(cw, ch) > 1200 else 50)
    zoom = args.zoom or max(1.0, min(3.0, 1500 / max(cw, ch)))

    out = cv2.cvtColor(cv2.resize(crop, (int(cw * zoom), int(ch * zoom)),
                                  interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
    oh, ow = out.shape[:2]

    # grid lines on canonical multiples of `step`
    for gx in range(int(np.ceil(x0 / step)) * step, x1 + 1, step):
        px = int((gx - x0) * zoom)
        cv2.line(out, (px, 0), (px, oh), (0, 90, 220), 1, cv2.LINE_AA)
        cv2.putText(out, str(gx), (px + 3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 90, 220), 1, cv2.LINE_AA)
    for gy in range(int(np.ceil(y0 / step)) * step, y1 + 1, step):
        py = int((gy - y0) * zoom)
        cv2.line(out, (0, py), (ow, py), (200, 60, 0), 1, cv2.LINE_AA)
        cv2.putText(out, str(gy), (3, py - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (200, 60, 0), 1, cv2.LINE_AA)

    OUT.mkdir(parents=True, exist_ok=True)
    dst = Path(args.out) if args.out else OUT / f"grid_{args.form}.png"
    cv2.imwrite(str(dst), out)
    print(f"{args.form}  region [{x0},{y0},{x1},{y1}]  step {step}  zoom {zoom:.2f}")
    try:
        shown = dst.resolve().relative_to(ROOT)
    except ValueError:
        shown = dst
    print(f"-> {shown}  ({ow}x{oh})")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
