"""Calibrate the form-ID confidence threshold honestly.

check_formid.py showed correct matches winning by 52-278x -- but that is an
artefact: each template IS the page under test, so inliers saturate. A real
second scan would score far lower and the margin would shrink.

So we simulate a second scan. Every page is perturbed by a random homography
and degraded before identification. Then we compare two distributions:

  TRUE  : perturbed pages whose form HAS a template  -> margin when correct
  FALSE : perturbed pages with NO template           -> margin of the wrong claim

A usable threshold sits in the gap. If there is no gap, inlier margin alone
cannot separate these forms and we need overlay-region masking instead.

Usage:  PYTHONPATH=src python scripts/calibrate_margin.py [--trials 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aihtr.formid import FormIdentifier                      # noqa: E402
from aihtr.normalize import CANONICAL_SIZE, normalize_page   # noqa: E402
from aihtr.register import warp                              # noqa: E402
from aihtr.synth import degrade, random_homography           # noqa: E402

RAW = ROOT / "data" / "raw"
TEMPLATES = ROOT / "data" / "templates"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tmpls = {p.stem: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
             for p in sorted(TEMPLATES.glob("*.png"))}
    ident = FormIdentifier(tmpls)
    corpus = yaml.safe_load((ROOT / "configs" / "corpus.yaml").read_text(encoding="utf-8"))

    true_margins, false_margins, misid = [], [], 0
    print(f"perturbing each of {len(corpus['pages'])} pages x{args.trials}, "
          f"degraded, against {len(tmpls)} templates\n")

    for page in corpus["pages"]:
        src = RAW / page["file"]
        if not src.exists():
            continue
        norm = normalize_page(src)
        known = page["form"] in tmpls
        margins = []

        for _ in range(args.trials):
            pert = degrade(warp(norm.image, random_homography(CANONICAL_SIZE, rng),
                                CANONICAL_SIZE), rng)
            res = ident.identify(pert)
            if len(res.ranked) < 2 or res.ranked[0].n_inliers == 0:
                continue
            margin = res.ranked[0].n_inliers / max(res.ranked[1].n_inliers, 1)
            top = res.ranked[0].form_id
            if known:
                if top == page["form"]:
                    true_margins.append(margin)
                else:
                    misid += 1
            else:
                false_margins.append(margin)
            margins.append(margin)

        tag = "KNOWN" if known else "unknown"
        if margins:
            print(f"  {page['form'][:26]:<28} {tag:<8} "
                  f"margins {min(margins):.1f}x - {max(margins):.1f}x")

    t, f = np.array(true_margins), np.array(false_margins)
    print("\n" + "=" * 62)
    if len(t):
        print(f"TRUE  (correct form present)  n={len(t):<4} "
              f"min={t.min():6.1f}x  median={np.median(t):6.1f}x")
    if len(f):
        print(f"FALSE (no template exists)    n={len(f):<4} "
              f"max={f.max():6.1f}x  median={np.median(f):6.1f}x")
    print(f"top-1 misidentifications among known forms: {misid}")

    if len(t) and len(f):
        if t.min() > f.max():
            lo, hi = f.max(), t.min()
            print(f"\nSEPARABLE. gap {lo:.1f}x .. {hi:.1f}x")
            print(f"suggested MARGIN_RATIO = {np.sqrt(lo * hi):.1f}  "
                  f"(geometric mean of the gap)")
        else:
            print(f"\nOVERLAP: worst true {t.min():.1f}x <= best false {f.max():.1f}x.")
            print("Inlier margin alone cannot separate these forms -- the shared")
            print("base-card layout dominates. Mask to the overlay region instead.")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
