"""Stage 1 — form identification. The keystone of the whole pipeline.

Which of the known templates is this page? We answer it by attempting
registration against every template and keeping whichever yields the most
RANSAC inliers.

Why this and not a trained classifier:
  * no training data and no labels needed;
  * it hands stage 2 its homography for free, as a by-product;
  * it can say "I don't know". A softmax classifier always names one of its
    classes, however wrong. Given the hospital's form family runs to dozens of
    variants we will never all have, honest abstention matters more than a
    fractionally better hit rate.

Cost is linear in template count -- fine for tens of forms, revisit near a
hundred (index the descriptors, or prefilter on coarse layout).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .register import MIN_INLIERS, RegistrationResult, detect, match
import cv2

# Best must beat the runner-up by this factor to be called confident.
#
# Calibrated empirically, not guessed -- see scripts/calibrate_margin.py.
# Perturbing and degrading all 12 pages against the 4 templates gave:
#     correct-form margins   14.3x .. 152x  (median 28.5x)
#     no-template margins     1.0x ..  3.1x (median  1.4x)
# 6.5 sits in the middle of that gap, ~2x clear of either side.
#
# The initial guess of 1.5x was far too low: seven of the twelve pages share the
# identical base OPD card background, so a wrong form still earns real inliers
# from shared printed furniture. Discrimination comes from the overlay content,
# and the margin has to be wide enough to ignore the shared layer.
MARGIN_RATIO = 6.5


@dataclass
class FormScore:
    form_id: str
    result: RegistrationResult

    @property
    def n_inliers(self) -> int:
        return self.result.n_inliers


@dataclass
class Identification:
    ranked: list[FormScore]
    confident: bool
    reason: str

    @property
    def best(self) -> FormScore | None:
        return self.ranked[0] if self.ranked else None

    @property
    def form_id(self) -> str | None:
        """The identified form, or None when we decline to guess."""
        return self.best.form_id if (self.confident and self.best) else None

    def __str__(self) -> str:
        if not self.ranked:
            return "no templates to compare against"
        head = "  ".join(f"{s.form_id}={s.n_inliers}" for s in self.ranked[:4])
        verdict = self.form_id or "UNKNOWN"
        return f"{verdict:<22} [{self.reason}]  {head}"


class FormIdentifier:
    """Holds templates with their descriptors pre-computed.

    Detecting features on every template for every page would dominate runtime,
    so we do it once at construction.
    """

    def __init__(self, templates: dict[str, np.ndarray]):
        self.templates = templates
        self._features = {fid: detect(img) for fid, img in templates.items()}

    def identify(self, page_gray: np.ndarray) -> Identification:
        kp_p, des_p = detect(page_gray)
        scores: list[FormScore] = []

        for fid, (kp_t, des_t) in self._features.items():
            good = match(des_p, des_t)
            base = dict(
                n_keypoints_src=len(kp_p), n_keypoints_dst=len(kp_t),
                n_matches=len(good),
            )
            if len(good) < 4:
                res = RegistrationResult(None, **base, n_inliers=0,
                                         reproj_error=float("inf"))
            else:
                pts_p = np.float32([kp_p[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                pts_t = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(pts_p, pts_t, cv2.RANSAC, 4.0)
                if H is None:
                    res = RegistrationResult(None, **base, n_inliers=0,
                                             reproj_error=float("inf"))
                else:
                    inl = mask.ravel().astype(bool)
                    n_in = int(inl.sum())
                    if n_in:
                        proj = cv2.perspectiveTransform(pts_p[inl], H)
                        err = float(np.mean(np.linalg.norm(proj - pts_t[inl], axis=2)))
                    else:
                        err = float("inf")
                    res = RegistrationResult(H, **base, n_inliers=n_in,
                                             reproj_error=err)
            scores.append(FormScore(fid, res))

        scores.sort(key=lambda s: s.n_inliers, reverse=True)

        if not scores or scores[0].n_inliers < MIN_INLIERS:
            got = scores[0].n_inliers if scores else 0
            return Identification(scores, False, f"too few inliers ({got}<{MIN_INLIERS})")

        if len(scores) > 1:
            runner_up = max(scores[1].n_inliers, 1)
            margin = scores[0].n_inliers / runner_up
            if margin < MARGIN_RATIO:
                return Identification(scores, False, f"ambiguous (margin {margin:.2f}x)")
            return Identification(scores, True, f"margin {margin:.1f}x")

        return Identification(scores, True, "only template")
