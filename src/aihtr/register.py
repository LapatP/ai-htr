"""Stage 2 — registration. Warp a normalized page onto its form template.

Once this succeeds, the page occupies the same coordinate system as the
template, so the fixed boxes in configs/forms/*.yaml become valid again. Every
later stage depends on this one being honest, which is why the result carries
its own diagnostics rather than just a matrix.

ORB rather than SIFT: free of patent baggage, fast, binary descriptors, and
plenty good on documents where the features are high-contrast printed corners
and rule intersections.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ORB_FEATURES = 5000
RATIO_TEST = 0.75      # Lowe's ratio; lower = stricter
RANSAC_THRESH = 4.0    # px in canonical space
MIN_INLIERS = 25       # below this we refuse to claim a match


@dataclass
class RegistrationResult:
    homography: np.ndarray | None
    n_keypoints_src: int
    n_keypoints_dst: int
    n_matches: int
    n_inliers: int
    reproj_error: float          # mean px error over inliers; inf if failed

    @property
    def inlier_ratio(self) -> float:
        return self.n_inliers / self.n_matches if self.n_matches else 0.0

    @property
    def ok(self) -> bool:
        return self.homography is not None and self.n_inliers >= MIN_INLIERS

    def __str__(self) -> str:
        if not self.ok:
            return f"FAILED (matches={self.n_matches}, inliers={self.n_inliers})"
        return (
            f"ok  inliers={self.n_inliers}/{self.n_matches} "
            f"({self.inlier_ratio:.0%})  reproj={self.reproj_error:.2f}px"
        )


def detect(gray: np.ndarray, n_features: int = ORB_FEATURES):
    orb = cv2.ORB_create(nfeatures=n_features)
    return orb.detectAndCompute(gray, None)


def match(des_src, des_dst) -> list:
    """Brute-force Hamming with Lowe's ratio test.

    The ratio test is what makes this work on documents: forms are full of
    near-identical repeated features (dotted rules, rows of checkboxes), and a
    descriptor whose best two matches are equally good is ambiguous by
    definition. Discarding those is what stops the homography drifting a row.
    """
    if des_src is None or des_dst is None:
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_src, des_dst, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)
    return good


def register(src_gray: np.ndarray, dst_gray: np.ndarray) -> RegistrationResult:
    """Find the homography mapping `src` onto `dst` (the template)."""
    kp_s, des_s = detect(src_gray)
    kp_d, des_d = detect(dst_gray)
    good = match(des_s, des_d)

    base = dict(
        n_keypoints_src=len(kp_s), n_keypoints_dst=len(kp_d), n_matches=len(good)
    )
    if len(good) < 4:
        return RegistrationResult(None, **base, n_inliers=0, reproj_error=float("inf"))

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts_s, pts_d, cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        return RegistrationResult(None, **base, n_inliers=0, reproj_error=float("inf"))

    inliers = mask.ravel().astype(bool)
    n_in = int(inliers.sum())
    if n_in == 0:
        return RegistrationResult(None, **base, n_inliers=0, reproj_error=float("inf"))

    # Mean residual over inliers only -- including outliers would make a good
    # fit look terrible and hide real degradation.
    proj = cv2.perspectiveTransform(pts_s[inliers], H)
    err = float(np.mean(np.linalg.norm(proj - pts_d[inliers], axis=2)))

    return RegistrationResult(H, **base, n_inliers=n_in, reproj_error=err)


def warp(gray: np.ndarray, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Apply a homography, filling uncovered area with paper-white."""
    return cv2.warpPerspective(
        gray, H, size, flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def overlay(warped: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Red/cyan overlay for eyeballing alignment.

    Template goes in red, warped page in green+blue. Where they agree the ink
    is near-black; where they disagree you get coloured ghosting. Misalignment
    is obvious at a glance -- far more informative than a single number.
    """
    return cv2.merge([warped, warped, template])
