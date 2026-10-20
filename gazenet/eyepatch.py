"""Eye patch geometry shared by data preparation, training and inference.

A patch is a PATCH_W x PATCH_H crop in which the eye corners sit on a horizontal line through the
patch center, EYE_FRAC of the patch width apart. The image-left eye (side 0) is mirrored so both
eyes appear in the same orientation, with the inner corner on the left.
"""
from __future__ import annotations

import cv2
import numpy as np

PATCH_W, PATCH_H = 144, 96
EYE_FRAC = 0.55

# MediaPipe face mesh indices; side 0 is the subject's right eye (image left).
EYES = [
    dict(outer=33, inner=133, iris=[468, 469, 470, 471, 472],
         contour=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
    dict(outer=263, inner=362, iris=[473, 474, 475, 476, 477],
         contour=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]),
]


def patch_transform(lm: np.ndarray, side: int) -> tuple[np.ndarray, float]:
    """2x3 affine from image pixels to patch pixels, and the eye width in image pixels."""
    e = EYES[side]
    inner, outer = lm[e["inner"]].astype(np.float64), lm[e["outer"]].astype(np.float64)
    width = float(np.linalg.norm(outer - inner))
    s = EYE_FRAC * PATCH_W / max(width, 1e-6)
    d = (outer - inner) / max(width, 1e-6)  # inner -> outer; must map to patch +x
    if side == 0:
        # Reflection: d -> +x while image-down stays patch-down (det < 0 mirrors the eye).
        R = s * np.array([[d[0], d[1]], [d[1], -d[0]]])
    else:
        # Rotation: d -> +x, normal (-d.y, d.x) -> +y.
        R = s * np.array([[d[0], d[1]], [-d[1], d[0]]])
    t = np.array([PATCH_W / 2, PATCH_H / 2]) - R @ ((inner + outer) / 2)
    return np.hstack([R, t[:, None]]), width


def transform_points(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ M[:, :2].T + M[:, 2]


def extract_patch(rgb: np.ndarray, lm: np.ndarray, side: int, supersample: int = 2):
    """Returns (patch uint8 HxWx3, eye landmarks in patch coordinates (21, 2), eye width px)."""
    M, width = patch_transform(lm, side)
    # Warp at a higher resolution and area-downsample, so large source eyes don't alias.
    S = np.diag([supersample, supersample]) @ M
    big = cv2.warpAffine(rgb, S, (PATCH_W * supersample, PATCH_H * supersample),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    patch = cv2.resize(big, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
    e = EYES[side]
    pts = transform_points(M, lm[e["contour"] + e["iris"]].astype(np.float64)).astype(np.float32)
    return patch, pts, width
