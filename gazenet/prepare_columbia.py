#!/usr/bin/env python3
"""Extracts aligned eye patches from the Columbia Gaze Data Set.

Every image is named <subject>_2m_<pose>P_<vertical>V_<horizontal>H.jpg. For each image both eyes
are located with MediaPipe, cut out with a similarity transform anchored on the eye corners (so
all patches of a subject share the same framing), and the image-left eye is mirrored so a single
model handles both eyes. Output is one .npz with patches, labels and patch-space landmarks.

    python gazenet/prepare_columbia.py data/columbia_gaze_data_set.zip data/columbia_patches.npz
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyepatch import EYES, PATCH_H, PATCH_W, extract_patch  # noqa: E402

NAME = re.compile(r"(\d{4})_2m_(-?\d+)P_(-?\d+)V_(-?\d+)H\.jpg$", re.I)

_mesh = None


def _landmarks(rgb: np.ndarray):
    import mediapipe as mp
    global _mesh
    if _mesh is None:
        _mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1)
    r = _mesh.process(rgb)
    if not r.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in r.multi_face_landmarks[0].landmark], np.float32)


def full_landmarks(rgb: np.ndarray):
    """MediaPipe landmarks in full-image pixels for a large photo, or None."""
    # Coarse pass at low resolution to find the face, then a fine pass on a face crop so the
    # iris landmarks get enough pixels.
    s = 1600 / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lm = _landmarks(small)
    if lm is None:
        return None
    lm /= s
    x0, y0 = lm.min(0)
    x1, y1 = lm.max(0)
    pad = 0.35 * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1, y1 = int(min(rgb.shape[1], x1 + pad)), int(min(rgb.shape[0], y1 + pad))
    face = rgb[y0:y1, x0:x1]
    fs = min(1.0, 1280 / max(face.shape[:2]))
    face_s = cv2.resize(face, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA) if fs < 1 else face
    lm2 = _landmarks(face_s)
    if lm2 is not None:
        lm = lm2 / fs + (x0, y0)
    return lm


def process(path: str):
    m = NAME.search(path)
    if not m:
        return None
    subject, pose, v, h = (int(g) for g in m.groups())
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lm = full_landmarks(rgb)
    if lm is None:
        return None

    out = []
    for side in (0, 1):
        patch, lm_patch, width = extract_patch(rgb, lm, side)
        out.append(dict(subject=subject, pose=pose, v=v, h=h, side=side, patch=patch,
                        lm=lm_patch, eye_width=width))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="columbia_gaze_data_set.zip or the extracted folder")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src = Path(args.source)
    if src.suffix == ".zip":
        root = src.with_suffix("")
        if not root.exists():
            print(f"extracting to {root} ...")
            with zipfile.ZipFile(src) as z:
                z.extractall(root)
        src = root
    paths = sorted(str(p) for p in src.rglob("*.jpg") if NAME.search(p.name))
    print(f"{len(paths)} images")
    if not paths:
        sys.exit("no images matching <subject>_2m_<P>P_<V>V_<H>H.jpg found")

    records, failed = [], 0
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, paths, chunksize=4)):
            if r is None:
                failed += 1
            else:
                records += r
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(paths)} ({failed} without a face)", flush=True)

    keys = ("subject", "pose", "v", "h", "side")
    np.savez_compressed(
        args.output,
        patches=np.stack([r["patch"] for r in records]),
        lm=np.stack([r["lm"] for r in records]).astype(np.float32),
        eye_width=np.array([r["eye_width"] for r in records], np.float32),
        **{k: np.array([r[k] for r in records], np.int16) for k in keys},
    )
    print(f"wrote {args.output}: {len(records)} eye patches ({PATCH_W}x{PATCH_H}), {failed} images skipped")


if __name__ == "__main__":
    main()
