#!/usr/bin/env python3
"""Builds the training set with eye patches framed by Apple Vision, the tracker the plug-in uses.

Patches and eye masks come from Vision's measurements (`visiondump` output) using exactly the math
in EyeContactCore/GazeNet.swift, so training and the plug-in see identical inputs. Iris positions
for the displacement labels come from MediaPipe, whose iris landmark is more precise; both are
expressed in the Vision patch frame. Also fits the mapping from Vision's iris offset to MediaPipe's
(used at runtime to measure gaze) and the eye-contact target, written to a JSON next to the output.

    python gazenet/prepare_vision.py data/columbia_vision.jsonl data/columbia_vision_patches.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyepatch import EYE_FRAC, EYES, PATCH_H, PATCH_W  # noqa: E402
from prepare_columbia import NAME, full_landmarks  # noqa: E402

MASK_GROW = 3.0  # patch pixels, matching the contour dilation used for MediaPipe masks


def lid(c, x):
    return (x * x - 0.25) * (c[0] + c[1] * x)


def corners(e, side):
    """Vision eye corners (inner, outer); side 0 is the image-left eye, whose inner corner is on
    the right."""
    mid, axis, w = np.array(e["mid"]), np.array(e["axis"]), e["width"]
    a, b = mid - axis * w / 2, mid + axis * w / 2  # image-left, image-right
    return (b, a) if side == 0 else (a, b)


def transform(inner, outer, side):
    """Frame -> patch affine (2x3), identical to EyePatchFrame in GazeNet.swift."""
    width = np.linalg.norm(outer - inner)
    d = (outer - inner) / width
    s = EYE_FRAC * PATCH_W / width
    R = s * (np.array([[d[0], d[1]], [d[1], -d[0]]]) if side == 0 else np.array([[d[0], d[1]], [-d[1], d[0]]]))
    t = np.array([PATCH_W / 2, PATCH_H / 2]) - R @ ((inner + outer) / 2)
    return np.hstack([R, t[:, None]]), s


def mask(e, M, s):
    """Eye-opening mask in patch space from Vision's lid curves, as EyePatchFrame.mask()."""
    Minv = cv2.invertAffineTransform(M)
    ys, xs = np.mgrid[0:PATCH_H, 0:PATCH_W].astype(np.float64)
    px = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
    py = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
    mid, axis, w = np.array(e["mid"]), np.array(e["axis"]), e["width"]
    normal = np.array([-axis[1], axis[0]])
    lx = ((px - mid[0]) * axis[0] + (py - mid[1]) * axis[1]) / w
    ly = ((px - mid[0]) * normal[0] + (py - mid[1]) * normal[1]) / w
    inside = np.minimum(np.minimum(ly - lid(e["upper"], lx), lid(e["lower"], lx) - ly), 0.5 - np.abs(lx))
    return (inside * w * s >= -MASK_GROW).astype(np.uint8) * 255


def offset(inner, outer, iris):
    """Iris offset from the corner midpoint: x toward image-right, y down, eye widths."""
    a, b = (inner, outer) if inner[0] < outer[0] else (outer, inner)
    w = np.linalg.norm(b - a)
    axis = (b - a) / w
    normal = np.array([-axis[1], axis[0]])
    d = np.asarray(iris) - (a + b) / 2
    return np.array([d @ axis, d @ normal]) / w


def process(item):
    path, veyes = item
    m = NAME.search(path)
    subject, pose, v, h = (int(g) for g in m.groups())
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lm = full_landmarks(rgb)
    if lm is None:
        return None
    out, raw_v, raw_mp = [], [], []
    for side in (0, 1):
        e = veyes[side]
        inner, outer = corners(e, side)
        M, s = transform(inner, outer, side)
        big = cv2.warpAffine(rgb, np.diag([2.0, 2.0]) @ M, (PATCH_W * 2, PATCH_H * 2),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        patch = cv2.resize(big, (PATCH_W, PATCH_H), interpolation=cv2.INTER_AREA)
        iris_mp = lm[EYES[side]["iris"][0]].astype(np.float64)
        to_patch = lambda p: M[:, :2] @ p + M[:, 2]
        out.append(dict(subject=subject, pose=pose, v=v, h=h, side=side, patch=patch, mask=mask(e, M, s),
                        iris=to_patch(iris_mp), iris_v=to_patch(np.array(e["iris"]))))
        raw_v.append(offset(inner, outer, e["iris"]))
        raw_mp.append(offset(inner, outer, iris_mp))
    return out, np.mean(raw_v, 0), np.mean(raw_mp, 0), (v, h, pose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vision_jsonl")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = []
    for line in open(args.vision_jsonl):
        r = json.loads(line)
        if len(r.get("eyes", [])) == 2 and NAME.search(r["path"]):
            items.append((r["path"], sorted(r["eyes"], key=lambda e: e["mid"][0])))
    print(f"{len(items)} images with two Vision eyes")

    records, rv, rm, labels = [], [], [], []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process, items, chunksize=4)):
            if res is not None:
                recs, a, b, lab = res
                records += recs
                rv.append(a); rm.append(b); labels.append(lab)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(items)}", flush=True)

    keys = ("subject", "pose", "v", "h", "side")
    np.savez_compressed(
        args.output,
        patches=np.stack([r["patch"] for r in records]),
        masks=np.stack([r["mask"] for r in records]),
        iris=np.stack([r["iris"] for r in records]).astype(np.float32),
        iris_v=np.stack([r["iris_v"] for r in records]).astype(np.float32),
        **{k: np.array([r[k] for r in records], np.int16) for k in keys},
    )
    print(f"wrote {args.output}: {len(records)} patches")

    # Runtime gaze measurement: Vision's iris offset mapped to MediaPipe's (same Vision frame).
    rv, rm, labels = np.array(rv), np.array(rm), np.array(labels)
    cal = {}
    for k, name in enumerate("xy"):
        a, b = np.polyfit(rv[:, k], rm[:, k], 1)
        r = np.corrcoef(rv[:, k], rm[:, k])[0, 1]
        cal[name] = dict(scale=float(a), bias=float(b), r=float(r))
        print(f"gaze {name}: mediapipe = {a:.3f} * vision + {b:+.4f} (r={r:.3f})")
    front = (labels[:, 0] == 0) & (labels[:, 1] == 0) & (labels[:, 2] == 0)
    target = np.median(rm[front], 0)
    cal["eye_contact_target"] = target.tolist()
    print(f"eye-contact target (frontal, looking at lens): {target.round(4)}")
    out_json = str(Path(args.output).with_suffix(".json"))
    json.dump(cal, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
