#!/usr/bin/env python3
"""Fits the correction from Apple Vision eye measurements (used by the FCP plug-in) to the
MediaPipe eye frame the gaze model was trained on.

Both trackers are run on the Columbia photos (Vision results come from `visiondump`). For each eye
the MediaPipe corners are expressed in the Vision eye frame, giving a mean offset and scale; then
gaze offsets measured from Vision's iris in the corrected frame are regressed onto MediaPipe's.

    python gazenet/calibrate_vision.py data/columbia_vision.jsonl data/vision_calibration.json
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
from eyepatch import EYES  # noqa: E402
from prepare_columbia import NAME, full_landmarks  # noqa: E402


def mp_eyes(path: str):
    bgr = cv2.imread(path)
    lm = None if bgr is None else full_landmarks(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if lm is None:
        return path, None
    return path, [dict(inner=lm[e["inner"]], outer=lm[e["outer"]], iris=lm[e["iris"][0]]) for e in EYES]


def frame(mid, axis, width):
    axis = np.asarray(axis, np.float64)
    return np.asarray(mid, np.float64), axis, np.array([-axis[1], axis[0]]), float(width)


def local(p, fr):
    mid, axis, normal, w = fr
    d = np.asarray(p, np.float64) - mid
    return np.array([d @ axis, d @ normal]) / w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vision_jsonl")
    ap.add_argument("output")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    vision = {}
    for line in open(args.vision_jsonl):
        r = json.loads(line)
        if "eyes" in r and len(r["eyes"]) == 2 and NAME.search(r["path"]):
            vision[r["path"]] = sorted(r["eyes"], key=lambda e: e["mid"][0])  # side 0 = image left
    print(f"{len(vision)} images with two Vision eyes")

    with Pool(args.workers) as pool:
        mp = dict(pool.imap_unordered(mp_eyes, sorted(vision), chunksize=8))

    # 1. Where MediaPipe's corners sit in Vision's eye frame (eye widths). Side 0 is mirrored so
    #    "inner" is always at negative x and the two eyes can be pooled.
    corners = {0: [], 1: []}
    rows = []
    for path, veyes in vision.items():
        meyes = mp.get(path)
        if meyes is None:
            continue
        v, h = (int(x) for x in NAME.search(path).groups()[2:4])
        pose = int(NAME.search(path).group(2))
        for side in (0, 1):
            ve, me = veyes[side], meyes[side]
            fr = frame(ve["mid"], ve["axis"], ve["width"])
            sx = -1 if side == 0 else 1
            ci, co = local(me["inner"], fr), local(me["outer"], fr)
            corners[side].append(np.r_[ci * (sx, 1), co * (sx, 1)])
        rows.append((path, veyes, meyes, v, h, pose))
    cal = {}
    for side in (0, 1):
        c = np.median(np.array(corners[side]), axis=0)
        cal[f"side{side}"] = dict(inner=c[:2].tolist(), outer=c[2:].tolist())
        print(f"side {side}: MediaPipe inner corner at {c[:2].round(3)}, outer at {c[2:].round(3)} "
              f"(Vision's are at (-0.5, 0) and (0.5, 0); x mirrored for side 0)")

    def corrected(ve, side):
        """MediaPipe-equivalent (inner, outer) corners from a Vision eye."""
        fr = frame(ve["mid"], ve["axis"], ve["width"])
        mid, axis, normal, w = fr
        sx = -1 if side == 0 else 1
        c = cal[f"side{side}"]
        to_img = lambda q: mid + (q[0] * sx * axis + q[1] * normal) * w
        return to_img(c["inner"]), to_img(c["outer"])

    def gaze(eyes_corners, irises):
        """Mean iris offset like eyecontact_lp.measure: frame from corners, x toward image-right."""
        offs = []
        for (inner, outer), iris in zip(eyes_corners, irises):
            a, b = (inner, outer) if inner[0] < outer[0] else (outer, inner)
            w = np.linalg.norm(b - a)
            axis = (b - a) / w
            normal = np.array([-axis[1], axis[0]])
            d = np.asarray(iris) - (a + b) / 2
            offs.append([d @ axis / w, d @ normal / w])
        return np.mean(offs, axis=0)

    # 2. Gaze offset from Vision's iris in the corrected frame vs MediaPipe's offset.
    gv, gm, labels = [], [], []
    for path, veyes, meyes, v, h, pose in rows:
        gv.append(gaze([corrected(veyes[s], s) for s in (0, 1)], [veyes[s]["iris"] for s in (0, 1)]))
        gm.append(gaze([(meyes[s]["inner"], meyes[s]["outer"]) for s in (0, 1)], [meyes[s]["iris"] for s in (0, 1)]))
        labels.append((v, h, pose))
    gv, gm, labels = np.array(gv), np.array(gm), np.array(labels)
    fit = {}
    for k, name in enumerate("xy"):
        a, b = np.polyfit(gv[:, k], gm[:, k], 1)
        resid = gm[:, k] - (a * gv[:, k] + b)
        corr = np.corrcoef(gv[:, k], gm[:, k])[0, 1]
        fit[name] = dict(scale=float(a), bias=float(b))
        print(f"gaze {name}: mediapipe = {a:.3f} * vision + {b:+.4f}  (r={corr:.3f}, residual std {resid.std():.4f})")
    cal["gaze"] = fit

    # 3. Sanity check: the eye-contact target in this convention (frontal, looking at the lens).
    m = (labels[:, 0] == 0) & (labels[:, 1] == 0) & (labels[:, 2] == 0)
    mapped = np.stack([gv[:, k] * fit[n]["scale"] + fit[n]["bias"] for k, n in enumerate("xy")], 1)
    print(f"looking at camera: MediaPipe offset {np.median(gm[m], 0).round(3)}, "
          f"mapped Vision offset {np.median(mapped[m], 0).round(3)} ({m.sum()} images)")
    cal["n_images"] = len(rows)
    json.dump(cal, open(args.output, "w"), indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
