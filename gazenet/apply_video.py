#!/usr/bin/env python3
"""Applies a trained GazeWarpNet to a video.

Tracking and gaze smoothing are the same as python/eyecontact_lp.py. For each frame and eye, the
network predicts a warp field on the eye patch; the field is mapped back into the frame and used
to resample the original full-resolution, high-bit-depth pixels, so the eyes stay as sharp as the
source.

    python gazenet/apply_video.py runs/gazenet/best.pt input.mov output.mov [--strength 1]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import eyecontact_lp as V  # noqa: E402  (video I/O, tracking, smoothing)
from eyepatch import EYES, PATCH_H, PATCH_W, extract_patch, patch_transform  # noqa: E402
from model import load_checkpoint  # noqa: E402
from train import eye_masks  # noqa: E402


@torch.no_grad()
def predict_fields(model, patches: np.ndarray, lms: np.ndarray, disp: np.ndarray, device: str):
    """patches (B,H,W,3) uint8, lms (B,21,2), disp (B,2) in patch orientation -> numpy fields at half res."""
    img = torch.from_numpy(patches).permute(0, 3, 1, 2).float().div(255).to(device)
    mask = torch.from_numpy(eye_masks(lms)).unsqueeze(1).float().div(255).to(device)
    cond = torch.from_numpy(disp).float().to(device)
    flow, alpha, color = model.fields(img, mask, cond)
    return flow.cpu().numpy(), alpha.cpu().numpy(), color.cpu().numpy()


TRAINED_RANGE = np.array([0.14, 0.05], np.float32)  # |dx|, |dy| in eye widths


def warp_eye(frame: np.ndarray, M: np.ndarray, flow, alpha, color, eye_w: float):
    """Applies one eye's fields to `frame` (float32 RGB 0..1) in place."""
    h, w = frame.shape[:2]
    Minv = cv2.invertAffineTransform(M)
    # Frame-space bounding box of the patch.
    corners = np.array([[0, 0], [PATCH_W, 0], [0, PATCH_H], [PATCH_W, PATCH_H]], np.float64)
    pts = corners @ Minv[:, :2].T + Minv[:, 2]
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # Frame pixel centers -> patch pixel coordinates.
    px = M[0, 0] * xs + M[0, 1] * ys + M[0, 2]
    py = M[1, 0] * xs + M[1, 1] * ys + M[1, 2]
    # Fields live at half resolution with align_corners=False sampling semantics.
    fh, fw = flow.shape[1:]
    fx = (px + 0.5) * fw / PATCH_W - 0.5
    fy = (py + 0.5) * fh / PATCH_H - 0.5
    sample = lambda ch: cv2.remap(ch.astype(np.float32), fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Flow is in normalized patch coordinates (align_corners=True): [-1, 1] spans 0..W-1.
    dpx = sample(flow[0]) * (PATCH_W - 1) / 2
    dpy = sample(flow[1]) * (PATCH_H - 1) / 2
    a = sample(alpha[0])[..., None]
    col = np.stack([sample(color[c]) for c in range(3)], -1)
    # Source position in patch space -> frame space.
    sx = Minv[0, 0] * (px + dpx) + Minv[0, 1] * (py + dpy) + Minv[0, 2]
    sy = Minv[1, 0] * (px + dpx) + Minv[1, 1] * (py + dpy) + Minv[1, 2]
    warped = cv2.remap(frame, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT)
    # Fade to the original toward the patch border so nothing outside the patch can change.
    edge = np.minimum.reduce([px, PATCH_W - 1 - px, py, PATCH_H - 1 - py])
    win = np.clip(edge / (0.12 * PATCH_H), 0, 1)[..., None]
    roi = frame[y0:y1, x0:x1]
    out = (1 - a) * warped + a * col
    frame[y0:y1, x0:x1] = roi * (1 - win) + out * win


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--target-x", type=float, default=0.0)
    ap.add_argument("--target-y", type=float, default=-0.07,
                    help="vertical eye-contact position, eye widths, + is down. Default is where Columbia Gaze subjects' irises sit when looking into the lens (above the corner line)")
    ap.add_argument("--max", type=float, default=0.3)
    ap.add_argument("--vertical", type=float, default=1.0, help="scale for vertical correction (0 = horizontal only)")
    ap.add_argument("--smooth", type=float, default=2.0, help="temporal smoothing sigma, frames")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_checkpoint(args.checkpoint, device)
    vi = V.probe(args.input)
    print(f"{args.input}: {vi.width}x{vi.height} @ {vi.fps}, matrix {vi.matrix}")

    # ---- pass 1: track and smooth (zero-phase) -- this is what removes jitter
    t0 = time.time()
    mesh = V.make_face_mesh(static=False)
    lms = []
    for i, rgb in enumerate(V.read_frames(args.input, vi, "rgb24")):
        if args.frames and i >= args.frames:
            break
        lms.append(V.landmarks(mesh, rgb))
    mesh.close()
    n = len(lms)
    valid = np.array([l is not None for l in lms])
    if not valid.any():
        sys.exit("no face found")
    L = np.stack([l if l is not None else np.zeros((478, 2), np.float32) for l in lms])
    gaze = np.zeros((n, 2), np.float32)
    openness = np.zeros(n, np.float32)
    yaw = np.zeros(n, np.float32)
    for i in np.nonzero(valid)[0]:
        gaze[i], openness[i], yaw[i] = V.measure(L[i])
    open_ = valid & (openness > 0.18)
    L_s = V.smooth_series(L, valid, args.smooth)
    gaze_s = V.smooth_series(gaze, open_ if open_.any() else valid, args.smooth * 1.5)
    open_s = V.smooth_series(openness, valid, args.smooth)
    yaw_s = V.smooth_series(yaw, valid, args.smooth)

    error = gaze_s - np.array([args.target_x, args.target_y], np.float32)
    amount = args.strength * np.ones(n, np.float32)
    amount *= 1 - V.smoothstep(args.max, args.max * 1.4, np.linalg.norm(error, axis=1))
    amount *= V.smoothstep(0.12, 0.2, open_s)
    amount *= 1 - V.smoothstep(0.25, 0.4, np.abs(yaw_s))
    amount *= valid
    delta = -error * amount[:, None]  # desired iris move, eye widths, image orientation
    delta[:, 1] *= args.vertical
    # Soft-limit to the range the model was trained on (95th percentile of Columbia pair
    # displacements); beyond it the network extrapolates, e.g. by closing the lids.
    delta = TRAINED_RANGE * np.tanh(delta / TRAINED_RANGE)
    print(f"tracked {valid.sum()}/{n} frames in {time.time() - t0:.1f}s")

    # ---- pass 2: warp
    t0 = time.time()
    writer = V.open_writer(args.output, args.input, vi)
    check = V.make_face_mesh(static=False)
    before, after = [], []
    for i, rgb16 in enumerate(V.read_frames(args.input, vi, "rgb48le")):
        if i >= n:
            break
        frame = rgb16.astype(np.float32) / 65535.0
        if valid[i] and np.abs(delta[i]).max() > 0.002:
            rgb8 = (frame * 255 + 0.5).astype(np.uint8)
            patches, plms, disps, Ms, widths = [], [], [], [], []
            for side in (0, 1):
                p, pl, w = extract_patch(rgb8, L_s[i], side)
                M, _ = patch_transform(L_s[i], side)
                # Side 0 patches are mirrored, so horizontal motion flips sign.
                d = delta[i] * (-1 if side == 0 else 1, 1)
                patches.append(p); plms.append(pl); disps.append(d.astype(np.float32)); Ms.append(M); widths.append(w)
            flow, alpha, color = predict_fields(model, np.stack(patches), np.stack(plms), np.stack(disps), device)
            for k in range(2):
                warp_eye(frame, Ms[k], flow[k], alpha[k], color[k], widths[k])
        out8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        l_out = V.landmarks(check, out8)
        if open_[i] and amount[i] > 0.5 and l_out is not None:
            tgt = (args.target_x, args.target_y)
            before.append(np.linalg.norm(gaze[i] - tgt))
            after.append(np.linalg.norm(V.measure(l_out)[0] - tgt))
        writer.stdin.write((np.clip(frame, 0, 1) * 65535 + 0.5).astype("<u2").tobytes())
        if (i + 1) % 100 == 0:
            print(f"  frame {i + 1}/{n} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/frame)", flush=True)
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("ffmpeg failed")
    check.close()
    print(f"wrote {args.output} in {time.time() - t0:.1f}s")
    if before:
        print(f"distance from target (eye widths): median {np.median(before):.3f} -> {np.median(after):.3f}, "
              f"90th pct {np.percentile(before, 90):.3f} -> {np.percentile(after, 90):.3f}")


if __name__ == "__main__":
    main()
