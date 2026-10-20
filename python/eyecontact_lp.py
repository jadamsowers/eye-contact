#!/usr/bin/env python3
"""Offline eye-contact correction with LivePortrait.

Pass 1 tracks every frame with MediaPipe (iris landmarks) and smooths the measurements with a
zero-phase filter. A short calibration measures how far LivePortrait's pupil controls move the
iris on this face. Pass 2 re-renders the face crop of each frame that needs correction with the
pupils redirected, and composites only the eye region back into the original high-bit-depth frame.

    python python/eyecontact_lp.py input.mov output.mov [--strength 1.0] [--debug]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
LP_DIR = ROOT.parent / "LivePortrait"
sys.path.insert(0, str(LP_DIR))
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# MediaPipe face mesh indices (subject's right eye = image left).
EYES = [
    dict(outer=33, inner=133, upper=159, lower=145, iris=468,
         contour=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]),
    dict(outer=263, inner=362, upper=386, lower=374, iris=473,
         contour=[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]),
]
NOSE, MOUTH_L, MOUTH_R, FACE_L, FACE_R = 1, 61, 291, 234, 454
# Implicit-keypoint coordinates that move the irises, found by perturbing each of LivePortrait's
# 21 keypoints and measuring both irises: 11.x / 15.x shift the image-left / image-right iris
# sideways; vertical gaze is spread thinly over 11.y, 15.y and 13.y.
CONTROLS = [(11, 0), (15, 0), (11, 1), (15, 1), (13, 1)]
CONTROL_LIMIT = 0.04  # canonical keypoint units; beyond this the eyes tear


# ----------------------------------------------------------------------------- video I/O

@dataclass
class VideoInfo:
    width: int
    height: int
    fps: str  # rational, e.g. "30000/1001"
    frames: int
    has_audio: bool
    matrix: str  # swscale colour matrix name
    tags: dict


def probe(path: str) -> VideoInfo:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         check=True, capture_output=True, text=True).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    w, h = int(v["width"]), int(v["height"])
    rotation = 0
    for sd in v.get("side_data_list", []):
        rotation = int(sd.get("rotation", rotation))
    if abs(rotation) in (90, 270):  # ffmpeg auto-rotates when decoding
        w, h = h, w
    frames = int(v.get("nb_frames", 0) or 0)
    if not frames:
        dur = float(v.get("duration", info["format"].get("duration", 0)) or 0)
        num, den = (int(x) for x in v["r_frame_rate"].split("/"))
        frames = int(round(dur * num / den))
    cs = v.get("color_space", "")
    matrix = {"bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601",
              "bt2020nc": "bt2020", "bt2020c": "bt2020"}.get(cs, "bt709" if h >= 720 else "bt601")
    tags = {k: v[k] for k in ("color_primaries", "color_transfer", "color_space", "color_range") if k in v}
    return VideoInfo(w, h, v["r_frame_rate"], frames,
                     any(s["codec_type"] == "audio" for s in info["streams"]), matrix, tags)


def read_frames(path: str, vi: VideoInfo, pix_fmt: str):
    """Yields frames as numpy arrays (rgb24 -> uint8, rgb48le -> uint16)."""
    bpc = 2 if pix_fmt == "rgb48le" else 1
    size = vi.width * vi.height * 3 * bpc
    cmd = ["ffmpeg", "-v", "fatal", "-i", path,
           "-vf", f"scale=in_color_matrix={vi.matrix}:in_range=tv",
           "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=size * 2)
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint16 if bpc == 2 else np.uint8).reshape(vi.height, vi.width, 3)
    finally:
        proc.kill()  # stopping early (--frames) is normal; don't let ffmpeg complain
        proc.stdout.close()
        proc.wait()


def open_writer(path: str, src: str, vi: VideoInfo):
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{vi.width}x{vi.height}", "-r", vi.fps, "-i", "-",
           "-i", src, "-map", "0:v", "-map", "1:a?",
           # Same matrix and range as decoding, so untouched pixels round-trip.
           "-vf", f"scale=out_color_matrix={vi.matrix}:out_range=tv",
           "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
           "-c:a", "copy"]
    for k, flag in (("color_primaries", "-color_primaries"), ("color_transfer", "-color_trc"),
                    ("color_space", "-colorspace")):
        if k in vi.tags:
            cmd += [flag, vi.tags[k]]
    cmd.append(path)
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# ----------------------------------------------------------------------------- tracking

def make_face_mesh(static: bool):
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=static, refine_landmarks=True, max_num_faces=1,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)


def landmarks(mesh, rgb: np.ndarray) -> np.ndarray | None:
    r = mesh.process(rgb)
    if not r.multi_face_landmarks:
        return None
    h, w = rgb.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in r.multi_face_landmarks[0].landmark], np.float32)


def eye_frame(lm: np.ndarray, e: dict):
    """Eye-local frame: midpoint between corners, unit axis toward image-right, normal pointing down."""
    a, b = lm[e["outer"]], lm[e["inner"]]
    if a[0] > b[0]:
        a, b = b, a
    width = float(np.linalg.norm(b - a))
    axis = (b - a) / max(width, 1e-6)
    normal = np.array([-axis[1], axis[0]], np.float32)
    return (a + b) / 2, axis, normal, width


def per_eye_offsets(lm: np.ndarray) -> np.ndarray:
    """[dx_A, dy_A, dx_B, dy_B]: each iris relative to its eye, in eye widths."""
    out = []
    for e in EYES:
        mid, axis, normal, width = eye_frame(lm, e)
        d = lm[e["iris"]] - mid
        out += [d @ axis / width, d @ normal / width]
    return np.array(out, np.float32)


def measure(lm: np.ndarray):
    """Returns (gaze offset (2,) in eye widths, min openness, yaw ratio)."""
    offsets, opens = [], []
    for e in EYES:
        mid, axis, normal, width = eye_frame(lm, e)
        d = lm[e["iris"]] - mid
        offsets.append([d @ axis / width, d @ normal / width])
        opens.append(np.linalg.norm(lm[e["lower"]] - lm[e["upper"]]) / width)
    l, r, n = lm[FACE_L], lm[FACE_R], lm[NOSE]
    face_w = max(np.linalg.norm(r - l), 1e-6)
    yaw = (np.linalg.norm(n - l) - np.linalg.norm(n - r)) / face_w
    return np.mean(offsets, axis=0), min(opens), yaw


def smooth_series(x: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 that ignores invalid samples (normalized convolution)."""
    if sigma <= 0:
        return x.copy()
    radius = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    flat = x.reshape(len(x), -1)
    w = valid.astype(np.float64)
    num = np.stack([np.convolve(np.where(valid, flat[:, i], 0), k, "same") for i in range(flat.shape[1])], 1)
    den = np.convolve(w, k, "same")[:, None]
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), flat)
    return out.reshape(x.shape).astype(x.dtype)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- LivePortrait

class Renderer:
    def __init__(self, fp16: bool):
        import torch
        import mps_conv3d
        import mps_grid_sample
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper
        from src.utils.camera import get_rotation_matrix
        from src.utils.crop import crop_image

        self.torch = torch
        self.crop_image = crop_image
        self.get_rotation_matrix = get_rotation_matrix
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.lp = LivePortraitWrapper(InferenceConfig())
        self.mps = self.lp.device == "mps"
        if self.mps:
            # Keep LivePortrait entirely on the GPU (see the two modules for why).
            mps_grid_sample.install()
            mps_conv3d.install_interpolate()
            mps_conv3d.convert(self.lp.warping_module)
            mps_conv3d.convert_pools(self.lp.warping_module)
        self.fp16 = fp16
        if fp16:
            self.lp.warping_module.half()
            self.lp.spade_generator.half()

    def crop(self, rgb8: np.ndarray, lm: np.ndarray):
        eyes = [(lm[e["outer"]] + lm[e["inner"]]) / 2 for e in EYES]
        pts5 = np.array([eyes[0], eyes[1], lm[NOSE], lm[MOUTH_L], lm[MOUTH_R]], np.float32)
        # Same framing LivePortrait uses for source images.
        return self.crop_image(rgb8, pts5, dsize=512, scale=2.3, vx_ratio=0, vy_ratio=-0.125, flag_do_rot=True)

    def render(self, crop512: np.ndarray, controls: list[np.ndarray]) -> list[np.ndarray]:
        """Renders the crop once per control vector (see CONTROLS). Returns 512x512 uint8 RGB images."""
        torch, lp = self.torch, self.lp
        x = lp.prepare_source(cv2.resize(crop512, (256, 256), interpolation=cv2.INTER_AREA))
        with torch.no_grad():
            info = lp.get_kp_info(x)
            f_s = lp.extract_feature_3d(x)
            x_s = lp.transform_keypoint(info)
            rot = self.get_rotation_matrix(info["pitch"], info["yaw"], info["roll"])
            outs = []
            for c in controls:
                delta = torch.zeros_like(info["exp"])
                for (k, ax), v in zip(CONTROLS, c):
                    delta[0, k, ax] += float(v)
                # Edits live in the canonical (unrotated) face, so they follow head pose.
                x_d = x_s + info["scale"][..., None] * (delta @ rot)
                x_d = lp.stitching(x_s, x_d)
                if self.fp16:
                    out = lp.warp_decode(f_s.half(), x_s.half(), x_d.half())["out"].float()
                else:
                    out = lp.warp_decode(f_s, x_s, x_d)["out"]
                outs.append(lp.parse_output(out)[0])
        return outs


# ----------------------------------------------------------------------------- calibration

def calibrate(renderer: Renderer, frames: list[tuple[np.ndarray, np.ndarray]], step=0.012):
    """Finite-difference Jacobian (4 x len(CONTROLS)) of per-eye iris offsets with respect to the
    controls, measured on LivePortrait's own renders of this face. Median over sample frames."""
    mesh = make_face_mesh(static=True)
    k = len(CONTROLS)
    tests = [np.zeros(k)] + [s_ * step * np.eye(k)[j] for j in range(k) for s_ in (1, -1)]
    jacobians = []
    for rgb8, lm in frames:
        crop = renderer.crop(rgb8, lm)["img_crop"]
        offs = []
        for img in renderer.render(crop, tests):
            l = landmarks(mesh, img)
            if l is None:
                break
            offs.append(per_eye_offsets(l))
        if len(offs) != len(tests):
            continue
        # Central differences.
        J = np.stack([(offs[1 + 2 * j] - offs[2 + 2 * j]) / (2 * step) for j in range(k)], axis=1)
        jacobians.append(J)
    mesh.close()
    return np.median(jacobians, axis=0) if jacobians else None


class GazeSolver:
    """Per-frame closed loop: render, measure where the irises actually went, correct, re-render.

    How strongly each control moves each iris depends on the face and pose, so the Jacobian
    from calibration is only a starting point. It's refined with Broyden updates from every
    render and carried from frame to frame (relaxed back toward the calibration), which also
    keeps consecutive frames' solutions consistent.
    """

    def __init__(self, renderer: Renderer, J0: np.ndarray, iterations: int, tol: float = 0.006):
        self.r, self.J0, self.J = renderer, J0, J0.copy()
        self.iterations, self.tol = iterations, tol
        self.mesh = make_face_mesh(static=True)

    def close(self):
        self.mesh.close()

    def _broyden(self, dc: np.ndarray, do: np.ndarray):
        if dc @ dc > 1e-10:
            self.J += np.outer(do - self.J @ dc, dc) / (dc @ dc)

    def __call__(self, crop512: np.ndarray, goal_delta: np.ndarray):
        """Returns (rendered image, residual eye widths) for moving both irises by `goal_delta` (2,)."""
        l0 = landmarks(self.mesh, crop512)
        target = np.tile(goal_delta, 2)
        c = solve_controls(self.J, target)
        if l0 is None:
            return self.r.render(crop512, [c])[0], float("nan")
        goal = per_eye_offsets(l0) + target
        # The unedited reconstruction stands in for the c = 0 sample (saves a render).
        prev_c, prev_o = np.zeros_like(c), per_eye_offsets(l0)
        best = None
        for _ in range(max(1, self.iterations)):
            img = self.r.render(crop512, [c])[0]
            l = landmarks(self.mesh, img)
            if l is None:
                best = best or (img, float("nan"))
                break
            o = per_eye_offsets(l)
            resid = float(np.abs(goal - o).max())
            if best is None or not resid >= best[1]:
                best = (img, resid)
            if resid < self.tol:
                break
            self._broyden(c - prev_c, o - prev_o)
            prev_c, prev_o = c, o
            c = np.clip(c + solve_controls(self.J, goal - o), -CONTROL_LIMIT, CONTROL_LIMIT)
        # Drift back toward the calibration so one bad measurement can't poison later frames.
        self.J = 0.7 * self.J + 0.3 * self.J0
        return best


def solve_controls(J: np.ndarray, target: np.ndarray, ridge: float = 0.02) -> np.ndarray:
    """Least-squares control vector moving the irises by `target` ([dxA, dyA, dxB, dyB])."""
    k = J.shape[1]
    scale = np.abs(J).max()
    c = np.linalg.solve(J.T @ J + (ridge * scale) ** 2 * np.eye(k), J.T @ target)
    return np.clip(c, -CONTROL_LIMIT, CONTROL_LIMIT)


# ----------------------------------------------------------------------------- compositing

def eye_mask(lm: np.ndarray, shape, grow: float, feather: float):
    """Soft mask (float32 HxW) over both eye openings, grown by `grow` eye widths."""
    mask = np.zeros(shape[:2], np.float32)
    widths = []
    for e in EYES:
        _, _, _, width = eye_frame(lm, e)
        widths.append(width)
        cv2.fillPoly(mask, [np.round(lm[e["contour"]]).astype(np.int32)], 1.0)
    w = float(np.mean(widths))
    if grow > 0:
        r = max(1, int(round(grow * w)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), max(0.5, feather * w))
    return mask, w


def composite(frame: np.ndarray, gen512: np.ndarray, M_c2o: np.ndarray, lm: np.ndarray,
              grow: float, feather: float) -> np.ndarray:
    """Pastes the eye region of the generated crop into `frame` (float32 RGB 0..1, modified copy)."""
    h, w = frame.shape[:2]
    mask, eye_w = eye_mask(lm, frame.shape, grow, feather)
    ys, xs = np.nonzero(mask > 1e-3)
    if len(xs) == 0:
        return frame
    pad = int(eye_w * 0.5)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)

    # Crop -> ROI transform: M_c2o followed by a translation to the ROI origin.
    M = M_c2o[:2].astype(np.float64).copy()
    M[:, 2] -= (x0, y0)
    gen = cv2.warpAffine(gen512.astype(np.float32) / 255.0, M, (x1 - x0, y1 - y0),
                         flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    gen = np.clip(gen, 0, 1)
    roi = frame[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1, None]

    # LivePortrait's reconstruction drifts slightly in colour; match it to the original on a
    # ring of skin around the eyes before blending.
    ring, _ = eye_mask(lm, frame.shape, grow + 0.25, 0)
    ring = (ring[y0:y1, x0:x1] > 0.5) & (m[..., 0] < 0.05)
    if ring.sum() > 20:
        gen = np.clip(gen + (roi[ring].mean(0) - gen[ring].mean(0)), 0, 1)

    out = frame.copy()
    out[y0:y1, x0:x1] = roi * (1 - m) + gen * m
    return out


def draw_debug(img8: np.ndarray, lm: np.ndarray, text: str):
    for e in EYES:
        cv2.polylines(img8, [np.round(lm[e["contour"]]).astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        c = tuple(int(v) for v in np.round(lm[e["iris"]]))
        cv2.circle(img8, c, 2, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.putText(img8, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output", help="ProRes 422 HQ .mov")
    ap.add_argument("--strength", type=float, default=1.0, help="0..1 amount of correction (default 1)")
    ap.add_argument("--target-x", type=float, default=0.0, help='"eye contact" iris position, eye widths (default 0)')
    ap.add_argument("--target-y", type=float, default=-0.07,
                    help="vertical eye-contact position, eye widths, + is down. Default is where Columbia Gaze subjects' irises sit when looking into the lens (above the corner line)")
    ap.add_argument("--max", type=float, default=0.3,
                    help="gaze offsets beyond this (eye widths) are treated as look-aways and left alone (default 0.3)")
    ap.add_argument("--smooth", type=float, default=2.0, help="temporal smoothing sigma in frames (default 2)")
    ap.add_argument("--grow", type=float, default=0.12, help="mask growth around the eye openings, eye widths")
    ap.add_argument("--feather", type=float, default=0.06, help="mask feather, eye widths")
    ap.add_argument("--iterations", type=int, default=3,
                    help="renders per frame for the closed-loop gaze solve (1 = open loop, default 3)")
    ap.add_argument("--fp32", action="store_true", help="full precision (slower, marginally cleaner)")
    ap.add_argument("--debug", action="store_true", help="draw tracking and per-frame values")
    ap.add_argument("--frames", type=int, default=0, help="only process the first N frames")
    args = ap.parse_args()

    vi = probe(args.input)
    total = min(vi.frames, args.frames) if args.frames else vi.frames
    print(f"{args.input}: {vi.width}x{vi.height} @ {vi.fps}, ~{total} frames, matrix {vi.matrix}")

    # ---- pass 1: track
    t0 = time.time()
    mesh = make_face_mesh(static=False)
    lms, samples = [], []
    for i, rgb in enumerate(read_frames(args.input, vi, "rgb24")):
        if args.frames and i >= args.frames:
            break
        lm = landmarks(mesh, rgb)
        lms.append(lm)
        if lm is not None and i % max(1, total // 12) == 0:
            samples.append((rgb.copy(), lm))
    mesh.close()
    n = len(lms)
    valid = np.array([l is not None for l in lms])
    print(f"tracked {valid.sum()}/{n} frames in {time.time() - t0:.1f}s")
    if not valid.any():
        sys.exit("no face found")

    L = np.stack([l if l is not None else np.zeros((478, 2), np.float32) for l in lms])
    gaze = np.zeros((n, 2), np.float32)
    openness = np.zeros(n, np.float32)
    yaw = np.zeros(n, np.float32)
    for i in np.nonzero(valid)[0]:
        gaze[i], openness[i], yaw[i] = measure(L[i])

    # Blinks make the iris landmark unreliable: smooth gaze only over open-eye frames.
    open_ = valid & (openness > 0.18)
    L_s = smooth_series(L, valid, args.smooth)
    gaze_s = smooth_series(gaze, open_ if open_.any() else valid, args.smooth * 1.5)
    open_s = smooth_series(openness, valid, args.smooth)
    yaw_s = smooth_series(yaw, valid, args.smooth)

    error = gaze_s - np.array([args.target_x, args.target_y], np.float32)
    err_len = np.linalg.norm(error, axis=1)
    amount = args.strength * np.ones(n, np.float32)
    amount *= 1 - smoothstep(args.max, args.max * 1.4, err_len)       # intentional look-aways
    amount *= smoothstep(0.12, 0.2, open_s)                            # blinks
    amount *= 1 - smoothstep(0.25, 0.4, np.abs(yaw_s))                 # turned heads
    amount *= valid
    delta = -error * amount[:, None]                                   # desired gaze change
    need = np.linalg.norm(delta, axis=1) > 0.004

    print(f"gaze offset (eye widths): median |x| {np.median(np.abs(gaze_s[valid, 0])):.3f}, "
          f"median |y| {np.median(np.abs(gaze_s[valid, 1])):.3f}; {need.sum()} frames need correction")

    # ---- calibration
    t0 = time.time()
    renderer = Renderer(fp16=not args.fp32)
    cal_frames = [(rgb, lm) for rgb, lm in samples if measure(lm)[1] > 0.22][:6] or samples[:6]
    J = calibrate(renderer, cal_frames)
    if J is None:
        sys.exit("calibration failed: MediaPipe couldn't find the face in LivePortrait's renders")
    np.set_printoptions(precision=2, suppress=True)
    print(f"calibration ({time.time() - t0:.1f}s), iris response per control unit:\n{J}")
    del samples, cal_frames

    # ---- pass 2: render + composite
    t0 = time.time()
    writer = open_writer(args.output, args.input, vi)
    check = make_face_mesh(static=False)
    solver = GazeSolver(renderer, J, args.iterations)
    rendered = 0
    before, after = [], []
    for i, rgb16 in enumerate(read_frames(args.input, vi, "rgb48le")):
        if i >= n:
            break
        frame = rgb16.astype(np.float32) / 65535.0
        info = ""
        if need[i]:
            lm = L_s[i]
            rgb8 = (frame * 255 + 0.5).astype(np.uint8)
            c = renderer.crop(rgb8, lm)
            gen, resid = solver(c["img_crop"], delta[i])
            frame = composite(frame, gen, c["M_c2o"], lm, args.grow, args.feather)
            rendered += 1
            info = f"d=({delta[i, 0]:+.3f},{delta[i, 1]:+.3f}) amt={amount[i]:.2f} resid={resid:.3f}"
        # Measure the result so the summary reports how close we got to the target.
        out8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
        lm_out = landmarks(check, out8)
        if need[i] and lm_out is not None and open_[i]:
            before.append(np.linalg.norm(gaze[i] - (args.target_x, args.target_y)))
            after.append(np.linalg.norm(measure(lm_out)[0] - (args.target_x, args.target_y)))
        if args.debug and valid[i]:
            img8 = (frame * 255 + 0.5).astype(np.uint8)
            draw_debug(img8, L_s[i], f"#{i} gaze=({gaze_s[i, 0]:+.3f},{gaze_s[i, 1]:+.3f}) {info}")
            frame = img8.astype(np.float32) / 255.0
        writer.stdin.write((np.clip(frame, 0, 1) * 65535 + 0.5).astype("<u2").tobytes())
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  frame {i + 1}/{n}  ({rendered} rendered, {el / (i + 1):.2f}s/frame)", flush=True)
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("ffmpeg failed while writing the output")
    check.close()
    solver.close()
    print(f"wrote {args.output}: {n} frames, {rendered} re-rendered, {time.time() - t0:.1f}s")
    if before:
        print(f"distance from target on corrected frames (eye widths): median {np.median(before):.3f} -> "
              f"{np.median(after):.3f}, 90th pct {np.percentile(before, 90):.3f} -> {np.percentile(after, 90):.3f}")


if __name__ == "__main__":
    main()
