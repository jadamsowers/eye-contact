#!/usr/bin/env python3
"""Trains GazeWarpNet on Columbia Gaze eye patches.

Pairs are two photos of the same subject, head pose and eye with different gaze. The network sees
the source patch plus the desired iris displacement (measured from the landmarks of the pair, in
eye widths) and must reproduce the target patch.

    python gazenet/train.py data/columbia_patches.npz runs/gazenet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyepatch import EYE_FRAC, PATCH_H, PATCH_W  # noqa: E402
from model import GazeWarpNet, bilinear_sample  # noqa: E402

EYE_PX = EYE_FRAC * PATCH_W  # eye width in patch pixels
IRIS = 16                    # index of the iris center in the stored landmarks (16 contour points first)


def eye_masks(lm: np.ndarray, grow: int = 3) -> np.ndarray:
    """Eye-opening masks from the 16 contour landmarks, slightly grown to cover the lid margin."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    out = np.zeros((len(lm), PATCH_H, PATCH_W), np.uint8)
    for i, pts in enumerate(lm):
        m = np.zeros((PATCH_H, PATCH_W), np.uint8)
        cv2.fillPoly(m, [np.round(pts[:16]).astype(np.int32)], 255)
        out[i] = cv2.dilate(m, k)
    return out


class PairData:
    def __init__(self, path: str, device: str, val_subjects: int):
        d = np.load(path)
        self.patches = torch.from_numpy(d["patches"]).permute(0, 3, 1, 2).contiguous().to(device)  # uint8
        # Vision-framed sets (prepare_vision.py) carry their own masks and iris positions;
        # MediaPipe-framed ones (prepare_columbia.py) derive them from the landmarks.
        masks = d["masks"] if "masks" in d else eye_masks(d["lm"])
        self.masks = torch.from_numpy(masks).unsqueeze(1).to(device)
        iris = d["iris"] if "iris" in d else d["lm"][:, IRIS]
        self.iris = torch.from_numpy(iris).float().to(device)
        self.meta = {k: d[k] for k in ("subject", "pose", "v", "h", "side")}
        subjects = np.unique(self.meta["subject"])
        val = set(subjects[-val_subjects:].tolist())
        groups = {}
        for i, key in enumerate(zip(self.meta["subject"], self.meta["pose"], self.meta["side"])):
            groups.setdefault(key, []).append(i)
        self.groups = {"train": [], "val": []}
        for key, idx in groups.items():
            if len(idx) > 1:
                self.groups["val" if key[0] in val else "train"].append(np.array(idx))
        self.device = device
        print(f"{len(self.patches)} patches, {len(subjects)} subjects "
              f"({len(val)} held out), {sum(map(len, self.groups['train']))} train patches")

    def sample(self, n: int, split: str, rng: np.random.Generator, identity_frac=0.1):
        groups = self.groups[split]
        src, tgt = np.empty(n, np.int64), np.empty(n, np.int64)
        for k in range(n):
            g = groups[rng.integers(len(groups))]
            a, b = rng.choice(g, 2, replace=False)
            src[k], tgt[k] = a, (a if rng.random() < identity_frac else b)
        return torch.from_numpy(src).to(self.device), torch.from_numpy(tgt).to(self.device)

    def batch(self, src, tgt, augment: bool, rng: np.random.Generator):
        to_f = lambda t: t.float() / 255.0
        s_img, t_img = to_f(self.patches[src]), to_f(self.patches[tgt])
        s_mask, t_mask = to_f(self.masks[src]), to_f(self.masks[tgt])
        disp = (self.iris[tgt] - self.iris[src]) / EYE_PX  # desired iris displacement, eye widths
        if augment:
            b = len(src)
            # Same photometric change for both images of a pair.
            gain = torch.from_numpy(rng.uniform(0.75, 1.25, (b, 1, 1, 1))).float().to(self.device)
            bias = torch.from_numpy(rng.uniform(-0.06, 0.06, (b, 1, 1, 1))).float().to(self.device)
            tint = torch.from_numpy(rng.uniform(0.92, 1.08, (b, 3, 1, 1))).float().to(self.device)
            gamma = torch.from_numpy(rng.uniform(0.8, 1.25, (b, 1, 1, 1))).float().to(self.device)
            photo = lambda x: ((x * gain * tint + bias).clamp(0, 1) ** gamma)
            s_img, t_img = photo(s_img), photo(t_img)
            # Same small zoom/shift for both, so the model tolerates imprecise patch framing.
            scale = torch.from_numpy(rng.uniform(0.9, 1.1, b)).float().to(self.device)
            shift = torch.from_numpy(rng.uniform(-0.04, 0.04, (b, 2))).float().to(self.device)
            theta = torch.zeros(b, 2, 3, device=self.device)
            theta[:, 0, 0] = theta[:, 1, 1] = 1 / scale
            theta[:, :, 2] = shift
            grid = torch.nn.functional.affine_grid(theta, s_img.shape, align_corners=True)
            s_img, t_img = bilinear_sample(s_img, grid), bilinear_sample(t_img, grid)
            s_mask, t_mask = bilinear_sample(s_mask, grid), bilinear_sample(t_mask, grid)
            disp = disp * scale[:, None]
        return s_img, s_mask, t_img, t_mask, disp


def grad(x):
    return x[..., :, 1:] - x[..., :, :-1], x[..., 1:, :] - x[..., :-1, :]


def losses(model, s_img, s_mask, t_img, t_mask, disp):
    out, flow, alpha = model(s_img, s_mask, disp)
    eye = torch.maximum(s_mask, t_mask)
    weight = 1 + 6 * eye
    l1 = ((out - t_img).abs() * weight).mean() / weight.mean()
    # Edge loss inside the eye: matching image gradients penalizes blur that plain L1 tolerates.
    (ox, oy), (tx, ty) = grad(out), grad(t_img)
    edge = ((ox - tx).abs() * eye[..., :, 1:]).mean() + ((oy - ty).abs() * eye[..., 1:, :]).mean()
    tv = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + (flow[..., 1:] - flow[..., :-1]).abs().mean()
    return l1 + 0.5 * edge, tv, alpha.mean(), out


@torch.no_grad()
def evaluate(model, data, rng, batches=20):
    model.eval()
    tot = base = 0.0
    for _ in range(batches):
        src, tgt = data.sample(64, "val", rng, identity_frac=0)
        s_img, s_mask, t_img, t_mask, disp = data.batch(src, tgt, False, rng)
        _, _, _, out = losses(model, s_img, s_mask, t_img, t_mask, disp)
        m = torch.maximum(s_mask, t_mask)
        tot += ((out - t_img).abs() * m).sum().item() / (m.sum().item() * 3)
        base += ((s_img - t_img).abs() * m).sum().item() / (m.sum().item() * 3)
    model.train()
    return tot / batches, base / batches


@torch.no_grad()
def preview(model, data, path: Path, rng):
    """Rows of: source | target | output, for held-out subjects."""
    model.eval()
    src, tgt = data.sample(8, "val", rng, identity_frac=0)
    s_img, s_mask, t_img, _, disp = data.batch(src, tgt, False, rng)
    out = model(s_img, s_mask, disp)[0]
    rows = [torch.cat([s_img[i], t_img[i], out[i]], 2) for i in range(len(src))]
    img = (torch.cat(rows, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    model.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-subjects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-weight", type=float, default=0.05, help="penalty on painted (non-warped) pixels")
    ap.add_argument("--resume", help="checkpoint to continue from (weights, optimizer if saved, step)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = PairData(args.data, device, args.val_subjects)
    model = GazeWarpNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))

    start, best = 1, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start, best = ckpt["step"] + 1, ckpt.get("best", ckpt["val"])
        for _ in range(ckpt["step"]):
            sched.step()
        print(f"resumed from {args.resume} at step {ckpt['step']}")
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    for step in range(start, args.steps + 1):
        src, tgt = data.sample(args.batch, "train", rng)
        l1, tv, alpha, _ = losses(model, *data.batch(src, tgt, True, rng))
        loss = l1 + 0.05 * tv + args.alpha_weight * alpha
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step}/{args.steps}  loss {loss.item():.4f}  l1 {l1.item():.4f}  "
                  f"alpha {alpha.item():.3f}  {(time.time() - t0) / (step - start + 1):.2f}s/step", flush=True)
        if step % 1000 == 0 or step == args.steps:
            val, baseline = evaluate(model, data, np.random.default_rng(123))
            preview(model, data, out / f"preview_{step:06d}.png", np.random.default_rng(7))
            print(f"  val eye-region L1 {val:.4f} (no-op baseline {baseline:.4f})", flush=True)
            log.write(json.dumps(dict(step=step, loss=loss.item(), val=val, baseline=baseline)) + "\n")
            log.flush()
            best = min(best, val)
            ckpt = dict(model=model.state_dict(), opt=opt.state_dict(), step=step, val=val, best=best,
                        config=dict(patch=(PATCH_W, PATCH_H), eye_frac=EYE_FRAC, gate=model.gate))
            torch.save(ckpt, out / "last.pt")
            if val <= best:
                torch.save(ckpt, out / "best.pt")
    print(f"done; best val {best:.4f}")


if __name__ == "__main__":
    main()
