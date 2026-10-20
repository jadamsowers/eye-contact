#!/usr/bin/env python3
"""Iris locator: finds the iris center in a Vision-framed eye patch.

Vision's pupil estimate is noisy and biased on narrow or squinting eyes (the visible part of the
iris is cut off by the upper lid). This small network is trained to reproduce MediaPipe's iris
center (which models the whole iris) from the same patches the gaze model sees. It predicts a
heatmap and reads the position with a soft-argmax, which is sub-pixel and smooth.

    python gazenet/iris_net.py train data/columbia_vision_patches.npz runs/iris_v1
    python gazenet/iris_net.py export runs/iris_v1/best.pt runs/iris_v1/IrisLocator.mlpackage
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyepatch import EYE_FRAC, PATCH_H, PATCH_W  # noqa: E402
from model import bilinear_sample  # noqa: E402

EYE_PX = EYE_FRAC * PATCH_W


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class IrisNet(nn.Module):
    """Patch (+ eye mask) -> iris center in patch pixels, via a 1/4-resolution heatmap."""

    def __init__(self, w=24):
        super().__init__()
        self.net = nn.Sequential(
            block(4, w, 2), block(w, w),                 # 1/2
            block(w, 2 * w, 2), block(2 * w, 2 * w),     # 1/4
            block(2 * w, 2 * w), block(2 * w, 2 * w),
            nn.Conv2d(2 * w, 1, 3, 1, 1),
        )
        hh, ww = PATCH_H // 4, PATCH_W // 4
        self.hh, self.ww = hh, ww
        # Patch-pixel coordinates of each heatmap cell center (cell k covers pixels 4k..4k+3).
        ys = (torch.arange(hh, dtype=torch.float32) + 0.5) * 4 - 0.5
        xs = (torch.arange(ww, dtype=torch.float32) + 0.5) * 4 - 0.5
        self.register_buffer("xs", xs.view(1, 1, ww), persistent=False)
        self.register_buffer("ys", ys.view(1, hh, 1), persistent=False)

    def forward(self, img, mask):
        logits = self.net(torch.cat([img, mask], 1))[:, 0]        # (B, H/4, W/4)
        p = torch.softmax(logits.reshape(-1, self.hh * self.ww), 1).reshape(-1, self.hh, self.ww)
        x = (p * self.xs).sum((1, 2))
        y = (p * self.ys).sum((1, 2))
        return torch.stack([x, y], 1)                              # patch pixels


def train(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    d = np.load(args.data)
    imgs = torch.from_numpy(d["patches"]).permute(0, 3, 1, 2).contiguous().to(device)
    masks = torch.from_numpy(d["masks"]).unsqueeze(1).to(device)
    iris = torch.from_numpy(d["iris"]).float().to(device)
    subjects = d["subject"]
    held = np.unique(subjects)[-6:]
    tr = torch.from_numpy(np.nonzero(~np.isin(subjects, held))[0]).to(device)
    va = torch.from_numpy(np.nonzero(np.isin(subjects, held))[0]).to(device)
    rng = np.random.default_rng(0)

    model = IrisNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / 300) * 0.5 * (1 + math.cos(math.pi * min(1, s / args.steps))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def batch(idx, augment):
        x = imgs[idx].float() / 255
        m = masks[idx].float() / 255
        y = iris[idx].clone()
        if augment:
            b = len(idx)
            g = lambda lo, hi, *s: torch.from_numpy(rng.uniform(lo, hi, (b, *s))).float().to(device)
            x = ((x * g(0.7, 1.3, 1, 1, 1) * g(0.9, 1.1, 3, 1, 1) + g(-0.08, 0.08, 1, 1, 1)).clamp(0, 1)
                 ** g(0.75, 1.3, 1, 1, 1))
            # Random zoom/shift/rotation, as imprecise eye framing would produce; labels follow.
            scale, ang = g(0.88, 1.12), g(-0.08, 0.08)
            shift = g(-0.06, 0.06, 2)
            cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
            theta = torch.stack([torch.stack([cos, -sin * PATCH_H / PATCH_W, shift[:, 0]], 1),
                                 torch.stack([sin * PATCH_W / PATCH_H, cos, shift[:, 1]], 1)], 1)
            grid = F.affine_grid(theta, x.shape, align_corners=True)
            x, m = bilinear_sample(x, grid), bilinear_sample(m, grid)
            # Map labels through the inverse of theta (normalized align_corners=True coords).
            nx = y[:, 0] / (PATCH_W - 1) * 2 - 1
            ny = y[:, 1] / (PATCH_H - 1) * 2 - 1
            A = theta[:, :, :2]
            inv = torch.linalg.inv(A)
            src = torch.stack([nx, ny], 1) - theta[:, :, 2]
            dst = (inv @ src.unsqueeze(2)).squeeze(2)
            y = torch.stack([(dst[:, 0] + 1) / 2 * (PATCH_W - 1), (dst[:, 1] + 1) / 2 * (PATCH_H - 1)], 1)
            # Occasionally lower the upper lid by painting skin color over the top of the opening,
            # so narrow / squinting eyes are covered (Columbia's subjects mostly have open eyes).
            squint = torch.from_numpy(rng.random(b) < 0.3).to(device)
            if squint.any():
                rows = torch.arange(PATCH_H, device=device).view(1, PATCH_H, 1).float()
                top = torch.from_numpy(rng.uniform(0.33, 0.47, b) * PATCH_H).float().to(device).view(b, 1, 1)
                cover = ((rows < top) & squint.view(b, 1, 1)).unsqueeze(1).float() * m
                skin = x[:, :, 2:8, :].mean((2, 3), keepdim=True)
                x = x * (1 - cover) + skin * cover
        return x, m, y

    best, t0 = float("inf"), time.time()
    for step in range(1, args.steps + 1):
        idx = tr[torch.from_numpy(rng.integers(0, len(tr), args.batch)).to(device)]
        x, m, y = batch(idx, True)
        loss = F.smooth_l1_loss(model(x, m), y, beta=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                x, m, y = batch(va, False)
                err = (model(x, m) - y).abs() / EYE_PX
            model.train()
            ex, ey = err[:, 0].mean().item(), err[:, 1].mean().item()
            print(f"step {step}/{args.steps} loss {loss.item():.3f}  val error x {ex:.4f} y {ey:.4f} eye widths "
                  f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
            torch.save(dict(model=model.state_dict(), step=step, err=(ex, ey)), out / "last.pt")
            if ex + ey < best:
                best = ex + ey
                torch.save(dict(model=model.state_dict(), step=step, err=(ex, ey)), out / "best.pt")


def export(args):
    import coremltools as ct
    net = IrisNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"])
    net.eval()
    ex = (torch.rand(2, 3, PATCH_H, PATCH_W), torch.rand(2, 1, PATCH_H, PATCH_W))
    ml = ct.convert(torch.jit.trace(net, ex),
                    inputs=[ct.TensorType("image", ex[0].shape), ct.TensorType("mask", ex[1].shape)],
                    outputs=[ct.TensorType("iris")], convert_to="mlprogram",
                    compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS14)
    ml.short_description = "Iris center (patch pixels) for two 144x96 Vision-framed eye patches"
    ml.save(args.output)
    with torch.no_grad():
        ref = net(*ex).numpy()
    got = ml.predict({"image": ex[0].numpy(), "mask": ex[1].numpy()})["iris"]
    print(f"max abs diff vs PyTorch {np.abs(got - ref).max():.4f} px; wrote {args.output}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("data")
    t.add_argument("out")
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch", type=int, default=64)
    e = sub.add_parser("export")
    e.add_argument("checkpoint")
    e.add_argument("output")
    args = ap.parse_args()
    train(args) if args.cmd == "train" else export(args)


if __name__ == "__main__":
    main()
