"""Gaze redirection network: a DeepWarp-style warp-field predictor (Ganin et al., ECCV 2016).

Given an eye patch, a mask of the eye opening and the desired iris displacement, the network
predicts a dense warp field plus a sparse "paint" layer for content the warp can't produce (sclera
revealed behind a moving iris, highlights). Most of the output is the input's own pixels moved, so
it stays sharp, and the low-resolution field can be upsampled to warp a 4K eye directly.

Output = (1 - alpha) * warp(input, flow) + alpha * color
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bilinear_sample(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """F.grid_sample(img, grid, align_corners=True, padding_mode="border") built from gathers.

    PyTorch 2.3 has no MPS backward for grid_sample; gather's backward (scatter-add) is supported.
    img (B,C,H,W), grid (B,H',W',2) in [-1, 1].
    """
    b, c, h, w = img.shape
    x = ((grid[..., 0] + 1) * 0.5 * (w - 1)).clamp(0, w - 1)
    y = ((grid[..., 1] + 1) * 0.5 * (h - 1)).clamp(0, h - 1)
    x0 = x.floor().clamp(max=w - 2)
    y0 = y.floor().clamp(max=h - 2)
    wx, wy = (x - x0).unsqueeze(1), (y - y0).unsqueeze(1)
    x0, y0 = x0.long(), y0.long()
    flat = img.reshape(b, c, h * w)

    def tap(yy, xx):
        idx = (yy * w + xx).reshape(b, 1, -1).expand(-1, c, -1)
        return flat.gather(2, idx).reshape(b, c, *grid.shape[1:3])

    top = tap(y0, x0) * (1 - wx) + tap(y0, x0 + 1) * wx
    bottom = tap(y0 + 1, x0) * (1 - wx) + tap(y0 + 1, x0 + 1) * wx
    return top * (1 - wy) + bottom * wy


def conv(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.SiLU())


class FiLM(nn.Module):
    """Feature-wise modulation by the conditioning vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.lin = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, cond):
        g, b = self.lin(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + g) + b


class GazeWarpNet(nn.Module):
    """U-Net predicting the fields at half the patch resolution; they're upsampled before warping,
    so the warp itself always samples full-resolution pixels."""

    def __init__(self, width=24, max_flow=0.35, gate=True):
        super().__init__()
        self.gate = gate
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 5
        cond_dim = 64
        self.cond_mlp = nn.Sequential(nn.Linear(2, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim), nn.SiLU())
        # Inputs: RGB + eye-opening mask + coordinate grid (2) + conditioning planes (2).
        self.stem = conv(3 + 1 + 2 + 2, c1, 2)                     # 1/2
        self.e2 = nn.Sequential(conv(c1, c2), conv(c2, c2))        # 1/2
        self.e3 = nn.Sequential(conv(c2, c3, 2), conv(c3, c3))     # 1/4
        self.e4 = nn.Sequential(conv(c3, c4, 2), conv(c4, c4))     # 1/8
        self.f4 = FiLM(cond_dim, c4)
        self.d3 = nn.Sequential(conv(c4 + c3, c3), conv(c3, c3))
        self.f3 = FiLM(cond_dim, c3)
        self.d2 = nn.Sequential(conv(c3 + c2, c2), conv(c2, c2))
        self.f2 = FiLM(cond_dim, c2)
        self.head = nn.Conv2d(c2, 2 + 1 + 3, 3, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[2] = -4.0  # start with alpha ~ 0: pure warp
        self.max_flow = max_flow  # normalized coordinates (fraction of the half-width)
        # Precomputed so the graph has no shape-dependent ops (keeps Core ML conversion static).
        from eyepatch import PATCH_H, PATCH_W
        self.register_buffer("coords", self._grid(1, PATCH_H, PATCH_W, "cpu", torch.float32), persistent=False)

    @staticmethod
    def _grid(b, h, w, device, dtype):
        ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], 0).unsqueeze(0).expand(b, -1, -1, -1)

    def fields(self, img, mask, cond):
        """Predicts (flow (B,2,h,w) in normalized coords, alpha (B,1,h,w), color (B,3,h,w)) at half
        the input resolution."""
        # Broadcasting against img[:, :2] * 0 gives batch-sized tensors without reading shapes.
        zero = img[:, :2] * 0
        grid = zero + self.coords
        planes = zero + cond[:, :, None, None]
        z = self.cond_mlp(cond)
        x2 = self.e2(self.stem(torch.cat([img, mask, grid, planes], 1)))
        x3 = self.e3(x2)
        x4 = self.f4(self.e4(x3), z)
        # Patch dimensions are multiples of 8, so every level is exactly half the one above.
        up = lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False)
        y = self.f3(self.d3(torch.cat([up(x4), x3], 1)), z)
        y = self.f2(self.d2(torch.cat([up(y), x2], 1)), z)
        out = self.head(y)  # half resolution
        flow = torch.tanh(out[:, :2]) * self.max_flow
        alpha = torch.sigmoid(out[:, 2:3])
        color = torch.sigmoid(out[:, 3:6])
        if self.gate:
            # Only the eye opening (plus a margin for the lid edge) may change: skin, brows and
            # lashes away from the eye are guaranteed untouched, so they stay perfectly sharp.
            g = F.max_pool2d(F.avg_pool2d(mask, 2), 9, stride=1, padding=4)  # half res, grown ~8 px
            g = F.avg_pool2d(g, 5, stride=1, padding=2)                     # soft edge
            flow, alpha = flow * g, alpha * g
        return flow, alpha, color

    @staticmethod
    def apply(img, flow, alpha, color):
        """Composes the output; works at any resolution if the fields are resized to `img`."""
        b, _, h, w = img.shape
        if flow.shape[-2:] != (h, w):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)
            color = F.interpolate(color, size=(h, w), mode="bilinear", align_corners=False)
        grid = GazeWarpNet._grid(b, h, w, img.device, img.dtype) + flow
        warped = bilinear_sample(img, grid.permute(0, 2, 3, 1))
        return (1 - alpha) * warped + alpha * color

    def forward(self, img, mask, cond):
        flow, alpha, color = self.fields(img, mask, cond)
        return self.apply(img, flow, alpha, color), flow, alpha


def load_checkpoint(path: str, device: str = "cpu") -> GazeWarpNet:
    """Loads a training checkpoint; checkpoints from before gating existed load ungated."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = GazeWarpNet(gate=ckpt.get("config", {}).get("gate", False))
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval()
