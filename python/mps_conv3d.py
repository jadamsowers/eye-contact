"""Conv3d as a batch of Conv2d calls.

PyTorch's MPS Conv3d is several times slower than its Conv2d. A stride-1 3-D convolution
with depth kernel kD equals a 2-D convolution over (channels x kD) once each output depth
slice's kD input slices are stacked as channels, so the whole thing becomes one big Conv2d.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dAs2d(nn.Module):
    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        assert conv.stride == (1, 1, 1) and conv.dilation == (1, 1, 1) and conv.groups == 1
        assert conv.padding_mode == "zeros"
        o, c, kd, kh, kw = conv.weight.shape
        self.kd = kd
        self.pd, self.ph, self.pw = conv.padding
        # (O, C, kD, kH, kW) -> (O, kD*C, kH, kW), depth-major to match the unfold below.
        self.weight = nn.Parameter(conv.weight.detach().permute(0, 2, 1, 3, 4).reshape(o, kd * c, kh, kw).contiguous())
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach().clone())

    def forward(self, x):
        n, c, d, h, w = x.shape
        if self.pd:
            x = F.pad(x, (0, 0, 0, 0, self.pd, self.pd))
        do = x.shape[2] - self.kd + 1
        # (N, C, Dp, H, W) -> (N, Do, kD, C, H, W): each output slice sees kD consecutive input slices.
        x = x.permute(0, 2, 1, 3, 4)  # (N, Dp, C, H, W)
        x = torch.stack([x[:, i:i + do] for i in range(self.kd)], dim=2)
        x = x.reshape(n * do, self.kd * c, h, w)
        y = F.conv2d(x, self.weight, self.bias, padding=(self.ph, self.pw))
        return y.reshape(n, do, -1, h, w).permute(0, 2, 1, 3, 4)


def convert(module: nn.Module) -> int:
    """Replace every eligible Conv3d in `module` in place. Returns how many were replaced."""
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Conv3d) and child.stride == (1, 1, 1) and child.groups == 1 \
                and child.dilation == (1, 1, 1) and child.padding_mode == "zeros":
            setattr(module, name, Conv3dAs2d(child).to(child.weight.device))
            count += 1
        else:
            count += convert(child)
    return count


class AvgPool3dAs2d(nn.Module):
    """AvgPool3d with a depth kernel of 1 is a 2-D pool on every depth slice (MPS has no avg_pool3d)."""

    def __init__(self, pool: nn.AvgPool3d):
        super().__init__()
        ks = pool.kernel_size if isinstance(pool.kernel_size, tuple) else (pool.kernel_size,) * 3
        st = pool.stride if isinstance(pool.stride, tuple) else (pool.stride,) * 3
        assert ks[0] == 1 and st[0] == 1 and pool.padding in (0, (0, 0, 0))
        self.kernel, self.stride = ks[1:], st[1:]

    def forward(self, x):
        n, c, d, h, w = x.shape
        y = F.avg_pool2d(x.transpose(1, 2).reshape(n * d, c, h, w), self.kernel, self.stride)
        return y.reshape(n, d, c, *y.shape[-2:]).transpose(1, 2)


_orig_interpolate = F.interpolate


def _interpolate(x, size=None, scale_factor=None, mode="nearest", align_corners=None, **kw):
    # 3-D nearest upsampling that leaves depth alone (scale (1, s, s)) is 2-D per slice.
    if (x.dim() == 5 and x.device.type == "mps" and mode == "nearest" and size is None
            and isinstance(scale_factor, tuple) and scale_factor[0] == 1):
        n, c, d, h, w = x.shape
        y = _orig_interpolate(x.transpose(1, 2).reshape(n * d, c, h, w), scale_factor=scale_factor[1:], mode="nearest")
        return y.reshape(n, d, c, *y.shape[-2:]).transpose(1, 2)
    return _orig_interpolate(x, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, **kw)


def convert_pools(module: nn.Module) -> int:
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.AvgPool3d):
            setattr(module, name, AvgPool3dAs2d(child))
            count += 1
        else:
            count += convert_pools(child)
    return count


def install_interpolate():
    F.interpolate = _interpolate
