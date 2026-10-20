"""3D grid_sample for Apple GPUs.

PyTorch has no MPS kernel for 5-D (volumetric) grid_sample, so LivePortrait's warping falls
back to the CPU and dominates render time. Trilinear sampling factors into a 2-D bilinear
sample of every depth slice followed by linear interpolation along depth, both of which run
on the GPU.
"""
import torch
import torch.nn.functional as F

_orig_grid_sample = F.grid_sample


def grid_sample_3d(inp: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Matches F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    for inp (N, C, D, H, W) and grid (N, Do, Ho, Wo, 3)."""
    n, c, d, h, w = inp.shape
    _, do, ho, wo, _ = grid.shape

    # Bilinear in x/y on every input depth slice: batch = N * D.
    slices = inp.permute(0, 2, 1, 3, 4).reshape(n * d, c, h, w)
    xy = grid[..., :2].reshape(n, 1, do * ho, wo, 2).expand(n, d, do * ho, wo, 2).reshape(n * d, do * ho, wo, 2)
    sampled = _orig_grid_sample(slices, xy, mode="bilinear", padding_mode="zeros", align_corners=False)
    sampled = sampled.reshape(n, d, c, do, ho, wo)

    # Linear in z: unnormalize (align_corners=False) and weight each slice with a hat function.
    # Slices outside [0, D-1] simply don't exist, which reproduces zero padding.
    z = ((grid[..., 2] + 1) * d - 1) / 2  # (N, Do, Ho, Wo)
    idx = torch.arange(d, device=inp.device, dtype=inp.dtype).view(1, d, 1, 1, 1)
    weight = (1 - (z.unsqueeze(1) - idx).abs()).clamp(min=0)  # (N, D, Do, Ho, Wo)
    return (sampled * weight.unsqueeze(2)).sum(dim=1)


def patched_grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    if (inp.dim() == 5 and inp.device.type == "mps" and mode == "bilinear"
            and padding_mode == "zeros" and not align_corners):
        return grid_sample_3d(inp, grid)
    return _orig_grid_sample(inp, grid, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


def install():
    F.grid_sample = patched_grid_sample
