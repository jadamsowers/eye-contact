#!/usr/bin/env python3
"""Exports GazeWarpNet's field predictor to Core ML for the Final Cut Pro plug-in.

The model takes both eyes at once (batch 2) and returns the half-resolution fields; the plug-in
applies them to full-resolution pixels in Metal, exactly like GazeWarpNet.apply.

    python gazenet/export_coreml.py runs/gazenet_v1/best.pt EyeContactFx/Plugin/GazeWarp.mlpackage
"""
import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eyepatch import PATCH_H, PATCH_W  # noqa: E402
from model import load_checkpoint  # noqa: E402


class Fields(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image, mask, displacement):
        return self.net.fields(image, mask, displacement)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    args = ap.parse_args()

    model = Fields(load_checkpoint(args.checkpoint)).eval()
    ex = (torch.rand(2, 3, PATCH_H, PATCH_W), torch.rand(2, 1, PATCH_H, PATCH_W), torch.zeros(2, 2))
    ex[1][:, :, 30:66, 35:110] = 1  # an eye-shaped mask, so the gate is exercised by the check
    traced = torch.jit.trace(model, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType("image", ex[0].shape), ct.TensorType("mask", ex[1].shape),
                ct.TensorType("displacement", ex[2].shape)],
        outputs=[ct.TensorType("flow"), ct.TensorType("alpha"), ct.TensorType("color")],
        convert_to="mlprogram",
        # Float32: the model is tiny, and FP16 visibly perturbs the paint layer (alpha/color).
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    ml.short_description = "Gaze redirection fields (DeepWarp-style) for two 144x96 eye patches"
    ml.save(args.output)

    # Check the conversion against PyTorch.
    with torch.no_grad():
        ref = [t.numpy() for t in model(*ex)]
    got = ml.predict({"image": ex[0].numpy(), "mask": ex[1].numpy(), "displacement": ex[2].numpy()})
    for name, r in zip(("flow", "alpha", "color"), ref):
        print(f"{name}: shape {got[name].shape}, max abs diff vs PyTorch {np.abs(got[name] - r).max():.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
