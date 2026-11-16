"""PROVISIONAL group objective. This is NOT the official SAGE training loss."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FusionLoss(nn.Module):
    """mean intensity MSE + max-Sobel L1 + frozen-Student output MSE anchor."""
    def __init__(self, intensity=1.0, gradient=0.2, anchor=1.0):
        super().__init__()
        self.weights = {"intensity": intensity, "gradient": gradient, "anchor": anchor}
        if (any(not math.isfinite(x) or x < 0 for x in self.weights.values())
                or sum(self.weights.values()) == 0):
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")
        self.register_buffer("kx", torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None])
        self.register_buffer("ky", torch.tensor(
            [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]])[None, None])

    def sobel(self, value):
        return (F.conv2d(value, self.kx.to(value), padding=1).abs()
                + F.conv2d(value, self.ky.to(value), padding=1).abs())

    def forward(self, fused, vis, ir, reference):
        values = (fused, vis, ir, reference)
        if (fused.ndim != 4 or fused.shape[1] != 1
                or any(x.shape != fused.shape or x.device != fused.device for x in values)):
            raise ValueError("Loss requires matching N1HW tensors")
        if any(not torch.isfinite(x).all().item() for x in values):
            raise FloatingPointError("Nonfinite loss input")
        vis, ir, reference = vis.detach(), ir.detach(), reference.detach()
        parts = {
            "intensity": 0.5 * F.mse_loss(fused, vis) + 0.5 * F.mse_loss(fused, ir),
            "gradient": F.l1_loss(self.sobel(fused), torch.maximum(self.sobel(vis), self.sobel(ir))),
            "anchor": F.mse_loss(fused, reference),
        }
        parts["total"] = sum(self.weights[k] * v for k, v in parts.items())
        if any(not torch.isfinite(v).item() for v in parts.values()):
            raise FloatingPointError("Nonfinite loss component")
        return parts
