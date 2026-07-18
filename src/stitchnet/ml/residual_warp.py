from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class ResidualWarpNet(nn.Module):
    """Identity-safe U-Net that predicts a bounded dense correction field.

    The final layer is zero initialized, so an untrained or newly loaded network
    cannot distort an input. Flow values are expressed in image pixels.
    """

    def __init__(self, max_flow_pixels: float = 8.0) -> None:
        super().__init__()
        self.max_flow_pixels = float(max_flow_pixels)
        self.enc1 = ConvBlock(6, 24)
        self.enc2 = ConvBlock(24, 48)
        self.enc3 = ConvBlock(48, 96)
        self.bottleneck = ConvBlock(96, 128)
        self.dec3 = ConvBlock(128 + 96, 96)
        self.dec2 = ConvBlock(96 + 48, 48)
        self.dec1 = ConvBlock(48 + 24, 32)
        self.flow_head = nn.Conv2d(32, 2, 3, padding=1)
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)

    def forward(self, reference: Tensor, moving: Tensor) -> tuple[Tensor, Tensor]:
        if reference.shape != moving.shape or reference.ndim != 4:
            raise ValueError("reference and moving must have identical NCHW shapes")
        first = self.enc1(torch.cat((reference, moving), dim=1))
        second = self.enc2(F.avg_pool2d(first, 2))
        third = self.enc3(F.avg_pool2d(second, 2))
        latent = self.bottleneck(F.avg_pool2d(third, 2))
        up3 = F.interpolate(latent, size=third.shape[-2:], mode="bilinear", align_corners=False)
        up3 = self.dec3(torch.cat((up3, third), dim=1))
        up2 = F.interpolate(up3, size=second.shape[-2:], mode="bilinear", align_corners=False)
        up2 = self.dec2(torch.cat((up2, second), dim=1))
        up1 = F.interpolate(up2, size=first.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(torch.cat((up1, first), dim=1))
        flow = torch.tanh(self.flow_head(features)) * self.max_flow_pixels
        return warp_image(moving, flow), flow


def warp_image(image: Tensor, flow_pixels: Tensor) -> Tensor:
    """Sample ``image`` using an output-to-input displacement in pixels."""

    if image.ndim != 4 or flow_pixels.shape != (image.shape[0], 2, *image.shape[-2:]):
        raise ValueError("flow must have shape [batch, 2, height, width]")
    if not torch.is_grad_enabled() and torch.count_nonzero(flow_pixels).item() == 0:
        return image
    batch, _, height, width = image.shape
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1).expand(batch, -1, -1, -1)
    scale_x = 2.0 / max(width - 1, 1)
    scale_y = 2.0 / max(height - 1, 1)
    normalized = torch.stack((flow_pixels[:, 0] * scale_x, flow_pixels[:, 1] * scale_y), dim=-1)
    return F.grid_sample(
        image,
        base + normalized,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def load_residual_warp(
    checkpoint_path: str | Path, device: str | torch.device = "cpu"
) -> tuple[ResidualWarpNet, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Residual-warp checkpoint is missing model metadata.")
    architecture = checkpoint.get("architecture", {})
    model = ResidualWarpNet(float(architecture.get("max_flow_pixels", 8.0)))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, checkpoint
