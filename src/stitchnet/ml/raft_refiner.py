from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.optical_flow import raft_small

from .residual_warp import warp_image


class GatedRaftRefiner(nn.Module):
    """RAFT-small correspondence with a per-image no-regression quality gate."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        num_flow_updates: int = 8,
        minimum_improvement: float = 0.01,
        max_flow_pixels: float = 12.0,
    ) -> None:
        super().__init__()
        self.model = model or raft_small(weights=None)
        self.num_flow_updates = num_flow_updates
        self.minimum_improvement = minimum_improvement
        self.max_flow_pixels = max_flow_pixels

    def estimate_flow(self, reference: Tensor, moving: Tensor) -> Tensor:
        if reference.shape != moving.shape or reference.ndim != 4:
            raise ValueError("reference and moving must have identical NCHW shapes")
        height, width = reference.shape[-2:]
        target_height = max(128, ((height + 7) // 8) * 8)
        target_width = max(128, ((width + 7) // 8) * 8)
        pad = (0, target_width - width, 0, target_height - height)
        reference_padded = F.pad(reference, pad, mode="replicate")
        moving_padded = F.pad(moving, pad, mode="replicate")
        predictions = self.model(
            reference_padded * 2.0 - 1.0,
            moving_padded * 2.0 - 1.0,
            num_flow_updates=self.num_flow_updates,
        )
        return predictions[-1][..., :height, :width].clamp(
            -self.max_flow_pixels, self.max_flow_pixels
        )

    def forward(self, reference: Tensor, moving: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        flow = self.estimate_flow(reference, moving)
        corrected = warp_image(moving, flow)
        baseline_error = (moving - reference).abs().flatten(1).mean(1)
        corrected_error = (corrected - reference).abs().flatten(1).mean(1)
        accepted = (
            (baseline_error > 1e-6)
            & (corrected_error <= baseline_error * (1.0 - self.minimum_improvement))
            & (flow.abs().flatten(1).amax(1) < self.max_flow_pixels)
        )
        accepted_image = accepted[:, None, None, None]
        accepted_flow = accepted[:, None, None, None]
        return (
            torch.where(accepted_image, corrected, moving),
            torch.where(accepted_flow, flow, torch.zeros_like(flow)),
            accepted,
        )


def load_raft_refiner(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    require_quality_gate: bool = True,
) -> tuple[GatedRaftRefiner, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("RAFT refiner checkpoint is missing model metadata.")
    if require_quality_gate and checkpoint.get("quality_gate_passed") is not True:
        raise ValueError("RAFT refiner checkpoint did not pass its deployment quality gate.")
    architecture = checkpoint.get("architecture", {})
    raft = raft_small(weights=None)
    raft.load_state_dict(checkpoint["model_state"], strict=True)
    refiner = GatedRaftRefiner(
        raft,
        num_flow_updates=int(architecture.get("num_flow_updates", 8)),
        minimum_improvement=float(architecture.get("minimum_improvement", 0.01)),
        max_flow_pixels=float(architecture.get("max_flow_pixels", 12.0)),
    )
    refiner.to(device).eval().requires_grad_(False)
    return refiner, checkpoint
