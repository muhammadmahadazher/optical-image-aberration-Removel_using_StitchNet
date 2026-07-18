from __future__ import annotations

import pytest
import torch
from torch import nn
from torchvision.models.optical_flow import raft_small

from stitchnet.ml import (
    GatedRaftRefiner,
    ResidualWarpNet,
    load_raft_refiner,
    load_residual_warp,
    warp_image,
)


class _ConstantFlow(nn.Module):
    def __init__(self, dx: float, dy: float) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.dx = dx
        self.dy = dy

    def forward(self, first, second, *, num_flow_updates):
        del second, num_flow_updates
        flow = torch.zeros(first.shape[0], 2, *first.shape[-2:], device=first.device)
        flow[:, 0] = self.dx + self.anchor * 0.0
        flow[:, 1] = self.dy + self.anchor * 0.0
        return [flow]


def test_untrained_residual_model_is_exactly_identity_safe() -> None:
    model = ResidualWarpNet(max_flow_pixels=6.0).eval()
    reference = torch.rand(1, 3, 32, 40)
    moving = torch.rand(1, 3, 32, 40)
    with torch.inference_mode():
        corrected, flow = model(reference, moving)
    assert torch.count_nonzero(flow) == 0
    assert torch.equal(corrected, moving)


def test_warp_image_uses_pixel_displacements() -> None:
    image = torch.zeros(1, 3, 12, 14)
    image[:, :, 5, 8] = 1.0
    flow = torch.zeros(1, 2, 12, 14)
    flow[:, 0] = 2.0
    shifted = warp_image(image, flow)
    maximum = torch.nonzero(shifted[0, 0] == shifted[0, 0].max())[0]
    assert maximum.tolist() == [5, 6]


def test_checkpoint_round_trip_preserves_metadata(tmp_path) -> None:
    model = ResidualWarpNet(max_flow_pixels=5.0)
    path = tmp_path / "residual.pt"
    torch.save(
        {
            "format_version": 1,
            "architecture": {"name": "ResidualWarpNet", "max_flow_pixels": 5.0},
            "model_state": model.state_dict(),
            "quality_gate_passed": False,
        },
        path,
    )
    loaded, metadata = load_residual_warp(path)
    assert loaded.max_flow_pixels == 5.0
    assert metadata["quality_gate_passed"] is False
    assert not loaded.training


def test_raft_refiner_accepts_only_an_improving_warp() -> None:
    reference = torch.zeros(1, 3, 128, 128)
    reference[:, :, 48:80, 52:76] = 1.0
    applied = torch.zeros(1, 2, 128, 128)
    applied[:, 0] = 2.0
    moving = warp_image(reference, applied)
    refiner = GatedRaftRefiner(_ConstantFlow(-2.0, 0.0), num_flow_updates=1)
    corrected, flow, accepted = refiner(reference, moving)
    assert accepted.tolist() == [True]
    assert (corrected - reference).abs().mean() < (moving - reference).abs().mean()
    assert torch.all(flow[:, 0] == -2.0)

    identity, identity_flow, identity_accepted = refiner(reference, reference)
    assert identity_accepted.tolist() == [False]
    assert torch.equal(identity, reference)
    assert torch.count_nonzero(identity_flow) == 0


def test_raft_loader_rejects_checkpoint_that_failed_gate(tmp_path) -> None:
    model = raft_small(weights=None)
    path = tmp_path / "failed-raft.pt"
    torch.save(
        {
            "architecture": {"name": "torchvision-raft-small-gated"},
            "model_state": model.state_dict(),
            "quality_gate_passed": False,
        },
        path,
    )
    with pytest.raises(ValueError, match="quality gate"):
        load_raft_refiner(path)
