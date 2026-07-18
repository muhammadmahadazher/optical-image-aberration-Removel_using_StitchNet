from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from stitchnet.io import SUPPORTED_EXTENSIONS, read_tile
from stitchnet.ml import warp_image


class RandomPatchDataset(Dataset[Tensor]):
    def __init__(
        self,
        paths: list[Path],
        crop_size: int,
        samples: int,
        seed: int,
    ) -> None:
        if not paths:
            raise ValueError("At least one readable source image is required.")
        self.paths = paths
        self.crop_size = crop_size
        self.samples = samples
        self.seed = seed
        self._cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self.samples

    def _read(self, path: Path) -> np.ndarray:
        if path not in self._cache:
            image = read_tile(path)
            height, width = image.shape[:2]
            if min(height, width) < self.crop_size:
                scale = self.crop_size / min(height, width)
                image = cv2.resize(
                    image,
                    (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_CUBIC,
                )
            self._cache[path] = image
        return self._cache[path]

    def __getitem__(self, index: int) -> Tensor:
        generator = random.Random(self.seed + index * 104729)
        image = self._read(self.paths[generator.randrange(len(self.paths))])
        height, width = image.shape[:2]
        top = generator.randrange(height - self.crop_size + 1)
        left = generator.randrange(width - self.crop_size + 1)
        patch = image[top : top + self.crop_size, left : left + self.crop_size]
        rotation = generator.randrange(4)
        patch = np.rot90(patch, rotation)
        if generator.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if generator.random() < 0.5:
            patch = np.flip(patch, axis=1)
        return torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))


def discover_sources(root: str | Path) -> list[Path]:
    folder = Path(root)
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def synthetic_aberration(
    clean: Tensor,
    *,
    max_flow_pixels: float,
    generator: torch.Generator,
    identity_fraction: float = 0.15,
) -> tuple[Tensor, Tensor]:
    """Create bounded smooth polynomial/radial distortions directly on the GPU."""

    batch, _, height, width = clean.shape
    dtype, device = clean.dtype, clean.device
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    basis = torch.stack(
        (
            torch.ones_like(x),
            x,
            y,
            x * y,
            x.square(),
            y.square(),
            x * (x.square() + y.square()),
            y * (x.square() + y.square()),
        )
    )
    coefficients = torch.randn(
        (batch, 2, basis.shape[0]), generator=generator, device=device, dtype=dtype
    )
    coefficients[:, :, 0] *= 0.50
    raw = torch.einsum("bck,khw->bchw", coefficients, basis)
    peak = raw.abs().flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-4)
    amplitude = torch.rand((batch, 1, 1, 1), generator=generator, device=device, dtype=dtype)
    flow = raw / peak * amplitude * max_flow_pixels
    identity = (
        torch.rand((batch, 1, 1, 1), generator=generator, device=device, dtype=dtype)
        < identity_fraction
    )
    flow = torch.where(identity, torch.zeros_like(flow), flow)
    moving = warp_image(clean, flow)
    brightness = 1.0 + 0.06 * torch.randn(
        (batch, 3, 1, 1), generator=generator, device=device, dtype=dtype
    )
    offset = 0.025 * torch.randn((batch, 3, 1, 1), generator=generator, device=device, dtype=dtype)
    moving = torch.where(identity, moving, (moving * brightness + offset).clamp(0.0, 1.0))
    return moving, flow
