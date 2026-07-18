from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_tiles(tmp_path: Path) -> tuple[Path, np.ndarray]:
    rng = np.random.default_rng(7231)
    height = width = 224
    yy, xx = np.mgrid[:height, :width]
    base = 0.24 + 0.22 * np.sin(xx / 11.0) + 0.14 * np.cos(yy / 17.0)
    texture = base + rng.normal(0.0, 0.075, (height, width))
    image = np.stack([texture, 0.85 * texture + 0.08, 0.68 * texture + 0.16], axis=-1)
    for center_x, center_y, radius in [(35, 48, 12), (116, 76, 17), (174, 151, 14)]:
        disk = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        image[disk] = (0.82, 0.24, 0.48)
    image = np.clip(image, 0.0, 1.0)

    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    tile_size = 128
    stride = 96
    for row in range(2):
        for column in range(2):
            y, x = row * stride, column * stride
            tile = image[y : y + tile_size, x : x + tile_size]
            encoded = np.round(tile * 255.0).astype(np.uint8)
            assert cv2.imwrite(
                str(tile_dir / f"specimen_r{row + 1:03d}_c{column + 1:03d}.png"),
                encoded,
            )
    return tile_dir, np.round(image * 255.0).astype(np.uint8)
