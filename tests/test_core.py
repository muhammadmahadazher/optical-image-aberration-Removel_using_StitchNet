from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stitchnet import StitchConfig, stitch_directory, stitch_tiles
from stitchnet.blending import blend_tiles, crop_to_valid_region
from stitchnet.io import load_tiles
from stitchnet.types import MosaicLayout, Tile


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("overlap", 0.01),
        ("columns", 0),
        ("max_shift_fraction", 0.5),
        ("registration_max_size", 64),
        ("max_canvas_megapixels", 0),
    ],
)
def test_configuration_rejects_unsafe_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        StitchConfig(**{field: value})


def test_hybrid_registration_reconstructs_textured_grid(
    synthetic_tiles: tuple[Path, np.ndarray],
) -> None:
    tile_dir, expected = synthetic_tiles
    result = stitch_directory(
        tile_dir,
        StitchConfig(overlap=0.25, compensate_exposure=False, feather_fraction=0.04),
    )

    assert result.report.status == "pass"
    assert result.report.tile_count == 4
    assert result.report.fallback_count == 0
    assert result.report.placement_p95_px < 1.5
    assert result.report.coverage_fraction == pytest.approx(1.0)
    assert abs(result.image.shape[0] - expected.shape[0]) <= 2
    assert abs(result.image.shape[1] - expected.shape[1]) <= 2
    common_height = min(result.image.shape[0], expected.shape[0])
    common_width = min(result.image.shape[1], expected.shape[1])
    error = np.mean(
        np.abs(
            result.image[:common_height, :common_width].astype(np.float32)
            - expected[:common_height, :common_width].astype(np.float32)
        )
    )
    assert error < 8.0


def test_single_tile_and_16_bit_output(tmp_path: Path) -> None:
    source = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)
    path = tmp_path / "tile_0001.tif"
    assert cv2.imwrite(str(path), source)
    result = stitch_directory(tmp_path, StitchConfig(columns=1, output_bit_depth=16))
    assert result.image.dtype == np.uint16
    assert result.image.shape == (64, 80, 3)
    assert result.report.registration_count == 0
    assert result.report.coverage_fraction == 1.0


def test_mixed_dimensions_are_normalized_or_rejected(tmp_path: Path) -> None:
    paths = []
    for index, shape in enumerate([(60, 80), (64, 84)], start=1):
        path = tmp_path / f"tile_{index:04d}.png"
        assert cv2.imwrite(str(path), np.full((*shape, 3), index * 40, np.uint8))
        paths.append(path)
    tiles, _, _, warnings = load_tiles(paths, StitchConfig(columns=2))
    assert {tile.image.shape for tile in tiles} == {(62, 82, 3)}
    assert any("Mixed tile dimensions" in warning for warning in warnings)
    with pytest.raises(ValueError, match="dimensions differ"):
        load_tiles(paths, StitchConfig(columns=2, normalize_tile_size=False))


def test_duplicate_and_missing_coordinates_are_checked(tmp_path: Path) -> None:
    duplicate = []
    for name in ["a_r1_c1.png", "b_r1_c1.png"]:
        path = tmp_path / name
        assert cv2.imwrite(str(path), np.zeros((20, 20, 3), np.uint8))
        duplicate.append(path)
    with pytest.raises(ValueError, match="Duplicate grid coordinates"):
        load_tiles(duplicate, StitchConfig())

    second = tmp_path / "b_r1_c3.png"
    duplicate[1].rename(second)
    with pytest.raises(ValueError, match="missing tile"):
        load_tiles([duplicate[0], second], StitchConfig(allow_missing_tiles=False))


def test_canvas_memory_limit_prevents_large_allocation() -> None:
    tile = Tile("one", Path("one.png"), 0, 0, np.zeros((8, 8, 3), np.float32))
    layout = MosaicLayout({"one": (0.0, 0.0)}, 20_000, 20_000)
    with pytest.raises(MemoryError, match="safety limit"):
        blend_tiles([tile], layout, StitchConfig(max_canvas_megapixels=1.0))


def test_crop_uses_coverage_not_black_pixel_intensity() -> None:
    image = np.zeros((8, 10, 3), np.uint8)
    coverage = np.zeros((8, 10), np.uint8)
    coverage[1:7, 2:9] = 255
    cropped, cropped_mask, origin = crop_to_valid_region(image, coverage)
    assert origin == (2, 1)
    assert cropped.shape == (6, 7, 3)
    assert np.all(cropped == 0)
    assert np.all(cropped_mask == 255)


def test_nominal_mode_handles_featureless_tiles() -> None:
    tiles = [
        Tile(
            f"r0_c{column}",
            Path(f"tile_{column}.png"),
            0,
            column,
            np.full((40, 40, 3), 0.3 + 0.1 * column, np.float32),
        )
        for column in range(2)
    ]
    result = stitch_tiles(
        tiles,
        StitchConfig(overlap=0.25, registration="nominal", crop_to_valid_region=False),
        grid_rows=1,
        grid_columns=2,
    )
    assert result.report.fallback_count == 1
    assert result.image.shape[1] == 70
