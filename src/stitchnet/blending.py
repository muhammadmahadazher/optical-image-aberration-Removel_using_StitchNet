from __future__ import annotations

import cv2
import numpy as np

from .config import StitchConfig
from .types import MosaicLayout, PairwiseConstraint, Tile


def _feather_mask(height: int, width: int, fraction: float) -> np.ndarray:
    if fraction <= 0:
        return np.ones((height, width), dtype=np.float32)
    binary = np.pad(np.ones((height, width), dtype=np.uint8), 1)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)[1:-1, 1:-1]
    feather_pixels = max(2.0, min(height, width) * fraction)
    return np.clip(distance / feather_pixels, 0.0, 1.0).astype(np.float32)


def _overlap_crops(
    first: Tile, second: Tile, layout: MosaicLayout
) -> tuple[np.ndarray, np.ndarray] | None:
    first_x, first_y = (int(round(value)) for value in layout.positions[first.key])
    second_x, second_y = (int(round(value)) for value in layout.positions[second.key])
    left = max(first_x, second_x)
    top = max(first_y, second_y)
    right = min(first_x + first.width, second_x + second.width)
    bottom = min(first_y + first.height, second_y + second.height)
    if right - left < 8 or bottom - top < 8:
        return None
    first_crop = first.image[top - first_y : bottom - first_y, left - first_x : right - first_x]
    second_crop = second.image[
        top - second_y : bottom - second_y, left - second_x : right - second_x
    ]
    return first_crop, second_crop


def _solve_graph_values(count: int, equations: list[tuple[int, int, float, float]]) -> np.ndarray:
    if count <= 1 or not equations:
        return np.zeros(count, dtype=np.float64)
    rows = len(equations) + count + 1
    matrix = np.zeros((rows, count), dtype=np.float64)
    target = np.zeros(rows, dtype=np.float64)
    cursor = 0
    for source, destination, value, weight in equations:
        scale = max(float(weight), 0.05) ** 0.5
        matrix[cursor, source] = -scale
        matrix[cursor, destination] = scale
        target[cursor] = value * scale
        cursor += 1
    regularization = 0.035
    for index in range(count):
        matrix[cursor, index] = regularization
        cursor += 1
    matrix[cursor, 0] = 8.0
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def _exposure_parameters(
    tiles: list[Tile],
    layout: MosaicLayout,
    constraints: list[PairwiseConstraint],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    index_by_key = {tile.key: index for index, tile in enumerate(tiles)}
    tile_by_key = {tile.key: tile for tile in tiles}
    observations: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []
    for constraint in constraints:
        if not constraint.used:
            continue
        first = tile_by_key[constraint.source]
        second = tile_by_key[constraint.target]
        crops = _overlap_crops(first, second, layout)
        if crops is None:
            continue
        first_crop, second_crop = crops
        first_pixels = first_crop.reshape(-1, 3)
        second_pixels = second_crop.reshape(-1, 3)
        first_median = np.median(first_pixels, axis=0)
        second_median = np.median(second_pixels, axis=0)
        first_scale = np.percentile(first_pixels, 90, axis=0) - np.percentile(
            first_pixels, 10, axis=0
        )
        second_scale = np.percentile(second_pixels, 90, axis=0) - np.percentile(
            second_pixels, 10, axis=0
        )
        observations.append(
            (
                index_by_key[first.key],
                index_by_key[second.key],
                first_median,
                second_median,
                first_scale,
                second_scale,
                0.25 + 0.75 * constraint.confidence,
            )
        )

    gains = np.ones((len(tiles), 3), dtype=np.float64)
    for channel in range(3):
        equations = []
        for source, destination, _, _, first_scale, second_scale, weight in observations:
            if first_scale[channel] > 1e-4 and second_scale[channel] > 1e-4:
                value = float(np.log(first_scale[channel] / second_scale[channel]))
                equations.append((source, destination, value, weight))
        log_gain = np.clip(_solve_graph_values(len(tiles), equations), -np.log(1.45), np.log(1.45))
        gains[:, channel] = np.exp(log_gain)

    biases = np.zeros((len(tiles), 3), dtype=np.float64)
    for channel in range(3):
        equations = []
        for source, destination, first_median, second_median, _, _, weight in observations:
            value = (
                gains[source, channel] * first_median[channel]
                - gains[destination, channel] * second_median[channel]
            )
            equations.append((source, destination, float(value), weight))
        biases[:, channel] = np.clip(_solve_graph_values(len(tiles), equations), -0.25, 0.25)

    return {
        tile.key: (gains[index].astype(np.float32), biases[index].astype(np.float32))
        for index, tile in enumerate(tiles)
    }


def blend_tiles(
    tiles: list[Tile],
    layout: MosaicLayout,
    config: StitchConfig,
    constraints: list[PairwiseConstraint] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    megapixels = layout.width * layout.height / 1_000_000
    if megapixels > config.max_canvas_megapixels:
        raise MemoryError(
            f"Estimated canvas is {megapixels:.1f} MP, above the configured "
            f"{config.max_canvas_megapixels:.1f} MP safety limit."
        )

    accumulator = np.zeros((layout.height, layout.width, 3), dtype=np.float32)
    weights = np.zeros((layout.height, layout.width), dtype=np.float32)
    cached_masks: dict[tuple[int, int, str], np.ndarray] = {}
    exposure = (
        _exposure_parameters(tiles, layout, constraints or []) if config.compensate_exposure else {}
    )

    for tile in tiles:
        x, y = layout.positions[tile.key]
        left, top = int(round(x)), int(round(y))
        right = min(layout.width, left + tile.width)
        bottom = min(layout.height, top + tile.height)
        if right <= left or bottom <= top:
            continue
        tile_view = tile.image[: bottom - top, : right - left]
        if tile.key in exposure:
            gain, bias = exposure[tile.key]
            tile_view = np.clip(tile_view * gain + bias, 0.0, 1.0)
        mask_key = (tile.height, tile.width, config.blend)
        if mask_key not in cached_masks:
            cached_masks[mask_key] = (
                np.ones((tile.height, tile.width), dtype=np.float32)
                if config.blend == "mean"
                else _feather_mask(tile.height, tile.width, config.feather_fraction)
            )
        mask = cached_masks[mask_key][: bottom - top, : right - left]
        accumulator[top:bottom, left:right] += tile_view * mask[..., None]
        weights[top:bottom, left:right] += mask

    coverage = weights > 0
    output = np.zeros_like(accumulator)
    output[coverage] = accumulator[coverage] / weights[coverage, None]
    output = np.clip(output, 0.0, 1.0)
    if config.output_bit_depth == 16:
        encoded = np.round(output * 65535.0).astype(np.uint16)
    else:
        encoded = np.round(output * 255.0).astype(np.uint8)
    return encoded, coverage.astype(np.uint8) * 255


def crop_to_valid_region(
    image: np.ndarray, coverage_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Crop only pixels proven to be covered, never intensity-thresholded content."""

    covered = coverage_mask > 0
    if not np.any(covered):
        return image, coverage_mask, (0, 0)

    height, width = covered.shape
    scale = min(1.0, 1200.0 / max(height, width))
    if scale < 1.0:
        small_width = max(1, int(width * scale))
        small_height = max(1, int(height * scale))
        density = cv2.resize(
            covered.astype(np.float32),
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        search_mask = density >= 0.999
    else:
        search_mask = covered

    histogram = np.zeros(search_mask.shape[1], dtype=np.int32)
    best_area = 0
    best = (0, 0, search_mask.shape[1], search_mask.shape[0])
    for row_index, row in enumerate(search_mask):
        histogram = np.where(row, histogram + 1, 0)
        stack: list[tuple[int, int]] = []
        for column in range(len(histogram) + 1):
            current = int(histogram[column]) if column < len(histogram) else 0
            start = column
            while stack and stack[-1][1] > current:
                start, rectangle_height = stack.pop()
                area = rectangle_height * (column - start)
                if area > best_area:
                    best_area = area
                    best = (
                        start,
                        row_index + 1 - rectangle_height,
                        column,
                        row_index + 1,
                    )
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))

    small_left, small_top, small_right, small_bottom = best
    left = int(np.ceil(small_left / scale))
    top = int(np.ceil(small_top / scale))
    right = int(np.floor(small_right / scale))
    bottom = int(np.floor(small_bottom / scale))
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right <= left or bottom <= top:
        return image, coverage_mask, (0, 0)

    valid = covered[top:bottom, left:right]
    if valid.size == 0 or not np.all(valid):
        zero_y, zero_x = np.where(~valid)
        if zero_x.size:
            candidates = (
                (left + int(np.max(zero_x)) + 1, top, right, bottom),
                (left, top, left + int(np.min(zero_x)), bottom),
                (left, top + int(np.max(zero_y)) + 1, right, bottom),
                (left, top, right, top + int(np.min(zero_y))),
            )
            candidates = [
                candidate
                for candidate in candidates
                if candidate[2] > candidate[0] and candidate[3] > candidate[1]
            ]
            if not candidates:
                return image, coverage_mask, (0, 0)
            left, top, right, bottom = max(
                candidates,
                key=lambda candidate: (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]),
            )
            valid = covered[top:bottom, left:right]
    if valid.size == 0 or not np.all(valid):
        return image, coverage_mask, (0, 0)
    return (
        image[top:bottom, left:right],
        coverage_mask[top:bottom, left:right],
        (left, top),
    )
