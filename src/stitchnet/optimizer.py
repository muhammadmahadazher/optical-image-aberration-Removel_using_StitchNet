from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from .config import StitchConfig
from .types import MosaicLayout, PairwiseConstraint, Tile


def _solve_positions(
    tiles: list[Tile],
    constraints: list[PairwiseConstraint],
    config: StitchConfig,
) -> tuple[dict[str, tuple[float, float]], np.ndarray]:
    keys = [tile.key for tile in tiles]
    index_by_key = {key: index for index, key in enumerate(keys)}
    first = tiles[0]
    step_x = first.width * (1.0 - config.overlap)
    step_y = first.height * (1.0 - config.overlap)
    nominal = np.asarray(
        [(tile.column * step_x, tile.row * step_y) for tile in tiles], dtype=np.float64
    )
    connected = {constraint.source for constraint in constraints if constraint.used}
    connected.update(constraint.target for constraint in constraints if constraint.used)

    def residual(vector: np.ndarray) -> np.ndarray:
        positions = vector.reshape(-1, 2)
        values: list[float] = []
        for constraint in constraints:
            if not constraint.used:
                continue
            source = index_by_key[constraint.source]
            target = index_by_key[constraint.target]
            measured = np.asarray((constraint.dx, constraint.dy))
            difference = positions[target] - positions[source] - measured
            weight = 0.10 if constraint.method == "nominal" else 0.35 + 1.65 * constraint.confidence
            values.extend((difference * math.sqrt(weight)).tolist())
        for index, tile in enumerate(tiles):
            prior_weight = 0.30 if tile.key not in connected else 0.025
            values.extend(((positions[index] - nominal[index]) * math.sqrt(prior_weight)).tolist())
        values.extend(((positions[0] - nominal[0]) * 8.0).tolist())
        return np.asarray(values, dtype=np.float64)

    result = least_squares(
        residual,
        nominal.ravel(),
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=150,
    )
    solved = result.x.reshape(-1, 2)

    errors = []
    for constraint in constraints:
        source = index_by_key[constraint.source]
        target = index_by_key[constraint.target]
        difference = solved[target] - solved[source] - (constraint.dx, constraint.dy)
        errors.append(float(np.linalg.norm(difference)))
    return {
        key: (float(position[0]), float(position[1]))
        for key, position in zip(keys, solved, strict=True)
    }, np.asarray(errors)


def optimize_layout(
    tiles: list[Tile],
    constraints: list[PairwiseConstraint],
    config: StitchConfig,
) -> tuple[MosaicLayout, np.ndarray]:
    if not tiles:
        raise ValueError("Cannot optimize an empty tile collection.")

    positions, errors = _solve_positions(tiles, constraints, config)
    if len(errors) >= 4:
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        threshold = max(6.0, median + 4.5 * max(mad, 1.0))
        changed = False
        for constraint, error in zip(constraints, errors, strict=True):
            if constraint.method != "nominal" and error > threshold:
                constraint.used = False
                changed = True
        if changed:
            positions, errors = _solve_positions(tiles, constraints, config)

    min_x = min(positions[tile.key][0] for tile in tiles)
    min_y = min(positions[tile.key][1] for tile in tiles)
    max_x = max(positions[tile.key][0] + tile.width for tile in tiles)
    max_y = max(positions[tile.key][1] + tile.height for tile in tiles)
    offset_x = -math.floor(min_x)
    offset_y = -math.floor(min_y)
    shifted = {
        key: (position[0] + offset_x, position[1] + offset_y) for key, position in positions.items()
    }
    width = max(1, math.ceil(max_x - math.floor(min_x)))
    height = max(1, math.ceil(max_y - math.floor(min_y)))
    return MosaicLayout(shifted, width, height, offset_x, offset_y), errors
