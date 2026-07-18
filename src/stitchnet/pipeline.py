from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .blending import blend_tiles, crop_to_valid_region
from .config import StitchConfig
from .io import load_tiles_from_directory
from .optimizer import optimize_layout
from .registration import estimate_adjacency_constraints
from .types import QualityReport, StitchResult, Tile

ProgressCallback = Callable[[float, str], None]


def _emit(progress: ProgressCallback | None, value: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(value, 0.0, 1.0)), message)


def stitch_tiles(
    tiles: list[Tile],
    config: StitchConfig | None = None,
    *,
    grid_rows: int | None = None,
    grid_columns: int | None = None,
    warnings: list[str] | None = None,
    progress: ProgressCallback | None = None,
) -> StitchResult:
    config = config or StitchConfig()
    if not tiles:
        raise ValueError("At least one tile is required.")
    rows = grid_rows or max(tile.row for tile in tiles) + 1
    columns = grid_columns or max(tile.column for tile in tiles) + 1
    report_warnings = list(warnings or [])

    _emit(progress, 0.08, "Estimating pairwise tile registration")
    constraints = estimate_adjacency_constraints(
        tiles,
        config,
        lambda value, message: _emit(progress, 0.08 + 0.52 * value, message),
    )
    if len(tiles) > 1 and not constraints:
        report_warnings.append(
            "No adjacent tile pairs were available; nominal grid placement was used."
        )

    _emit(progress, 0.64, "Optimizing globally consistent tile positions")
    layout, placement_errors = optimize_layout(tiles, constraints, config)
    _emit(progress, 0.78, "Feathering aligned overlaps")
    image, coverage = blend_tiles(tiles, layout, config, constraints)
    if config.crop_to_valid_region:
        image, coverage, (crop_x, crop_y) = crop_to_valid_region(image, coverage)
        if crop_x or crop_y or image.shape[1] != layout.width or image.shape[0] != layout.height:
            layout.positions = {
                key: (position[0] - crop_x, position[1] - crop_y)
                for key, position in layout.positions.items()
            }
            layout.width = image.shape[1]
            layout.height = image.shape[0]

    confidences = [constraint.confidence for constraint in constraints]
    used_errors = [
        error
        for error, constraint in zip(placement_errors, constraints, strict=True)
        if constraint.used
    ]
    fallback_count = sum(constraint.method == "nominal" for constraint in constraints)
    median_confidence = float(np.median(confidences)) if confidences else 1.0
    placement_rmse = float(math.sqrt(np.mean(np.square(used_errors)))) if used_errors else 0.0
    placement_p95 = float(np.percentile(used_errors, 95)) if used_errors else 0.0
    seam_mae = (
        float(np.mean([constraint.overlap_mae for constraint in constraints]))
        if constraints
        else 0.0
    )
    coverage_fraction = float(np.count_nonzero(coverage) / coverage.size)

    fallback_fraction = fallback_count / max(len(constraints), 1)
    status = "pass"
    if (
        fallback_fraction > 0.35
        or placement_p95 > 6.0
        or median_confidence < 0.20
        or seam_mae > 0.16
    ):
        status = "review"
    rejected = sum(not constraint.used for constraint in constraints)
    if fallback_count:
        report_warnings.append(
            f"{fallback_count} low-confidence neighbor registration(s) used nominal placement."
        )
    if rejected:
        report_warnings.append(
            f"{rejected} inconsistent pairwise constraint(s) were rejected by global optimization."
        )

    report = QualityReport(
        status=status,
        tile_count=len(tiles),
        grid_rows=rows,
        grid_columns=columns,
        output_width=image.shape[1],
        output_height=image.shape[0],
        registration_count=len(constraints),
        fallback_count=fallback_count,
        median_confidence=median_confidence,
        placement_rmse_px=placement_rmse,
        placement_p95_px=placement_p95,
        seam_mae=seam_mae,
        coverage_fraction=coverage_fraction,
        warnings=report_warnings,
        constraints=[constraint.to_dict() for constraint in constraints],
    )
    _emit(progress, 1.0, "Mosaic ready")
    return StitchResult(image, coverage, layout, report)


def stitch_directory(
    folder: str | Path,
    config: StitchConfig | None = None,
    progress: ProgressCallback | None = None,
) -> StitchResult:
    config = config or StitchConfig()
    _emit(progress, 0.01, "Reading and validating image tiles")
    tiles, rows, columns, warnings = load_tiles_from_directory(folder, config)
    return stitch_tiles(
        tiles,
        config,
        grid_rows=rows,
        grid_columns=columns,
        warnings=warnings,
        progress=progress,
    )
