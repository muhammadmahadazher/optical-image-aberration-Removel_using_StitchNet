from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Direction = Literal["horizontal", "vertical"]


@dataclass(slots=True)
class Tile:
    key: str
    path: Path
    row: int
    column: int
    image: np.ndarray
    source_index: int | None = None

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass(slots=True)
class PairwiseConstraint:
    source: str
    target: str
    direction: Direction
    dx: float
    dy: float
    confidence: float
    method: str
    overlap_mae: float
    inliers: int = 0
    used: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class MosaicLayout:
    positions: dict[str, tuple[float, float]]
    width: int
    height: int
    offset_x: float = 0.0
    offset_y: float = 0.0


@dataclass(slots=True)
class QualityReport:
    status: Literal["pass", "review", "failed"]
    tile_count: int
    grid_rows: int
    grid_columns: int
    output_width: int
    output_height: int
    registration_count: int
    fallback_count: int
    median_confidence: float
    placement_rmse_px: float
    placement_p95_px: float
    seam_mae: float
    coverage_fraction: float
    warnings: list[str] = field(default_factory=list)
    constraints: list[dict[str, object]] = field(default_factory=list)
    intended_use: str = "Research-use microscopy preprocessing; not clinically validated."

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class StitchResult:
    image: np.ndarray
    coverage_mask: np.ndarray
    layout: MosaicLayout
    report: QualityReport

    def preview(self, max_side: int = 1600) -> np.ndarray:
        import cv2

        height, width = self.image.shape[:2]
        scale = min(1.0, max_side / max(height, width))
        if scale == 1.0:
            return self.image.copy()
        return cv2.resize(
            self.image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
