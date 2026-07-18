from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

RegistrationMode = Literal["hybrid", "learned", "features", "correlation", "nominal"]
BlendMode = Literal["feather", "mean"]


@dataclass(frozen=True, slots=True)
class StitchConfig:
    """Validated configuration shared by the CLI, API, and tests."""

    overlap: float = 0.10
    columns: int | None = None
    rows: int | None = None
    registration: RegistrationMode = "hybrid"
    learned_checkpoint: str = "models/raft_microscopy_v2.pt"
    blend: BlendMode = "feather"
    max_shift_fraction: float = 0.12
    min_registration_confidence: float = 0.18
    registration_max_size: int = 1024
    allow_missing_tiles: bool = True
    normalize_tile_size: bool = True
    compensate_exposure: bool = True
    feather_fraction: float = 0.08
    crop_to_valid_region: bool = True
    max_canvas_megapixels: float = 300.0
    output_bit_depth: Literal[8, 16] = 8

    def __post_init__(self) -> None:
        if not 0.02 <= self.overlap <= 0.80:
            raise ValueError("overlap must be between 0.02 and 0.80")
        if self.columns is not None and self.columns < 1:
            raise ValueError("columns must be positive")
        if self.rows is not None and self.rows < 1:
            raise ValueError("rows must be positive")
        if not 0.01 <= self.max_shift_fraction <= 0.40:
            raise ValueError("max_shift_fraction must be between 0.01 and 0.40")
        if not 0.0 <= self.min_registration_confidence <= 1.0:
            raise ValueError("min_registration_confidence must be between 0 and 1")
        if self.registration_max_size < 128:
            raise ValueError("registration_max_size must be at least 128")
        if not 0.0 <= self.feather_fraction <= 0.50:
            raise ValueError("feather_fraction must be between 0 and 0.50")
        if self.max_canvas_megapixels <= 0:
            raise ValueError("max_canvas_megapixels must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
