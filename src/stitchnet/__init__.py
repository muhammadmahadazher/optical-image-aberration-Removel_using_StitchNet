"""StitchNet Laboratory's confidence-aware microscopy stitching core."""

from .config import StitchConfig
from .pipeline import stitch_directory, stitch_tiles
from .types import QualityReport, StitchResult

__all__ = [
    "QualityReport",
    "StitchConfig",
    "StitchResult",
    "stitch_directory",
    "stitch_tiles",
]

__version__ = "2.0.0"
