from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import StitchConfig
from .types import Tile

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

_ROW_COL_PATTERNS = (
    re.compile(r"(?:^|[_-])r(?:ow)?[_-]?(\d+)[_-]c(?:ol)?[_-]?(\d+)(?:$|[_-])", re.I),
    re.compile(r"(?:^|[_-])y[_-]?(\d+)[_-]x[_-]?(\d+)(?:$|[_-])", re.I),
    re.compile(r"(?:^|[_-])tile[_-](\d+)[_-](\d+)(?:$|[_-])", re.I),
)
_TRAILING_INDEX = re.compile(r"(?:^|[_-])(\d+)$")


@dataclass(frozen=True, slots=True)
class ParsedName:
    path: Path
    row: int | None = None
    column: int | None = None
    index: int | None = None


def discover_image_paths(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Tile folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Tile input is not a directory: {root}")
    paths = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda path: path.as_posix().lower(),
    )
    if not paths:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"No supported image tiles found in {root}. Supported: {supported}")
    return paths


def _parse_name(path: Path) -> ParsedName:
    stem = path.stem
    for pattern in _ROW_COL_PATTERNS:
        match = pattern.search(stem)
        if match:
            return ParsedName(path, row=int(match.group(1)), column=int(match.group(2)))
    match = _TRAILING_INDEX.search(stem)
    if match:
        return ParsedName(path, index=int(match.group(1)))
    return ParsedName(path)


def _factor_columns(cell_count: int) -> int:
    if cell_count <= 1:
        return 1
    lower = int(math.sqrt(cell_count))
    for rows in range(lower, 0, -1):
        if cell_count % rows == 0:
            return cell_count // rows
    return cell_count


def _assign_coordinates(
    paths: Iterable[Path], config: StitchConfig
) -> tuple[list[tuple[Path, int, int, int | None]], int, int, list[str]]:
    parsed = [_parse_name(path) for path in paths]
    warnings: list[str] = []
    explicit_count = sum(item.row is not None for item in parsed)

    if explicit_count and explicit_count != len(parsed):
        raise ValueError(
            "Mixed filename layouts detected. Use row/column names for every tile or index names for every tile."
        )

    assigned: list[tuple[Path, int, int, int | None]] = []
    if explicit_count:
        min_row = min(int(item.row) for item in parsed if item.row is not None)
        min_column = min(int(item.column) for item in parsed if item.column is not None)
        for item in parsed:
            row = int(item.row) - min_row
            column = int(item.column) - min_column
            assigned.append((item.path, row, column, None))
        rows = max(row for _, row, _, _ in assigned) + 1
        columns = max(column for _, _, column, _ in assigned) + 1
    else:
        if any(item.index is None for item in parsed):
            unknown = [item.path.name for item in parsed if item.index is None][:5]
            raise ValueError(
                "Could not infer tile positions from filenames: "
                + ", ".join(unknown)
                + ". Use names such as slide_r001_c001.tif or slide_0001.png."
            )
        indices = [int(item.index) for item in parsed if item.index is not None]
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate trailing tile indices were detected.")
        index_origin = min(indices)
        span = max(indices) - index_origin + 1
        columns = config.columns or _factor_columns(span)
        rows = config.rows or math.ceil(span / columns)
        if config.columns is None:
            warnings.append(
                f"Grid width was inferred as {columns}. Confirm it for index-only filenames."
            )
        for item in parsed:
            normalized = int(item.index) - index_origin
            row, column = divmod(normalized, columns)
            if row >= rows:
                raise ValueError(
                    f"Tile index {item.index} falls outside configured grid {rows}x{columns}."
                )
            assigned.append((item.path, row, column, int(item.index)))

    if config.rows is not None and rows > config.rows:
        raise ValueError(f"Detected {rows} rows but configuration allows {config.rows}.")
    if config.columns is not None and explicit_count and columns > config.columns:
        raise ValueError(f"Detected {columns} columns but configuration allows {config.columns}.")
    rows = config.rows or rows
    columns = config.columns or columns

    coordinates = [(row, column) for _, row, column, _ in assigned]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError(
            "Duplicate grid coordinates were detected. The upload may contain multiple slide series."
        )
    missing = rows * columns - len(assigned)
    if missing:
        if not config.allow_missing_tiles:
            raise ValueError(f"Grid contains {missing} missing tile position(s).")
        warnings.append(f"Grid contains {missing} empty position(s); they will remain transparent.")

    return (
        sorted(assigned, key=lambda item: (item[1], item[2], item[0].name)),
        rows,
        columns,
        warnings,
    )


def read_tile(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"OpenCV could not decode image tile: {image_path.name}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.ndim == 3 and raw.shape[2] == 1:
        raw = np.repeat(raw, 3, axis=2)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    elif raw.ndim != 3 or raw.shape[2] != 3:
        raise ValueError(f"Unsupported channel layout {raw.shape} in {image_path.name}")

    if np.issubdtype(raw.dtype, np.integer):
        normalized = raw.astype(np.float32) / float(np.iinfo(raw.dtype).max)
    elif np.issubdtype(raw.dtype, np.floating):
        normalized = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        low, high = np.percentile(normalized, (0.1, 99.9))
        if low < 0.0 or high > 1.0:
            span = max(float(high - low), 1e-6)
            normalized = (normalized - float(low)) / span
    else:
        raise ValueError(f"Unsupported dtype {raw.dtype} in {image_path.name}")
    return np.ascontiguousarray(np.clip(normalized, 0.0, 1.0))


def load_tiles(
    paths: Iterable[str | Path], config: StitchConfig
) -> tuple[list[Tile], int, int, list[str]]:
    resolved = [Path(path) for path in paths]
    if len(resolved) < 1:
        raise ValueError("At least one tile is required.")
    assigned, rows, columns, warnings = _assign_coordinates(resolved, config)

    loaded: list[tuple[Path, int, int, int | None, np.ndarray]] = []
    for path, row, column, index in assigned:
        loaded.append((path, row, column, index, read_tile(path)))

    effective_high = float(np.median([np.percentile(image, 99.9) for *_, image in loaded]))
    if 1e-6 < effective_high < 0.50:
        loaded = [
            (path, row, column, index, np.clip(image / effective_high, 0.0, 1.0))
            for path, row, column, index, image in loaded
        ]
        warnings.append(
            "Underfilled high-bit-depth intensity range was normalized consistently across the tile set."
        )

    heights = [image.shape[0] for *_, image in loaded]
    widths = [image.shape[1] for *_, image in loaded]
    target_height = int(np.median(heights))
    target_width = int(np.median(widths))
    if len(set(zip(heights, widths, strict=True))) > 1:
        if not config.normalize_tile_size:
            raise ValueError("Tile dimensions differ and normalize_tile_size is disabled.")
        warnings.append(
            f"Mixed tile dimensions were normalized to {target_width}x{target_height} pixels."
        )

    tiles: list[Tile] = []
    for path, row, column, index, image in loaded:
        if image.shape[:2] != (target_height, target_width):
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        key = f"r{row:04d}_c{column:04d}"
        tiles.append(Tile(key, path, row, column, image, index))
    return tiles, rows, columns, warnings


def load_tiles_from_directory(
    folder: str | Path, config: StitchConfig
) -> tuple[list[Tile], int, int, list[str]]:
    return load_tiles(discover_image_paths(folder), config)
