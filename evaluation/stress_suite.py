"""Run deterministic stitching and learned-gate stress scenarios."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from stitchnet import StitchConfig, stitch_directory
from stitchnet.ml import load_raft_refiner, warp_image


def _scene(rows: int, columns: int, tile_size: int, overlap: float) -> np.ndarray:
    step = round(tile_size * (1.0 - overlap))
    height = tile_size + step * (rows - 1)
    width = tile_size + step * (columns - 1)
    rng = np.random.default_rng(9137)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 2.2)
    for index in range(55):
        center = (int(rng.integers(width)), int(rng.integers(height)))
        radius = int(rng.integers(3, 18))
        color = tuple(int(value) for value in rng.integers(20, 236, 3))
        cv2.circle(image, center, radius, color, -1, lineType=cv2.LINE_AA)
        if index % 4 == 0:
            end = (int(rng.integers(width)), int(rng.integers(height)))
            cv2.line(image, center, end, color, 2, lineType=cv2.LINE_AA)
    return image


def _write_grid(
    root: Path,
    image: np.ndarray,
    rows: int,
    columns: int,
    tile_size: int,
    overlap: float,
    *,
    exposure: bool = False,
    sixteen_bit: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    step = round(tile_size * (1.0 - overlap))
    rng = np.random.default_rng(2183)
    for row in range(rows):
        for column in range(columns):
            tile = image[
                row * step : row * step + tile_size,
                column * step : column * step + tile_size,
            ].copy()
            if exposure:
                gain = rng.uniform(0.82, 1.18, (1, 1, 3))
                bias = rng.uniform(-16.0, 16.0, (1, 1, 3))
                tile = np.clip(tile.astype(np.float32) * gain + bias, 0, 255).astype(
                    np.uint8
                )
            suffix = ".tif" if sixteen_bit else ".png"
            if sixteen_bit:
                tile = tile.astype(np.uint16) * 257
            path = root / f"tile_r{row:04d}_c{column:04d}{suffix}"
            if not cv2.imwrite(str(path), tile):
                raise OSError(f"Could not write stress tile: {path}")


def _textured_case(root: Path) -> dict[str, Any]:
    rows, columns, tile_size, overlap = 5, 5, 160, 0.25
    source = _scene(rows, columns, tile_size, overlap)
    tile_root = root / "textured"
    _write_grid(
        tile_root,
        source,
        rows,
        columns,
        tile_size,
        overlap,
        exposure=True,
    )
    started = time.perf_counter()
    result = stitch_directory(
        tile_root,
        StitchConfig(overlap=overlap, registration="hybrid", compensate_exposure=True),
    )
    elapsed = time.perf_counter() - started
    common_height = min(source.shape[0], result.image.shape[0])
    common_width = min(source.shape[1], result.image.shape[1])
    direct_mae = float(
        np.mean(
            np.abs(
                result.image[:common_height, :common_width].astype(np.float32)
                - source[:common_height, :common_width].astype(np.float32)
            )
        )
        / 255.0
    )
    passed = (
        result.report.status == "pass"
        and result.report.fallback_count == 0
        and result.report.coverage_fraction == 1.0
        and result.report.placement_p95_px < 1.5
        and result.report.seam_mae < 0.06
    )
    return {
        "passed": passed,
        "tiles": rows * columns,
        "runtime_seconds": elapsed,
        "output_shape": list(result.image.shape),
        "placement_p95_px": result.report.placement_p95_px,
        "seam_mae": result.report.seam_mae,
        "coverage_fraction": result.report.coverage_fraction,
        "fallback_count": result.report.fallback_count,
        "direct_source_mae": direct_mae,
    }


def _low_texture_case(root: Path) -> dict[str, Any]:
    tile_root = root / "low-texture"
    tile_root.mkdir(parents=True)
    for column in range(12):
        value = np.uint8(12 + column * 18)
        tile = np.full((64, 64, 3), value, np.uint8)
        if not cv2.imwrite(str(tile_root / f"tile_r0000_c{column:04d}.png"), tile):
            raise OSError("Could not write low-texture stress tile.")
    result = stitch_directory(
        tile_root,
        StitchConfig(
            overlap=0.20,
            registration="nominal",
            compensate_exposure=False,
        ),
    )
    minimum_width = 64 + round(64 * 0.80) * 11
    maximum_width = int(np.ceil(64 + 64 * 0.80 * 11))
    return {
        "passed": (
            minimum_width <= result.image.shape[1] <= maximum_width
            and result.report.coverage_fraction == 1.0
            and result.report.fallback_count == 11
        ),
        "tiles": 12,
        "output_shape": list(result.image.shape),
        "expected_width_range": [minimum_width, maximum_width],
        "coverage_fraction": result.report.coverage_fraction,
        "fallback_count": result.report.fallback_count,
        "quality_status": result.report.status,
    }


def _sixteen_bit_case(root: Path) -> dict[str, Any]:
    rows, columns, tile_size, overlap = 3, 3, 144, 0.25
    source = _scene(rows, columns, tile_size, overlap)
    tile_root = root / "sixteen-bit"
    _write_grid(
        tile_root,
        source,
        rows,
        columns,
        tile_size,
        overlap,
        sixteen_bit=True,
    )
    result = stitch_directory(
        tile_root,
        StitchConfig(overlap=overlap, output_bit_depth=16),
    )
    return {
        "passed": (
            result.image.dtype == np.uint16
            and result.report.status == "pass"
            and result.report.coverage_fraction == 1.0
            and result.report.fallback_count == 0
        ),
        "tiles": rows * columns,
        "dtype": str(result.image.dtype),
        "output_shape": list(result.image.shape),
        "placement_p95_px": result.report.placement_p95_px,
        "coverage_fraction": result.report.coverage_fraction,
        "fallback_count": result.report.fallback_count,
    }


def _learned_gate_case(checkpoint: Path) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(4771)
    texture = torch.rand((3, 128, 128), generator=generator)
    dark = torch.rand((3, 128, 128), generator=generator) * 0.035
    flat = torch.full((3, 128, 128), 0.42)
    checker = torch.zeros((3, 128, 128))
    checker[:, 20:108:12, :] = 1.0
    checker[:, :, 18:110:15] = 0.65
    reference = torch.stack((texture, dark, flat, checker)).to(device)
    applied = torch.zeros((4, 2, 128, 128), device=device)
    for index, (dx, dy) in enumerate(((2, 1), (-3, 2), (0, 0), (1, -2))):
        applied[index, 0] = dx
        applied[index, 1] = dy
    moving = warp_image(reference, applied)
    refiner, metadata = load_raft_refiner(checkpoint, device=device)
    with torch.inference_mode():
        corrected, _, accepted = refiner(reference, moving)
        identity, identity_flow, identity_accepted = refiner(reference, reference)
    baseline = (moving - reference).abs().flatten(1).mean(1)
    corrected_error = (corrected - reference).abs().flatten(1).mean(1)
    no_regression = corrected_error <= baseline + 1e-7
    exact_identity = bool(torch.equal(identity, reference))
    zero_identity_flow = bool(torch.count_nonzero(identity_flow).item() == 0)
    identity_rejected = bool(not torch.any(identity_accepted).item())
    return {
        "passed": bool(
            torch.all(no_regression).item()
            and exact_identity
            and zero_identity_flow
            and identity_rejected
            and metadata.get("quality_gate_passed") is True
        ),
        "device": str(device),
        "patterns": ["texture", "near-black", "flat", "hard-edges"],
        "baseline_mae": baseline.detach().cpu().tolist(),
        "corrected_mae": corrected_error.detach().cpu().tolist(),
        "accepted": accepted.detach().cpu().tolist(),
        "no_regression": no_regression.detach().cpu().tolist(),
        "exact_identity": exact_identity,
        "zero_identity_flow": zero_identity_flow,
        "identity_rejected": identity_rejected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/raft_microscopy_v2.pt"))
    parser.add_argument("--output", type=Path, default=Path("reports/stress-suite.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="stitchnet-stress-") as temporary:
        root = Path(temporary)
        cases = {
            "textured_exposure_5x5": _textured_case(root),
            "low_texture_1x12": _low_texture_case(root),
            "sixteen_bit_3x3": _sixteen_bit_case(root),
            "learned_no_regression_gate": _learned_gate_case(args.checkpoint),
        }
    report = {
        "suite": "deterministic-local-stress",
        "passed": all(case["passed"] for case in cases.values()),
        "runtime_seconds": time.perf_counter() - started,
        "cases": cases,
        "intended_use": "Engineering stress evidence only; not clinical validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
