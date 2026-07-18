"""Build and score a perturbed open-world H&E mosaic with known stage geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from stitchnet import StitchConfig, stitch_directory
from stitchnet.io import read_tile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/open-world-mosaic"))
    parser.add_argument("--report", type=Path, default=Path("reports/open-world-mosaic.json"))
    parser.add_argument("--checkpoint", default="models/raft_microscopy_v2.pt")
    parser.add_argument("--seed", type=int, default=4117)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_tiles(
    source: np.ndarray, tile_dir: Path, seed: int
) -> tuple[dict[str, tuple[int, int]], float]:
    rng = np.random.default_rng(seed)
    tile_size = 384
    overlap = 0.25
    stride = round(tile_size * (1.0 - overlap))
    extent = tile_size + 2 * stride
    margin = 12
    if source.shape[0] < extent + 2 * margin or source.shape[1] < extent + 2 * margin:
        raise ValueError("Open-world source is too small for the benchmark grid.")
    origin_y = (source.shape[0] - extent) // 2
    origin_x = (source.shape[1] - extent) // 2
    tile_dir.mkdir(parents=True, exist_ok=True)
    positions: dict[str, tuple[int, int]] = {}
    for row in range(3):
        for column in range(3):
            jitter_x = 0 if row == 0 and column == 0 else int(rng.integers(-5, 6))
            jitter_y = 0 if row == 0 and column == 0 else int(rng.integers(-5, 6))
            x = origin_x + column * stride + jitter_x
            y = origin_y + row * stride + jitter_y
            tile = source[y : y + tile_size, x : x + tile_size].copy()
            gain = rng.uniform(0.88, 1.12, (1, 1, 3))
            bias = rng.uniform(-0.025, 0.025, (1, 1, 3))
            tile = np.clip(tile * gain + bias, 0.0, 1.0)
            key = f"r{row:04d}_c{column:04d}"
            positions[key] = (x, y)
            encoded = np.round(tile * 255.0).astype(np.uint8)
            path = tile_dir / f"open_he_r{row + 1:03d}_c{column + 1:03d}.png"
            if not cv2.imwrite(str(path), encoded):
                raise RuntimeError(f"Could not write benchmark tile: {path}")
    return positions, overlap


def score_result(
    result, source: np.ndarray, truth: dict[str, tuple[int, int]]
) -> dict[str, object]:
    anchor = "r0000_c0000"
    predicted_anchor = result.layout.positions[anchor]
    truth_anchor = truth[anchor]
    errors = []
    for key, true_position in truth.items():
        predicted = result.layout.positions[key]
        predicted_relative = np.asarray(predicted) - np.asarray(predicted_anchor)
        truth_relative = np.asarray(true_position) - np.asarray(truth_anchor)
        errors.append(float(np.linalg.norm(predicted_relative - truth_relative)))

    output = result.image.astype(np.float32) / 255.0
    height, width = output.shape[:2]
    offset_x = truth_anchor[0] - predicted_anchor[0]
    offset_y = truth_anchor[1] - predicted_anchor[1]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) + offset_x,
        np.arange(height, dtype=np.float32) + offset_y,
    )
    expected = cv2.remap(
        source, grid_x, grid_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    margin = min(24, height // 8, width // 8)
    region = np.s_[margin : height - margin, margin : width - margin]
    mosaic_mae = float(np.mean(np.abs(output[region] - expected[region])))
    return {
        "ground_truth_placement_rmse_px": float(np.sqrt(np.mean(np.square(errors)))),
        "ground_truth_placement_p95_px": float(np.percentile(errors, 95)),
        "mosaic_mae": mosaic_mae,
        "quality_report": result.report.to_dict(),
    }


def main() -> int:
    args = parse_args()
    source = read_tile(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    truth, overlap = create_tiles(source, args.output_dir / "tiles", args.seed)
    results: dict[str, object] = {}
    for mode in ("hybrid", "learned"):
        result = stitch_directory(
            args.output_dir / "tiles",
            StitchConfig(
                overlap=overlap,
                registration=mode,
                learned_checkpoint=args.checkpoint,
                compensate_exposure=True,
                crop_to_valid_region=True,
            ),
        )
        mosaic_path = args.output_dir / f"mosaic-{mode}.png"
        if not cv2.imwrite(str(mosaic_path), result.image):
            raise RuntimeError(f"Could not write benchmark mosaic: {mosaic_path}")
        results[mode] = score_result(result, source, truth)

    report = {
        "benchmark": "perturbed-open-world-h-and-e-3x3",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "conditions": {
            "grid": "3x3",
            "tile_size": 384,
            "overlap": overlap,
            "maximum_stage_jitter_px": 5,
            "per_tile_gain_range": [0.88, 1.12],
            "per_tile_bias_range": [-0.025, 0.025],
            "seed": args.seed,
        },
        "results": results,
        "intended_use": "Research benchmark only; not evidence of clinical validity.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    passed = all(
        metrics["ground_truth_placement_p95_px"] < 3.0
        and metrics["mosaic_mae"] < 0.10
        and metrics["quality_report"]["coverage_fraction"] == 1.0
        for metrics in results.values()
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
