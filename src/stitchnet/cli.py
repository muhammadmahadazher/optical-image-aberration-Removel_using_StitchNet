from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import StitchConfig
from .pipeline import stitch_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Confidence-aware microscopy mosaic reconstruction."
    )
    parser.add_argument("input", type=Path, help="Folder containing microscopy tiles")
    parser.add_argument("-o", "--output", type=Path, default=Path("stitched_result.png"))
    parser.add_argument("--report", type=Path, default=None, help="Quality report JSON path")
    parser.add_argument("--overlap", type=float, default=0.10)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument(
        "--registration",
        choices=("hybrid", "learned", "features", "correlation", "nominal"),
        default="hybrid",
    )
    parser.add_argument("--blend", choices=("feather", "mean"), default="feather")
    parser.add_argument("--bit-depth", type=int, choices=(8, 16), default=8)
    parser.add_argument("--no-exposure-compensation", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--max-canvas-mp", type=float, default=300.0)
    parser.add_argument(
        "--learned-checkpoint",
        default="models/raft_microscopy_v2.pt",
        help="Quality-gated RAFT checkpoint used by --registration learned",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = StitchConfig(
        overlap=args.overlap,
        columns=args.columns,
        rows=args.rows,
        registration=args.registration,
        learned_checkpoint=args.learned_checkpoint,
        blend=args.blend,
        compensate_exposure=not args.no_exposure_compensation,
        crop_to_valid_region=not args.no_crop,
        output_bit_depth=args.bit_depth,
        max_canvas_megapixels=args.max_canvas_mp,
    )

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:5.1f}%] {message}", flush=True)

    result = stitch_directory(args.input, config, progress)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result.image):
        raise RuntimeError(f"OpenCV could not write output: {args.output}")
    report_path = args.report or args.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report.to_dict(), indent=2), encoding="utf-8")
    print(f"Saved mosaic: {args.output}")
    print(f"Saved report: {report_path}")
    print(f"Quality status: {result.report.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
