"""Evaluate a residual-warp checkpoint without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from stitchnet.ml import load_residual_warp
from training.residual_data import RandomPatchDataset
from training.train_residual import evaluate_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, checkpoint = load_residual_warp(args.checkpoint, device)
    max_flow = float(checkpoint["architecture"]["max_flow_pixels"])
    dataset = RandomPatchDataset([args.image], args.crop_size, args.samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_model(model, loader, device, max_flow, args.seed + 1)
    print(json.dumps({"device": str(device), **asdict(metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    from dataclasses import asdict

    raise SystemExit(main())
