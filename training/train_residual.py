"""Train the optional residual aberration corrector on local microscopy imagery."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from stitchnet.ml import ResidualWarpNet
from training.residual_data import RandomPatchDataset, discover_sources, synthetic_aberration


@dataclass(slots=True)
class Evaluation:
    baseline_mae: float
    corrected_mae: float
    improvement_fraction: float
    identity_mae: float
    flow_mae_pixels: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--open-world-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/residual_warp_v2.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/residual-warp-metrics.json"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--max-flow", type=float, default=7.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=7319)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _edge_loss(first: Tensor, second: Tensor) -> Tensor:
    first_x = first[:, :, :, 1:] - first[:, :, :, :-1]
    second_x = second[:, :, :, 1:] - second[:, :, :, :-1]
    first_y = first[:, :, 1:, :] - first[:, :, :-1, :]
    second_y = second[:, :, 1:, :] - second[:, :, :-1, :]
    return (first_x - second_x).abs().mean() + (first_y - second_y).abs().mean()


def _smoothness(flow: Tensor) -> Tensor:
    return (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() + (
        flow[:, :, 1:, :] - flow[:, :, :-1, :]
    ).abs().mean()


@torch.inference_mode()
def evaluate_model(
    model: ResidualWarpNet,
    loader: DataLoader[Tensor],
    device: torch.device,
    max_flow: float,
    seed: int,
) -> Evaluation:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    baseline_total = corrected_total = identity_total = flow_total = 0.0
    pixel_count = identity_count = flow_count = sample_count = 0
    amp_enabled = device.type == "cuda"
    for clean in loader:
        clean = clean.to(device, non_blocking=True)
        moving, applied_flow = synthetic_aberration(
            clean, max_flow_pixels=max_flow, generator=generator, identity_fraction=0.0
        )
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            corrected, predicted_flow = model(clean, moving)
            identity_corrected, identity_flow = model(clean, clean)
        baseline_total += float((moving - clean).abs().sum())
        corrected_total += float((corrected - clean).abs().sum())
        identity_total += float((identity_corrected - clean).abs().sum())
        flow_total += float((predicted_flow + applied_flow).abs().sum())
        pixel_count += clean.numel()
        identity_count += identity_flow.numel()
        flow_count += predicted_flow.numel()
        sample_count += clean.shape[0]
    baseline = baseline_total / max(pixel_count, 1)
    corrected = corrected_total / max(pixel_count, 1)
    return Evaluation(
        baseline_mae=baseline,
        corrected_mae=corrected,
        improvement_fraction=(baseline - corrected) / max(baseline, 1e-9),
        identity_mae=identity_total / max(pixel_count, 1),
        flow_mae_pixels=flow_total / max(flow_count, 1),
        samples=sample_count,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise SystemExit("epochs, steps-per-epoch, and batch-size must be positive")
    if args.crop_size < 96 or args.crop_size % 8:
        raise SystemExit("crop-size must be at least 96 and divisible by 8")
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    sources = discover_sources(args.train_dir)
    if len(sources) < 4:
        raise SystemExit("Training requires at least four source images.")
    split = max(1, min(len(sources) - 1, math.floor(len(sources) * 0.8)))
    train_sources, validation_sources = sources[:split], sources[split:]
    if not args.open_world_image.is_file():
        raise SystemExit(f"Open-world image is missing: {args.open_world_image}")

    train_dataset = RandomPatchDataset(
        train_sources,
        args.crop_size,
        args.steps_per_epoch * args.batch_size,
        args.seed,
    )
    validation_dataset = RandomPatchDataset(
        validation_sources, args.crop_size, max(32, args.batch_size * 4), args.seed + 1
    )
    open_world_dataset = RandomPatchDataset(
        [args.open_world_image], args.crop_size, max(32, args.batch_size * 4), args.seed + 2
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    open_world_loader = DataLoader(open_world_dataset, shuffle=False, **loader_options)

    model = ResidualWarpNet(args.max_flow).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed + 10)
    amp_enabled = device.type == "cuda"
    best_score = float("inf")
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    print(f"device={device} sources={len(train_sources)}/{len(validation_sources)}/{1}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for clean in train_loader:
            clean = clean.to(device, non_blocking=True)
            with torch.no_grad():
                moving, applied_flow = synthetic_aberration(
                    clean,
                    max_flow_pixels=args.max_flow,
                    generator=generator,
                    identity_fraction=0.18,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                corrected, predicted_flow = model(clean, moving)
                photometric = torch.sqrt((corrected - clean).square() + 1e-6).mean()
                loss = (
                    photometric
                    + 0.08 * F.smooth_l1_loss(predicted_flow, -applied_flow, beta=0.25)
                    + 0.12 * _edge_loss(corrected, clean)
                    + 0.012 * _smoothness(predicted_flow)
                    + 0.0006 * predicted_flow.square().mean()
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before_step:
                scheduler.step()
            running += float(loss.detach())

        validation = evaluate_model(
            model, validation_loader, device, args.max_flow, args.seed + 100 + epoch
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": running / len(train_loader),
            **{f"validation_{key}": value for key, value in asdict(validation).items()},
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch:02d} loss={epoch_record['train_loss']:.5f} "
            f"val={validation.corrected_mae:.5f} "
            f"improvement={validation.improvement_fraction * 100:.1f}% "
            f"identity={validation.identity_mae:.5f}"
        )
        if validation.corrected_mae < best_score:
            best_score = validation.corrected_mae
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    model.load_state_dict(best_state)
    validation = evaluate_model(model, validation_loader, device, args.max_flow, args.seed + 500)
    open_world = evaluate_model(model, open_world_loader, device, args.max_flow, args.seed + 600)
    deployable = (
        validation.improvement_fraction >= 0.10
        and open_world.improvement_fraction >= 0.05
        and max(validation.identity_mae, open_world.identity_mae) <= 0.01
    )
    metadata = {
        "format_version": 1,
        "architecture": {
            "name": "ResidualWarpNet",
            "max_flow_pixels": args.max_flow,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size,
            "crop_size": args.crop_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "train_source_count": len(train_sources),
            "validation_source_count": len(validation_sources),
        },
        "validation": asdict(validation),
        "open_world": asdict(open_world),
        "quality_gate_passed": deployable,
        "data": {
            "training_root": str(args.train_dir),
            "open_world_source": str(args.open_world_image),
            "open_world_sha256": file_sha256(args.open_world_image),
        },
        "intended_use": (
            "Experimental residual alignment for microscopy preprocessing; "
            "not clinically validated and disabled by default."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, args.output)
    metrics = {key: value for key, value in metadata.items() if key != "model_state"}
    metrics["history"] = history
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps({"validation": asdict(validation), "open_world": asdict(open_world)}, indent=2)
    )
    print(f"checkpoint={args.output} quality_gate_passed={deployable}")
    return 0 if deployable else 2


if __name__ == "__main__":
    raise SystemExit(main())
