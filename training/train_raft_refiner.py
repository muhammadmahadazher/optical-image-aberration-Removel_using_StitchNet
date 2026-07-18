"""Fine-tune a gated RAFT-small microscopy correspondence refiner locally."""

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
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

from stitchnet.ml import GatedRaftRefiner, warp_image
from training.residual_data import RandomPatchDataset, discover_sources, synthetic_aberration


@dataclass(slots=True)
class Evaluation:
    baseline_mae: float
    corrected_mae: float
    improvement_fraction: float
    identity_mae: float
    flow_mae_pixels: float
    acceptance_fraction: float
    samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--open-world-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/raft_microscopy_v2.pt"))
    parser.add_argument(
        "--metrics", type=Path, default=Path("reports/raft-microscopy-metrics.json")
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--max-flow", type=float, default=7.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--flow-updates", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7823)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(
    refiner: GatedRaftRefiner,
    loader: DataLoader[Tensor],
    device: torch.device,
    max_flow: float,
    seed: int,
) -> Evaluation:
    refiner.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    baseline_total = corrected_total = identity_total = flow_total = 0.0
    pixel_count = flow_count = accepted_count = sample_count = 0
    for clean in loader:
        clean = clean.to(device, non_blocking=True)
        moving, applied = synthetic_aberration(
            clean, max_flow_pixels=max_flow, generator=generator, identity_fraction=0.0
        )
        corrected, flow, accepted = refiner(clean, moving)
        identity, _, _ = refiner(clean, clean)
        baseline_total += float((moving - clean).abs().sum())
        corrected_total += float((corrected - clean).abs().sum())
        identity_total += float((identity - clean).abs().sum())
        flow_total += float((flow + applied).abs().sum())
        pixel_count += clean.numel()
        flow_count += flow.numel()
        accepted_count += int(accepted.sum())
        sample_count += clean.shape[0]
    baseline = baseline_total / max(pixel_count, 1)
    corrected = corrected_total / max(pixel_count, 1)
    return Evaluation(
        baseline_mae=baseline,
        corrected_mae=corrected,
        improvement_fraction=(baseline - corrected) / max(baseline, 1e-9),
        identity_mae=identity_total / max(pixel_count, 1),
        flow_mae_pixels=flow_total / max(flow_count, 1),
        acceptance_fraction=accepted_count / max(sample_count, 1),
        samples=sample_count,
    )


def main() -> int:
    args = parse_args()
    if args.crop_size < 128 or args.crop_size % 8:
        raise SystemExit("crop-size must be at least 128 and divisible by 8")
    if min(args.epochs, args.steps_per_epoch, args.batch_size, args.flow_updates) < 1:
        raise SystemExit("training counts must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    sources = discover_sources(args.train_dir)
    if len(sources) < 4:
        raise SystemExit("Training requires at least four source images.")
    split = max(1, min(len(sources) - 1, math.floor(len(sources) * 0.8)))
    train_sources, validation_sources = sources[:split], sources[split:]
    train_dataset = RandomPatchDataset(
        train_sources, args.crop_size, args.steps_per_epoch * args.batch_size, args.seed
    )
    validation_dataset = RandomPatchDataset(
        validation_sources, args.crop_size, max(16, args.batch_size * 8), args.seed + 1
    )
    open_dataset = RandomPatchDataset(
        [args.open_world_image], args.crop_size, max(16, args.batch_size * 8), args.seed + 2
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    open_loader = DataLoader(open_dataset, shuffle=False, **loader_options)

    raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True).to(device)
    for parameter in raft.feature_encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in raft.context_encoder.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in raft.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader), eta_min=args.learning_rate * 0.1
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 20)
    best_score = float("inf")
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    print(
        f"device={device} trainable={sum(p.numel() for p in trainable):,} "
        f"sources={len(train_sources)}/{len(validation_sources)}/1"
    )

    for epoch in range(1, args.epochs + 1):
        raft.train()
        running = 0.0
        for clean in train_loader:
            clean = clean.to(device, non_blocking=True)
            with torch.no_grad():
                moving, applied = synthetic_aberration(
                    clean,
                    max_flow_pixels=args.max_flow,
                    generator=generator,
                    identity_fraction=0.10,
                )
            target = -applied
            optimizer.zero_grad(set_to_none=True)
            predictions = raft(
                clean * 2.0 - 1.0,
                moving * 2.0 - 1.0,
                num_flow_updates=args.flow_updates,
            )
            flow_loss = sum(
                (0.8 ** (len(predictions) - index - 1))
                * F.smooth_l1_loss(prediction, target, beta=0.25)
                for index, prediction in enumerate(predictions)
            )
            corrected = warp_image(moving, predictions[-1].clamp(-12.0, 12.0))
            photometric = torch.sqrt((corrected - clean).square() + 1e-6).mean()
            loss = flow_loss + 0.25 * photometric
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())

        refiner = GatedRaftRefiner(
            raft,
            num_flow_updates=args.flow_updates,
            minimum_improvement=0.01,
            max_flow_pixels=12.0,
        )
        validation = evaluate(
            refiner, validation_loader, device, args.max_flow, args.seed + 100 + epoch
        )
        record = {
            "epoch": float(epoch),
            "train_loss": running / len(train_loader),
            **{f"validation_{key}": value for key, value in asdict(validation).items()},
        }
        history.append(record)
        print(
            f"epoch={epoch:02d} loss={record['train_loss']:.4f} "
            f"val={validation.corrected_mae:.5f} "
            f"improvement={validation.improvement_fraction * 100:.1f}% "
            f"accepted={validation.acceptance_fraction * 100:.0f}%"
        )
        if validation.corrected_mae < best_score:
            best_score = validation.corrected_mae
            best_state = copy.deepcopy(raft.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint.")
    raft.load_state_dict(best_state)
    refiner = GatedRaftRefiner(
        raft,
        num_flow_updates=args.flow_updates,
        minimum_improvement=0.01,
        max_flow_pixels=12.0,
    ).eval()
    validation = evaluate(refiner, validation_loader, device, args.max_flow, args.seed + 500)
    open_world = evaluate(refiner, open_loader, device, args.max_flow, args.seed + 600)
    deployable = (
        validation.improvement_fraction >= 0.0
        and open_world.improvement_fraction >= 0.05
        and validation.identity_mae == 0.0
        and open_world.identity_mae == 0.0
    )
    metadata = {
        "format_version": 1,
        "architecture": {
            "name": "torchvision-raft-small-gated",
            "num_flow_updates": args.flow_updates,
            "minimum_improvement": 0.01,
            "max_flow_pixels": 12.0,
            "parameters": sum(parameter.numel() for parameter in raft.parameters()),
            "pretrained_weights": "Raft_Small_Weights.DEFAULT",
        },
        "model_state": {key: value.detach().cpu() for key, value in raft.state_dict().items()},
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
            "Quality-gated residual correspondence for microscopy preprocessing; "
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
