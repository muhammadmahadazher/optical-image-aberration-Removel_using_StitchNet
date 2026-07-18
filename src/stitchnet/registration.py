from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import StitchConfig
from .types import Direction, PairwiseConstraint, Tile

ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class _Candidate:
    dx: float
    dy: float
    confidence: float
    method: str
    inliers: int = 0


_LEARNED_REFINERS: dict[str, tuple[object, object]] = {}


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2GRAY)


def _registration_view(image: np.ndarray, max_size: int) -> tuple[np.ndarray, np.ndarray, float]:
    gray = _gray(image)
    height, width = gray.shape
    scale = min(1.0, max_size / max(height, width))
    if scale < 1.0:
        gray = cv2.resize(
            gray,
            (max(16, round(width * scale)), max(16, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    low, high = np.percentile(gray, (1.0, 99.0))
    gray = np.clip((gray - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    gray_u8 = np.round(gray * 255.0).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_u8)
    gx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    gradient /= max(float(np.percentile(gradient, 99.0)), 1e-6)
    feature_view = 0.72 * np.clip(gradient, 0.0, 1.0) + 0.28 * (clahe / 255.0)
    return np.ascontiguousarray(feature_view.astype(np.float32)), clahe, scale


def nominal_translation(tile: Tile, direction: Direction, overlap: float) -> tuple[float, float]:
    if direction == "horizontal":
        return tile.width * (1.0 - overlap), 0.0
    return 0.0, tile.height * (1.0 - overlap)


def overlap_mae(first: np.ndarray, second: np.ndarray, dx: float, dy: float) -> float:
    """Robust photometric disagreement in the translated intersection."""

    height = min(first.shape[0], second.shape[0])
    width = min(first.shape[1], second.shape[1])
    shift_x, shift_y = int(round(dx)), int(round(dy))
    left = max(0, shift_x)
    top = max(0, shift_y)
    right = min(width, shift_x + width)
    bottom = min(height, shift_y + height)
    if right - left < 8 or bottom - top < 8:
        return 1.0

    first_crop = _gray(first[top:bottom, left:right])
    second_crop = _gray(second[top - shift_y : bottom - shift_y, left - shift_x : right - shift_x])
    if first_crop.shape != second_crop.shape or first_crop.size == 0:
        return 1.0
    first_centered = first_crop - float(np.median(first_crop))
    second_centered = second_crop - float(np.median(second_crop))
    return float(np.mean(np.abs(first_centered - second_centered)))


def _correlation_candidate(
    first: Tile, second: Tile, direction: Direction, config: StitchConfig
) -> _Candidate | None:
    first_view, _, scale = _registration_view(first.image, config.registration_max_size)
    second_view, _, _ = _registration_view(second.image, config.registration_max_size)
    height, width = first_view.shape
    overlap_axis = width if direction == "horizontal" else height
    overlap_px = max(12, round(overlap_axis * config.overlap))
    search = max(4, round(overlap_axis * config.max_shift_fraction))
    search = min(search, max(8, overlap_px))

    if direction == "horizontal":
        template_width = max(8, min(overlap_px - 2, round(overlap_px * 0.60)))
        if template_width >= overlap_px:
            return None
        template_x = width - overlap_px + (overlap_px - template_width) // 2
        margin_y = min(search, max(0, height // 4))
        template = first_view[
            margin_y : height - margin_y, template_x : template_x + template_width
        ]
        search_view = second_view[:, : min(width, overlap_px + 2 * search)]
        nominal = ((width - overlap_px), 0.0)
    else:
        template_height = max(8, min(overlap_px - 2, round(overlap_px * 0.60)))
        if template_height >= overlap_px:
            return None
        template_y = height - overlap_px + (overlap_px - template_height) // 2
        margin_x = min(search, max(0, width // 4))
        template = first_view[
            template_y : template_y + template_height, margin_x : width - margin_x
        ]
        search_view = second_view[: min(height, overlap_px + 2 * search), :]
        nominal = (0.0, (height - overlap_px))

    if (
        template.size == 0
        or search_view.shape[0] < template.shape[0]
        or search_view.shape[1] < template.shape[1]
    ):
        return None

    response = cv2.matchTemplate(search_view, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, location = cv2.minMaxLoc(response)
    if not math.isfinite(peak):
        return None
    if direction == "horizontal":
        dx_scaled = float(template_x - location[0])
        dy_scaled = float(margin_y - location[1])
    else:
        dx_scaled = float(margin_x - location[0])
        dy_scaled = float(template_y - location[1])

    response_mean = float(np.mean(response))
    response_std = float(np.std(response))
    peak_z = max(0.0, (float(peak) - response_mean) / max(response_std, 1e-6))
    confidence = float(np.clip(max(0.0, peak) * min(1.0, peak_z / 6.0), 0.0, 1.0))
    distance = math.hypot(dx_scaled - nominal[0], dy_scaled - nominal[1])
    if distance > 1.5 * search:
        confidence *= 0.1
    elif distance > search:
        confidence *= 0.5
    return _Candidate(
        dx=dx_scaled / scale,
        dy=dy_scaled / scale,
        confidence=confidence,
        method="correlation",
    )


def _feature_candidate(
    first: Tile, second: Tile, direction: Direction, config: StitchConfig
) -> _Candidate | None:
    _, first_u8, scale = _registration_view(first.image, config.registration_max_size)
    _, second_u8, _ = _registration_view(second.image, config.registration_max_size)
    height, width = first_u8.shape
    overlap_axis = width if direction == "horizontal" else height
    overlap_px = max(12, round(overlap_axis * config.overlap))
    search = max(6, round(overlap_axis * config.max_shift_fraction))
    band = min(overlap_axis, overlap_px + search)

    first_mask = np.zeros_like(first_u8)
    second_mask = np.zeros_like(second_u8)
    if direction == "horizontal":
        first_mask[:, width - band :] = 255
        second_mask[:, :band] = 255
        nominal = np.array([width - overlap_px, 0.0], dtype=np.float32)
    else:
        first_mask[height - band :, :] = 255
        second_mask[:band, :] = 255
        nominal = np.array([0.0, height - overlap_px], dtype=np.float32)

    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.01, edgeThreshold=12)
    first_points, first_desc = sift.detectAndCompute(first_u8, first_mask)
    second_points, second_desc = sift.detectAndCompute(second_u8, second_mask)
    if first_desc is None or second_desc is None or len(first_points) < 4 or len(second_points) < 4:
        return None

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(first_desc, second_desc, k=2)
    deltas: list[tuple[float, float]] = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        best, runner_up = pair
        if best.distance >= 0.76 * runner_up.distance:
            continue
        first_xy = np.asarray(first_points[best.queryIdx].pt)
        second_xy = np.asarray(second_points[best.trainIdx].pt)
        delta = first_xy - second_xy
        if np.all(np.abs(delta - nominal) <= max(search, 5)):
            deltas.append((float(delta[0]), float(delta[1])))
    if len(deltas) < 3:
        return None

    array = np.asarray(deltas, dtype=np.float32)
    median = np.median(array, axis=0)
    residuals = np.linalg.norm(array - median, axis=1)
    median_residual = float(np.median(residuals))
    threshold = max(2.5, 2.5 * median_residual)
    inlier_mask = residuals <= threshold
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 3:
        return None
    estimate = np.median(array[inlier_mask], axis=0)
    confidence = min(1.0, inliers / 24.0) * max(
        0.0, 1.0 - median_residual / max(float(search), 1.0)
    )
    return _Candidate(
        dx=float(estimate[0] / scale),
        dy=float(estimate[1] / scale),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method="features",
        inliers=inliers,
    )


def _learned_candidate(
    first: Tile, second: Tile, direction: Direction, config: StitchConfig
) -> _Candidate | None:
    try:
        import torch

        from .ml import load_raft_refiner
    except ImportError as exc:
        raise RuntimeError(
            "Learned registration requires the optional torch and torchvision dependencies."
        ) from exc

    checkpoint = str(Path(config.learned_checkpoint).resolve())
    if checkpoint not in _LEARNED_REFINERS:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LEARNED_REFINERS[checkpoint] = load_raft_refiner(checkpoint, device)
    refiner, _ = _LEARNED_REFINERS[checkpoint]
    device = next(refiner.parameters()).device

    overlap_axis = first.width if direction == "horizontal" else first.height
    overlap_px = max(32, round(overlap_axis * config.overlap))
    if direction == "horizontal":
        first_band = first.image[:, -overlap_px:]
        second_band = second.image[:, :overlap_px]
        long_size = min(first_band.shape[0], second_band.shape[0])
    else:
        first_band = first.image[-overlap_px:, :]
        second_band = second.image[:overlap_px, :]
        long_size = min(first_band.shape[1], second_band.shape[1])

    window = min(long_size, max(128, overlap_px))
    window_count = min(5, max(1, math.ceil(long_size / window)))
    starts = np.linspace(0, max(0, long_size - window), window_count).round().astype(int)
    estimates: list[np.ndarray] = []
    improvements: list[float] = []
    for start in np.unique(starts):
        if direction == "horizontal":
            reference_crop = first_band[start : start + window]
            moving_crop = second_band[start : start + window]
        else:
            reference_crop = first_band[:, start : start + window]
            moving_crop = second_band[:, start : start + window]
        height, width = reference_crop.shape[:2]
        scale = min(1.0, 512.0 / max(height, width))
        if scale < 1.0:
            size = (max(32, round(width * scale)), max(32, round(height * scale)))
            reference_crop = cv2.resize(reference_crop, size, interpolation=cv2.INTER_AREA)
            moving_crop = cv2.resize(moving_crop, size, interpolation=cv2.INTER_AREA)
        reference_tensor = torch.from_numpy(
            np.ascontiguousarray(reference_crop[..., ::-1].transpose(2, 0, 1))
        ).to(device=device, dtype=torch.float32)[None]
        moving_tensor = torch.from_numpy(
            np.ascontiguousarray(moving_crop[..., ::-1].transpose(2, 0, 1))
        ).to(device=device, dtype=torch.float32)[None]
        with torch.inference_mode():
            corrected, flow, accepted = refiner(reference_tensor, moving_tensor)
        if not bool(accepted.item()):
            continue
        margin_y = max(1, flow.shape[-2] // 8)
        margin_x = max(1, flow.shape[-1] // 8)
        core = flow[0, :, margin_y:-margin_y, margin_x:-margin_x]
        estimate = core.flatten(1).median(dim=1).values.detach().cpu().numpy() / scale
        baseline = float((moving_tensor - reference_tensor).abs().mean())
        corrected_error = float((corrected - reference_tensor).abs().mean())
        estimates.append(estimate)
        improvements.append((baseline - corrected_error) / max(baseline, 1e-6))

    if not estimates:
        return None
    array = np.asarray(estimates, dtype=np.float32)
    flow_x, flow_y = np.median(array, axis=0)
    residual = np.linalg.norm(array - np.median(array, axis=0), axis=1)
    consistency = max(0.0, 1.0 - float(np.median(residual)) / 3.0)
    confidence = float(np.clip(0.45 + np.median(improvements) + 0.25 * consistency, 0.0, 1.0))
    nominal_dx, nominal_dy = nominal_translation(first, direction, config.overlap)
    return _Candidate(
        dx=float(nominal_dx - flow_x),
        dy=float(nominal_dy - flow_y),
        confidence=confidence,
        method="learned-gated",
        inliers=len(estimates),
    )


def estimate_pairwise_constraint(
    first: Tile, second: Tile, direction: Direction, config: StitchConfig
) -> PairwiseConstraint:
    nominal_dx, nominal_dy = nominal_translation(first, direction, config.overlap)
    if config.registration == "nominal":
        return PairwiseConstraint(
            first.key,
            second.key,
            direction,
            nominal_dx,
            nominal_dy,
            0.0,
            "nominal",
            overlap_mae(first.image, second.image, nominal_dx, nominal_dy),
        )

    candidates: list[_Candidate] = []
    if config.registration == "learned":
        candidate = _learned_candidate(first, second, direction, config)
        if candidate is not None:
            candidates.append(candidate)
    if config.registration in {"hybrid", "learned", "correlation"}:
        candidate = _correlation_candidate(first, second, direction, config)
        if candidate is not None:
            candidates.append(candidate)
    if config.registration in {"hybrid", "learned", "features"}:
        candidate = _feature_candidate(first, second, direction, config)
        if candidate is not None:
            candidates.append(candidate)

    scored: list[tuple[float, float, _Candidate]] = []
    for candidate in candidates:
        mae = overlap_mae(first.image, second.image, candidate.dx, candidate.dy)
        score = candidate.confidence - 0.45 * mae
        scored.append((score, mae, candidate))

    if not scored:
        best = None
    else:
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0]
        if len(scored) > 1:
            _, _, left = scored[0]
            _, _, right = scored[1]
            agreement = math.hypot(left.dx - right.dx, left.dy - right.dy)
            if agreement <= 0.02 * max(first.width, first.height):
                total = max(left.confidence + right.confidence, 1e-6)
                combined = _Candidate(
                    dx=(left.dx * left.confidence + right.dx * right.confidence) / total,
                    dy=(left.dy * left.confidence + right.dy * right.confidence) / total,
                    confidence=min(1.0, max(left.confidence, right.confidence) + 0.10),
                    method="hybrid",
                    inliers=max(left.inliers, right.inliers),
                )
                combined_mae = overlap_mae(first.image, second.image, combined.dx, combined.dy)
                best = (combined.confidence - 0.45 * combined_mae, combined_mae, combined)

    if best is None or best[2].confidence < config.min_registration_confidence:
        dx, dy, confidence, method, inliers = nominal_dx, nominal_dy, 0.0, "nominal", 0
        mae = overlap_mae(first.image, second.image, dx, dy)
    else:
        _, mae, candidate = best
        dx, dy = candidate.dx, candidate.dy
        confidence, method, inliers = candidate.confidence, candidate.method, candidate.inliers

    return PairwiseConstraint(
        source=first.key,
        target=second.key,
        direction=direction,
        dx=float(dx),
        dy=float(dy),
        confidence=float(confidence),
        method=method,
        overlap_mae=float(mae),
        inliers=inliers,
    )


def estimate_adjacency_constraints(
    tiles: list[Tile],
    config: StitchConfig,
    progress: ProgressCallback | None = None,
) -> list[PairwiseConstraint]:
    by_coordinate = {(tile.row, tile.column): tile for tile in tiles}
    pairs: list[tuple[Tile, Tile, Direction]] = []
    for tile in tiles:
        right = by_coordinate.get((tile.row, tile.column + 1))
        below = by_coordinate.get((tile.row + 1, tile.column))
        if right is not None:
            pairs.append((tile, right, "horizontal"))
        if below is not None:
            pairs.append((tile, below, "vertical"))
    constraints: list[PairwiseConstraint] = []
    for index, (first, second, direction) in enumerate(pairs, start=1):
        constraints.append(estimate_pairwise_constraint(first, second, direction, config))
        if progress is not None:
            progress(index / max(len(pairs), 1), f"Registered neighbor pair {index}/{len(pairs)}")
    return constraints
