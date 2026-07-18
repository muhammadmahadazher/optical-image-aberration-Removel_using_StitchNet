from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from stitchnet import StitchConfig, __version__
from stitchnet.io import SUPPORTED_EXTENSIONS

from .jobs import JobManager
from .security import safe_extract_zip, sanitize_filename

manager = JobManager(workers=int(os.environ.get("STITCHNET_JOB_WORKERS", "1")))
MAX_UPLOAD_BYTES = int(os.environ.get("STITCHNET_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
LEARNED_CHECKPOINT = Path(
    os.environ.get("STITCHNET_RAFT_CHECKPOINT", "models/raft_microscopy_v2.pt")
)


class JobConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlap: float = Field(default=0.10, ge=0.02, le=0.80)
    columns: int | None = Field(default=None, ge=1, le=10000)
    rows: int | None = Field(default=None, ge=1, le=10000)
    registration: Literal["hybrid", "learned", "features", "correlation", "nominal"] = "hybrid"
    blend: Literal["feather", "mean"] = "feather"
    max_shift_fraction: float = Field(default=0.12, ge=0.01, le=0.40)
    min_registration_confidence: float = Field(default=0.18, ge=0.0, le=1.0)
    registration_max_size: int = Field(default=1024, ge=128, le=4096)
    allow_missing_tiles: bool = True
    normalize_tile_size: bool = True
    compensate_exposure: bool = True
    feather_fraction: float = Field(default=0.08, ge=0.0, le=0.50)
    crop_to_valid_region: bool = True
    max_canvas_megapixels: float = Field(default=300.0, gt=0.0, le=2000.0)
    output_bit_depth: Literal[8, 16] = 8

    def to_core(self) -> StitchConfig:
        return StitchConfig(
            **self.model_dump(), learned_checkpoint=str(LEARNED_CHECKPOINT)
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    manager.shutdown()


app = FastAPI(
    title="StitchNet Laboratory API",
    version=__version__,
    description=(
        "Confidence-aware microscopy mosaic preprocessing for research use. "
        "This service is not a clinically validated diagnostic device."
    ),
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _job_or_404(job_id: str):
    try:
        return manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


async def _save_upload(upload: UploadFile, destination: Path, remaining: int) -> int:
    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > remaining:
                raise ValueError("Upload exceeds the configured total size limit.")
            output.write(chunk)
    await upload.close()
    return written


@lru_cache(maxsize=1)
def _learned_model_status() -> dict[str, object]:
    state: dict[str, object] = {
        "available": LEARNED_CHECKPOINT.is_file(),
        "quality_gate_passed": False,
        "checkpoint": LEARNED_CHECKPOINT.name,
    }
    if not state["available"]:
        return state
    try:
        import torch

        metadata = torch.load(LEARNED_CHECKPOINT, map_location="cpu", weights_only=True)
        state["quality_gate_passed"] = metadata.get("quality_gate_passed") is True
        state["architecture"] = metadata.get("architecture", {}).get("name")
    except (OSError, RuntimeError, ValueError, TypeError):
        state["available"] = False
    return state


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    cuda = False
    device = None
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda else None
    except ImportError:
        pass
    return {
        "status": "ok",
        "version": __version__,
        "opencv": cv2.__version__,
        "cuda": cuda,
        "device": device,
        "learned_model": _learned_model_status(),
        "intended_use": "research_only",
    }


@app.get("/api/v1/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "formats": sorted(SUPPORTED_EXTENSIONS),
        "archives": [".zip"],
        "registration_modes": ["hybrid", "learned", "features", "correlation", "nominal"],
        "output_bit_depths": [8, 16],
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "learned_model": _learned_model_status(),
    }


@app.post("/api/v1/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    files: Annotated[list[UploadFile], File(description="Image tiles or ZIP archives")],
    configuration: Annotated[str, Form()] = "{}",
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=422, detail="At least one file is required.")
    try:
        parsed = JobConfiguration.model_validate_json(configuration)
        core_config = parsed.to_core()
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid configuration: {exc}") from exc

    job = manager.allocate(core_config)
    total_written = 0
    try:
        for file_index, upload in enumerate(files):
            original = upload.filename or f"upload_{file_index}"
            name = sanitize_filename(original, f"upload_{file_index}")
            suffix = Path(name).suffix.lower()
            if suffix != ".zip" and suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported upload type: {original}")
            destination = job.input_dir / name
            if destination.exists():
                raise ValueError(f"Duplicate upload filename: {name}")
            total_written += await _save_upload(
                upload, destination, MAX_UPLOAD_BYTES - total_written
            )
            if suffix == ".zip":
                extraction_root = job.input_dir / f"archive_{file_index:04d}"
                safe_extract_zip(destination, extraction_root)
                destination.unlink(missing_ok=True)
        manager.submit(job.id)
        return manager.get(job.id).public()
    except (OSError, ValueError) as exc:
        manager.fail_upload(job.id, str(exc))
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "job_id": job.id},
        ) from exc


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, object]:
    return _job_or_404(job_id).public()


def _artifact(job_id: str, attribute: str, media_type: str, filename: str) -> FileResponse:
    job = _job_or_404(job_id)
    if job.status != "succeeded":
        raise HTTPException(status_code=409, detail="Job artifacts are not ready.")
    path = getattr(job, attribute)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact is missing.")
    return FileResponse(path, media_type=media_type, filename=filename)


@app.get("/api/v1/jobs/{job_id}/preview")
def preview(job_id: str) -> FileResponse:
    return _artifact(job_id, "preview_path", "image/jpeg", f"{job_id}-preview.jpg")


@app.get("/api/v1/jobs/{job_id}/result")
def result(job_id: str) -> FileResponse:
    return _artifact(job_id, "result_path", "image/png", f"{job_id}-mosaic.png")


@app.get("/api/v1/jobs/{job_id}/coverage")
def coverage(job_id: str) -> FileResponse:
    return _artifact(job_id, "coverage_path", "image/png", f"{job_id}-coverage.png")


@app.get("/api/v1/jobs/{job_id}/report")
def report(job_id: str) -> JSONResponse:
    job = _job_or_404(job_id)
    if job.status != "succeeded" or not job.report_path.exists():
        raise HTTPException(status_code=409, detail="Quality report is not ready.")
    return JSONResponse(json.loads(job.report_path.read_text(encoding="utf-8")))
