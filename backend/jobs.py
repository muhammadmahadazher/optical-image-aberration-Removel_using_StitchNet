from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from stitchnet import StitchConfig, stitch_directory


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Job:
    id: str
    root: Path
    config: StitchConfig
    status: str = "uploading"
    progress: float = 0.0
    message: str = "Receiving tiles"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    report: dict[str, Any] | None = None
    error: str | None = None

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def result_path(self) -> Path:
        return self.output_dir / "mosaic.png"

    @property
    def preview_path(self) -> Path:
        return self.output_dir / "preview.jpg"

    @property
    def coverage_path(self) -> Path:
        return self.output_dir / "coverage.png"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "quality-report.json"

    def public(self) -> dict[str, Any]:
        artifacts = None
        if self.status == "succeeded":
            artifacts = {
                "preview": f"/api/v1/jobs/{self.id}/preview",
                "result": f"/api/v1/jobs/{self.id}/result",
                "coverage": f"/api/v1/jobs/{self.id}/coverage",
                "report": f"/api/v1/jobs/{self.id}/report",
            }
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config.to_dict(),
            "report": self.report,
            "error": self.error,
            "artifacts": artifacts,
        }


class JobManager:
    def __init__(self, root: str | Path | None = None, workers: int = 1) -> None:
        configured = root or os.environ.get("STITCHNET_JOB_DIR", "var/jobs")
        self.root = Path(configured).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="stitchnet-job"
        )
        self._load_existing()

    def _manifest_path(self, job: Job) -> Path:
        return job.root / "job.json"

    def _persist(self, job: Job) -> None:
        job.updated_at = _now()
        temporary = self._manifest_path(job).with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job.public(), indent=2), encoding="utf-8")
        temporary.replace(self._manifest_path(job))

    def _load_existing(self) -> None:
        for manifest in self.root.glob("*/job.json"):
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                config = StitchConfig(**data.get("config", {}))
                job = Job(
                    id=data["id"],
                    root=manifest.parent,
                    config=config,
                    status=data.get("status", "failed"),
                    progress=float(data.get("progress", 0.0)),
                    message=data.get("message", ""),
                    created_at=data.get("created_at", _now()),
                    updated_at=data.get("updated_at", _now()),
                    report=data.get("report"),
                    error=data.get("error"),
                )
                if job.status in {"uploading", "queued", "running"}:
                    job.status = "failed"
                    job.error = "Job was interrupted by an application restart."
                    job.message = "Interrupted"
                    self._persist(job)
                self._jobs[job.id] = job
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue

    def allocate(self, config: StitchConfig) -> Job:
        job_id = uuid.uuid4().hex
        job = Job(job_id, self.root / job_id, config)
        job.input_dir.mkdir(parents=True)
        job.output_dir.mkdir(parents=True)
        with self._lock:
            self._jobs[job_id] = job
            self._persist(job)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return job

    def fail_upload(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self.get(job_id)
            job.status = "failed"
            job.progress = 0.0
            job.message = "Upload validation failed"
            job.error = error
            self._persist(job)

    def submit(self, job_id: str) -> None:
        with self._lock:
            job = self.get(job_id)
            job.status = "queued"
            job.progress = 0.01
            job.message = "Queued for reconstruction"
            self._persist(job)
        self._executor.submit(self._run, job_id)

    def _update_progress(self, job_id: str, progress: float, message: str) -> None:
        with self._lock:
            job = self.get(job_id)
            job.progress = float(np.clip(progress, 0.0, 0.99))
            job.message = message
            self._persist(job)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self.get(job_id)
            job.status = "running"
            job.message = "Validating tiles"
            job.progress = 0.02
            self._persist(job)
        started = time.perf_counter()
        try:
            result = stitch_directory(
                job.input_dir,
                job.config,
                lambda value, message: self._update_progress(job_id, value, message),
            )
            preview = result.preview(1800)
            if preview.dtype == np.uint16:
                preview = np.round(preview / 257.0).astype(np.uint8)
            if not cv2.imwrite(str(job.result_path), result.image):
                raise RuntimeError("Failed to encode mosaic PNG.")
            if not cv2.imwrite(str(job.preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError("Failed to encode preview JPEG.")
            if not cv2.imwrite(str(job.coverage_path), result.coverage_mask):
                raise RuntimeError("Failed to encode coverage mask.")
            report = result.report.to_dict()
            report["runtime_seconds"] = round(time.perf_counter() - started, 3)
            job.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            with self._lock:
                job = self.get(job_id)
                job.status = "succeeded"
                job.progress = 1.0
                job.message = "Mosaic ready"
                job.report = report
                job.error = None
                self._persist(job)
        except Exception as exc:  # job failures must be reflected through the API
            with self._lock:
                job = self.get(job_id)
                job.status = "failed"
                job.message = "Reconstruction failed"
                job.error = str(exc)
                self._persist(job)
            traceback.print_exc()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=False)
