from __future__ import annotations

import io
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

import backend.app as api_module
from backend.jobs import JobManager


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_and_end_to_end_job(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path / "jobs", workers=1)
    monkeypatch.setattr(api_module, "manager", manager)
    tile = np.zeros((72, 96, 3), np.uint8)
    cv2.circle(tile, (40, 36), 18, (180, 80, 210), -1)
    payload = encode_png(tile)

    with TestClient(api_module.app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["intended_use"] == "research_only"

        created = client.post(
            "/api/v1/jobs",
            data={"configuration": '{"columns": 1, "output_bit_depth": 16}'},
            files={"files": ("slide_0001.png", io.BytesIO(payload), "image/png")},
        )
        assert created.status_code == 202
        job_id = created.json()["id"]

        deadline = time.monotonic() + 10
        job = created.json()
        while job["status"] not in {"succeeded", "failed"} and time.monotonic() < deadline:
            time.sleep(0.05)
            job = client.get(f"/api/v1/jobs/{job_id}").json()
        assert job["status"] == "succeeded", job.get("error")
        assert job["report"]["coverage_fraction"] == 1.0

        result = client.get(f"/api/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        decoded = cv2.imdecode(np.frombuffer(result.content, np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded.dtype == np.uint16
        assert decoded.shape == tile.shape
        assert client.get(f"/api/v1/jobs/{job_id}/report").status_code == 200


def test_api_rejects_unknown_file_type(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(tmp_path / "jobs", workers=1)
    monkeypatch.setattr(api_module, "manager", manager)
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/v1/jobs",
            data={"configuration": "{}"},
            files={"files": ("payload.exe", b"bad", "application/octet-stream")},
        )
    assert response.status_code == 422
    assert "Unsupported upload type" in str(response.json()["detail"])
