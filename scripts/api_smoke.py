"""Submit local tiles to the running API and wait for a terminal job state."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

from stitchnet.io import SUPPORTED_EXTENSIONS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--overlap", type=float, default=0.10)
    parser.add_argument(
        "--registration",
        choices=("hybrid", "learned", "features", "correlation", "nominal"),
        default="hybrid",
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"No image tiles found in {args.input}")
    configuration = json.dumps(
        {
            "overlap": args.overlap,
            "registration": args.registration,
            "compensate_exposure": True,
            "crop_to_valid_region": True,
        }
    )
    with httpx.Client(base_url=args.api, timeout=60.0) as client:
        handles = [path.open("rb") for path in paths]
        try:
            files = [
                ("files", (path.name, handle, "application/octet-stream"))
                for path, handle in zip(paths, handles, strict=True)
            ]
            response = client.post(
                "/api/v1/jobs", data={"configuration": configuration}, files=files
            )
            response.raise_for_status()
            job = response.json()
        finally:
            for handle in handles:
                handle.close()

        deadline = time.monotonic() + args.timeout
        while job["status"] not in {"succeeded", "failed"}:
            if time.monotonic() >= deadline:
                raise SystemExit(f"Job {job['id']} did not finish before timeout.")
            time.sleep(0.25)
            response = client.get(f"/api/v1/jobs/{job['id']}")
            response.raise_for_status()
            job = response.json()
    print(json.dumps(job, indent=2))
    return 0 if job["status"] == "succeeded" else 2


if __name__ == "__main__":
    raise SystemExit(main())
