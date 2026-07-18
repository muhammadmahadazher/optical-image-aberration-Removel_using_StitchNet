"""Cross-platform development launcher for StitchNet Laboratory."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--ui-port", type=int, default=3000)
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.backend_only and args.frontend_only:
        parser.error("--backend-only and --frontend-only cannot be combined")
    return args


def require_command(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise SystemExit(
            f"Required command '{name}' was not found. See README.md for setup instructions."
        )
    return executable


def _frontend_cache_root() -> Path:
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "StitchNet" / "frontend-runtime-v2"


def stage_frontend(npm: str) -> Path:
    """Stage the UI outside sync folders and install only when the lockfile changes."""

    runtime = _frontend_cache_root()
    runtime.mkdir(parents=True, exist_ok=True)
    lockfile = FRONTEND / "package-lock.json"
    lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
    marker = runtime / ".stitchnet-lock-sha256"

    for filename in (
        "package.json",
        "package-lock.json",
        "index.html",
        "tsconfig.json",
        "vite.config.ts",
    ):
        shutil.copy2(FRONTEND / filename, runtime / filename)
    for directory in ("src", "public", "tests"):
        shutil.copytree(FRONTEND / directory, runtime / directory, dirs_exist_ok=True)

    installed_hash = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if installed_hash != lock_hash or not (runtime / "node_modules").is_dir():
        print("Installing locked frontend dependencies in the local cache …")
        completed = subprocess.run(
            [npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
            cwd=runtime,
            check=False,
        )
        if completed.returncode:
            raise SystemExit("Frontend dependency installation failed; see npm output above.")
        marker.write_text(lock_hash, encoding="utf-8")
    return runtime


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.3)
        return probe.connect_ex((host, port)) != 0


def wait_for_url(url: str, stop: threading.Event, timeout: float = 45.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not stop.is_set():
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status < 500:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.4)
    return False


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def main() -> int:
    args = parse_args()
    if not port_available(args.host, args.api_port) and not args.frontend_only:
        raise SystemExit(f"API port {args.api_port} is already in use on {args.host}.")
    if not port_available(args.host, args.ui_port) and not args.backend_only:
        raise SystemExit(f"UI port {args.ui_port} is already in use on {args.host}.")

    processes: list[subprocess.Popen[bytes]] = []
    stop = threading.Event()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["VITE_API_URL"] = f"http://{args.host}:{args.api_port}"

    if not args.frontend_only:
        try:
            import uvicorn  # noqa: F401
        except ImportError as error:
            raise SystemExit(
                'Backend dependencies are missing. Run: python -m pip install -e ".[ml,dev]"'
            ) from error
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.app:app",
                    "--host",
                    args.host,
                    "--port",
                    str(args.api_port),
                ],
                cwd=ROOT,
                env=environment,
            )
        )

    if not args.backend_only:
        npm = require_command("npm.cmd" if os.name == "nt" else "npm")
        frontend_runtime = stage_frontend(npm)
        processes.append(
            subprocess.Popen(
                [npm, "run", "dev", "--", "--host", args.host, "--port", str(args.ui_port)],
                cwd=frontend_runtime,
                env=environment,
            )
        )

    def handle_signal(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, handle_signal)

    ui_url = f"http://{args.host}:{args.ui_port}"
    api_url = f"http://{args.host}:{args.api_port}"
    print("\nStitchNet Laboratory")
    if not args.frontend_only:
        print(f"  API: {api_url}/docs")
    if not args.backend_only:
        print(f"  UI:  {ui_url}")
    print("  Press Ctrl+C to stop.\n")

    if not args.backend_only and not args.no_browser:
        threading.Thread(
            target=lambda: webbrowser.open(ui_url) if wait_for_url(ui_url, stop) else None,
            daemon=True,
        ).start()

    try:
        while not stop.is_set():
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    stop.set()
                    return exit_code or 1
            time.sleep(0.25)
    finally:
        stop.set()
        for process in reversed(processes):
            stop_process(process)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
