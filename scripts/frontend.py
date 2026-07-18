"""Build or test the frontend using the sync-folder-safe local runtime cache."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from start import require_command, stage_frontend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "test", "typecheck"))
    parser.add_argument("--base", help="Vite base path, for example /repository/")
    parser.add_argument(
        "--static-demo",
        action="store_true",
        help="Build the read-only hosted demonstration mode.",
    )
    args = parser.parse_args()
    npm = require_command("npm.cmd" if os.name == "nt" else "npm")
    runtime = stage_frontend(npm)
    environment = os.environ.copy()
    if args.static_demo:
        environment["VITE_STATIC_DEMO"] = "true"
    if args.command == "test" and args.base:
        typecheck = subprocess.call(
            [npm, "run", "typecheck"], cwd=runtime, env=environment
        )
        if typecheck:
            return typecheck
        build = subprocess.call(
            [npm, "run", "build", "--", f"--base={args.base}"],
            cwd=runtime,
            env=environment,
        )
        if build:
            return build
        node = require_command("node.exe" if os.name == "nt" else "node")
        return subprocess.call(
            [node, "--test", "tests/ui.test.mjs"], cwd=runtime, env=environment
        )
    command = [npm, "run", args.command]
    if args.base and args.command == "build":
        command.extend(("--", f"--base={args.base}"))
    return subprocess.call(command, cwd=runtime, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
