from __future__ import annotations

import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from stitchnet.io import SUPPORTED_EXTENSIONS

MAX_ARCHIVE_FILES = 4096
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250.0

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str, fallback: str = "upload") -> str:
    cleaned = _SAFE_COMPONENT.sub("_", Path(name).name).strip("._")
    return (cleaned or fallback)[:180]


def _safe_relative_path(name: str) -> Path:
    posix = PurePosixPath(name.replace("\\", "/"))
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    if any(":" in part for part in posix.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    return Path(*(sanitize_filename(part, "item") for part in posix.parts))


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_files: int = MAX_ARCHIVE_FILES,
    max_uncompressed_bytes: int = MAX_ARCHIVE_BYTES,
) -> list[Path]:
    archive = Path(archive_path)
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    target_root = target_root.resolve()
    extracted: list[Path] = []

    with zipfile.ZipFile(archive) as bundle:
        entries = [entry for entry in bundle.infolist() if not entry.is_dir()]
        if len(entries) > max_files:
            raise ValueError(f"Archive contains {len(entries)} files; limit is {max_files}.")
        total_size = sum(entry.file_size for entry in entries)
        if total_size > max_uncompressed_bytes:
            raise ValueError("Archive exceeds the uncompressed size limit.")

        for entry in entries:
            if entry.flag_bits & 0x1:
                raise ValueError("Encrypted ZIP entries are not supported.")
            mode = entry.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise ValueError("ZIP symbolic links are not supported.")
            if entry.compress_size == 0 and entry.file_size:
                raise ValueError("Archive contains an invalid zero-size compressed entry.")
            ratio = entry.file_size / max(entry.compress_size, 1)
            if ratio > MAX_COMPRESSION_RATIO and entry.file_size > 1_000_000:
                raise ValueError("Archive contains a suspicious compression ratio.")
            relative = _safe_relative_path(entry.filename)
            if relative.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            output = (target_root / relative).resolve()
            if target_root not in output.parents:
                raise ValueError(f"Archive entry escapes extraction root: {entry.filename}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, output.open("wb") as destination_file:
                shutil.copyfileobj(source, destination_file, length=1024 * 1024)
            extracted.append(output)
    if not extracted:
        raise ValueError("ZIP archive contains no supported image tiles.")
    return extracted
