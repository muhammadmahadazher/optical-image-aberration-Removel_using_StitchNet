from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from backend.security import safe_extract_zip, sanitize_filename


def test_sanitize_filename_removes_paths_and_unsafe_characters() -> None:
    assert sanitize_filename("../../patient 12/slide?.tif") == "slide_.tif"
    assert sanitize_filename("...") == "upload"


@pytest.mark.parametrize("member", ["../escape.png", "/root.png", "C:/drive.png"])
def test_zip_path_traversal_is_rejected(tmp_path: Path, member: str) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"not-an-image")
    with pytest.raises(ValueError, match="Unsafe archive path"):
        safe_extract_zip(archive, tmp_path / "out")


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    archive = tmp_path / "link.zip"
    entry = zipfile.ZipInfo("tile.png")
    entry.create_system = 3
    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(entry, "target.png")
    with pytest.raises(ValueError, match="symbolic links"):
        safe_extract_zip(archive, tmp_path / "out")


def test_zip_skips_metadata_but_extracts_supported_images(tmp_path: Path) -> None:
    archive = tmp_path / "tiles.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("notes.txt", "metadata")
        bundle.writestr("tiles/tile_0001.png", b"image-bytes")
    paths = safe_extract_zip(archive, tmp_path / "out")
    assert [path.name for path in paths] == ["tile_0001.png"]
    assert paths[0].read_bytes() == b"image-bytes"
