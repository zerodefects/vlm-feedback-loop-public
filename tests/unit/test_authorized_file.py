# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral tests for authorization-bound filesystem reads."""

from __future__ import annotations

import os
import socket
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import make_settings
from vlm_feedback_loop.services import authorized_file
from vlm_feedback_loop.services.authorized_file import (
    open_authorized_image,
    open_regular_file_beneath,
)


def test_open_descriptor_keeps_authorized_inode_after_path_replacement(
    tmp_path: Path,
) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"authorized bytes")

    with open_regular_file_beneath(target, tmp_path) as opened:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(b"replacement bytes")
        os.replace(replacement, target)

        with opened.open_binary() as stream:
            assert stream.read() == b"authorized bytes"

    assert target.read_bytes() == b"replacement bytes"


def test_missing_out_of_root_path_is_denied_without_revealing_existence(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    settings = make_settings(tmp_path / "workspace", IMAGE_ROOT=str(image_root))

    with pytest.raises(PermissionError, match="outside IMAGE_ROOT"):
        open_authorized_image(tmp_path / "private" / "missing.png", settings)


@pytest.mark.parametrize("kind", ["directory", "fifo", "unix_socket"])
def test_special_files_are_rejected_as_non_regular(
    tmp_path: Path,
    kind: str,
) -> None:
    candidate = tmp_path / kind
    owner: socket.socket | None = None
    if kind == "directory":
        candidate.mkdir()
    elif kind == "fifo":
        os.mkfifo(candidate)
    else:
        owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        owner.bind(str(candidate))

    try:
        with pytest.raises(FileNotFoundError, match="regular file"):
            open_regular_file_beneath(candidate, tmp_path)
    finally:
        if owner is not None:
            owner.close()


def test_final_symlink_retarget_cannot_change_authorized_bytes(tmp_path: Path) -> None:
    inside = tmp_path / "inside.bin"
    inside.write_bytes(b"inside")
    outside_root = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_root.mkdir()
    outside = outside_root / "outside.bin"
    outside.write_bytes(b"outside")
    link = tmp_path / "current.bin"
    link.symlink_to(inside)

    original_open = authorized_file._open_canonical_regular_file

    def retarget_then_open(
        target: Path,
        root: Path,
        *,
        display_path: Path,
    ) -> authorized_file.OpenedRegularFile:
        link.unlink()
        link.symlink_to(outside)
        return original_open(target, root, display_path=display_path)

    with patch.object(
        authorized_file,
        "_open_canonical_regular_file",
        side_effect=retarget_then_open,
    ):
        with open_regular_file_beneath(link, tmp_path) as opened:
            with opened.open_binary() as stream:
                assert stream.read() == b"inside"


def test_opened_file_rejects_stream_after_close(tmp_path: Path) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"bytes")
    opened = open_regular_file_beneath(target, tmp_path)
    opened.close()

    with pytest.raises(ValueError, match="closed"):
        opened.open_binary()
