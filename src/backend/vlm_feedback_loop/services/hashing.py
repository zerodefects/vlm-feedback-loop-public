# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared file-hashing helpers.

Single home for the streamed SHA-256 file digest used by dataset export,
and the TAO workspace uploads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO


def sha256_stream(stream: BinaryIO, *, chunk_size: int = 65536) -> str:
    """Digest bytes from the stream's current position without closing it."""
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(chunk_size), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Stream a file through SHA-256 and return the hex digest.

    Memory-safe for large archives — reads ``chunk_size`` bytes at a time.
    """
    with open(path, "rb") as fh:
        return sha256_stream(fh, chunk_size=chunk_size)
