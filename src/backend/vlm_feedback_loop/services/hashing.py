# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared file-hashing helpers.

Single home for the streamed SHA-256 file digest used by dataset export,
and the TAO workspace uploads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, *, chunk_size: int = 65536) -> str:
    """Stream a file through SHA-256 and return the hex digest.

    Memory-safe for large archives — reads ``chunk_size`` bytes at a time.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
