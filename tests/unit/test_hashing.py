# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared streamed SHA-256 helper (services/hashing.py)."""

from __future__ import annotations

import hashlib
import random

import pytest

from vlm_feedback_loop.services.hashing import sha256_file

CHUNK = 65536  # the helper's default chunk_size


@pytest.mark.parametrize(
    "size",
    [0, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 17],
    ids=["empty", "one-below-chunk", "exact-chunk", "one-over-chunk", "multi-chunk"],
)
def test_streamed_digest_matches_whole_file_sha256(tmp_path, size):
    """Chunked streaming must be invisible in the result: for any file size
    relative to the chunk boundary, the digest equals the standard SHA-256 of
    the whole file — the value outside developers check with ``sha256sum``
    against export manifests, and that TAO upload idempotency compares
    against S3 object metadata.
    """
    # Deterministic but chunk-distinct bytes, so reordered/dropped/duplicated
    # chunks cannot collide back to the right digest.
    data = random.Random(size).randbytes(size)
    path = tmp_path / "archive.bin"
    path.write_bytes(data)

    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
