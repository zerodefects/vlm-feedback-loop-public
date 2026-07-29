# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO artifact names must stay inside the job's local cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_feedback_loop.services import tao_polling_service


class _S3Client:
    def __init__(self) -> None:
        self.calls = 0

    def download_fileobj(self, bucket: str, key: str, destination) -> None:
        self.calls += 1
        destination.write(f"{bucket}/{key}".encode())


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["../outside.bin", "/absolute.bin", "a//b", "a\\b"])
async def test_tao_artifact_download_rejects_paths_outside_its_cache(
    tmp_path: Path,
    name: str,
) -> None:
    """A remote artifact name cannot select a local path outside its job cache."""
    client = _S3Client()

    result = await tao_polling_service._download_workspace_s3_object(
        client,
        bucket="workspace",
        key="results/job/file",
        cache_dir=tmp_path / "cache",
        relative_name=name,
    )

    assert result["success"] is False
    assert client.calls == 0
    assert not (tmp_path / "outside.bin").exists()


@pytest.mark.asyncio
async def test_tao_artifact_download_preserves_nested_checkpoint_layout(
    tmp_path: Path,
) -> None:
    """Valid nested checkpoint files are downloaded beneath the job cache."""
    client = _S3Client()
    cache_dir = tmp_path / "cache"

    result = await tao_polling_service._download_workspace_s3_object(
        client,
        bucket="workspace",
        key="results/job/config.json",
        cache_dir=cache_dir,
        relative_name="checkpoint/config.json",
    )

    assert result["success"] is True
    assert Path(result["local_path"]) == cache_dir / "checkpoint" / "config.json"
    assert (cache_dir / "checkpoint" / "config.json").read_bytes() == (
        b"workspace/results/job/config.json"
    )
