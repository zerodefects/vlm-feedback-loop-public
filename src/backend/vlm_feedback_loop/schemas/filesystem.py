# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for filesystem browse and scan endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ── Browse ──────────────────────────────────────────────────────────────────


class BrowseEntry(BaseModel):
    """Single entry (file or directory) in a browse response."""

    name: str
    type: Literal["directory", "file"]
    path: str
    size_bytes: int | None = None


class BrowseResponse(BaseModel):
    """Response body for ``GET /v1/filesystem/browse``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    parent: str | None
    entries: list[BrowseEntry]
    bundled_sample_path: str | None = None


# ── Scan ────────────────────────────────────────────────────────────────────


class ScanRequest(BaseModel):
    """Request body for ``POST /v1/filesystem/scan``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    recursive: bool = True
    project_id: str | None = None


class ScanImageEntry(BaseModel):
    """Single discovered image in a scan response."""

    storage_ref: str
    suggested_example_key: str
    size_bytes: int
    key_status: Literal[
        "available", "already_exists_same_path", "collision_different_path"
    ]
    existing_storage_ref: str | None = None


class ScanSkippedEntry(BaseModel):
    """A file skipped during scan with the reason."""

    path: str
    reason: str


class ScanResponse(BaseModel):
    """Response body for ``POST /v1/filesystem/scan``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    images: list[ScanImageEntry]
    skipped: list[ScanSkippedEntry]
    total_images: int
    total_skipped: int
    total_collisions: int
