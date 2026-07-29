# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem browse and scan routes.

Deployment-scoped — no ``project_id`` in the path prefix.  The scan
endpoint accepts an *optional* ``project_id`` in the request body for
collision checking against a specific project's example keys.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.filesystem import (
    BrowseResponse,
    ScanRequest,
    ScanResponse,
)
from vlm_feedback_loop.services import filesystem_service

filesystem_router = APIRouter(prefix="/filesystem", tags=["filesystem"])


@filesystem_router.get("/browse", response_model=BrowseResponse)
def browse_filesystem(
    path: str | None = Query(
        default=None,
        description="Absolute directory path to list; defaults to IMAGE_ROOT",
    ),
    show_files: bool = Query(default=True, description="Include files in listing"),
    image_formats_only: bool = Query(
        default=True,
        description="When true, filter files to supported image formats",
    ),
    settings: Settings = Depends(get_current_settings),
) -> BrowseResponse:
    """Browse the backend host's filesystem.

    Deployment-scoped (no project_id).
    Returns directory entries sorted: directories first (alpha), then files (alpha).
    Hidden files excluded. Enforces the single ``IMAGE_ROOT`` boundary.
    """
    try:
        result = filesystem_service.browse_directory(
            path=path,
            settings=settings,
            show_files=show_files,
            image_formats_only=image_formats_only,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None

    return BrowseResponse(**result)


@filesystem_router.post("/scan", response_model=ScanResponse)
def scan_filesystem(
    body: ScanRequest,
    settings: Settings = Depends(get_current_settings),
) -> ScanResponse:
    """Recursively scan a directory for supported images.

    Generates deterministic ``suggested_example_key`` values for each image.
    When ``project_id`` is provided, checks each key for collisions against
    existing example keys in that project.
    """
    try:
        result = filesystem_service.scan_directory(
            path=body.path,
            settings=settings,
            recursive=body.recursive,
            project_id=body.project_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None

    return ScanResponse(**result)
