# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset export REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.file_response import FileDescriptorResponse
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.dataset_export import (
    DatasetExportCreateRequest,
    DatasetExportListResponse,
    DatasetExportResponse,
)
from vlm_feedback_loop.services import dataset_export_service
from vlm_feedback_loop.services.errors import map_service_error

dataset_exports_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["dataset-exports"],
)


# ── Create ─────────────────────────────────────────────────────────────────


@dataset_exports_router.post(
    "/dataset_exports",
    response_model=DatasetExportResponse,
    status_code=201,
)
async def create_dataset_export(
    project_id: str,
    body: DatasetExportCreateRequest,
    settings: Settings = Depends(get_current_settings),
) -> DatasetExportResponse:
    """Create a dataset export; the archive builds in the background.

    Returns 201 with ``status="running"`` and ``artifact_refs: null`` —
    multi-GB archives take minutes to build, so the request must not
    block on the build. Poll ``GET .../dataset_exports/{id}`` or follow
    the ``export_*`` SSE events for the terminal state.
    """
    sel_filters = (
        body.selection_filters.model_dump() if body.selection_filters else None
    )
    result = await dataset_export_service.start_dataset_export(
        project_id,
        dataset_intent=body.dataset_intent,
        label_tier_filter=body.label_tier_filter,
        export_field_mode=body.export_field_mode,
        batch_label_run_id=body.batch_label_run_id,
        selection_filters=sel_filters,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return DatasetExportResponse(**result)


# ── Get ────────────────────────────────────────────────────────────────────


@dataset_exports_router.get("/dataset_exports/{dataset_export_id}/archive")
def download_dataset_export_archive(
    project_id: str,
    dataset_export_id: str,
    settings: Settings = Depends(get_current_settings),
) -> FileDescriptorResponse:
    """Stream a completed project-scoped export through the public edge."""

    result = dataset_export_service.get_dataset_export_archive(
        project_id,
        dataset_export_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return FileDescriptorResponse(
        result.opened_file,
        media_type="application/gzip",
        filename=result.path.name,
        headers={"X-Checksum-SHA256": result.checksum_sha256},
    )


@dataset_exports_router.get(
    "/dataset_exports/{dataset_export_id}",
    response_model=DatasetExportResponse,
)
def get_dataset_export(
    project_id: str,
    dataset_export_id: str,
    settings: Settings = Depends(get_current_settings),
) -> DatasetExportResponse:
    """Get a dataset export record."""
    result = dataset_export_service.get_dataset_export(
        project_id,
        dataset_export_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return DatasetExportResponse(**result)


# ── List ───────────────────────────────────────────────────────────────────


@dataset_exports_router.get(
    "/dataset_exports",
    response_model=DatasetExportListResponse,
)
def list_dataset_exports(
    project_id: str,
    dataset_intent: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> DatasetExportListResponse:
    """List dataset exports with cursor pagination."""
    items, next_cursor = dataset_export_service.list_dataset_exports(
        project_id,
        dataset_intent_filter=dataset_intent,
        cursor=cursor,
        limit=limit,
        settings=settings,
    )
    return DatasetExportListResponse(
        items=[DatasetExportResponse(**item) for item in items],
        next_cursor=next_cursor,
    )
