# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch labeling run REST endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import (
    get_current_settings,
    require_not_archived,
)
from vlm_feedback_loop.schemas.batch_label import (
    BatchLabelRunCancelResponse,
    BatchLabelRunCreateRequest,
    BatchLabelRunCreateResponse,
    BatchLabelRunListResponse,
    BatchLabelRunResponse,
    BatchLabelRunResumeResponse,
)
from vlm_feedback_loop.schemas.dataset_export import SchemaInvalidManifestResponse
from vlm_feedback_loop.services import batch_label_service, dataset_export_service
from vlm_feedback_loop.services.errors import map_service_error

batch_label_runs_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["batch-label-runs"],
)


# ── Start ──────────────────────────────────────────────────────────────────


@batch_label_runs_router.post(
    "/batch_label_runs",
    response_model=BatchLabelRunCreateResponse,
    status_code=201,
    dependencies=[Depends(require_not_archived)],
)
async def create_batch_label_run(
    project_id: str,
    body: BatchLabelRunCreateRequest,
    settings: Settings = Depends(get_current_settings),
) -> BatchLabelRunCreateResponse:
    """Start a new batch labeling run."""
    result = await batch_label_service.start_batch_label_run(
        project_id,
        include_auto_labeled=body.include_auto_labeled,
        run_limit=body.run_limit,
        structured_generation_mode=body.structured_generation_mode,
        ingested_after=body.ingested_after,
        ingested_before=body.ingested_before,
        concurrency=body.concurrency,
        icl_mode=body.icl_mode,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return BatchLabelRunCreateResponse(**result)


# ── Get ────────────────────────────────────────────────────────────────────


@batch_label_runs_router.get(
    "/batch_label_runs/{run_id}",
    response_model=BatchLabelRunResponse,
)
def get_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> BatchLabelRunResponse:
    """Get batch labeling run status and counters."""
    result = batch_label_service.get_batch_label_run(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return BatchLabelRunResponse(**result)


# ── List ───────────────────────────────────────────────────────────────────


@batch_label_runs_router.get(
    "/batch_label_runs",
    response_model=BatchLabelRunListResponse,
)
def list_batch_label_runs(
    project_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> BatchLabelRunListResponse:
    """List batch labeling runs with cursor pagination."""
    items, next_cursor = batch_label_service.list_batch_label_runs(
        project_id,
        status_filter=status,
        cursor=cursor,
        limit=limit,
        settings=settings,
    )
    return BatchLabelRunListResponse(
        items=[BatchLabelRunResponse(**item) for item in items],
        next_cursor=next_cursor,
    )


# ── Resume ─────────────────────────────────────────────────────────────────


@batch_label_runs_router.post(
    "/batch_label_runs/{run_id}:resume",
    response_model=BatchLabelRunResumeResponse,
)
async def resume_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Resume a paused batch labeling run."""
    result = await batch_label_service.resume_batch_label_run(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


# ── Cancel ─────────────────────────────────────────────────────────────────


@batch_label_runs_router.post(
    "/batch_label_runs/{run_id}:cancel",
    response_model=BatchLabelRunCancelResponse,
)
async def cancel_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Cancel a batch labeling run."""
    result = await batch_label_service.cancel_batch_label_run(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


# ── Schema-Invalid Manifest ─────────────────────────────────────


@batch_label_runs_router.get(
    "/batch_label_runs/{run_id}/schema_invalid_manifest",
    response_model=SchemaInvalidManifestResponse,
)
def get_schema_invalid_manifest(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> SchemaInvalidManifestResponse:
    """Download schema-invalid manifest for a batch labeling run."""
    result = dataset_export_service.get_schema_invalid_manifest(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return SchemaInvalidManifestResponse(**result)
