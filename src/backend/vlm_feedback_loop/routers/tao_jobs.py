# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAO job REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.tao_job import (
    TAOJobCreateRequest,
    TAOJobListResponse,
    TAOJobResponse,
)
from vlm_feedback_loop.services import tao_job_service
from vlm_feedback_loop.services.errors import map_service_error

tao_jobs_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["tao-jobs"],
)


# ── Create ────────────────────────────────────────────────────────────────


@tao_jobs_router.post(
    "/tao_jobs",
    response_model=TAOJobResponse,
    status_code=201,
)
async def create_tao_job(
    project_id: str,
    body: TAOJobCreateRequest,
    settings: Settings = Depends(get_current_settings),
) -> TAOJobResponse:
    """Submit a new TAO job."""
    result = await tao_job_service.create_tao_job(
        project_id,
        body=body,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TAOJobResponse(**result)


# ── Get ───────────────────────────────────────────────────────────────────


@tao_jobs_router.get(
    "/tao_jobs/{tao_job_id}",
    response_model=TAOJobResponse,
)
async def get_tao_job(
    project_id: str,
    tao_job_id: str,
    refresh: bool = Query(default=False),
    settings: Settings = Depends(get_current_settings),
) -> TAOJobResponse:
    """Get a TAO job, optionally refreshing from TAO."""
    result = await tao_job_service.get_tao_job(
        project_id,
        tao_job_id,
        refresh=refresh,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TAOJobResponse(**result)


# ── Cancel ────────────────────────────────────────────────────────────────


@tao_jobs_router.post(
    "/tao_jobs/{tao_job_id}:cancel",
    response_model=TAOJobResponse,
)
async def cancel_tao_job(
    project_id: str,
    tao_job_id: str,
    force_local: bool = Query(
        default=False,
        description=(
            "Skip the TAO POST :cancel call entirely and transition "
            "the local row to ``canceled`` regardless. Use ONLY when the "
            "external TAO is unreachable or the ``tao_external_job_id`` "
            "has been orphaned by a TAO server rebuild. The canceled "
            "row's ``poll_error_ref`` is stamped so the audit trail "
            "records why this was used."
        ),
    ),
    settings: Settings = Depends(get_current_settings),
) -> TAOJobResponse:
    """Cancel an in-flight TAO job and halt downstream chain siblings.

    Wired to the ``[Cancel Job]`` button on the Training Job Monitor's
    Paused state. Returns 404 if the job does not
    exist, 409 if the job is already in a terminal status (succeeded,
    failed, canceled, deleted). A failed TAO round-trip maps per the
    error contract: 502 ``tao_error`` (provider refused), 503
    ``tao_unreachable``, 504 ``tao_timeout`` — pass ``?force_local=true``
    to override when the external job is orphaned.
    """
    result = await tao_job_service.cancel_tao_job(
        project_id,
        tao_job_id,
        settings=settings,
        force_local=force_local,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TAOJobResponse(**result)


# ── List ──────────────────────────────────────────────────────────────────


@tao_jobs_router.get(
    "/tao_jobs",
    response_model=TAOJobListResponse,
)
def list_tao_jobs(
    project_id: str,
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> TAOJobListResponse:
    """List TAO jobs with cursor pagination."""
    result = tao_job_service.list_tao_jobs(
        project_id,
        status_filter=status,
        cursor=cursor,
        limit=limit,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    items, next_cursor = result
    return TAOJobListResponse(
        items=[TAOJobResponse(**item) for item in items],
        next_cursor=next_cursor,
    )
