# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label save, skip, restore, and rationale regeneration router."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.label import (
    LabelSaveRequest,
    LabelSaveResponse,
    RationaleRegenerationRequest,
    RationaleRegenerationResponse,
    RestoreOmittedResponse,
    SkipResponse,
)
from vlm_feedback_loop.services import label_service, rationale_service
from vlm_feedback_loop.services.errors import map_service_error
from vlm_feedback_loop.services.priority import priority_dispatch
from vlm_feedback_loop.services.project_db_locks import get_project_write_lock

labels_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["labels"],
)


# ── Save ─────────────────────────────────────────────────────────────────────


@labels_router.post("/labels", response_model=LabelSaveResponse, status_code=200)
async def save_label_endpoint(
    project_id: str,
    body: LabelSaveRequest,
    settings: Settings = Depends(get_current_settings),
) -> LabelSaveResponse:
    """Save a Verified label for an example.

    The backend computes a deterministic diff between the proposal and
    the submitted label to classify as Accept (no diff) or Edit (any diff).
    """
    async with get_project_write_lock(project_id):
        result = await asyncio.to_thread(
            label_service.save_label,
            project_id,
            example_key=body.example_key,
            inference_invocation_id=body.inference_invocation_id,
            label_json=body.label_json,
            rationale_source=body.rationale_source,
            rationale_regeneration_invocation_id=body.rationale_regeneration_invocation_id,
            workspace_root=settings.WORKSPACE_ROOT,
        )

    if isinstance(result, str):
        raise map_service_error(result)

    return LabelSaveResponse(**result)


# ── Skip ─────────────────────────────────────────────────────────────────────


@labels_router.post(
    "/examples/{example_key}:skip",
    response_model=SkipResponse,
    status_code=200,
)
async def skip_endpoint(
    project_id: str,
    example_key: str,
    settings: Settings = Depends(get_current_settings),
) -> SkipResponse:
    """Skip (omit) an example from the labeling workflow."""
    async with get_project_write_lock(project_id):
        result = await asyncio.to_thread(
            label_service.skip_example,
            project_id,
            example_key,
            settings.WORKSPACE_ROOT,
        )

    if isinstance(result, str):
        raise map_service_error(result)

    return SkipResponse(**result)


# ── Restore Omitted ──────────────────────────────────────────────────────────


@labels_router.post(
    "/examples:restore_omitted",
    response_model=RestoreOmittedResponse,
    status_code=200,
)
async def restore_omitted_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> RestoreOmittedResponse:
    """Bulk-restore all Omitted examples to Unlabeled."""
    async with get_project_write_lock(project_id):
        result = await asyncio.to_thread(
            label_service.restore_omitted,
            project_id,
            settings.WORKSPACE_ROOT,
        )

    if isinstance(result, str):
        raise map_service_error(result)

    return RestoreOmittedResponse(**result)


# ── Rationale Regeneration ───────────────────────────────────────────────────


@labels_router.post(
    "/examples/{example_key}:regenerate_rationale",
    response_model=RationaleRegenerationResponse,
    status_code=200,
)
async def regenerate_rationale_endpoint(
    project_id: str,
    example_key: str,
    body: RationaleRegenerationRequest,
    settings: Settings = Depends(get_current_settings),
) -> RationaleRegenerationResponse:
    """Regenerate rationale for an edited label.

    Calls the Teacher with the image and active task context to produce a
    fresh, independently observed rationale.
    """
    # Foreground-priority dispatch: rationale regeneration is an
    # interactive Teacher call, so hold background dispatch while it runs.
    async with priority_dispatch.foreground():
        result = await rationale_service.regenerate_rationale(
            project_id,
            example_key,
            body.teacher_model_config_id,
            settings.WORKSPACE_ROOT,
            settings,
        )

    if isinstance(result, str):
        raise map_service_error(result)

    return RationaleRegenerationResponse(**result)
