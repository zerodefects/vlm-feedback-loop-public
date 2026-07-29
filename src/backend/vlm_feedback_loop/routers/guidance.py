# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance CRUD, draft validation, and edit endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.guidance import (
    DraftValidationRequest,
    DraftValidationResponse,
    GuidanceCreate,
    GuidanceListResponse,
    GuidanceResponse,
    IclCountResponse,
    ReminderDismissResponse,
    ReminderStatusResponse,
    SchemaIssueResponse,
)
from vlm_feedback_loop.schemas.schema_evolution import (
    EditExecuteResponse,
    EditPreviewResponse,
    FieldChangeResponse,
    GuidanceEditRequest,
)
from vlm_feedback_loop.services import (
    guidance_service,
    schema_evolution_service,
)
from vlm_feedback_loop.services.edit_classification import FieldChange
from vlm_feedback_loop.services.errors import map_service_error
from vlm_feedback_loop.services.project_db_locks import get_project_write_lock

guidance_router = APIRouter(
    prefix="/projects/{project_id}/guidance",
    tags=["guidance"],
)


@guidance_router.post("", status_code=201, response_model=GuidanceResponse)
async def create_guidance_endpoint(
    project_id: str,
    body: GuidanceCreate,
    settings: Settings = Depends(get_current_settings),
) -> GuidanceResponse:
    """Create a new immutable Guidance version."""
    # Join the same short-write queue used by ingest/pHash/embeddings.
    # Without this, a Guidance save can enter SQLite's busy wait while a
    # 200-image ingest batch owns the writer slot.
    async with get_project_write_lock(project_id):
        result = await asyncio.to_thread(
            guidance_service.create_guidance,
            project_id=project_id,
            description=body.description,
            schema_fields=[f.model_dump() for f in body.schema_fields],
            rules=body.rules,
            workspace_root=settings.WORKSPACE_ROOT,
        )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if isinstance(result, str):
        raise map_service_error(result)
    return GuidanceResponse.from_guidance(result)


@guidance_router.get("/{guidance_id}", response_model=GuidanceResponse)
def get_guidance_endpoint(
    project_id: str,
    guidance_id: str,
    settings: Settings = Depends(get_current_settings),
) -> GuidanceResponse:
    """Retrieve a specific Guidance version."""
    guidance = guidance_service.get_guidance(
        project_id=project_id,
        guidance_id=guidance_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if guidance is None:
        raise HTTPException(status_code=404, detail="Guidance not found")
    return GuidanceResponse.from_guidance(guidance)


@guidance_router.get("", response_model=GuidanceListResponse)
def list_guidances_endpoint(
    project_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> GuidanceListResponse:
    """List Guidance versions newest-first with cursor pagination."""
    items, next_cursor = guidance_service.list_guidances(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        cursor=cursor,
        limit=limit,
    )
    return GuidanceListResponse(
        items=[GuidanceResponse.from_guidance(g) for g in items],
        next_cursor=next_cursor,
    )


@guidance_router.post(":validate_draft", response_model=DraftValidationResponse)
def validate_draft_endpoint(
    project_id: str,
    body: DraftValidationRequest,
    settings: Settings = Depends(get_current_settings),
) -> DraftValidationResponse:
    """Validate a draft Guidance without saving.

    Uses the same ``validate_and_derive`` function as the create endpoint.
    """
    result = guidance_service.validate_draft(
        project_id=project_id,
        description=body.description,
        schema_fields=[f.model_dump() for f in body.schema_fields],
        rules=body.rules,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return DraftValidationResponse(
        issues=[
            SchemaIssueResponse(
                severity=i.severity,
                code=i.code,
                message=i.message,
                field_path=i.field_path,
            )
            for i in result.issues
        ],
        derived_json_schema=result.derived_json_schema,
        schema_hash=result.schema_hash,
        save_allowed=result.save_allowed,
    )


@guidance_router.get(":icl_count", response_model=IclCountResponse)
def icl_count_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> IclCountResponse:
    """Count ICL-eligible Edits: non-pool Verified Edits under the active Guidance."""
    result = guidance_service.get_icl_eligible_count(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return IclCountResponse(eligible_count=result)


# ── Edit ────────────────────────────────────────────────────


def _changes_to_response(changes: list[FieldChange]) -> list[FieldChangeResponse]:
    """Convert FieldChange dataclasses to response models."""
    return [
        FieldChangeResponse(
            field_id=c.field_id,
            change_type=c.change_type,
            classification=c.classification,
            detail=c.detail,
        )
        for c in changes
    ]


@guidance_router.post(":edit")
async def edit_guidance_endpoint(
    project_id: str,
    body: GuidanceEditRequest,
    settings: Settings = Depends(get_current_settings),
) -> EditPreviewResponse | EditExecuteResponse:
    """Edit existing guidance: classify, preview, or execute.

    ``dry_run=true``:  returns edit classification and affected counts.
    ``dry_run=false``: executes in-place propagation or atomic evolution.
    """
    if body.dry_run:
        result = await asyncio.to_thread(
            schema_evolution_service.preview_edit,
            project_id=project_id,
            description=body.description,
            new_schema_fields=[f.model_dump() for f in body.schema_fields],
            rules=body.rules,
            workspace_root=settings.WORKSPACE_ROOT,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if isinstance(result, str):
            raise map_service_error(result)

        cls = result.classification
        edit_type = (
            "no_change"
            if not cls.changes
            else "semantic"
            if cls.has_semantic_changes
            else "in_place"
        )

        return EditPreviewResponse(
            edit_type=edit_type,
            changes=_changes_to_response(cls.changes),
            verified_count=result.verified_count,
            auto_labeled_count=result.auto_labeled_count,
            change_summary=cls.change_summary,
        )
    else:
        async with get_project_write_lock(project_id):
            result = await asyncio.to_thread(
                schema_evolution_service.execute_edit,
                project_id=project_id,
                description=body.description,
                new_schema_fields=[f.model_dump() for f in body.schema_fields],
                rules=body.rules,
                workspace_root=settings.WORKSPACE_ROOT,
                schema_change_context_example_key=body.schema_change_context_example_key,
            )
        if result is None:
            raise HTTPException(status_code=404, detail="Project not found")
        if isinstance(result, str):
            raise map_service_error(result)

        return EditExecuteResponse(
            guidance=GuidanceResponse.from_guidance(result.guidance),
            edit_type=result.edit_type,
            verified_reverted_count=result.verified_reverted_count,
            auto_labeled_reverted_count=result.auto_labeled_reverted_count,
            changes=_changes_to_response(result.classification.changes),
        )


# ── Schema Refinement Reminders ───────────────────────────────────


@guidance_router.get(":reminder_status", response_model=ReminderStatusResponse)
def reminder_status_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ReminderStatusResponse:
    """Check schema refinement reminder status."""
    result = guidance_service.get_reminder_status(
        project_id,
        settings.WORKSPACE_ROOT,
        settings,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")

    return ReminderStatusResponse(**result)


@guidance_router.post(":dismiss_reminder", response_model=ReminderDismissResponse)
def dismiss_reminder_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ReminderDismissResponse:
    """Dismiss the current schema refinement reminder."""
    result = guidance_service.dismiss_reminder(
        project_id,
        settings.WORKSPACE_ROOT,
    )

    if isinstance(result, str):
        raise map_service_error(result)

    return ReminderDismissResponse(**result)
