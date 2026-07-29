# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project CRUD endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from vlm_feedback_loop.config import Settings, get_settings
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.schemas.action_request import (
    ActionRequestGenerateRequest,
    ActionRequestGenerateResponse,
    ActionRequestLogCopyRequest,
    ActionRequestLogCopyResponse,
)
from vlm_feedback_loop.schemas.project import (
    MarkSetupCompletedRequest,
    MarkSetupCompletedResponse,
    ProjectCounts,
    ProjectCreate,
    ProjectListItem,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from vlm_feedback_loop.services import action_requests, project_service
from vlm_feedback_loop.services.errors import not_found
from vlm_feedback_loop.services.project_db_locks import get_project_write_lock
from vlm_feedback_loop.services.project_service import ProjectArchivedError
from vlm_feedback_loop.services.sse import sse_manager

router = APIRouter(prefix="/projects", tags=["projects"])


def get_current_settings() -> Settings:
    """Dependency — importable for ``app.dependency_overrides`` in tests."""
    return get_settings()


def require_not_archived(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> None:
    """FastAPI dependency: raise ``ProjectArchivedError`` if archived.

    Mounted on the long-running or job-launching mutating endpoints that must
    refuse work on an archived (paused) project: evaluation-run,
    batch-label-run, and training-suite creation, plus project PATCH. Lighter
    interactive edits (Guidance create, image ingest, label save) are not gated
    here — archiving is an operator action on an idle project, and startup
    recovery plus the one-active-session invariant already keep an archived
    project from running background work. The marker file is consulted first
    as a cheap pre-check; the DB column is authoritative.
    """
    if project_service.is_project_archived(project_id, settings.WORKSPACE_ROOT):
        raise ProjectArchivedError(project_id)


def require_project(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> Project:
    """FastAPI dependency: 404 unless the project exists.

    Returns the (detached) Project row for handlers that use it; mount as
    ``dependencies=[Depends(require_project)]`` for a bare existence guard.
    Single-process lock: if another process holds the project lock,
    project_service raises ProjectLockedError, which the global handler in
    main.py converts to 409 so the UI can show the project-locked screen.
    """
    project = project_service.get_project(project_id, settings.WORKSPACE_ROOT)
    if project is None:
        raise not_found("Project not found")
    return project


@router.post("", status_code=201)
async def create_project(
    body: ProjectCreate,
    settings: Settings = Depends(get_current_settings),
) -> ProjectResponse:
    # Hardware size is an eligibility gate, not the quality policy. Resolve
    # the curated local recommendation before seeding so a fresh project only
    # adopts a running resident when it is still the preferred model on this
    # host; a different resident remains an explicit keep/replace choice.
    from vlm_feedback_loop.services.environment import (
        pick_local_teacher_recommendation,
        probe_gpu_inventory,
    )

    local_recommendation = pick_local_teacher_recommendation(
        await probe_gpu_inventory()
    )
    project = project_service.create_project(
        name=body.name,
        description=body.description,
        settings=settings,
        preferred_local_teacher_model_name=(
            str(local_recommendation["model_name"])
            if local_recommendation is not None
            else None
        ),
    )
    # Embedding-NIM probe at project creation
    from vlm_feedback_loop.services.clip_embedding_service import (
        probe_and_set_embedding_provider,
    )

    engine = project_service.get_project_engine(
        project.project_id, settings.WORKSPACE_ROOT
    )
    if engine is not None:
        updated = await probe_and_set_embedding_provider(
            project.project_id, engine, settings
        )
        if updated is not None:
            project = updated

    return _build_project_response(project, settings.WORKSPACE_ROOT)


def _build_project_response(project: Project, workspace_root: str) -> ProjectResponse:
    """Compose a ProjectResponse including current example-state counts."""
    counts_dict = project_service.get_project_counts(project.project_id, workspace_root)
    response = ProjectResponse.model_validate(project)
    if counts_dict is not None:
        return response.model_copy(update={"counts": ProjectCounts(**counts_dict)})
    return response


@router.get("", response_model=ProjectListResponse)
def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    include_archived: bool = Query(
        default=False,
        description=(
            "Include soft-archived projects in the response. "
            "Default false matches the project list screen (active projects only)."
        ),
    ),
    settings: Settings = Depends(get_current_settings),
) -> ProjectListResponse:
    items, next_cursor = project_service.list_projects(
        workspace_root=settings.WORKSPACE_ROOT,
        cursor=cursor,
        limit=limit,
        include_archived=include_archived,
    )
    # After list_projects so any marker reconciliation it performed is
    # already visible to the marker scan.
    has_archived = project_service.has_archived_projects(settings.WORKSPACE_ROOT)
    return ProjectListResponse(
        items=[
            ProjectListItem(
                project_id=item["project_id"],
                name=item["name"],
                description=item["description"],
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                counts=ProjectCounts(**item["counts"]),
                archived_at=item.get("archived_at"),
                setup_completed_at=item.get("setup_completed_at"),
            )
            for item in items
        ],
        next_cursor=next_cursor,
        has_archived=has_archived,
    )


@router.get("/{project_id}")
def read_project(
    project: Project = Depends(require_project),
    settings: Settings = Depends(get_current_settings),
) -> ProjectResponse:
    return _build_project_response(project, settings.WORKSPACE_ROOT)


@router.patch("/{project_id}", dependencies=[Depends(require_not_archived)])
async def update_project(
    project_id: str,
    body: ProjectUpdate,
    project: Project = Depends(require_project),
    settings: Settings = Depends(get_current_settings),
) -> ProjectResponse:
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        # Nothing to update — return current state
        return _build_project_response(project, settings.WORKSPACE_ROOT)

    try:
        async with get_project_write_lock(project_id):
            updated = await asyncio.to_thread(
                project_service.update_project,
                project_id=project_id,
                updates=updates,
                workspace_root=settings.WORKSPACE_ROOT,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _build_project_response(updated, settings.WORKSPACE_ROOT)


@router.post("/{project_id}:archive")
async def archive_project_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ProjectResponse:
    """Soft-archive a project.

    Returns 409 ``already_archived`` if already archived, 409
    ``project_busy`` if any RunRecord/TAOJob/LocalNimDeployment is
    non-terminal, 409 ``project_in_use`` if another process holds the
    file lock (mapped via ``ProjectLockedError``).
    """
    try:
        project = project_service.archive_project(project_id, settings.WORKSPACE_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    await sse_manager.emit(
        project_id=project_id,
        event_type="project_archived",
        data={"project_id": project_id, "archived_at": project.archived_at},
    )
    return _build_project_response(project, settings.WORKSPACE_ROOT)


@router.post("/{project_id}:unarchive")
async def unarchive_project_endpoint(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ProjectResponse:
    """Unarchive a project.

    Returns 409 ``not_archived`` if the project is not currently archived.
    """
    try:
        project = project_service.unarchive_project(project_id, settings.WORKSPACE_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    await sse_manager.emit(
        project_id=project_id,
        event_type="project_unarchived",
        data={"project_id": project_id},
    )
    return _build_project_response(project, settings.WORKSPACE_ROOT)


@router.post("/{project_id}:mark_setup_completed")
async def mark_setup_completed_endpoint(
    project_id: str,
    body: MarkSetupCompletedRequest,
    settings: Settings = Depends(get_current_settings),
) -> MarkSetupCompletedResponse:
    """Stamp ``setup_completed_at`` on first transition; idempotent on repeat.

    Called by the frontend whenever the SME exits onboarding (NIM
    connection auto-skip or manual [Continue], embedding setup auto-skip
    or manual Save). ``ProjectIndexRedirect`` gates the setup route on this
    field — once stamped, the SME proceeds straight to labeling on
    subsequent project opens.

    Idempotency is enforced in the service layer: if the field is already
    non-null, the call is a no-op (no double-stamp, no duplicate
    ``setup_completed`` AuditEvent). The response's ``transitioned`` flag
    tells the caller whether work was actually persisted.
    """
    try:
        project, transitioned = project_service.mark_setup_completed(
            project_id,
            settings.WORKSPACE_ROOT,
            auto_skip=body.auto_skip,
            teacher_mode=body.teacher_mode,
            embedding_mode=body.embedding_mode,
            embedding_provider=body.embedding_provider,
            local_deploy_queued=body.local_deploy_queued,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Project not found") from exc

    return MarkSetupCompletedResponse(
        transitioned=transitioned,
        project=_build_project_response(project, settings.WORKSPACE_ROOT),
    )


@router.get("/{project_id}/events", dependencies=[Depends(require_project)])
async def project_events(project_id: str) -> StreamingResponse:
    """SSE stream for project-scoped events."""
    return StreamingResponse(
        sse_manager.stream(project_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Action Requests ──────────────────────────────────────────────


@router.post("/{project_id}/action_requests:generate")
def generate_action_request_endpoint(
    project_id: str,
    body: ActionRequestGenerateRequest,
    project: Project = Depends(require_project),
) -> ActionRequestGenerateResponse:
    """Generate a pre-filled Action Request for infrastructure handoff."""
    try:
        result = action_requests.generate_action_request(
            request_type=body.request_type,
            project_name=project.name,
            project_id=project_id,
            context=body.context.model_dump() if body.context else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ActionRequestGenerateResponse(**result)


@router.post(
    "/{project_id}/action_requests:log_copy",
    dependencies=[Depends(require_project)],
)
def log_action_request_copy(
    project_id: str,
    body: ActionRequestLogCopyRequest,
    settings: Settings = Depends(get_current_settings),
) -> ActionRequestLogCopyResponse:
    """Log that the SME copied an Action Request to clipboard."""
    engine = project_service.get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        raise HTTPException(status_code=404, detail="Project not found")
    audit_event_id = action_requests.log_copy(
        engine,
        project_id,
        request_type=body.request_type,
        rendered_text=body.rendered_text,
    )
    return ActionRequestLogCopyResponse(audit_event_id=audit_event_id)
