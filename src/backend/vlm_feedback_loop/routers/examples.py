# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image ingestion, example query, image serving, and path remapping routes."""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.file_response import FileDescriptorResponse
from vlm_feedback_loop.routers.projects import get_current_settings, require_project
from vlm_feedback_loop.schemas.example import (
    EmbeddingStatusResponse,
    ExampleIngestRequest,
    ExampleIngestResponse,
    ExampleIngestResultItem,
    ExampleQueryItem,
    ExampleQueryResponse,
    ExampleResponse,
    RemapPathsRequest,
    RemapResponse,
    RemapSampleItem,
    RemapValidation,
    VerifiedLabelResponse,
)
from vlm_feedback_loop.services import example_service, filesystem_service
from vlm_feedback_loop.services.authorized_file import open_authorized_image
from vlm_feedback_loop.services.image_transport import EXT_TO_MIME
from vlm_feedback_loop.services.pagination import InvalidCursorError

examples_router = APIRouter(
    prefix="/projects/{project_id}/examples",
    tags=["examples"],
)


# ── Ingestion ───────────────────────────────────────────────────────────────


@examples_router.post(
    ":ingest",
    response_model=ExampleIngestResponse,
    status_code=202,
    dependencies=[Depends(require_project)],
)
async def ingest_examples(
    project_id: str,
    body: ExampleIngestRequest,
    settings: Settings = Depends(get_current_settings),
) -> ExampleIngestResponse:
    """Batch ingest images into a project.

    Per-item processing with partial success.  Idempotent by example_key:
    same key + same path returns existing record; same key + different path
    is rejected with ``example_key_collision``.

    This endpoint creates skeleton ``Example`` rows
    with ``phash=None`` and returns **202 Accepted** in ~1s, then schedules
    two background workers:

    * ``trigger_ingest_processing`` — pHash sweeper. Reads
      ``Example WHERE phash IS NULL``, computes pHash per row, writes
      back in short transactions, emits ``ingest_progress`` /
      ``ingest_completed`` SSE events.
    * ``trigger_embedding_computation`` — CLIP embedding worker.
      Reads ``Example WHERE example_key NOT IN ClipEmbedding`` and emits
      ``embedding_progress`` / ``embedding_completed`` SSE events.

    The two workers run concurrently and write disjoint columns. Image
    validation (PIL header parse + multi-page TIFF / animated GIF /
    unsupported format rejection) still happens synchronously inside
    the endpoint so a bad file surfaces as ``status="error"`` instead
    of creating a skeleton row the sweeper can never populate.

    The route is ``async def`` so the post-ingest worker triggers run
    from the main event loop — ``asyncio.create_task`` requires a
    running loop and would raise ``RuntimeError: no running event loop``
    from FastAPI's threadpool that runs sync routes.
    """
    items = [item.model_dump() for item in body.examples]
    # Serialize against the pHash sweeper + CLIP worker so we don't fight
    # over the SQLite write lock and time out at busy_timeout. See
    # ``services/project_db_locks.py`` for the contract.
    from vlm_feedback_loop.services.project_db_locks import get_project_write_lock

    async with get_project_write_lock(project_id):
        raw_results = await asyncio.to_thread(
            example_service.ingest_examples,
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
            items=items,
            settings=settings,
        )

    results: list[ExampleIngestResultItem] = []
    for r in raw_results:
        ex_resp = None
        if r.get("example") is not None:
            ex_resp = ExampleResponse(**r["example"])

        results.append(
            ExampleIngestResultItem(
                example_key=r["example_key"],
                status=r["status"],
                error=r.get("error"),
                error_code=r.get("error_code"),
                warnings=r.get("warnings", []),
                example=ex_resp,
            )
        )

    created_count = sum(1 for r in raw_results if r.get("status") == "created")
    if created_count > 0:
        # Kick off the pHash sweeper. Non-blocking; sweeper picks
        # up the freshly-created skeleton rows on its first pass.
        from vlm_feedback_loop.services.ingest_sweeper_service import (
            trigger_ingest_processing,
        )

        trigger_ingest_processing(project_id, settings.WORKSPACE_ROOT, settings)

        # Trigger CLIP embedding computation.
        from vlm_feedback_loop.services.clip_embedding_service import (
            trigger_embedding_computation,
        )

        trigger_embedding_computation(project_id, settings.WORKSPACE_ROOT, settings)

    return ExampleIngestResponse(results=results)


# ── Image Serving ───────────────────────────────────────────────────────────


@examples_router.get("/{example_key}/image")
def serve_example_image(
    project_id: str,
    example_key: str,
    settings: Settings = Depends(get_current_settings),
) -> FileDescriptorResponse:
    """Stream an image from its persisted storage_ref.

    MUST NOT accept arbitrary filesystem paths — only the ``storage_ref``
    already persisted on the project-scoped Example record is served.
    """
    example = example_service.get_example(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        example_key=example_key,
    )
    if example is None:
        raise HTTPException(status_code=404, detail="Example not found")

    path = Path(example.storage_ref)

    try:
        opened = open_authorized_image(example.storage_ref, settings)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from None
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Image file not found at stored path: {example.storage_ref}",
        ) from None

    ext = path.suffix.lower()
    media_type = EXT_TO_MIME.get(ext)
    if media_type is None:
        media_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"

    return FileDescriptorResponse(
        opened,
        media_type=media_type,
        filename=path.name,
    )


# ── Example Query ───────────────────────────────────────────────────────────


@examples_router.get(
    "",
    response_model=ExampleQueryResponse,
    dependencies=[Depends(require_project)],
)
def query_examples(
    project_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    state: str | None = Query(default=None),
    verified_after: str | None = Query(default=None),
    verified_before: str | None = Query(default=None),
    verified_outcome: str | None = Query(default=None),
    guidance_id: str | None = Query(default=None),
    pool_membership: str | None = Query(default=None),
    include: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> ExampleQueryResponse:
    """Query examples with cursor pagination and filters.

    Stable ordering: ``verified_at desc, example_key asc`` when label
    filters are active; ``ingested_at desc, example_key asc`` otherwise.
    """
    try:
        items, next_cursor = example_service.query_examples(
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
            limit=limit,
            cursor=cursor,
            state=state,
            verified_after=verified_after,
            verified_before=verified_before,
            verified_outcome=verified_outcome,
            guidance_id=guidance_id,
            pool_membership=pool_membership,
            include=include,
        )
    except InvalidCursorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    query_items: list[ExampleQueryItem] = []
    for item in items:
        ex_resp = ExampleResponse(**item["example"])
        lbl_resp = None
        if item.get("verified_label") is not None:
            lbl_resp = VerifiedLabelResponse(**item["verified_label"])
        query_items.append(ExampleQueryItem(example=ex_resp, verified_label=lbl_resp))

    return ExampleQueryResponse(items=query_items, next_cursor=next_cursor)


# ── Embedding status ──────────────────────────────────────────────────────────


@examples_router.get(":embedding_status", response_model=EmbeddingStatusResponse)
async def embedding_status(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> EmbeddingStatusResponse:
    """Report CLIP embedding completion for a project.

    Authoritative REST counterpart to the ``embedding_progress`` /
    ``embedding_completed`` SSE events. ``complete`` is True once every example
    is embedded and no worker is still running — or when embeddings are
    disabled (auto-compute off / provider ``none``), so an embeddings-first
    seeding barrier proceeds on pHash diversity instead of hanging.

    Polling this endpoint is also the self-heal path for a drain that died
    mid-flight: with examples still pending, no worker running, and
    embeddings enabled, the worker is re-triggered (rate-limited per
    project). The route must stay ``async def`` — the trigger registers an
    asyncio task and silently no-ops without a running event loop.
    """
    from vlm_feedback_loop.services.clip_embedding_service import (
        get_embedding_status,
        maybe_restart_dead_embedding_worker,
    )

    status = get_embedding_status(project_id, settings.WORKSPACE_ROOT, settings)
    if status is None:
        raise HTTPException(status_code=404, detail="Project not found")
    maybe_restart_dead_embedding_worker(
        project_id, settings.WORKSPACE_ROOT, settings, status
    )
    return EmbeddingStatusResponse(**status)


# ── Path Remapping ──────────────────────────────────────────────────────────


@examples_router.post(
    ":remap_paths",
    response_model=RemapResponse,
    dependencies=[Depends(require_project)],
)
def remap_paths_endpoint(
    project_id: str,
    body: RemapPathsRequest,
    settings: Settings = Depends(get_current_settings),
) -> RemapResponse:
    """Bulk-remap storage_ref paths by prefix replacement.

    Dry-run (default) previews changes without modifying records.
    Commit mode replaces prefixes in a single transaction.
    """
    # Reject a new_prefix outside IMAGE_ROOT so remap can't
    # repoint an Example at an out-of-root path (e.g. /etc/passwd or the
    # secrets .env) that would then be readable via the image endpoint. No-op
    # on the default loopback posture (IMAGE_ROOT unset).
    prefix_err = filesystem_service.check_path_allowed(Path(body.new_prefix), settings)
    if prefix_err is not None:
        raise HTTPException(status_code=403, detail=prefix_err)

    try:
        result = example_service.remap_paths(
            project_id=project_id,
            workspace_root=settings.WORKSPACE_ROOT,
            old_prefix=body.old_prefix,
            new_prefix=body.new_prefix,
            dry_run=body.dry_run,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if result["dry_run"]:
        return RemapResponse(
            dry_run=True,
            matched_count=result["matched_count"],
            sample_remappings=[
                RemapSampleItem(**s) for s in result["sample_remappings"]
            ],
            unmatched_count=result["unmatched_count"],
            validation=RemapValidation(**result["validation"]),
        )

    return RemapResponse(
        dry_run=False,
        remapped_count=result["remapped_count"],
        audit_event_id=result["audit_event_id"],
    )
