# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review selector endpoint."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.routers.projects import get_current_settings, require_project
from vlm_feedback_loop.schemas.review_selector import ReviewSelectorNextResponse
from vlm_feedback_loop.services import review_selector_service
from vlm_feedback_loop.services.project_db_locks import get_project_write_lock
from vlm_feedback_loop.services.project_service import get_project_engine

review_selector_router = APIRouter(
    prefix="/projects/{project_id}/review_selector",
    tags=["review-selector"],
)


@review_selector_router.get("/next", response_model=ReviewSelectorNextResponse)
async def get_next_review_item(
    project_id: str,
    project: Project = Depends(require_project),
    settings: Settings = Depends(get_current_settings),
) -> ReviewSelectorNextResponse:
    """Select the next image for the SME to review.

    Uses a diversity-driven selector (pHash or CLIP) to maximize visual
    variety across the labeling session.  After a schema change, examples
    with prior labels are prioritized (Edits first).
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        raise HTTPException(status_code=404, detail="Project not found")

    # Resolve feature-flag overrides for selection mode and switchover
    ff = project.feature_flags or {}
    review_mode = ff.get("REVIEW_SELECTION_MODE", settings.REVIEW_SELECTION_MODE)
    clip_switchover = ff.get(
        "CLIP_SWITCHOVER_MIN_COUNT", settings.CLIP_SWITCHOVER_MIN_COUNT
    )

    # Selection persists the recent-window cursor, so it is a writer even
    # though the API operation is a GET. Queue it with background batches
    # instead of letting the SME's next-image request enter SQLite busy wait.
    async with get_project_write_lock(project_id):
        result = await asyncio.to_thread(
            review_selector_service.select_next,
            engine=engine,
            project=project,
            review_selection_mode=review_mode,
            review_recent_window_k=settings.REVIEW_RECENT_WINDOW_K,
            clip_switchover_min_count=clip_switchover,
        )

    return ReviewSelectorNextResponse(
        example_key=result.example_key,
        example_state=result.example_state,
        has_existing_label=result.has_existing_label,
        selection_mode=result.selection_mode,
        queue_empty=result.queue_empty,
        storage_ref=result.storage_ref,
        prior_verified_label_ref=result.prior_verified_label_ref,
    )
