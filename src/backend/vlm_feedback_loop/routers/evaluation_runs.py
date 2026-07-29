# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation run endpoints and trigger status/dismiss."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import (
    get_current_settings,
    require_not_archived,
)
from vlm_feedback_loop.schemas.evaluation import (
    EvaluationRunCancelResponse,
    EvaluationRunCreateRequest,
    EvaluationRunCreateResponse,
    EvaluationRunListResponse,
    EvaluationRunResponse,
    ScaleUpGateResponse,
    TriggerDismissRequest,
    TriggerDismissResponse,
    TriggerStatusResponse,
)
from vlm_feedback_loop.services import evaluation_service
from vlm_feedback_loop.services.errors import map_service_error

evaluation_runs_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["evaluation-runs"],
)


# ── Create evaluation run ───────────────────────────────────────────────────


@evaluation_runs_router.post(
    "/evaluation_runs",
    response_model=EvaluationRunCreateResponse,
    status_code=201,
    dependencies=[Depends(require_not_archived)],
)
async def create_evaluation_run(
    project_id: str,
    body: EvaluationRunCreateRequest,
    settings: Settings = Depends(get_current_settings),
) -> EvaluationRunCreateResponse:
    """Trigger a new evaluation run."""
    result = await evaluation_service.start_evaluation_run(
        project_id,
        icl_mode=body.icl_mode,
        structured_generation_mode=body.structured_generation_mode,
        icl_max_examples=body.icl_max_examples,
        icl_candidate_limit=body.icl_candidate_limit,
        eval_concurrency=body.eval_concurrency,
        icl_sim_gap=body.icl_sim_gap,
        icl_abs_threshold=body.icl_abs_threshold,
        generation_preset_key=body.generation_preset_key,
        thinking_on=body.thinking_on,
        visual_budget_preset_key=body.visual_budget_preset_key,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return EvaluationRunCreateResponse(**result)


# ── Get evaluation run ──────────────────────────────────────────────────────


@evaluation_runs_router.get(
    "/evaluation_runs/{run_id}",
    response_model=EvaluationRunResponse,
)
def get_evaluation_run(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> EvaluationRunResponse:
    """Get evaluation run status and metrics."""
    result = evaluation_service.get_evaluation_run(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return EvaluationRunResponse(**result)


# ── List evaluation runs ────────────────────────────────────────────────────


@evaluation_runs_router.get(
    "/evaluation_runs",
    response_model=EvaluationRunListResponse,
)
def list_evaluation_runs(
    project_id: str,
    status: str | None = Query(default=None),
    basis: Literal["gate", "benchmark"] | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> EvaluationRunListResponse:
    """List evaluation runs with cursor pagination.

    ``basis`` scopes by provenance: ``gate`` = gate-basis Teacher runs
    (the eval strip's view), ``benchmark`` = Student benchmark runs (§9.5.2).
    """
    items, next_cursor = evaluation_service.list_evaluation_runs(
        project_id,
        status_filter=status,
        basis=basis,
        cursor=cursor,
        limit=limit,
        settings=settings,
    )
    return EvaluationRunListResponse(
        items=[EvaluationRunResponse(**item) for item in items],
        next_cursor=next_cursor,
    )


# ── Cancel evaluation run ───────────────────────────────────────────────────


@evaluation_runs_router.post(
    "/evaluation_runs/{run_id}:cancel",
    response_model=EvaluationRunCancelResponse,
)
async def cancel_evaluation_run(
    project_id: str,
    run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Cancel a running evaluation."""
    result = await evaluation_service.cancel_evaluation_run(
        project_id,
        run_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


# ── Trigger status ──────────────────────────────────────────────────────────


@evaluation_runs_router.get(
    "/evaluation_trigger_status",
    response_model=TriggerStatusResponse,
)
def get_trigger_status(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Compute evaluation trigger state from persisted data."""
    result = evaluation_service.compute_trigger_status(
        project_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


# ── Trigger dismiss ─────────────────────────────────────────────────────────


@evaluation_runs_router.post(
    "/evaluation_trigger_status:dismiss",
    response_model=TriggerDismissResponse,
)
def dismiss_trigger(
    project_id: str,
    body: TriggerDismissRequest,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Dismiss an evaluation trigger."""
    result = evaluation_service.dismiss_trigger(
        project_id,
        body.trigger_type,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


# ── Scale-Up Readiness Gate ─────────────────────────────────────────────────


@evaluation_runs_router.get(
    "/scaleup_gate",
    response_model=ScaleUpGateResponse,
)
def get_scaleup_gate(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, Any]:
    """Evaluate the Scale-Up Readiness Gate."""
    result = evaluation_service.compute_scaleup_gate(
        project_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result
