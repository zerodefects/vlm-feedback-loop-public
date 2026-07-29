# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training suite REST endpoints.

One POST, one atomic suite: dataset exports + pre-created TAO chain jobs +
TrainingSuite record + first-chain kickoff.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import (
    get_current_settings,
    require_not_archived,
)
from vlm_feedback_loop.schemas.training_suite import (
    TrainingPresetResolveRequest,
    TrainingPresetResolveResponse,
    TrainingSuiteCancelResponse,
    TrainingSuiteCreateRequest,
    TrainingSuiteListResponse,
    TrainingSuiteResponse,
)
from vlm_feedback_loop.services import training_suite_service
from vlm_feedback_loop.services.errors import map_service_error

training_suites_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["training-suites"],
)


# ── Create ────────────────────────────────────────────────────────────────


@training_suites_router.post(
    "/training_suites",
    response_model=TrainingSuiteResponse,
    status_code=201,
    dependencies=[Depends(require_not_archived)],
)
async def create_training_suite(
    project_id: str,
    body: TrainingSuiteCreateRequest,
    settings: Settings = Depends(get_current_settings),
) -> TrainingSuiteResponse:
    """Create a training suite.

    Idempotent: re-POST with the same ``idempotency_key`` returns the
    existing suite (still 201 — the resource exists after the call either
    way).
    """
    result = await training_suite_service.launch_training_suite(
        project_id,
        student_base_model_config_ids=list(body.student_base_model_config_ids),
        training_preset=body.training_preset,
        include_auto_labeled=body.include_auto_labeled,
        export_field_mode=body.export_field_mode,
        quantization_schemes=list(body.quantization_schemes),
        enable_lora=body.enable_lora,
        idempotency_key=body.idempotency_key,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TrainingSuiteResponse(**result)


@training_suites_router.post(
    "/training_presets:resolve",
    response_model=TrainingPresetResolveResponse,
)
def resolve_training_presets(
    project_id: str,
    body: TrainingPresetResolveRequest,
    settings: Settings = Depends(get_current_settings),
) -> TrainingPresetResolveResponse:
    """Resolve read-only Advanced hyperparameters without starting setup."""
    result = training_suite_service.resolve_training_presets_for_models(
        project_id,
        student_base_model_config_ids=list(body.student_base_model_config_ids),
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TrainingPresetResolveResponse(**result)


# ── Cancel ────────────────────────────────────────────────────────────────


@training_suites_router.post(
    "/training_suites/{training_suite_id}:cancel",
    response_model=TrainingSuiteCancelResponse,
    dependencies=[Depends(require_not_archived)],
)
async def cancel_training_suite(
    project_id: str,
    training_suite_id: str,
    settings: Settings = Depends(get_current_settings),
) -> TrainingSuiteCancelResponse:
    """Best-effort cancel all remaining work in a Training Suite."""
    result = await training_suite_service.cancel_training_suite(
        project_id,
        training_suite_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TrainingSuiteCancelResponse(**result)


# ── Get ───────────────────────────────────────────────────────────────────


@training_suites_router.get(
    "/training_suites/{training_suite_id}",
    response_model=TrainingSuiteResponse,
)
def get_training_suite(
    project_id: str,
    training_suite_id: str,
    settings: Settings = Depends(get_current_settings),
) -> TrainingSuiteResponse:
    """Get a training suite with live chain status."""
    result = training_suite_service.get_training_suite(
        project_id,
        training_suite_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TrainingSuiteResponse(**result)


# ── List ──────────────────────────────────────────────────────────────────


@training_suites_router.get(
    "/training_suites",
    response_model=TrainingSuiteListResponse,
)
def list_training_suites(
    project_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> TrainingSuiteListResponse:
    """List training suites with cursor pagination, newest-first."""
    result = training_suite_service.list_training_suites(
        project_id,
        cursor=cursor,
        limit=limit,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    items, next_cursor = result
    return TrainingSuiteListResponse(
        items=[TrainingSuiteResponse(**item) for item in items],
        next_cursor=next_cursor,
    )
