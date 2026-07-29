# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelConfig CRUD and capability re-probe routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.model_config import (
    ModelAvailability,
    ModelConfigCreate,
    ModelConfigListResponse,
    ModelConfigResponse,
    ModelConfigUpdate,
)
from vlm_feedback_loop.services import model_config_service
from vlm_feedback_loop.services.environment import get_cached_environment
from vlm_feedback_loop.services.errors import map_service_error

model_configs_router = APIRouter(
    prefix="/projects/{project_id}/model_configs",
    tags=["model_configs"],
)


def _attach_availability(
    mc: ModelConfig,
    endpoint: NimEndpoint | None,
    env: dict[str, Any],
) -> ModelConfigResponse:
    """Build a ModelConfigResponse with a freshly computed availability field."""
    response = ModelConfigResponse.model_validate(mc)
    response.availability = ModelAvailability(
        **model_config_service.compute_availability(mc, endpoint, env)
    )
    return response


@model_configs_router.post("", status_code=201, response_model=ModelConfigResponse)
async def create_model_config_endpoint(
    project_id: str,
    body: ModelConfigCreate,
    settings: Settings = Depends(get_current_settings),
) -> ModelConfigResponse:
    """Create a new ModelConfig entry."""
    result = model_config_service.create_model_config(
        project_id=project_id,
        data=body.model_dump(),
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Project not found")
    if isinstance(result, str):
        raise map_service_error(result)
    env = await get_cached_environment(settings)
    endpoints = model_config_service.get_endpoints_for_model_configs(
        project_id, [result], settings.WORKSPACE_ROOT
    )
    return _attach_availability(result, endpoints.get(result.endpoint_id), env)


@model_configs_router.get("", response_model=ModelConfigListResponse)
async def list_model_configs_endpoint(
    project_id: str,
    eligible_role: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> ModelConfigListResponse:
    """List ModelConfig entries with optional role filtering and per-entry availability."""
    items, next_cursor = model_config_service.list_model_configs(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        eligible_role=eligible_role,
        limit=limit,
        cursor=cursor,
    )
    env = await get_cached_environment(settings)
    endpoints = model_config_service.get_endpoints_for_model_configs(
        project_id, items, settings.WORKSPACE_ROOT
    )
    return ModelConfigListResponse(
        items=[
            _attach_availability(mc, endpoints.get(mc.endpoint_id), env) for mc in items
        ],
        next_cursor=next_cursor,
    )


@model_configs_router.get("/{model_config_id}", response_model=ModelConfigResponse)
async def get_model_config_endpoint(
    project_id: str,
    model_config_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ModelConfigResponse:
    """Retrieve a single ModelConfig by ID."""
    mc = model_config_service.get_model_config(
        project_id=project_id,
        model_config_id=model_config_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if mc is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    env = await get_cached_environment(settings)
    endpoints = model_config_service.get_endpoints_for_model_configs(
        project_id, [mc], settings.WORKSPACE_ROOT
    )
    return _attach_availability(mc, endpoints.get(mc.endpoint_id), env)


@model_configs_router.patch("/{model_config_id}", response_model=ModelConfigResponse)
async def update_model_config_endpoint(
    project_id: str,
    model_config_id: str,
    body: ModelConfigUpdate,
    settings: Settings = Depends(get_current_settings),
) -> ModelConfigResponse:
    """Partial update for a ModelConfig entry."""
    updates = body.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = model_config_service.update_model_config(
        project_id=project_id,
        model_config_id=model_config_id,
        updates=updates,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    if isinstance(result, str):
        raise map_service_error(result)
    env = await get_cached_environment(settings)
    endpoints = model_config_service.get_endpoints_for_model_configs(
        project_id, [result], settings.WORKSPACE_ROOT
    )
    return _attach_availability(result, endpoints.get(result.endpoint_id), env)


@model_configs_router.post(
    "/{model_config_id}:reprobe", response_model=ModelConfigResponse
)
async def reprobe_model_config_endpoint(
    project_id: str,
    model_config_id: str,
    settings: Settings = Depends(get_current_settings),
) -> ModelConfigResponse:
    """Re-probe all three capabilities for a model config.

    Returns 409 if the model is referenced by an active run or TAO job.
    """
    result = await model_config_service.reprobe_model_config(
        project_id=project_id,
        model_config_id=model_config_id,
        workspace_root=settings.WORKSPACE_ROOT,
        settings=settings,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Model config not found")
    if isinstance(result, str):  # "active_use" — blocked by active run/job
        raise HTTPException(
            status_code=409,
            detail="Cannot re-probe: model is referenced by an active run or training job",
        )
    env = await get_cached_environment(settings)
    endpoints = model_config_service.get_endpoints_for_model_configs(
        project_id, [result], settings.WORKSPACE_ROOT
    )
    return _attach_availability(result, endpoints.get(result.endpoint_id), env)
