# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training preflight endpoint.

See the service docstring for the boundary against the NIM deployment
preflight (no Docker / GPU / NGC checks here).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.training_preflight import (
    TrainingPreflightRequest,
    TrainingPreflightResponse,
)
from vlm_feedback_loop.services import training_preflight_service

training_preflight_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["training_preflight"],
)


@training_preflight_router.post(
    "/training_preflight",
    response_model=TrainingPreflightResponse,
)
async def run_training_preflight_endpoint(
    project_id: str,
    body: TrainingPreflightRequest,
    settings: Settings = Depends(get_current_settings),
) -> TrainingPreflightResponse:
    """Run the training preflight."""
    result = await training_preflight_service.run_training_preflight(
        project_id=project_id,
        student_base_model_config_ids=body.student_base_model_config_ids,
        settings=settings,
        include_auto_labeled=body.include_auto_labeled,
        enable_lora=body.enable_lora,
    )
    return TrainingPreflightResponse.model_validate(result)
