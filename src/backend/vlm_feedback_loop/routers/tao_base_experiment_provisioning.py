# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped entry points for deployment-scoped TAO base provisioning."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import (
    get_current_settings,
    require_not_archived,
)
from vlm_feedback_loop.schemas.tao_base_experiment_provisioning import (
    TAOBaseExperimentProvisioningRequest,
    TAOBaseExperimentProvisioningResponse,
)
from vlm_feedback_loop.services import (
    tao_base_experiment_provisioning_run_service as provisioning_service,
)
from vlm_feedback_loop.services.errors import map_service_error

tao_base_experiment_provisioning_router = APIRouter(
    prefix="/projects/{project_id}",
    tags=["training-base-provisioning"],
)


@tao_base_experiment_provisioning_router.post(
    "/tao_base_experiment_provisioning",
    response_model=TAOBaseExperimentProvisioningResponse,
    status_code=202,
    dependencies=[Depends(require_not_archived)],
)
async def start_tao_base_experiment_provisioning(
    project_id: str,
    body: TAOBaseExperimentProvisioningRequest,
    settings: Settings = Depends(get_current_settings),
) -> TAOBaseExperimentProvisioningResponse:
    """Ensure the selected missing Student bases in the background."""
    result = provisioning_service.start_provisioning_run(
        project_id,
        list(body.student_base_model_config_ids),
        settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TAOBaseExperimentProvisioningResponse.model_validate(result)


@tao_base_experiment_provisioning_router.get(
    "/tao_base_experiment_provisioning/{provisioning_run_id}",
    response_model=TAOBaseExperimentProvisioningResponse,
)
def get_tao_base_experiment_provisioning(
    project_id: str,
    provisioning_run_id: str,
    settings: Settings = Depends(get_current_settings),
) -> TAOBaseExperimentProvisioningResponse:
    """Get durable status for one first-use provisioning attempt."""
    result = provisioning_service.get_provisioning_run(
        project_id,
        provisioning_run_id,
        settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return TAOBaseExperimentProvisioningResponse.model_validate(result)
