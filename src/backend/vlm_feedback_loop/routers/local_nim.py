# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local NIM deployment router.

Project-scoped endpoints for deploying, managing, and querying local NIM
containers.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings, require_project
from vlm_feedback_loop.schemas.local_nim import (
    ActiveNimResidentResponse,
    LocalNimDeploymentListResponse,
    LocalNimDeploymentResponse,
    LocalNimDeployRequest,
    LocalNimDeployResponse,
    LocalNimPreflightRequest,
    PreflightCheckSchema,
    PreflightResponse,
)
from vlm_feedback_loop.services import local_nim_service
from vlm_feedback_loop.services.errors import map_service_error

logger = logging.getLogger("vlm_feedback_loop.routers.local_nim")

local_nim_router = APIRouter(
    prefix="/projects/{project_id}/local_nim",
    tags=["local_nim"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_deploy_params_or_raise(
    project_id: str,
    role: str,
    model_config_id: str | None,
    nim_container_image: str | None,
    gpu_assignment: str | None,
    preferred_port: int | None,
    settings: Settings,
) -> dict[str, Any]:
    """Delegate to the service resolver, mapping its errors to HTTP statuses.

    ``ValueError`` carries client-fixable messages (mapped 400/404 by
    ``map_service_error``); ``RuntimeError`` flags the missing
    EmbeddingDeploymentConfig singleton (server-side invariant → 500).
    """
    try:
        return local_nim_service.resolve_deploy_params(
            project_id=project_id,
            role=role,
            model_config_id=model_config_id,
            nim_container_image=nim_container_image,
            gpu_assignment=gpu_assignment,
            preferred_port=preferred_port,
            settings=settings,
        )
    except ValueError as exc:
        raise map_service_error(str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _gpu_conflict_detail(
    *,
    code: str,
    message: str,
    project_id: str,
    role: str,
    model_config_id: str,
    nim_container_image: str,
    workspace_root: str,
    gpu_assignment: str | None,
) -> dict[str, Any]:
    """Build actionable, non-secret context for an occupied-GPU response."""

    residents = local_nim_service.list_active_nim_residents(workspace_root)
    if gpu_assignment is not None:
        target_device = local_nim_service.extract_device_index(gpu_assignment)
        residents = [
            resident
            for resident in residents
            if local_nim_service.extract_device_index(resident.gpu_assignment)
            == target_device
        ]

    resident = residents[0] if residents else None
    matches_requested_model = False
    if resident is not None and role == "teacher":
        matches_requested_model = local_nim_service.teacher_resident_matches_request(
            resident,
            project_id=project_id,
            model_config_id=model_config_id,
            nim_container_image=nim_container_image,
            workspace_root=workspace_root,
        )

    effective_code = (
        "resident_starting"
        if resident is not None
        and resident.status == "starting"
        and matches_requested_model
        else code
    )
    can_replace = resident is not None and not matches_requested_model
    detail: dict[str, Any] = {
        "code": effective_code,
        "message": message,
        "can_replace": can_replace,
        "matches_requested_model": matches_requested_model,
    }
    if resident is not None:
        detail["resident"] = resident.public_summary()
    return detail


# ── Endpoints ─────────────────────────────────────────────────────────────────


@local_nim_router.post(
    "/preflight",
    response_model=PreflightResponse,
    dependencies=[Depends(require_project)],
)
async def run_preflight(
    project_id: str,
    body: LocalNimPreflightRequest,
    settings: Settings = Depends(get_current_settings),
) -> PreflightResponse:
    """Run preflight checks only (dry run) without starting a container."""
    params = _resolve_deploy_params_or_raise(
        project_id=project_id,
        role=body.role,
        model_config_id=body.model_config_id,
        nim_container_image=body.nim_container_image,
        gpu_assignment=body.gpu_assignment,
        preferred_port=None,
        settings=settings,
    )

    gpu = params["gpu_assignment"]
    if gpu is None:
        try:
            gpu = await local_nim_service.resolve_gpu_placement(
                role=body.role,
                explicit_gpu=None,
                workspace_root=settings.WORKSPACE_ROOT,
                min_gpu_memory_gb=(
                    params["gpu_memory_minimum_gb"]
                    if body.role in ("teacher", "embedding")
                    else None
                ),
                min_compute_capability=params["gpu_compute_capability_minimum"],
            )
        except (ValueError, local_nim_service.GpuExhaustedError) as exc:
            return PreflightResponse(
                all_passed=False,
                checks=[
                    PreflightCheckSchema(
                        check_name="gpu_placement",
                        passed=False,
                        diagnostic=str(exc),
                    )
                ],
            )

    model_size: str | None = None
    model_profile: str | None = None
    served_name: str | None = None
    if body.role == "teacher":
        model_size, model_profile, served_name = (
            local_nim_service.resolve_shared_image_preflight_env(
                project_id,
                params["model_config_id"],
                settings.WORKSPACE_ROOT,
            )
        )

    result = await local_nim_service.run_preflight_checks(
        nim_container_image=params["nim_container_image"],
        gpu_memory_minimum_gb=params["gpu_memory_minimum_gb"],
        gpu_assignment=gpu,
        role=body.role,
        settings=settings,
        gpu_compute_capability_minimum=params["gpu_compute_capability_minimum"],
        nim_model_size=model_size,
        nim_model_profile=model_profile,
        nim_served_model_name=served_name,
    )

    return PreflightResponse(
        all_passed=result.all_passed,
        checks=[
            PreflightCheckSchema(
                check_name=c.check_name,
                passed=c.passed,
                diagnostic=c.diagnostic,
            )
            for c in result.checks
        ],
        docker_run_command=result.docker_run_command,
        resolved_port=params["preferred_port"],
        gpu_assignment=gpu,
    )


@local_nim_router.post(
    "/deploy",
    response_model=LocalNimDeployResponse,
    status_code=201,
    dependencies=[Depends(require_project)],
)
async def deploy(
    project_id: str,
    body: LocalNimDeployRequest,
    settings: Settings = Depends(get_current_settings),
) -> LocalNimDeployResponse:
    """Deploy a local NIM container."""
    if body.activate_on_success and body.role != "teacher":
        raise HTTPException(
            status_code=400,
            detail="activate_on_success is supported only for teacher deployments",
        )
    params = _resolve_deploy_params_or_raise(
        project_id=project_id,
        role=body.role,
        model_config_id=body.model_config_id,
        nim_container_image=body.nim_container_image,
        gpu_assignment=body.gpu_assignment,
        preferred_port=body.preferred_port,
        settings=settings,
    )

    # A Teacher NIM is host infrastructure, not something every project must
    # cold-start independently. If an exact compatible Teacher is already
    # running, attach this project's ModelConfig to that resident endpoint and
    # return immediately. Explicit replace requests deliberately skip reuse:
    # they mean the operator asked to restart/replace infrastructure.
    if body.role == "teacher" and not body.replace_resident:
        reused = local_nim_service.reuse_compatible_running_teacher(
            project_id=project_id,
            model_config_id=params["model_config_id"],
            nim_container_image=params["nim_container_image"],
            workspace_root=settings.WORKSPACE_ROOT,
        )
        if reused is not None:
            if body.activate_on_success:
                activated = local_nim_service.activate_teacher_model_config(
                    project_id,
                    params["model_config_id"],
                    settings.WORKSPACE_ROOT,
                )
                if not activated:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "The running local Teacher was attached, but the "
                            "project could not select its ModelConfig."
                        ),
                    )
            return LocalNimDeployResponse(
                deployment=None,
                preflight=PreflightResponse(
                    all_passed=True,
                    checks=[
                        PreflightCheckSchema(
                            check_name="resident_reused",
                            passed=True,
                            diagnostic=(
                                f"Reusing the running {reused.model_name} NIM "
                                f"from project {reused.project_name}."
                            ),
                        )
                    ],
                    resolved_port=reused.host_port,
                    gpu_assignment=reused.gpu_assignment,
                ),
                disposition="reused",
                resident=ActiveNimResidentResponse(**reused.public_summary()),
            )

    gpu = params["gpu_assignment"]
    if gpu is None:
        try:
            gpu = await local_nim_service.resolve_gpu_placement(
                role=body.role,
                explicit_gpu=None,
                workspace_root=settings.WORKSPACE_ROOT,
                # Embedding-only floor: a heterogeneous host must not land
                # the embedding NIM on a device below the config minimum.
                # Teacher placement semantics are unchanged (preflight
                # enforces fit on its target device).
                min_gpu_memory_gb=(
                    params["gpu_memory_minimum_gb"]
                    if body.role in ("teacher", "embedding")
                    else None
                ),
                min_compute_capability=params["gpu_compute_capability_minimum"],
            )
        except local_nim_service.GpuExhaustedError as exc:
            # One-NIM-per-GPU invariant: every GPU has an active
            # resident. The caller must explicitly opt into replace
            # semantics to displace one. The FTUE sends it only after
            # the SME confirms the named resident replacement; the
            # Student NIM lifecycle defaults to true.
            if not body.replace_resident:
                # Resolve the same deterministic floor-qualified device a
                # confirmed retry would replace. The 409 must name that exact
                # resident so the UI never confirms one GPU and displaces
                # another on a heterogeneous multi-GPU host.
                conflict_gpu: str | None = None
                with suppress(ValueError, local_nim_service.GpuExhaustedError):
                    conflict_gpu = await local_nim_service.resolve_replace_target(
                        role=body.role,
                        min_gpu_memory_gb=(
                            params["gpu_memory_minimum_gb"]
                            if body.role in ("teacher", "embedding")
                            else None
                        ),
                        min_compute_capability=params["gpu_compute_capability_minimum"],
                    )
                raise HTTPException(
                    status_code=409,
                    detail=_gpu_conflict_detail(
                        code="gpu_exhausted",
                        message=str(exc),
                        project_id=project_id,
                        role=body.role,
                        model_config_id=params["model_config_id"],
                        nim_container_image=params["nim_container_image"],
                        workspace_root=settings.WORKSPACE_ROOT,
                        gpu_assignment=conflict_gpu,
                    ),
                ) from exc
            # replace_resident=true with no explicit_gpu and no free
            # qualifying GPU: pick the lowest-indexed device meeting
            # the role's memory floor and let stop_gpu_residents below
            # displace its resident (mirrors the Student lifecycle's
            # single-GPU fall-back path). The embedding floor stays
            # binding — replace semantics never land the embedding NIM
            # on a device that cannot hold it.
            try:
                gpu = await local_nim_service.resolve_replace_target(
                    role=body.role,
                    min_gpu_memory_gb=(
                        params["gpu_memory_minimum_gb"]
                        if body.role in ("teacher", "embedding")
                        else None
                    ),
                    min_compute_capability=params["gpu_compute_capability_minimum"],
                )
            except local_nim_service.GpuExhaustedError as floor_exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "gpu_exhausted", "message": str(floor_exc)},
                ) from floor_exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # If the target device has any active residents, route through
    # replace semantics. The router rejects with 409 when the caller
    # did not opt in.
    device_index = local_nim_service.extract_device_index(gpu)
    residents = local_nim_service.scan_active_residents_by_device(
        settings.WORKSPACE_ROOT
    )
    if device_index in residents and not body.replace_resident:
        raise HTTPException(
            status_code=409,
            detail=_gpu_conflict_detail(
                code="gpu_occupied",
                message=(
                    f"GPU {gpu} is occupied by an active NIM deployment. "
                    f"Pass replace_resident=true to stop the resident "
                    f"and reuse the GPU (one-NIM-per-GPU invariant)."
                ),
                project_id=project_id,
                role=body.role,
                model_config_id=params["model_config_id"],
                nim_container_image=params["nim_container_image"],
                workspace_root=settings.WORKSPACE_ROOT,
                gpu_assignment=gpu,
            ),
        )

    try:
        result = await local_nim_service.deploy_local_nim(
            project_id=project_id,
            model_config_id=params["model_config_id"],
            role=body.role,
            nim_container_image=params["nim_container_image"],
            gpu_assignment=gpu,
            gpu_memory_minimum_gb=params["gpu_memory_minimum_gb"],
            preferred_port=params["preferred_port"],
            settings=settings,
            workspace_root=settings.WORKSPACE_ROOT,
            replace_resident=body.replace_resident,
            activate_on_success=body.activate_on_success,
            background=True,
        )
    except local_nim_service.GpuExhaustedError as exc:
        # The host-wide deploy lock re-checked occupancy and found a resident
        # that appeared on the target GPU since the pre-check above (the
        # TOCTOU this closes). Surface the same 409 the pre-check returns.
        raise HTTPException(
            status_code=409,
            detail=_gpu_conflict_detail(
                code="gpu_occupied",
                message=str(exc),
                project_id=project_id,
                role=body.role,
                model_config_id=params["model_config_id"],
                nim_container_image=params["nim_container_image"],
                workspace_root=settings.WORKSPACE_ROOT,
                gpu_assignment=gpu,
            ),
        ) from exc

    deployment = result["deployment"]
    preflight = result["preflight"]

    return LocalNimDeployResponse(
        deployment=LocalNimDeploymentResponse.model_validate(deployment),
        preflight=PreflightResponse(
            all_passed=preflight.all_passed,
            checks=[
                PreflightCheckSchema(
                    check_name=c.check_name,
                    passed=c.passed,
                    diagnostic=c.diagnostic,
                )
                for c in preflight.checks
            ],
            docker_run_command=preflight.docker_run_command,
            # deploy_local_nim() may reserve a different free host port when
            # the preferred port is occupied.  Return the authoritative
            # reservation rather than the caller's preference so this field
            # agrees with deployment.host_port and the docker command.
            resolved_port=deployment.host_port,
            gpu_assignment=gpu,
        ),
        disposition="queued",
    )


@local_nim_router.get(
    "/deployments",
    response_model=LocalNimDeploymentListResponse,
    dependencies=[Depends(require_project)],
)
def list_deployments(
    project_id: str,
    settings: Settings = Depends(get_current_settings),
) -> LocalNimDeploymentListResponse:
    """List all local NIM deployments for a project."""
    deployments = local_nim_service.list_local_deployments(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    active_map = local_nim_service.matches_active_role_config(
        project_id, settings.WORKSPACE_ROOT, deployments
    )
    return LocalNimDeploymentListResponse(
        items=[
            LocalNimDeploymentResponse.model_validate(d).model_copy(
                update={
                    "matches_active_role_config": active_map.get(
                        d.local_nim_deployment_id, True
                    )
                }
            )
            for d in deployments
        ]
    )


@local_nim_router.get(
    "/deployments/{deployment_id}",
    response_model=LocalNimDeploymentResponse,
)
def get_deployment(
    project_id: str,
    deployment_id: str,
    settings: Settings = Depends(get_current_settings),
) -> LocalNimDeploymentResponse:
    """Get a single local NIM deployment."""
    dep = local_nim_service.get_local_deployment(
        project_id=project_id,
        deployment_id=deployment_id,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    if dep is None:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return LocalNimDeploymentResponse.model_validate(dep)


@local_nim_router.post(
    "/deployments/{deployment_id}:stop",
    response_model=LocalNimDeploymentResponse,
)
async def stop_deployment(
    project_id: str,
    deployment_id: str,
    settings: Settings = Depends(get_current_settings),
) -> LocalNimDeploymentResponse:
    """Stop a running local NIM deployment."""
    dep = await local_nim_service.stop_local_nim(
        deployment_id=deployment_id,
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        settings=settings,
    )
    if dep is None:
        raise HTTPException(status_code=404, detail="Deployment not found")

    return LocalNimDeploymentResponse.model_validate(dep)
