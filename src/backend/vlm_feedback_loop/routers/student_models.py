# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StudentModel list + detail + deploy_nim endpoints.

Read-side endpoints (list + detail) plus the ``:deploy_nim`` mutation
that triggers the Student NIM lifecycle (local Docker orchestration or
external endpoint registration + evaluation).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.schemas.student_model import (
    StudentModelListResponse,
    StudentModelResponse,
)
from vlm_feedback_loop.schemas.student_nim_deploy import (
    DeployNimRequest,
    DeployNimResponse,
)
from vlm_feedback_loop.services import (
    deployment_bundle_service,
    deployment_handoff_generator,
    student_model_service,
    tao_rescoring_service,
)
from vlm_feedback_loop.services.errors import conflict, map_service_error

student_models_router = APIRouter(
    prefix="/projects/{project_id}/student_models",
    tags=["student_models"],
)


@student_models_router.get("", response_model=StudentModelListResponse)
def list_student_models_endpoint(
    project_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    cursor: str | None = Query(default=None),
    settings: Settings = Depends(get_current_settings),
) -> StudentModelListResponse:
    """List StudentModel records, newest-first, with cursor pagination."""
    items, next_cursor = student_model_service.list_student_models(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        limit=limit,
        cursor=cursor,
        expected_benchmark_concurrencies=tuple(
            settings.STUDENT_LATENCY_TEST_CONCURRENCIES
        ),
    )
    return StudentModelListResponse(
        items=[StudentModelResponse.model_validate(row) for row in items],
        next_cursor=next_cursor,
    )


@student_models_router.get("/{student_model_id}", response_model=StudentModelResponse)
def get_student_model_endpoint(
    project_id: str,
    student_model_id: str,
    settings: Settings = Depends(get_current_settings),
) -> StudentModelResponse:
    """Return a single StudentModel record."""
    row = student_model_service.get_student_model(
        project_id=project_id,
        student_model_id=student_model_id,
        workspace_root=settings.WORKSPACE_ROOT,
        expected_benchmark_concurrencies=tuple(
            settings.STUDENT_LATENCY_TEST_CONCURRENCIES
        ),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Student model not found")
    return StudentModelResponse.model_validate(row)


@student_models_router.post(
    "/{student_model_id}:deploy_nim",
    response_model=DeployNimResponse,
    status_code=202,
)
async def deploy_nim_endpoint(
    project_id: str,
    student_model_id: str,
    body: DeployNimRequest,
    settings: Settings = Depends(get_current_settings),
) -> DeployNimResponse:
    """Trigger NIM deployment for a Student.

    ``body.nim_endpoint_url`` decides the mode: ``None`` runs the local
    Docker lifecycle (preflight → docker run → health → smoke → register
    temp endpoint → evaluation → benchmark sweep → stop); a URL skips
    local orchestration, registers the endpoint as external, and then runs
    the same evaluation and benchmark sweep.

    Returns 202 with the dispatch dict — the lifecycle runs in the
    background and emits SSE events (``nim_benchmark_progress`` /
    ``nim_benchmark_completed`` / ``run_failed``).
    """
    result = await student_model_service.deploy_nim(
        project_id=project_id,
        student_model_id=student_model_id,
        nim_endpoint_url=body.nim_endpoint_url,
        nim_container_image=body.nim_container_image,
        nim_release_version=body.nim_release_version,
        gpu_assignment=body.gpu_assignment,
        auth_mode=body.auth_mode,
        benchmark_kv_cache_reuse=body.benchmark_kv_cache_reuse,
        settings=settings,
    )

    if "error" in result:
        error = result["error"]
        if error == "student_not_found":
            raise HTTPException(status_code=404, detail="Student model not found")
        if error == "checkpoint_not_validated":
            raise HTTPException(
                status_code=400,
                detail=(
                    "Student checkpoint is not validated; cannot deploy. "
                    "checkpoint_packaging_status must be 'validated'."
                ),
            )
        if error == "invalid_external_url":
            raise HTTPException(
                status_code=400,
                detail="nim_endpoint_url must be an http:// or https:// URL.",
            )
        if error == "benchmark_cache_policy_unconfirmed":
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "benchmark_cache_policy_unconfirmed",
                    "message": (
                        "External Student endpoints must be launched with "
                        "NIM_ENABLE_KV_CACHE_REUSE=0 and registered with "
                        "benchmark_kv_cache_reuse='disabled'."
                    ),
                },
            )
        if error == "deploy_in_progress":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Another Student NIM deployment is already in progress for "
                    "this project. Wait for it to complete before retrying."
                ),
            )
        # Unknown error — surface as 500 with a structured detail (same
        # {code, message} shape as the local-NIM 409s) instead of an ad-hoc
        # "deploy_nim_error:<key>" token clients would have to parse.
        raise HTTPException(
            status_code=500,
            detail={"code": "deploy_nim_error", "message": str(error)},
        )

    return DeployNimResponse(**result)


@student_models_router.post(
    "/{student_model_id}:deployment_handoff",
    status_code=200,
)
def deployment_handoff_endpoint(
    project_id: str,
    student_model_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, object]:
    """Generate a ``deployment_handoff`` Action Request.

    Dual gates: requires both ``quality_status="validated"``
    AND ``serving_status="validated"``. The Student's training
    Inference Contract MUST equal the evaluation Contract on its
    ``serving_evaluation_run``.

    Status codes:
        200 — gates passed; AR payload returned
        404 — Student not found in project
        409 — quality_status_not_validated / serving_status_not_validated /
              serving_evaluation_run_missing /
              serving_benchmark_requires_aiperf /
              INFERENCE_CONTRACT_MISMATCH
    """
    result = deployment_handoff_generator.generate_deployment_handoff_for_student(
        project_id=project_id,
        student_model_id=student_model_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return result


@student_models_router.get(
    "/{student_model_id}/deployment_bundle",
    response_class=StreamingResponse,
)
def deployment_bundle_endpoint(
    project_id: str,
    student_model_id: str,
    settings: Settings = Depends(get_current_settings),
) -> StreamingResponse:
    """Stream a portable Student NIM deployment bundle.

    The same quality, serving, checkpoint-packaging, and Inference Contract
    gates as ``:deployment_handoff`` apply. The checkpoint must also resolve
    beneath the requested project and contain only regular files.
    """
    result = deployment_bundle_service.prepare_deployment_bundle(
        project_id=project_id,
        student_model_id=student_model_id,
        settings=settings,
    )
    if isinstance(result, str):
        raise map_service_error(result)
    return StreamingResponse(
        deployment_bundle_service.stream_deployment_bundle(result),
        media_type="application/x-tar",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{result.archive_filename}"'
            ),
            "X-Content-Type-Options": "nosniff",
        },
    )


@student_models_router.post(
    "/{student_model_id}:repackage",
    status_code=200,
)
async def repackage_endpoint(
    project_id: str,
    student_model_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, str | None]:
    """Replay checkpoint packaging after an environment fix.

    Packaging can fail for reasons outside the checkpoint itself —
    canonically the LoRA-merge interpreter being unprovisioned when the
    training output is adapter-only (§9.5.1). After the operator fixes
    the environment, this endpoint replays packaging in place via the
    idempotent registration path, without re-running the training chain.

    Status codes:
        200 — packaging replayed (inspect ``checkpoint_packaging_status``)
        404 — Student not found in project
        409 — Student's packaging is not currently ``failed``
        502 — the Student's TAOJob could not be resolved
    """
    result = await student_model_service.repackage_student_model(
        project_id=project_id,
        student_model_id=student_model_id,
        settings=settings,
    )
    err = result.get("error")
    if err in ("student_not_found", "project_not_found"):
        raise HTTPException(status_code=404, detail="Student model not found")
    if err == "packaging_not_failed":
        raise HTTPException(
            status_code=409,
            detail="conflict: checkpoint_packaging_status is not 'failed'",
        )
    if err == "tao_job_unresolved":
        raise HTTPException(status_code=502, detail="TAOJob could not be resolved")
    if err == "artifact_refresh_failed":
        raise HTTPException(
            status_code=502,
            detail=(
                "The quantized checkpoint could not be refreshed from TAO workspace "
                "storage. Verify TAO/S3 reachability and retry repackage."
            ),
        )
    return result


@student_models_router.post(
    "/{student_model_id}:rerescore",
    status_code=200,
)
async def rerescore_endpoint(
    project_id: str,
    student_model_id: str,
    settings: Settings = Depends(get_current_settings),
) -> dict[str, str | None]:
    """Re-rescore a Student whose ``quality_status="failed"`` after a
    fix to the rescoring service.

    The endpoint replays the canonical rescore against the existing
    per-sample predictions on disk under current code (including
    markdown-fence stripping and the quantize-parent prefix glob).
    Safety: the Student MUST be ``quality_status="failed"`` — the
    endpoint refuses to overwrite ``"validated"``, ``"pending"``, or
    ``"partial"`` Students.

    ``partial`` Students cannot use ``:rerescore``. The endpoint's
    purpose is to replay a TAO rescore against on-disk per-sample
    predictions; ``partial`` is set by NIM-eval and there is
    no on-disk TAO rescore to replay. The remediation path for a
    ``partial`` Student is to re-run NIM eval (the underlying model
    output may improve on a fresh inference pass) until the run lands
    ``completed`` and promotes to ``validated``.

    Status codes:
        200 — rescore ran (run_id null when there were no parseable
              predictions to replay; check ``quality_status``)
        400 — no paired evaluate TAOJob found in this Student's chain
        404 — Student not found in project
        409 — Student is not currently ``quality_status="failed"``
    """
    result = await tao_rescoring_service.rerescore_student_model_quality(
        project_id=project_id,
        student_model_id=student_model_id,
        settings=settings,
    )
    err = result.get("error")
    if err == "student_not_found":
        raise HTTPException(status_code=404, detail="Student model not found")
    if err == "project_not_found":
        raise HTTPException(status_code=404, detail="Project not found")
    if err == "student_not_failed":
        raise conflict(
            f"Student quality_status is "
            f"{result.get('quality_status')!r}, not 'failed'. "
            "Rerescore is only permitted on failed Students."
        )
    if err == "no_paired_evaluate_job":
        raise HTTPException(
            status_code=400,
            detail=(
                "No paired evaluate TAOJob found for this Student "
                "(checked parent_tao_job_id matching the Student's "
                "quantize_tao_job_id or tao_job_id with action='evaluate' "
                "and status='succeeded')."
            ),
        )
    return result
