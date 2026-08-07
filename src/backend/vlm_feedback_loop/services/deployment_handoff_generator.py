# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``deployment_handoff`` Action Request generator.

Generates the production-deployment handoff content for a Student that has
both ``quality_status="validated"`` AND ``serving_status="validated"`` AND
matching training-time vs evaluation-time Inference Contracts. The
generator is registered at import time; ``main.py`` imports this module as
a side-effect to populate the registry at startup.

The dual-gate + Inference-Contract checks live on top of the generic
Action Request framework (``services/action_requests.py``). The actual
content rendering (technical requirements, current environment, copy-pastable
docker run line) is the generator's responsibility.

This module also exposes the gate logic via
``generate_deployment_handoff_for_student`` for the Student-scoped HTTP
endpoint (``POST /student_models/{id}:deployment_handoff``) — the gate's
return shape mirrors the rest of the service layer (``dict | str``) so the
router maps error strings to HTTP codes uniformly.

Error strings produced (mapped by the router):

  - ``"not found: StudentModel {id}"`` → 404
  - ``"conflict: quality_status_not_validated"`` → 409
  - ``"conflict: serving_status_not_validated"`` → 409
  - ``"conflict: serving_evaluation_run_missing"`` → 409
  - ``"conflict: serving_benchmark_requires_aiperf"`` → 409
  - ``"conflict: INFERENCE_CONTRACT_MISMATCH"`` → 409
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import ValidationError
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.schemas.inference_contract import InferenceContract
from vlm_feedback_loop.services.action_requests import (
    generate_action_request,
    register_generator,
)
from vlm_feedback_loop.services.dataset_export_service import (
    resolve_paired_test_pool_dataset_sha,
)
from vlm_feedback_loop.services.gpu_memory_floor import (
    resolve_gpu_memory_floor_gb,
)
from vlm_feedback_loop.services.inference_contract_resolver import (
    resolve_training_inference_contract,
)
from vlm_feedback_loop.services.local_nim_service import (
    build_student_docker_run_args,
    build_student_docker_run_display,
    release_version_from_image,
    resolve_extra_container_env,
)
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.serving_validation_service import (
    assess_aiperf_serving_run,
)

logger = logging.getLogger("vlm_feedback_loop.services.deployment_handoff_generator")


# ── Inference Contract equivalence check ────────────────────────────────────


def _contracts_equivalent(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Validate and compare two canonical Inference Contract snapshots."""
    if not a and not b:
        return True
    if not a or not b:
        return False
    try:
        left = InferenceContract.model_validate(a)
        right = InferenceContract.model_validate(b)
    except ValidationError:
        return False
    return left == right


# ── Public gate + AR generation (Student-scoped) ────────────────────────────


def generate_deployment_handoff_for_student(
    *,
    project_id: str,
    student_model_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Validate the dual + Contract gates; on pass, generate the AR payload.

    Returns:
      - ``dict`` shaped like the generic AR response on success
      - ``str`` error code on gate failure (mapped to HTTP status by router)
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"
        student = (
            session.query(StudentModel)
            .filter_by(project_id=project_id, student_model_id=student_model_id)
            .first()
        )
        if student is None:
            return f"not found: StudentModel {student_model_id}"

        # The Action Request and portable bundle are production handoffs.
        # A quality/serving result must never make an absent or failed
        # checkpoint appear deployable.
        if (
            student.checkpoint_packaging_status != "validated"
            or not student.nim_checkpoint_ref
        ):
            return "conflict: checkpoint_packaging_not_validated"

        # ── Gate 1: quality_status ──────────────────────────────────────────
        # ``partial`` is informational, NOT gate-passing. The
        # production-handoff bar still requires a fully-validated run.
        # Distinguishing the two conflict codes lets the frontend
        # ActionRequestPanel render different messages for the two cases.
        if student.quality_status == "partial":
            return "conflict: quality_status_partial"
        if student.quality_status != "validated":
            return "conflict: quality_status_not_validated"

        # ── Gate 2: serving_status ──────────────────────────────────────────
        if student.serving_status != "validated":
            return "conflict: serving_status_not_validated"

        # ── Gate 3: serving evaluation run exists ───────────────────────────
        if student.serving_evaluation_run_id is None:
            return "conflict: serving_evaluation_run_missing"
        serving_run = (
            session.query(RunRecord)
            .filter_by(run_id=student.serving_evaluation_run_id)
            .first()
        )
        if serving_run is None:
            return "conflict: serving_evaluation_run_missing"

        # ``serving_status`` is retained historical state. Workspaces created
        # before the production AIPerf contract can contain a successful
        # synthetic httpx latency sweep marked validated. That evidence stays
        # visible for audit, but it cannot unlock a production handoff or
        # portable bundle until the Student is revalidated through AIPerf.
        serving_assessment = assess_aiperf_serving_run(
            serving_run,
            expected_concurrencies=settings.STUDENT_LATENCY_TEST_CONCURRENCIES,
        )
        if not serving_assessment.current:
            return "conflict: serving_benchmark_requires_aiperf"

        # ── Gate 4: Inference Contract parity ───────────────────────────────
        training_contract = student.training_inference_contract
        if not isinstance(training_contract, dict):
            # Defensive recovery for an incomplete Student row: the training
            # DatasetExport remains the canonical derivation source.
            training_contract = resolve_training_inference_contract(
                session, list(student.dataset_export_ids or [])
            )
        evaluation_contract = serving_run.inference_contract or {}
        if not _contracts_equivalent(training_contract, evaluation_contract):
            return "conflict: INFERENCE_CONTRACT_MISMATCH"

        # ── Compose context for AR rendering (read-only) ────────────────────
        # Quality eval (TAO-source)
        quality_metrics: dict[str, Any] | None = None
        rescored_metrics: dict[str, Any] | None = None
        if student.quality_evaluation_run_id is not None:
            qr = (
                session.query(RunRecord)
                .filter_by(run_id=student.quality_evaluation_run_id)
                .first()
            )
            if qr is not None:
                quality_metrics = qr.metrics
                rescored_metrics = qr.rescored_metrics
        # Serving metrics (NIM-source) — same RunRecord structure but
        # ``metrics`` carries Exact Match aggregates re-scored through the
        # full evaluation pipeline.
        serving_metrics = serving_run.metrics

        # NIM deployment metadata
        local_deployment = (
            session.query(LocalNimDeployment)
            .filter_by(student_model_id=student_model_id, project_id=project_id)
            .order_by(LocalNimDeployment.deployed_at.desc())
            .first()
        )
        nim_container_image = (
            local_deployment.nim_container_image if local_deployment else None
        )
        nim_served_model_name = (
            local_deployment.nim_served_model_name if local_deployment else None
        )
        nim_model_name_path = (
            local_deployment.nim_model_name_path if local_deployment else None
        )

        # Base ModelConfig for context window + image input flag
        base_mc = (
            session.query(ModelConfig)
            .filter_by(model_config_id=student.student_base_model_config_id)
            .first()
        )
        base_local_deploy_metadata = (
            base_mc.local_deploy_metadata
            if base_mc and isinstance(base_mc.local_deploy_metadata, dict)
            else {}
        )
        if not nim_container_image:
            catalog_image = base_local_deploy_metadata.get("nim_container_image")
            if isinstance(catalog_image, str) and catalog_image:
                nim_container_image = catalog_image
        nim_release_version = student.nim_vlm_release_version or (
            release_version_from_image(nim_container_image)
        )
        nim_model_size_raw = base_local_deploy_metadata.get("nim_model_size")
        nim_model_size = (
            str(nim_model_size_raw) if nim_model_size_raw is not None else None
        )
        extra_container_env = resolve_extra_container_env(
            project_id,
            student.student_base_model_config_id,
            settings.WORKSPACE_ROOT,
        )

        # Training lineage
        train_job = (
            session.query(TAOJob).filter_by(tao_job_id=student.tao_job_id).first()
        )
        quantize_job = None
        if student.quantize_tao_job_id is not None:
            quantize_job = (
                session.query(TAOJob)
                .filter_by(tao_job_id=student.quantize_tao_job_id)
                .first()
            )

        # Test Pool DatasetExport SHA-256 (provenance for reproducibility).
        # StudentModel.dataset_export_ids is training-only by contract; follow
        # the paired evaluate job from the artifact-producing train/quantize
        # job. Prefer the serving run's persisted value when present.
        artifact_parent_job_id = student.quantize_tao_job_id or student.tao_job_id
        dataset_sha = serving_run.dataset_manifest_sha256
        if not dataset_sha:
            dataset_sha = resolve_paired_test_pool_dataset_sha(
                session,
                artifact_parent_tao_job_id=artifact_parent_job_id,
                fallback_export_ids=list(student.dataset_export_ids or []),
            )

        # GPU memory minimum (drives the gpu_requirements string and
        # differentiates 2B vs 8B handoffs on identical-GPU rentals).
        base_model_name = base_mc.model_name if base_mc else ""
        # Shared policy (gpu_memory_floor) — one implementation for the
        # deploy preflight and the handoff, so they cannot disagree.
        gpu_memory_minimum_gb = resolve_gpu_memory_floor_gb(
            base_model_name=base_model_name,
            quantization_method=student.quantization_method,
            base_local_deploy_metadata=(
                base_mc.local_deploy_metadata if base_mc else None
            ),
            settings=settings,
        )

        # Canonical docker run via the local_nim_service builders. This is the
        # exact safe command :deploy_nim ran for this Student: Docker receives
        # name-only ``-e NGC_API_KEY`` and resolves the exported value without
        # placing it in argv.
        docker_run_args: list[str] | None = None
        docker_run_command: str | None = None
        if local_deployment is not None:
            # Action Request payloads (which include this
            # technical_requirements block) MUST NOT contain secrets.
            docker_run_args = build_student_docker_run_args(
                deployment=local_deployment,
                settings=settings,
            )
            docker_run_command = build_student_docker_run_display(
                deployment=local_deployment, settings=settings
            )

        # nim_model_profile_recommended — what the operator pastes into
        # ``NIM_MODEL_PROFILE`` when redeploying. Sourced from the live
        # selection (the value NIM auto-picked during our deploy); fall back
        # to the explicit request, then to None (let NIM auto-pick again).
        nim_model_profile_recommended = (
            student.nim_model_profile_selected or student.nim_model_profile_requested
        )

        context: dict[str, Any] = {
            "student_model_id": student.student_model_id,
            "base_model_name": base_model_name or None,
            "nim_checkpoint_ref": student.nim_checkpoint_ref,
            "nim_container_image": nim_container_image,
            "nim_served_model_name": nim_served_model_name,
            "nim_model_name_path": nim_model_name_path,
            "nim_release_version": nim_release_version,
            "nim_model_size": nim_model_size,
            "extra_container_env": extra_container_env,
            # Audit (sourced from StudentModel as-is).
            "nim_model_profile_requested": student.nim_model_profile_requested,
            "nim_model_profile_selected": student.nim_model_profile_selected,
            # Customer-facing recommendation.
            "nim_model_profile_recommended": nim_model_profile_recommended,
            "nim_profile_metadata": student.nim_profile_metadata,
            "quantization_method": student.quantization_method,
            "gpu_type": student.gpu_type,
            "gpu_count": student.gpu_count,
            "gpu_memory_minimum_gb": gpu_memory_minimum_gb,
            "training_preset": student.training_preset,
            "lora_config": student.lora_config,
            "dataset_export_ids": list(student.dataset_export_ids or []),
            "dataset_manifest_sha256": dataset_sha,
            "inference_contract": training_contract,
            "training_tao_job_id": train_job.tao_job_id if train_job else None,
            "quantize_tao_job_id": quantize_job.tao_job_id if quantize_job else None,
            "quality_metrics": quality_metrics,
            "rescored_metrics": rescored_metrics,
            "serving_metrics": serving_metrics,
            "quality_evaluation_run_id": student.quality_evaluation_run_id,
            "serving_evaluation_run_id": student.serving_evaluation_run_id,
            "visual_budget_preset_key": serving_run.visual_budget_preset_key,
            "generation_preset_key": serving_run.generation_preset_key,
            "thinking_mode_effective": serving_run.thinking_mode_effective,
            "docker_run_args": docker_run_args,
            "docker_run_command": docker_run_command,
        }

        return generate_action_request(
            "deployment_handoff",
            project_name=project.name or "(unnamed project)",
            project_id=project_id,
            context=context,
        )


# ── Generator registered with the AR framework ──────────────────────────────


def _generate_deployment_handoff(
    project_name: str,
    project_id: str,  # noqa: ARG001 — uniform Action Request generator signature
    context: dict[str, Any],
) -> dict[str, Any]:
    """Render the deployment_handoff content from the validated context.

    The gates are NOT re-checked here — the caller
    (``generate_deployment_handoff_for_student``) has already enforced them.
    Direct callers of ``generate_action_request("deployment_handoff", ...)``
    that bypass the Student-scoped helper get content shaped by whatever
    context they supplied; this is intentional — the AR framework is generic
    and the Student-scoped gate lives in the per-Student endpoint, not in
    the generator itself.
    """
    student_model_id = context.get("student_model_id", "(unknown)")
    base_model_name = context.get("base_model_name") or "(unknown)"
    quantization_method = context.get("quantization_method") or "none"
    gpu_type = context.get("gpu_type") or "(unknown)"
    gpu_count = context.get("gpu_count") or "?"
    gpu_memory_minimum_gb = context.get("gpu_memory_minimum_gb") or 0
    nim_checkpoint_ref = context.get("nim_checkpoint_ref") or "(packaging pending)"
    nim_container_image = context.get("nim_container_image") or "(not yet deployed)"
    nim_served_model_name = (
        context.get("nim_served_model_name") or f"student-{student_model_id[:8]}"
    )
    nim_model_name_path = (
        context.get("nim_model_name_path") or "/opt/checkpoints/student"
    )
    nim_release_version = context.get("nim_release_version") or "(unknown)"
    nim_model_size = context.get("nim_model_size")
    extra_container_env_raw: Any = context.get("extra_container_env") or {}
    extra_container_env: dict[str, str] = (
        cast("dict[str, str]", extra_container_env_raw)
        if isinstance(extra_container_env_raw, dict)
        else {}
    )
    nim_model_profile_recommended = context.get("nim_model_profile_recommended")
    # Display fallback for the rendered_text only — the structured field stays
    # null when no profile has been selected, so consumers can detect the
    # "let NIM auto-pick" case.
    nim_model_profile_display = (
        nim_model_profile_recommended or "(auto-selected by NIM)"
    )
    nim_profile_metadata_raw: Any = context.get("nim_profile_metadata") or {}
    nim_profile_metadata: dict[str, Any] = (
        cast("dict[str, Any]", nim_profile_metadata_raw)
        if isinstance(nim_profile_metadata_raw, dict)
        else {}
    )
    # tensor_parallelism — sourced from the live profile metadata. Defaults
    # to 1 (single-GPU deploys, the v1 default).
    tensor_parallelism_raw: Any = nim_profile_metadata.get("tp") or 1
    tensor_parallelism: int = (
        tensor_parallelism_raw if isinstance(tensor_parallelism_raw, int) else 1
    )
    inference_contract_raw: Any = context.get("inference_contract") or {}
    inference_contract: dict[str, Any] = (
        cast("dict[str, Any]", inference_contract_raw)
        if isinstance(inference_contract_raw, dict)
        else {}
    )
    visual_budget_preset_key = context.get("visual_budget_preset_key") or "balanced"
    generation_preset_key = context.get("generation_preset_key") or "precise"
    thinking_mode_effective = context.get("thinking_mode_effective") or "on"
    dataset_manifest_sha256 = (
        context.get("dataset_manifest_sha256") or "(test pool not exported)"
    )
    # Canonical docker_run via local_nim_service builders. When the context
    # arrives without these (direct-call path that bypasses the gate helper),
    # synthesize a placeholder so the rendered_text is still readable.
    docker_run_args_raw: Any = context.get("docker_run_args") or []
    docker_run_args: list[str] = (
        cast("list[str]", docker_run_args_raw)
        if isinstance(docker_run_args_raw, list)
        else []
    )
    docker_run_command = (
        context.get("docker_run_command")
        or "(deployment not yet recorded — re-deploy and re-export)"
    )

    # Quality + serving evaluation snapshot
    quality_metrics_raw: Any = context.get("quality_metrics") or {}
    quality_metrics: dict[str, Any] = (
        cast("dict[str, Any]", quality_metrics_raw)
        if isinstance(quality_metrics_raw, dict)
        else {}
    )
    rescored_metrics_raw: Any = context.get("rescored_metrics") or {}
    rescored_metrics: dict[str, Any] = (
        cast("dict[str, Any]", rescored_metrics_raw)
        if isinstance(rescored_metrics_raw, dict)
        else {}
    )
    serving_metrics_raw: Any = context.get("serving_metrics") or {}
    serving_metrics: dict[str, Any] = (
        cast("dict[str, Any]", serving_metrics_raw)
        if isinstance(serving_metrics_raw, dict)
        else {}
    )
    rescored_overall_raw: Any = rescored_metrics.get("overall") or {}
    rescored_overall: dict[str, Any] = (
        cast("dict[str, Any]", rescored_overall_raw)
        if isinstance(rescored_overall_raw, dict)
        else {}
    )
    overall_quality_em: Any = rescored_overall.get("exact_match_rate")
    if overall_quality_em is None:
        quality_overall_raw: Any = quality_metrics.get("overall") or {}
        quality_overall: dict[str, Any] = (
            cast("dict[str, Any]", quality_overall_raw)
            if isinstance(quality_overall_raw, dict)
            else {}
        )
        overall_quality_em = quality_overall.get("exact_match_rate")
    serving_overall_raw: Any = serving_metrics.get("overall") or {}
    serving_overall: dict[str, Any] = (
        cast("dict[str, Any]", serving_overall_raw)
        if isinstance(serving_overall_raw, dict)
        else {}
    )
    serving_em: Any = serving_overall.get("exact_match_rate")

    # Training lineage
    training_tao_job_id = context.get("training_tao_job_id") or "(unknown)"
    quantize_tao_job_id = context.get("quantize_tao_job_id") or "(none — baseline)"
    training_preset = context.get("training_preset") or "(unknown)"
    lora_config_raw: Any = context.get("lora_config") or {}
    lora_config: dict[str, Any] = (
        cast("dict[str, Any]", lora_config_raw)
        if isinstance(lora_config_raw, dict)
        else {}
    )
    dataset_export_ids_raw: Any = context.get("dataset_export_ids") or []
    dataset_export_ids: list[str] = (
        cast("list[str]", dataset_export_ids_raw)
        if isinstance(dataset_export_ids_raw, list)
        else []
    )

    # Recommended NIM env vars — what the operator pastes into a fresh deploy.
    # Mirrors the env values _build_docker_run_command emits for role=student.
    nim_env_vars_recommended: dict[str, str] = {
        "NGC_API_KEY": "$NGC_API_KEY",
        "NIM_MODEL_NAME": nim_model_name_path,
        "NIM_ENABLE_KV_CACHE_REUSE": "0",
    }
    if nim_served_model_name:
        nim_env_vars_recommended["NIM_SERVED_MODEL_NAME"] = nim_served_model_name
    if nim_model_size:
        nim_env_vars_recommended["NIM_MODEL_SIZE"] = str(nim_model_size)
    if nim_model_profile_recommended:
        nim_env_vars_recommended["NIM_MODEL_PROFILE"] = nim_model_profile_recommended
    nim_env_vars_recommended.update(extra_container_env)

    # gpu_requirements — example shape: "8× A100 80 GB". On a
    # single-GPU host both 2B and 8B land "1× A100"; the per-variant
    # memory hint is what differentiates them.
    if gpu_memory_minimum_gb:
        gpu_requirements = f"{gpu_count}× {gpu_type} (≥{gpu_memory_minimum_gb} GB)"
    else:
        gpu_requirements = f"{gpu_count}× {gpu_type}"

    technical_requirements: dict[str, Any] = {
        "nim_container_image": nim_container_image,
        "nim_release_version": nim_release_version,
        "nim_served_model_name": nim_served_model_name,
        "nim_model_name_path": nim_model_name_path,
        "nim_model_size": nim_model_size,
        "extra_container_env": extra_container_env,
        # Customer-facing recommendation. Audit value
        # ``nim_model_profile_selected`` is preserved on the StudentModel row.
        "nim_model_profile_recommended": nim_model_profile_recommended,
        "nim_profile_metadata": nim_profile_metadata,
        "tensor_parallelism": tensor_parallelism,
        "nim_env_vars_recommended": nim_env_vars_recommended,
        "checkpoint_reference": nim_checkpoint_ref,
        "checkpoint_directory_structure": [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "*.safetensors (sharded)",
            "runtime_params.json (NIM-injected)",
        ],
        "gpu_requirements": gpu_requirements,
        "quantization_method": quantization_method,
        "auth_mode": "none (assume internal network)",
        "health_check": "GET /v1/health/ready",
        "smoke_test": (
            f"POST /v1/chat/completions with model={nim_served_model_name!r}"
        ),
        "inference_contract": inference_contract,
        "decoding_params": {
            "generation_preset_key": generation_preset_key,
            "thinking_mode_effective": thinking_mode_effective,
        },
        "visual_budget_preset_key": visual_budget_preset_key,
        "docker_run_command": docker_run_command,
        "docker_run_args": list(docker_run_args),
    }

    current_environment: dict[str, Any] = {
        "student_model_id": student_model_id,
        "base_model_name": base_model_name,
        "quality_status": "validated",
        "serving_status": "validated",
        "quality_evaluation_overall_exact_match": overall_quality_em,
        "serving_evaluation_overall_exact_match": serving_em,
        "quality_evaluation_run_id": context.get("quality_evaluation_run_id"),
        "serving_evaluation_run_id": context.get("serving_evaluation_run_id"),
        "quality_metrics": quality_metrics,
        "rescored_metrics": rescored_metrics,
        "serving_metrics": serving_metrics,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "training_tao_job_id": training_tao_job_id,
        "quantize_tao_job_id": quantize_tao_job_id,
        "training_preset": training_preset,
        "lora_config": lora_config,
        "dataset_export_ids": dataset_export_ids,
        "checkpoint_packaging_status": "validated",
    }

    rendered_text = (
        "Deployment Handoff Request\n\n"
        f"Project: {project_name}\n"
        f"Student: {student_model_id}\n"
        f"Base: {base_model_name} · Quantization: {quantization_method} · "
        f"GPU: {gpu_requirements}\n\n"
        "Both quality and serving readiness gates have passed.\n"
        "Training-time and evaluation-time Inference Contracts match.\n\n"
        "The receiving infrastructure team owns the permanent service, "
        "access controls, scaling, monitoring, and operations.\n\n"
        "Checkpoint\n"
        f"    Validated source: {nim_checkpoint_ref}\n"
        "    Packaging status: validated\n"
        "    Portable artifact: use Download portable NIM deployment bundle "
        "in this panel; "
        "it includes the checkpoint, manifest, and SHA-256 checksums.\n\n"
        "NIM Configuration\n"
        f"    Runtime image: {nim_container_image}\n"
        f"    NIM release: {nim_release_version}\n"
        f"    NIM_MODEL_NAME: {nim_model_name_path}\n"
        f"    NIM_SERVED_MODEL_NAME: {nim_served_model_name}\n"
        f"    NIM_MODEL_SIZE: {nim_model_size or '(not required)'}\n"
        f"    NIM_MODEL_PROFILE: {nim_model_profile_display}\n"
        "Production deployment command:\n"
        f"{docker_run_command}\n\n"
        "Endpoint health: GET /v1/health/ready\n"
        f"Smoke test: POST /v1/chat/completions with model={nim_served_model_name!r}\n\n"
        "Model\n"
        f"    Base model: {base_model_name}\n"
        f"    Quantization: {quantization_method}\n"
        f"    GPU requirement: {gpu_requirements}\n"
        f"    Tensor parallelism: {tensor_parallelism}\n\n"
        "Evaluation\n"
        "Quality (TAO-rescored):\n"
        f"    Overall Exact Match: {overall_quality_em}\n"
        f"    Inference Contract: {inference_contract}\n"
        f"    Test Pool dataset SHA-256: {dataset_manifest_sha256}\n\n"
        "Serving (NIM-validated):\n"
        f"    Overall Exact Match: {serving_em}\n"
        f"    Profile recommended: {nim_model_profile_display}\n"
        f"    Profile metadata: {nim_profile_metadata}\n"
        f"    Tensor parallelism: {tensor_parallelism}\n\n"
        "Training Lineage\n"
        f"    Training TAO job: {training_tao_job_id}\n"
        f"    Quantize TAO job: {quantize_tao_job_id}\n"
        f"    Training preset: {training_preset}\n"
        f"    LoRA: {lora_config}\n"
        f"    Dataset exports: {dataset_export_ids}\n"
    )

    return {
        "technical_requirements": technical_requirements,
        "current_environment": current_environment,
        "rendered_text": rendered_text,
    }


# Register at import time (side-effect import in main.py).
register_generator("deployment_handoff", _generate_deployment_handoff)
