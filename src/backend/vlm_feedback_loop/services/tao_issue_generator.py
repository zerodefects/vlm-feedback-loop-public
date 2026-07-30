# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""tao_issue Action Request content generator.

Registered at import time.  Consumed by the Training Job Monitor's
[Report TAO Issue] button.

Unlike most generators, this one reads project DB state:
given ``context.tao_job_id`` it fetches the TAOJob row and the associated
student-base ModelConfig to pre-fill the Action Request with job-specific
details.  Rendered text passes through the framework's ``redact`` filter
before the response is returned (no secrets leak even if the persisted
``error_ref`` contained a token).
"""

from __future__ import annotations

from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import get_settings
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services.action_requests import register_generator
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.tao_job_service import (
    extract_actionable_failure_from_logs,
)


def _render_text(
    *,
    project_name: str,
    tao_base_url: str,
    tao_org_name: str,
    tao_job_id: str | None,
    tao_external_job_id: str | None,
    action: str,
    base_model_name: str,
    training_preset: str | None,
    training_policy_type: str | None,
    status: str,
    error_ref: str | None,
    epoch: Any,
    dataset_export_ids: list[str],
    tao_release_version: str | None,
    cosmos_rl_container_tag: str | None,
) -> str:
    logs_hint = (
        f"  1. Check TAO logs: GET {tao_base_url}/orgs/{tao_org_name}/jobs/"
        f"{tao_external_job_id}:logs\n"
        if tao_external_job_id
        else "  1. Check TAO logs (no external job id available — job was not "
        "submitted before the error).\n"
    )

    return (
        f"TAO Job Issue Report\n"
        f"\n"
        f"Project: {project_name}\n"
        f"TAO Endpoint: {tao_base_url}\n"
        f"TAO Organization: {tao_org_name}\n"
        f"\n"
        f"Job ID: {tao_job_id or '(job id not provided)'}\n"
        f"TAO External Job ID: {tao_external_job_id or '(not submitted)'}\n"
        f"Action: {action}\n"
        f"Base Model: {base_model_name}\n"
        f"Training Preset: {training_preset or 'n/a'}\n"
        f"Status: {status}\n"
        f"\n"
        f"Error: {error_ref or '(none)'}\n"
        f"\n"
        f"Job Configuration:\n"
        f"  Policy: {training_policy_type or 'n/a'}\n"
        f"  Training preset: {training_preset or 'n/a'}\n"
        f"  Epochs: {epoch if epoch is not None else 'n/a'}\n"
        f"  Dataset Exports: {', '.join(dataset_export_ids) or '(none)'}\n"
        f"  TAO release: {tao_release_version or 'n/a'}\n"
        f"  Cosmos-RL tag: {cosmos_rl_container_tag or 'n/a'}\n"
        f"\n"
        f"Suggested Diagnostics:\n"
        f"{logs_hint}"
        f"  2. Verify GPU memory / disk on the TAO workspace\n"
        f"  3. Confirm driver branch compatibility "
        f"(R580+ for TAO 6.26.3 / CUDA 13.0)\n"
        f"  4. Validate base_experiment_ids resolve on the TAO side\n"
    )


def _generate_tao_issue(
    project_name: str,
    project_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Generate a tao_issue Action Request.

    Requires ``context.tao_job_id``.  If the TAOJob is not found, renders
    a clear placeholder message so the SME can still copy the request and
    hand off with partial info.
    """
    settings = get_settings()
    tao_base_url = settings.TAO_API_BASE_URL or "(TAO_API_BASE_URL not configured)"
    tao_org_name = settings.TAO_ORG_NAME or "(TAO_ORG_NAME not configured)"

    tao_job_id = context.get("tao_job_id")

    # Defaults when the record cannot be resolved.
    defaults: dict[str, Any] = {
        "tao_external_job_id": None,
        "action": "(unknown)",
        "base_model_name": "(unknown)",
        "training_preset": None,
        "training_policy_type": None,
        "status": "(unknown)",
        "error_ref": None,
        "epoch": None,
        "dataset_export_ids": [],
        "tao_release_version": None,
        "cosmos_rl_container_tag": None,
    }

    if not tao_job_id:
        rendered = _render_text(
            project_name=project_name,
            tao_base_url=tao_base_url,
            tao_org_name=tao_org_name,
            tao_job_id=None,
            **defaults,
        )
        return {
            "technical_requirements": {
                "tao_endpoint": tao_base_url,
                "tao_org": tao_org_name,
                "tao_job_id": None,
                "note": "context.tao_job_id was not provided",
            },
            "current_environment": {},
            "rendered_text": rendered,
        }

    # Open the project DB to resolve the job + associated model config.
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    job_fields: dict[str, Any] = dict(defaults)
    base_model_name = "(unknown)"

    if engine is not None:
        with Session(engine) as session:
            job = (
                session.query(TAOJob)
                .filter_by(tao_job_id=tao_job_id, project_id=project_id)
                .first()
            )
            if job is not None:
                job_config: dict[str, Any] = job.job_config or {}
                hyperparameters_raw: Any = job_config.get("hyperparameters") or {}
                hyperparameters: dict[str, Any] = (
                    cast("dict[str, Any]", hyperparameters_raw)
                    if isinstance(hyperparameters_raw, dict)
                    else {}
                )
                train_params_raw: Any = hyperparameters.get("train")
                train_params: dict[str, Any] | None = (
                    cast("dict[str, Any]", train_params_raw)
                    if isinstance(train_params_raw, dict)
                    else None
                )
                epoch: Any = (
                    train_params.get("epoch") if train_params is not None else None
                )

                dataset_export_ids = [str(x) for x in job.dataset_export_ids]
                error_ref = job.error_ref
                logs_text = (job.outputs or {}).get("tao_logs_text")
                actionable_error = extract_actionable_failure_from_logs(
                    logs_text if isinstance(logs_text, str) else None
                )
                if actionable_error and (
                    not error_ref
                    or error_ref.strip().lower()
                    == f"{job.action} action failed for cosmos-rl"
                ):
                    error_ref = actionable_error

                job_fields = {
                    "tao_external_job_id": job.tao_external_job_id,
                    "action": job.action,
                    "base_model_name": "(unknown)",
                    "training_preset": job_config.get("training_preset"),
                    "training_policy_type": job.training_policy_type,
                    "status": job.status,
                    "error_ref": error_ref,
                    "epoch": epoch,
                    "dataset_export_ids": dataset_export_ids,
                    "tao_release_version": job_config.get("tao_release_version"),
                    "cosmos_rl_container_tag": job_config.get(
                        "cosmos_rl_container_tag"
                    ),
                }

                # Look up the base model name.
                model_config = (
                    session.query(ModelConfig)
                    .filter_by(
                        model_config_id=job.student_base_model_config_id,
                        project_id=project_id,
                    )
                    .first()
                )
                if model_config is not None:
                    base_model_name = model_config.model_name
                    job_fields["base_model_name"] = base_model_name
            else:
                # Job id provided but not resolvable.
                job_fields["status"] = "(TAO job record not found)"

    rendered = _render_text(
        project_name=project_name,
        tao_base_url=tao_base_url,
        tao_org_name=tao_org_name,
        tao_job_id=tao_job_id,
        **job_fields,
    )

    technical_requirements: dict[str, Any] = {
        "tao_endpoint": tao_base_url,
        "tao_org": tao_org_name,
        "tao_job_id": tao_job_id,
        "tao_external_job_id": job_fields["tao_external_job_id"],
        "action": job_fields["action"],
        "base_model_name": job_fields["base_model_name"],
        "training_preset": job_fields["training_preset"],
        "status": job_fields["status"],
        "error": job_fields["error_ref"],
        "dataset_export_ids": job_fields["dataset_export_ids"],
        "tao_release_version": job_fields["tao_release_version"],
        "cosmos_rl_container_tag": job_fields["cosmos_rl_container_tag"],
        "diagnostic_endpoint": (
            f"GET {tao_base_url}/orgs/{tao_org_name}/jobs/"
            f"{job_fields['tao_external_job_id']}:logs"
            if job_fields["tao_external_job_id"]
            else None
        ),
    }

    return {
        "technical_requirements": technical_requirements,
        "current_environment": {},
        "rendered_text": rendered,
    }


# Register at import time (side-effect import in main.py)
register_generator("tao_issue", _generate_tao_issue)
