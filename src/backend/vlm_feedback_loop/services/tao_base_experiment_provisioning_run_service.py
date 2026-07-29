# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracked, on-demand TAO base-experiment provisioning.

The Student Training UI calls :func:`start_provisioning_run` before creating a
suite when any selected base is not registered. The request returns
immediately; a tracked in-process task runs the existing idempotent
``provision_base_experiments`` flow, while the UI polls the durable deployment
record. A successful run patches the matching ModelConfig rows across every
project before the UI continues with the normal training-suite endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.deployment_models import (
    TAOBaseExperimentProvisioningRun,
)
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.tao_base_experiment_provisioning_service import (
    PULL_REQUIREMENTS_PATH,
    PULL_SCRIPT_PATH,
    ProvisioningResult,
    provision_base_experiments,
)
from vlm_feedback_loop.services.tao_bootstrap_service import (
    patch_model_pull_status_across_projects,
)
from vlm_feedback_loop.services.tao_workspace_service import (
    read_tao_deployment_config,
)

logger = logging.getLogger(
    "vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service"
)

ACTIVE_STATUSES = frozenset({"queued", "running"})
TERMINAL_STATUSES = frozenset({"succeeded", "failed"})


def _redact(text: str, settings: Settings) -> str:
    """Remove every provisioning credential from a persisted/returned error."""
    values = {
        get_effective_secret("TAO_API_KEY", settings) or "",
        settings.HF_TOKEN or "",
        settings.TAO_WORKSPACE_S3_ACCESS_KEY or "",
        settings.TAO_WORKSPACE_S3_SECRET_KEY or "",
    }
    redacted = text
    for value in sorted({value for value in values if value}, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def _roles(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(role) for role in cast("list[Any]", value)]


def _serialize(run: TAOBaseExperimentProvisioningRun) -> dict[str, Any]:
    return {
        "provisioning_run_id": run.provisioning_run_id,
        "project_id": run.project_id,
        "requested_model_config_ids": list(run.requested_model_config_ids or []),
        "requested_model_names": list(run.requested_model_names or []),
        "status": run.status,
        "registered": list(run.registered or []),
        "already_registered": list(run.already_registered or []),
        "failures": list(run.failures or []),
        "error_ref": run.error_ref,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _selected_models(
    project_id: str,
    model_config_ids: list[str],
    settings: Settings,
) -> tuple[list[ModelConfig], str | None]:
    """Load and role-check the selected project-local catalog rows."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return [], f"not found: project {project_id!r} not found"

    by_id: dict[str, ModelConfig] = {}
    with Session(engine) as session:
        rows = (
            session.query(ModelConfig)
            .filter(ModelConfig.project_id == project_id)
            .filter(ModelConfig.model_config_id.in_(model_config_ids))
            .all()
        )
        for row in rows:
            # Detach before the session closes; only scalar/JSON fields are read.
            session.expunge(row)
            by_id[row.model_config_id] = row

    ordered: list[ModelConfig] = []
    for model_config_id in model_config_ids:
        row = by_id.get(model_config_id)
        if row is None:
            return (
                [],
                f"not found: ModelConfig {model_config_id!r} not found in "
                f"project {project_id!r}",
            )
        if "student_base" not in _roles(row.eligible_roles):
            return (
                [],
                f"validation: ModelConfig {model_config_id!r} does not have "
                "the student_base role",
            )
        ordered.append(row)
    return ordered, None


def _provisioning_prerequisite_error(settings: Settings) -> str | None:
    """Return an actionable error when automatic provisioning cannot run."""
    tao_cfg = read_tao_deployment_config(settings)
    if (
        tao_cfg is None
        or tao_cfg.bootstrap_status != "bootstrapped"
        or not tao_cfg.tao_workspace_id
    ):
        return (
            "validation: TAO workspace is not bootstrapped. Run "
            "`vlm-feedback-loop tao-bootstrap` first."
        )
    if not tao_cfg.tao_workspace_bucket:
        return "validation: the TAO workspace bucket is not configured"
    if not tao_cfg.tao_workspace_s3_endpoint_url_external:
        return "validation: the TAO workspace external S3 endpoint is not configured"
    if (
        not settings.TAO_WORKSPACE_S3_ACCESS_KEY
        or not settings.TAO_WORKSPACE_S3_SECRET_KEY
    ):
        return (
            "validation: TAO_WORKSPACE_S3_ACCESS_KEY and "
            "TAO_WORKSPACE_S3_SECRET_KEY are required for automatic base provisioning"
        )
    if not (settings.HF_TOKEN or "").strip():
        return (
            "validation: HF_TOKEN is required to provision gated Cosmos "
            "Student bases. Add it to the Blueprint deployment secrets and retry."
        )
    if not PULL_SCRIPT_PATH.is_file() or not PULL_REQUIREMENTS_PATH.is_file():
        return (
            "validation: the TAO base-provisioning helper is missing from this "
            "Blueprint installation"
        )
    if shutil.which("uv") is None:
        return (
            "validation: `uv` is required for the isolated TAO provisioning "
            "helper but was not found on PATH"
        )
    return None


def get_provisioning_run(
    project_id: str,
    provisioning_run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Return one project-originated provisioning run."""
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        run = session.get(TAOBaseExperimentProvisioningRun, provisioning_run_id)
        if run is None or run.project_id != project_id:
            return (
                "not found: TAO base-experiment provisioning run "
                f"{provisioning_run_id!r} not found"
            )
        return _serialize(run)


def _set_run_failed(
    provisioning_run_id: str,
    settings: Settings,
    *,
    error: str,
    failures: list[dict[str, str]] | None = None,
) -> None:
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        run = session.get(TAOBaseExperimentProvisioningRun, provisioning_run_id)
        if run is None:
            return
        run.status = "failed"
        run.failures = failures or []
        run.error_ref = error[:4096]
        run.completed_at = utc_now()
        session.commit()


async def execute_provisioning_run(
    provisioning_run_id: str,
    settings: Settings,
    *,
    _provisioner: Any = None,
) -> None:
    """Execute one persisted run and record its terminal outcome."""
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        run = session.get(TAOBaseExperimentProvisioningRun, provisioning_run_id)
        if run is None or run.status not in ACTIVE_STATUSES:
            return
        run.status = "running"
        run.started_at = utc_now()
        project_id = run.project_id
        model_config_ids = list(run.requested_model_config_ids)
        requested_names = list(run.requested_model_names)
        session.commit()

    provisioner = _provisioner or provision_base_experiments
    try:
        result: ProvisioningResult = await provisioner(
            settings,
            student_base_model_config_ids=model_config_ids,
            project_id=project_id,
        )
    except asyncio.CancelledError:
        patch_model_pull_status_across_projects(
            Path(settings.WORKSPACE_ROOT),
            model_names=requested_names,
            pull_status="failed",
            preserve_pull_complete=True,
        )
        _set_run_failed(
            provisioning_run_id,
            settings,
            error="backend shutdown interrupted base provisioning; start a new Training Jobs run",
        )
        raise
    except Exception as exc:
        logger.exception("TAO base provisioning run %s failed", provisioning_run_id)
        patch_model_pull_status_across_projects(
            Path(settings.WORKSPACE_ROOT),
            model_names=requested_names,
            pull_status="failed",
            preserve_pull_complete=True,
        )
        _set_run_failed(
            provisioning_run_id,
            settings,
            error=_redact(f"{type(exc).__name__}: {exc}", settings),
        )
        return

    failures = [
        {"target": target, "error": _redact(error, settings)}
        for target, error in result.failed
    ]
    if failures:
        failed_ids = {failure["target"] for failure in failures}
        failed_names = [
            name
            for model_config_id, name in zip(
                model_config_ids, requested_names, strict=True
            )
            if model_config_id in failed_ids
        ]
        # A deployment-level failure (for example a workspace/S3 error)
        # may use a non-model target such as ``workspace``. In that case
        # every requested model was interrupted and must leave ``pulling``.
        if not failed_names:
            failed_names = requested_names
        patch_model_pull_status_across_projects(
            Path(settings.WORKSPACE_ROOT),
            model_names=failed_names,
            pull_status="failed",
            preserve_pull_complete=True,
        )

    with Session(engine) as session:
        run = session.get(TAOBaseExperimentProvisioningRun, provisioning_run_id)
        if run is None:
            return
        run.registered = list(result.registered)
        run.already_registered = list(result.already_registered)
        run.failures = failures
        run.status = "failed" if failures else "succeeded"
        run.error_ref = (
            "; ".join(f"{item['target']}: {item['error']}" for item in failures)[:4096]
            if failures
            else None
        )
        run.completed_at = utc_now()
        session.commit()


def start_provisioning_run(
    project_id: str,
    student_base_model_config_ids: list[str],
    settings: Settings,
) -> dict[str, Any] | str:
    """Ensure selected missing bases in a tracked background task."""
    # Preserve caller order while rejecting duplicate work.
    ids = list(dict.fromkeys(student_base_model_config_ids))
    models, error = _selected_models(project_id, ids, settings)
    if error is not None:
        return error

    missing = [
        model
        for model in models
        if not (
            model.tao_base_experiment_id
            and model.tao_base_experiment_pull_status == "pull_complete"
        )
    ]
    missing_ids = [model.model_config_id for model in missing]
    missing_names = [model.model_name for model in missing]

    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    if missing:
        with Session(engine) as session:
            active = (
                session.query(TAOBaseExperimentProvisioningRun)
                .filter(TAOBaseExperimentProvisioningRun.status.in_(ACTIVE_STATUSES))
                .order_by(TAOBaseExperimentProvisioningRun.created_at.desc())
                .first()
            )
            if active is not None:
                active_names = set(active.requested_model_names or [])
                if active.project_id == project_id and set(missing_names).issubset(
                    active_names
                ):
                    return _serialize(active)
                return (
                    "conflict: another TAO base-experiment provisioning run is "
                    "already active; wait for it to finish, then retry"
                )

    if missing:
        prerequisite_error = _provisioning_prerequisite_error(settings)
        if prerequisite_error is not None:
            return prerequisite_error

    run_id = generate_uuid4()
    now = utc_now()
    run = TAOBaseExperimentProvisioningRun(
        provisioning_run_id=run_id,
        project_id=project_id,
        requested_model_config_ids=missing_ids,
        requested_model_names=missing_names,
        status="queued" if missing else "succeeded",
        registered=[],
        already_registered=[],
        failures=[],
        error_ref=None,
        started_at=None,
        completed_at=None if missing else now,
        created_at=now,
        updated_at=now,
    )
    with Session(engine) as session:
        session.add(run)
        session.commit()
        response = _serialize(run)

    if not missing:
        return response

    patch_model_pull_status_across_projects(
        Path(settings.WORKSPACE_ROOT),
        model_names=missing_names,
        pull_status="pulling",
        preserve_pull_complete=True,
    )
    worker = execute_provisioning_run(run_id, settings)
    try:
        background_manager.register(
            f"tao-base-provision-{run_id}",
            worker,
        )
    except RuntimeError as exc:
        worker.close()
        patch_model_pull_status_across_projects(
            Path(settings.WORKSPACE_ROOT),
            model_names=missing_names,
            pull_status="failed",
            preserve_pull_complete=True,
        )
        _set_run_failed(run_id, settings, error=f"could not start provisioning: {exc}")
        return get_provisioning_run(project_id, run_id, settings)
    return response


def recover_interrupted_provisioning_runs(settings: Settings) -> int:
    """Fail active runs left behind by a backend restart."""
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    interrupted_names: list[str] = []
    recovered = 0
    with Session(engine) as session:
        runs = (
            session.query(TAOBaseExperimentProvisioningRun)
            .filter(TAOBaseExperimentProvisioningRun.status.in_(ACTIVE_STATUSES))
            .all()
        )
        for run in runs:
            interrupted_names.extend(run.requested_model_names or [])
            run.status = "failed"
            run.error_ref = (
                "backend restart interrupted base provisioning; "
                "start a new Training Jobs run"
            )
            run.completed_at = utc_now()
            recovered += 1
        if runs:
            session.commit()

    if interrupted_names:
        patch_model_pull_status_across_projects(
            Path(settings.WORKSPACE_ROOT),
            model_names=list(dict.fromkeys(interrupted_names)),
            pull_status="failed",
            preserve_pull_complete=True,
        )
    return recovered


__all__ = [
    "ACTIVE_STATUSES",
    "TERMINAL_STATUSES",
    "execute_provisioning_run",
    "get_provisioning_run",
    "recover_interrupted_provisioning_runs",
    "start_provisioning_run",
]
