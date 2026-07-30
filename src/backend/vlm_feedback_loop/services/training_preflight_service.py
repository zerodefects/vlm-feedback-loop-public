# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training preflight service.

Confirms TAO reachability + safe job-timeout support + workspace readiness +
per-student-base experiment readiness + ``student_base`` role on each selected
base model config. Hardware and
environment constraints (GPU count/memory, driver/CUDA, disk) are the
TAO deployment's responsibility; this system communicates with TAO via
REST API only and cannot inspect remote infrastructure. Any hardware
failure surfaces later through TAO job polling.

That boundary differentiates the training preflight from the NIM
deployment preflight. This module MUST NOT run Docker / NVIDIA
Container Toolkit / GPU memory / NGC key checks.

The timeout, workspace, and base-experiment checks run between
``tao_reachable`` and ``student_base_role``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from sqlalchemy import func
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.tao_auth import TaoAuthError
from vlm_feedback_loop.services.tao_client import probe_tao_connection
from vlm_feedback_loop.services.tao_workspace_service import (
    get_workspace,
    read_tao_deployment_config,
)
from vlm_feedback_loop.services.training_preset import (
    TRAINING_PRESETS,
    resolve_training_preset,
)

logger = logging.getLogger("vlm_feedback_loop.services.training_preflight_service")


async def run_training_preflight(
    *,
    project_id: str,
    student_base_model_config_ids: list[str],
    settings: Settings,
    include_auto_labeled: bool = True,
    enable_lora: bool = True,
) -> dict[str, Any]:
    """Run the training preflight checks.

    Returns a dict shaped for :class:`TrainingPreflightResponse`::

        {
            "status": "passed" | "failed",
            "checks": [
                {"check_name": "tao_reachable", "passed": bool,
                 "message": str, "model_config_id": None},
                {"check_name": "student_base_role", "passed": bool,
                 "message": str, "model_config_id": "<id>"},
                ...
            ],
        }

    Ordering: ``tao_reachable`` first (global), then per-model
    ``student_base_role`` checks in the order the caller supplied. Every
    check runs independently — a failure on one MUST NOT prevent the
    others from executing (matches the capability-probe pattern).
    Overall ``status`` is ``passed`` only when every check
    passes.
    """
    checks: list[dict[str, Any]] = []

    # 1. TAO reachability — global check, no per-model scope.
    probe = await probe_tao_connection(settings)
    if probe.get("success"):
        checks.append(
            {
                "check_name": "tao_reachable",
                "passed": True,
                "message": "TAO endpoint reachable.",
                "model_config_id": None,
            }
        )
    else:
        checks.append(
            {
                "check_name": "tao_reachable",
                "passed": False,
                "message": probe.get("error") or "TAO endpoint unreachable.",
                "model_config_id": None,
            }
        )

    # 2. TAO job-timeout safety. cosmos-rl does not heartbeat while training,
    #    and unpatched FTMS defaults to killing a quiet job after 60 minutes.
    #    Fail closed before suite creation unless the server declares support
    #    for the Blueprint's required per-job timeout override.
    if probe.get("job_timeout_supported"):
        checks.append(
            {
                "check_name": "tao_job_timeout_supported",
                "passed": True,
                "message": (
                    "TAO accepts the required "
                    f"{settings.TAO_JOB_TIMEOUT_MINUTES}-minute stale-job timeout."
                ),
                "model_config_id": None,
            }
        )
    else:
        compatibility_error = probe.get("job_timeout_error")
        if not compatibility_error and not probe.get("success"):
            compatibility_error = (
                "TAO job-timeout compatibility cannot be verified until the "
                "endpoint is reachable."
            )
        checks.append(
            {
                "check_name": "tao_job_timeout_supported",
                "passed": False,
                "message": compatibility_error
                or "TAO does not support the required safe job timeout.",
                "model_config_id": None,
                "remediation": (
                    "Apply the v2 timeout_minutes schema patch in "
                    "docs/tao-ftms-install.md, restart the "
                    "TAO API and workflow services, then rerun preflight."
                ),
            }
        )

    # 3. TAO workspace readiness. bootstrap_status must be
    #    "bootstrapped" AND GET /workspaces/{id} must return 200.
    tao_config = read_tao_deployment_config(settings)
    workspace_result = await _check_tao_workspace_reachable(
        settings=settings, tao_config=tao_config
    )
    checks.append(workspace_result)

    # 4. Per-model base-experiment readiness. Requires a
    #    non-null `tao_base_experiment_id` AND cached pull status
    #    `pull_complete` on the ModelConfig.
    be_checks = _check_tao_base_experiment_ready(
        project_id=project_id,
        student_base_model_config_ids=student_base_model_config_ids,
        workspace_root=settings.WORKSPACE_ROOT,
    )
    checks.extend(be_checks)

    # 5. Gated-model credential. First-use TAO provisioning needs it remotely;
    # LoRA training also needs it locally after training so the Blueprint can
    # load the gated base and merge the adapter before baseline NIM evaluation.
    provisioning_required = any(
        check.get("provisioning_required") is True for check in be_checks
    )
    hf_token_configured = bool((settings.HF_TOKEN or "").strip())
    hf_required = provisioning_required or enable_lora
    if hf_required:
        checks.append(
            {
                "check_name": "hf_token_configured",
                "passed": hf_token_configured,
                "message": (
                    "Hugging Face token configured for gated Cosmos Student "
                    "base access and LoRA checkpoint merge."
                    if hf_token_configured
                    else (
                        "HF_TOKEN is required to access the selected gated "
                        "Cosmos Student base for provisioning or LoRA merge. "
                        "Add it to the Blueprint deployment secrets and retry "
                        "readiness."
                    )
                ),
                "model_config_id": None,
                "remediation": (
                    None
                    if hf_token_configured
                    else (
                        "Set HF_TOKEN in ~/.vlm_feedback_loop/.env, restart "
                        "the Blueprint backend, then rerun readiness."
                    )
                ),
            }
        )
    else:
        checks.append(
            {
                "check_name": "hf_token_configured",
                "passed": True,
                "message": (
                    "Hugging Face token is not required because LoRA is "
                    "disabled and the selected Student bases are already "
                    "provisioned."
                ),
                "model_config_id": None,
            }
        )

    # 6. Blueprint-owned LoRA merge runtime. TAO cannot evaluate an adapter
    # checkpoint directly, so the local merge + Student NIM baseline is a
    # required part of a LoRA chain rather than an optional deployment step.
    if enable_lora:
        from vlm_feedback_loop.services import student_model_service

        (
            merge_ready,
            merge_message,
        ) = await student_model_service.check_lora_merge_readiness(settings)
        checks.append(
            {
                "check_name": "lora_merge_runtime",
                "passed": merge_ready,
                "message": merge_message,
                "model_config_id": None,
                "remediation": (
                    None
                    if merge_ready
                    else (
                        "Run scripts/setup-dev.sh on the Blueprint host (or "
                        "configure MERGE_LORA_PYTHON), restart the backend, "
                        "then rerun readiness."
                    )
                ),
            }
        )
    else:
        checks.append(
            {
                "check_name": "lora_merge_runtime",
                "passed": True,
                "message": "LoRA merge runtime is not required for full-weight training.",
                "model_config_id": None,
            }
        )

    # 7. Per-model student_base role. Query in a single short session.
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    role_results: dict[str, dict[str, Any]] = {}
    resolved_presets: dict[str, dict[str, Any]] = {}
    if engine is not None:
        with Session(engine) as session:
            rows = (
                session.query(ModelConfig)
                .filter(ModelConfig.project_id == project_id)
                .filter(ModelConfig.model_config_id.in_(student_base_model_config_ids))
                .all()
            )
            for row in rows:
                # Server-resolved training-preset patches for
                # the Advanced expander — the backend resolver is the ONE
                # source; the UI renders these instead of recomputing (a
                # TS mirror drifted: wrong max_keep, wrong cosmos3-super
                # epochs).
                resolved_presets[row.model_config_id] = {
                    preset: resolve_training_preset(preset, row.model_name)
                    for preset in sorted(TRAINING_PRESETS)
                }
                roles_raw: Any = row.eligible_roles
                if isinstance(roles_raw, str):
                    try:
                        roles_raw = json.loads(roles_raw)
                    except json.JSONDecodeError:
                        roles_raw = []
                roles: list[str] = (
                    [str(r) for r in cast("list[Any]", roles_raw)]
                    if isinstance(roles_raw, list)
                    else []
                )
                has_role = "student_base" in roles
                role_results[row.model_config_id] = {
                    "passed": has_role,
                    "message": (
                        f"Model {row.model_name!r} has the student_base role."
                        if has_role
                        else (
                            f"Model {row.model_name!r} does not have the "
                            f"student_base role (roles: {sorted(roles)})."
                        )
                    ),
                }

    for mcid in student_base_model_config_ids:
        result = role_results.get(mcid)
        if result is None:
            checks.append(
                {
                    "check_name": "student_base_role",
                    "passed": False,
                    "message": f"ModelConfig {mcid!r} not found in project.",
                    "model_config_id": mcid,
                }
            )
        else:
            checks.append(
                {
                    "check_name": "student_base_role",
                    "passed": result["passed"],
                    "message": result["message"],
                    "model_config_id": mcid,
                }
            )

    # 8. Training data availability. Training exports select Verified
    #    labels under the ACTIVE guidance outside the Test Pool
    #    (dataset_export_service) — an empty selection means a
    #    training suite cannot start, so surface that here instead of
    #    letting the SME discover it on the Training screen. The Scale-Up
    #    screen branches on this check to show "No Verified training
    #    examples yet. Continue labeling." rather than a TAO-setup CTA.
    checks.append(
        _check_verified_train_examples(
            project_id=project_id, workspace_root=settings.WORKSPACE_ROOT
        )
    )
    data_summary = _get_training_data_summary(
        project_id=project_id,
        workspace_root=settings.WORKSPACE_ROOT,
        include_auto_labeled=include_auto_labeled,
    )

    overall = "passed" if all(c["passed"] for c in checks) else "failed"
    logger.info(
        "training_preflight project=%s status=%s check_count=%d",
        project_id,
        overall,
        len(checks),
    )
    return {
        "status": overall,
        "checks": checks,
        "data_summary": data_summary,
        "resolved_presets": resolved_presets,
    }


def _get_training_data_summary(
    *,
    project_id: str,
    workspace_root: str,
    include_auto_labeled: bool,
) -> dict[str, int]:
    """Return counts that mirror the training DatasetExport selection.

    Verified Test Pool members are always excluded from training. Auto-Labeled
    examples count only when they remain in the ``Auto-Labeled`` example state,
    matching :func:`dataset_export_service._select_labels`. Distinct example
    keys keep historical Verified rows from inflating the SME-facing total.
    """
    verified_training_count = 0
    test_pool_count = 0
    auto_labeled_eligible_count = 0
    engine = get_project_engine(project_id, workspace_root)
    if engine is not None:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            active_gid = project.active_guidance_id if project else None
            if active_gid:
                verified_training_count = (
                    session.query(func.count(func.distinct(Label.example_key)))
                    .filter(
                        Label.project_id == project_id,
                        Label.label_status == "verified",
                        Label.guidance_id == active_gid,
                        Label.pool_assignment.is_(None),
                    )
                    .scalar()
                    or 0
                )
                test_pool_count = (
                    session.query(func.count(func.distinct(Label.example_key)))
                    .filter(
                        Label.project_id == project_id,
                        Label.label_status == "verified",
                        Label.guidance_id == active_gid,
                        Label.pool_assignment == "test_pool",
                    )
                    .scalar()
                    or 0
                )
                auto_labeled_eligible_count = (
                    session.query(func.count(func.distinct(Label.example_key)))
                    .join(
                        Example,
                        (Example.project_id == Label.project_id)
                        & (Example.example_key == Label.example_key),
                    )
                    .filter(
                        Label.project_id == project_id,
                        Label.label_status == "auto_labeled",
                        Label.guidance_id == active_gid,
                        Example.state == "Auto-Labeled",
                    )
                    .scalar()
                    or 0
                )

    auto_labeled_included_count = (
        auto_labeled_eligible_count if include_auto_labeled else 0
    )
    return {
        "verified_training_count": verified_training_count,
        "test_pool_count": test_pool_count,
        "auto_labeled_eligible_count": auto_labeled_eligible_count,
        "auto_labeled_included_count": auto_labeled_included_count,
        "excluded_test_pool_count": test_pool_count,
        "excluded_auto_labeled_count": (
            0 if include_auto_labeled else auto_labeled_eligible_count
        ),
        "usable_training_count": (
            verified_training_count + auto_labeled_included_count
        ),
    }


def _check_verified_train_examples(
    *, project_id: str, workspace_root: str
) -> dict[str, Any]:
    """Verify the training-export selection is non-empty.

    Mirrors the training-export filter exactly: Verified labels
    under the project's active guidance with ``pool_assignment IS NULL``
    (Test Pool members are evaluation-only). Global check —
    ``model_config_id`` is null.
    """
    count = 0
    engine = get_project_engine(project_id, workspace_root)
    if engine is not None:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            active_gid = project.active_guidance_id if project else None
            if active_gid:
                count = (
                    session.query(Label)
                    .filter(
                        Label.project_id == project_id,
                        Label.label_status == "verified",
                        Label.guidance_id == active_gid,
                        Label.pool_assignment.is_(None),
                    )
                    .count()
                )
    passed = count > 0
    return {
        "check_name": "verified_train_examples",
        "passed": passed,
        "message": (
            f"{count} Verified training example"
            f"{'s' if count != 1 else ''} available (Test Pool excluded)."
            if passed
            else "No Verified training examples yet. Continue labeling."
        ),
        "model_config_id": None,
    }


# ── Workspace / base-experiment check helpers ────────────────────────────────


async def _check_tao_workspace_reachable(
    *,
    settings: Settings,
    tao_config: TAODeploymentConfig | None,
) -> dict[str, Any]:
    """Validate workspace bootstrap + live reachability.

    Fails with plain-language guidance:
      - when ``TAODeploymentConfig.bootstrap_status != "bootstrapped"``
        OR ``tao_workspace_id`` is unset, the message references the
        bootstrap CLI.
      - when ``GET /workspaces/{id}`` returns an error, the message
        repeats the underlying TAO error.
    """
    if tao_config is None:
        return {
            "check_name": "tao_workspace_reachable",
            "passed": False,
            "message": (
                "TAO workspace is not provisioned. Run "
                "`vlm-feedback-loop tao-bootstrap` (self-service, default) "
                "on the Blueprint host, or use `--admin-managed` if your "
                "deployment is air-gapped (see "
                "docs/tao-ftms-install.md, base-experiment "
                "registration)."
            ),
            "model_config_id": None,
        }

    if tao_config.bootstrap_status != "bootstrapped" or not tao_config.tao_workspace_id:
        return {
            "check_name": "tao_workspace_reachable",
            "passed": False,
            "message": (
                "TAO workspace is not provisioned. Run "
                "`vlm-feedback-loop tao-bootstrap` (self-service, default) "
                "on the Blueprint host, or use `--admin-managed` if your "
                "deployment is air-gapped (see "
                "docs/tao-ftms-install.md, base-experiment "
                "registration)."
            ),
            "model_config_id": None,
        }

    try:
        ws_result = await get_workspace(
            settings, workspace_id=tao_config.tao_workspace_id
        )
    except TaoAuthError as exc:
        # Auth failures (NGC key rejected, endpoint unreachable, /login
        # exchange failed) must render as a structured check failure —
        # not propagate to the UI as a 500. Mirrors the pattern in
        # probe_tao_connection (services/tao_client.py).
        return {
            "check_name": "tao_workspace_reachable",
            "passed": False,
            "message": f"Cannot reach TAO: {exc}",
            "model_config_id": None,
        }
    if not ws_result.success:
        return {
            "check_name": "tao_workspace_reachable",
            "passed": False,
            "message": (
                f"TAO workspace {tao_config.tao_workspace_id!r} is "
                f"unreachable: {ws_result.error}"
            ),
            "model_config_id": None,
        }

    return {
        "check_name": "tao_workspace_reachable",
        "passed": True,
        "message": (f"TAO workspace {tao_config.tao_workspace_id!r} is reachable."),
        "model_config_id": None,
    }


def _check_tao_base_experiment_ready(
    *,
    project_id: str,
    student_base_model_config_ids: list[str],
    workspace_root: str,
) -> list[dict[str, Any]]:
    """Report whether each selected Student base needs first-use provisioning.

    A registered ``pull_complete`` experiment passes as ready. A valid
    project-local ``student_base`` row without one also passes preflight with
    ``provisioning_required=true`` because Start Training now owns the
    idempotent provisioning step. Missing ModelConfig rows remain failures.
    """
    results: list[dict[str, Any]] = []
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        for mcid in student_base_model_config_ids:
            results.append(
                {
                    "check_name": "tao_base_experiment_ready",
                    "passed": False,
                    "message": (
                        f"Project {project_id!r} not found; cannot verify "
                        f"base experiment for {mcid!r}."
                    ),
                    "model_config_id": mcid,
                }
            )
        return results

    row_lookup: dict[str, ModelConfig] = {}
    with Session(engine) as session:
        rows = (
            session.query(ModelConfig)
            .filter(ModelConfig.project_id == project_id)
            .filter(ModelConfig.model_config_id.in_(student_base_model_config_ids))
            .all()
        )
        for row in rows:
            row_lookup[row.model_config_id] = row

    for mcid in student_base_model_config_ids:
        row = row_lookup.get(mcid)
        if row is None:
            results.append(
                {
                    "check_name": "tao_base_experiment_ready",
                    "passed": False,
                    "message": (
                        f"ModelConfig {mcid!r} not found in project {project_id!r}."
                    ),
                    "model_config_id": mcid,
                }
            )
            continue

        exp_id = row.tao_base_experiment_id
        pull_status = row.tao_base_experiment_pull_status

        if not exp_id:
            state_copy = (
                "Automatic provisioning is already running."
                if pull_status == "pulling"
                else "Start Training will provision it automatically."
            )
            results.append(
                {
                    "check_name": "tao_base_experiment_ready",
                    "passed": True,
                    "message": (
                        f"Base experiment for {row.model_name!r} not yet "
                        f"registered in the workspace. {state_copy}"
                    ),
                    "model_config_id": mcid,
                    "provisioning_required": True,
                }
            )
            continue

        if pull_status != "pull_complete":
            results.append(
                {
                    "check_name": "tao_base_experiment_ready",
                    "passed": True,
                    "message": (
                        f"Base experiment for {row.model_name!r} is in "
                        f"state {pull_status!r}; Start Training will ensure "
                        "it reaches pull_complete."
                    ),
                    "model_config_id": mcid,
                    "provisioning_required": True,
                }
            )
            continue

        results.append(
            {
                "check_name": "tao_base_experiment_ready",
                "passed": True,
                "message": (
                    f"Base experiment for {row.model_name!r} is ready (id={exp_id})."
                ),
                "model_config_id": mcid,
            }
        )

    return results
