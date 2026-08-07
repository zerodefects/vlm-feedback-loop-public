#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared mechanics for the operator-run TAO validation drivers.

This module is not a standalone smoke.  It keeps deployment probing, suite
submission, polling, and terminal-state semantics independent from either
fixture driver:

* ``tao_live_smoke.py`` exercises the TAO wiring quickly with three generated
  images.
* ``rps_e2e.py`` exercises training quality and quantization with the complete
  372-image Rock Paper Scissors validation dataset.

Keeping these mechanics here avoids making one executable script an implicit
library of another.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
)
from vlm_feedback_loop.services import (
    tao_client,
    tao_polling_service,
    tao_workspace_service,
    training_suite_service,
)
from vlm_feedback_loop.services.project_service import get_project_engine

MODEL_NAME_2B = COSMOS_REASON2_2B
MODEL_NAME_8B = COSMOS_REASON2_8B
DEFAULT_BASE_EXPERIMENT_ID_2B = "cosmos-reason-2-2b"

logger = logging.getLogger("tao_validation")


def log_banner(message: str) -> None:
    """Print a consistently formatted validation-stage banner."""
    line = "─" * max(30, len(message) + 4)
    print(f"\n{line}\n  {message}\n{line}", flush=True)


@dataclass(frozen=True, slots=True)
class ResolvedWorkspaceState:
    """Non-secret TAO workspace identity loaded from ``deployment.db``."""

    tao_workspace_id: str
    tao_workspace_bucket: str | None
    tao_workspace_s3_endpoint_url_external: str | None
    tao_workspace_s3_endpoint_url_internal: str | None


def resolve_workspace_state(settings: Any) -> ResolvedWorkspaceState:
    """Return the bootstrapped TAO workspace state from ``deployment.db``."""
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        config = session.query(TAODeploymentConfig).first()

    if (
        config is not None
        and config.bootstrap_status == "bootstrapped"
        and config.tao_workspace_id
    ):
        return ResolvedWorkspaceState(
            tao_workspace_id=config.tao_workspace_id,
            tao_workspace_bucket=config.tao_workspace_bucket,
            tao_workspace_s3_endpoint_url_external=(
                config.tao_workspace_s3_endpoint_url_external
            ),
            tao_workspace_s3_endpoint_url_internal=(
                config.tao_workspace_s3_endpoint_url_internal
            ),
        )

    raise SystemExit(
        "✗ TAO workspace not bootstrapped.\n"
        "  Set TAO_WORKSPACE_S3_ACCESS_KEY and "
        "TAO_WORKSPACE_S3_SECRET_KEY in the environment, then run "
        "`vlm-feedback-loop tao-bootstrap --workspace-name <name> "
        "--bucket <bucket>` once "
        "per deployment to populate deployment.db.tao_deployment_configs."
    )


async def probe_and_confirm_workspace(settings: Any) -> str:
    """Probe TAO authentication and confirm the bootstrapped workspace."""
    log_banner("Probe TAO + confirm workspace")
    probe = await tao_client.probe_tao_connection(settings)
    if not probe.get("success"):
        raise SystemExit(f"TAO probe failed: {probe.get('error')}")
    print(f"✓ TAO probe OK (status={probe.get('status_code')})")

    state = resolve_workspace_state(settings)
    workspace = await tao_workspace_service.get_workspace(
        settings, workspace_id=state.tao_workspace_id
    )
    if not workspace.success:
        raise SystemExit(f"get_workspace failed: {workspace.error}")
    detail = workspace.workspace_detail or {}
    print(
        f"✓ Workspace reachable: id={detail.get('id')} "
        f"name={detail.get('name')} cloud={detail.get('cloud_type')}"
    )
    return state.tao_workspace_id


async def find_reason2_2b_base_experiment(
    settings: Any,
    fallback_experiment_id: str,
    *,
    deadline_s: float,
) -> str:
    """Resolve an already-indexed Cosmos Reason2 2B base experiment."""
    log_banner("Discover Cosmos Reason2 base experiment in workspace")
    # Airgapped loading registers the CSV display name (``Cosmos Reason2
    # 2B``), while older admin-managed workspaces may expose one of the
    # model-path slugs.  FTMS performs literal substring matching, so try the
    # current display spelling first and retain the legacy spellings for
    # already-provisioned workspaces.
    experiment = None
    for name_substring in (
        "cosmos reason2 2b",
        "cosmos-reason2-2b",
        "cosmos-reason-2-2b",
    ):
        experiment = await tao_workspace_service.find_base_experiment_by_arch(
            settings,
            network_arch="cosmos-rl",
            name_substring=name_substring,
        )
        if experiment is not None:
            break
    del deadline_s  # FTMS 6.26.3 catalog discovery is a single request.
    if experiment is None:
        print(
            "✗ No Cosmos Reason2 base experiment is indexed on this TAO "
            "workspace.\n"
            "  FTMS 6.26.3 does not expose a client-driven :pull_from_ngc "
            "endpoint — the\n"
            "  workspace's NGC catalog feed must be populated by the TAO "
            "admin before\n"
            "  training can proceed. Ask the admin to pull Cosmos Reason2 "
            "2B (and optionally 8B) into\n"
            "  the workspace, then re-run this smoke with "
            "`--base-experiment-id-2b=<uuid>`.\n"
            "\n"
            "  Once at least one base experiment is indexed, "
            "`jobs:list_base_experiments`\n"
            "  will return a non-empty list and this discovery step will "
            "find the right UUID."
        )
        raise SystemExit(2)

    experiment_id = (
        experiment.get("id")
        or experiment.get("experiment_id")
        or fallback_experiment_id
    )
    print(
        f"✓ Found base experiment: id={experiment_id} "
        f"name={experiment.get('name') or experiment.get('display_name')} "
        f"pull={experiment.get('base_experiment_pull_complete')}"
    )
    return str(experiment_id)


async def submit_training_suite(
    settings: Any,
    assembly: dict[str, Any],
    *,
    base_model: str,
    training_preset: str,
    quantization_schemes: list[str],
    export_field_mode: str,
    idempotency_prefix: str,
) -> dict[str, Any]:
    """Submit a validation training suite through the production service."""
    model_config_id = assembly[f"mc_{base_model}_id"]
    quants_label = (
        ",".join(quantization_schemes) if quantization_schemes else "[baseline only]"
    )
    log_banner(
        f"Submit training suite (base={base_model}, preset={training_preset}, "
        f"quants={quants_label}, export_field_mode={export_field_mode})"
    )
    result = await training_suite_service.create_training_suite(
        assembly["project_id"],
        student_base_model_config_ids=[model_config_id],
        training_preset=training_preset,
        include_auto_labeled=False,
        export_field_mode=export_field_mode,
        quantization_schemes=quantization_schemes,
        idempotency_key=f"{idempotency_prefix}-{uuid.uuid4()}",
        settings=settings,
    )
    if isinstance(result, str):
        raise SystemExit(f"create_training_suite failed: {result}")
    print(
        f"✓ TrainingSuite {result['training_suite_id']} created with "
        f"{len(result['chains'])} chain(s) and "
        f"{sum(len(chain['jobs']) for chain in result['chains'])} TAOJob(s)"
    )
    return result


def compute_training_outcome(
    suite_jobs: list[Any],
    *,
    accept_eval_failure: bool,
) -> str:
    """Return ``succeeded``, ``failed``, or ``pending`` for TAO job state.

    Strict mode requires the last evaluation to succeed and treats any job
    failure as fatal.  Chain-isolation mode tolerates evaluation failures:
    every job must be terminal, training must have succeeded, and at least one
    requested quantization must have succeeded.
    """

    def attribute(job: Any, name: str) -> str:
        return job[name] if isinstance(job, dict) else getattr(job, name)

    statuses_by_action: dict[str, list[str]] = {}
    for job in suite_jobs:
        statuses_by_action.setdefault(attribute(job, "action"), []).append(
            attribute(job, "status")
        )

    if not accept_eval_failure:
        if statuses_by_action.get("evaluate", [None])[-1] == "succeeded":
            return "succeeded"
        if any(attribute(job, "status") == "failed" for job in suite_jobs):
            return "failed"
        return "pending"

    terminal = {"succeeded", "failed", "canceled", "deleted"}
    if not all(attribute(job, "status") in terminal for job in suite_jobs):
        return "pending"
    if "succeeded" not in statuses_by_action.get("train", []):
        return "failed"

    quantize_statuses = statuses_by_action.get("quantize", [])
    if quantize_statuses and "succeeded" not in quantize_statuses:
        return "failed"
    return "succeeded"


async def poll_training_suite(
    settings: Any,
    suite: dict[str, Any],
    *,
    deadline_s: float,
    accept_eval_failure: bool = False,
) -> str:
    """Drive polling ticks until the suite succeeds, fails, or times out."""
    banner_target = (
        "all jobs terminal under chain-isolation"
        if accept_eval_failure
        else "evaluate(baseline)=succeeded"
    )
    log_banner(f"Poll until {banner_target} (deadline={int(deadline_s)}s)")
    start = time.monotonic()
    last_summary = ""
    while True:
        elapsed = time.monotonic() - start
        if elapsed >= deadline_s:
            print(f"⏱ deadline exceeded after {int(elapsed)}s — stopping poll.")
            return "deadline_exceeded"

        try:
            await tao_polling_service.tick(settings)
        except Exception as exc:  # pragma: no cover - live-service defense
            logger.exception("tick crashed: %s", exc)

        engine = get_project_engine(suite["project_id"], settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            training_suite = (
                session.query(TrainingSuite)
                .filter_by(training_suite_id=suite["training_suite_id"])
                .one()
            )
            jobs = (
                session.query(TAOJob)
                .filter_by(project_id=suite["project_id"])
                .order_by(TAOJob.chain_sequence.asc())
                .all()
            )

        summary = f"suite={training_suite.status} " + " ".join(
            f"{job.action}:{job.status}" for job in jobs
        )
        if summary != last_summary:
            print(f"  [{int(elapsed):4d}s] {summary}")
            last_summary = summary

        outcome = compute_training_outcome(
            jobs, accept_eval_failure=accept_eval_failure
        )
        if outcome == "succeeded":
            if accept_eval_failure:
                print(
                    "✓ chain terminal under chain-isolation; train+quantize succeeded"
                )
            else:
                print("✓ evaluate(baseline) = succeeded")
            return "succeeded"
        if outcome == "failed":
            failed = [job for job in jobs if job.status == "failed"]
            print(
                "✗ One or more jobs failed: "
                + ", ".join(
                    f"{job.action}/{job.tao_job_id} → {job.error_ref}" for job in failed
                )
            )
            return "failed"

        await asyncio.sleep(30)


__all__ = [
    "DEFAULT_BASE_EXPERIMENT_ID_2B",
    "MODEL_NAME_2B",
    "MODEL_NAME_8B",
    "ResolvedWorkspaceState",
    "compute_training_outcome",
    "find_reason2_2b_base_experiment",
    "log_banner",
    "poll_training_suite",
    "probe_and_confirm_workspace",
    "resolve_workspace_state",
    "submit_training_suite",
]
