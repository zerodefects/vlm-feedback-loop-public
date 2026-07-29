# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Student NIM deployment lifecycle orchestrator.

Drives the per-Student state machine end to end:

  preflight → docker_run → health_poll → smoke_inference →
  registering_endpoint → evaluation → benchmark → stopping

Local mode: spins a Docker container via ``local_nim_service.deploy_local_nim``
(extended for the Student role with checkpoint mount + served-model-name +
NIM_MODEL_NAME path), polls health, smoke-tests, registers a temporary
NimEndpoint + ModelConfig (both flagged so the Teacher picker / role
filters don't pick them up), runs the evaluation pipeline against the Test
Pool with ``evaluation_source="nim"``, sweeps latency benchmarks at
``STUDENT_LATENCY_TEST_CONCURRENCIES``, then stops the container.

External mode: registers the supplied URL as a permanent NimEndpoint and
runs only the evaluation phase (no docker, no benchmark sweep). The
Compare & Benchmark UI may follow up with a benchmark request later.

State writes are durable BEFORE any SSE event fires; both StudentModel
fields (``serving_status``, ``nim_*``, ``serving_evaluation_run_id``) and
LocalNimDeployment columns are kept in sync so restart recovery and the
Compare & Benchmark UI can read consistent state at any moment.

Per-stage timeouts (``asyncio.wait_for``):
  - preflight + docker_run + health = ``NIM_STARTUP_TIMEOUT_S`` (600s shared)
  - benchmark per concurrency level = ``NIM_BENCHMARK_TIMEOUT_S`` (1200s)

Failure categories (persisted in ``StudentModel.nim_preflight_details``):
  preflight_failed | docker_run_failed | health_timeout | smoke_failed |
  register_endpoint_failed | eval_failed | benchmark_timeout | internal_error
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.services import local_nim_service, nim_client
from vlm_feedback_loop.services.action_requests import generate_action_request
from vlm_feedback_loop.services.benchmark_adapter import (
    BenchmarkAdapter,
    BenchmarkResult,
    select_adapter,
)
from vlm_feedback_loop.services.dataset_export_service import (
    resolve_test_pool_dataset_sha,
)
from vlm_feedback_loop.services.evaluation_service import (
    TERMINAL_STATUSES,
    start_evaluation_run,
)
from vlm_feedback_loop.services.gpu_memory_floor import (
    resolve_gpu_memory_floor_gb,
)
from vlm_feedback_loop.services.inference_contract_resolver import (
    resolve_training_inference_contract,
)
from vlm_feedback_loop.services.nim_endpoint_service import create_nim_endpoint
from vlm_feedback_loop.services.nim_metrics_scraper import scrape_prometheus
from vlm_feedback_loop.services.project_service import (
    get_project_engine,
    project_dir_path,
)
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.services.student_nim_lifecycle")

# Sentinel role assigned to the temporary ModelConfig so the standard
# role filters (teacher / student_base) skip it.
STUDENT_INFERENCE_ROLE = "student_inference"

# In-container path the Student checkpoint is mounted at via -v.
STUDENT_CHECKPOINT_IN_CONTAINER = "/opt/checkpoints/student"

# Stage names used in nim_benchmark_progress SSE events.
STAGE_PREFLIGHT = "preflight"
STAGE_DOCKER_RUN = "docker_run"
STAGE_HEALTH_POLL = "health_poll"
STAGE_SMOKE = "smoke_inference"
STAGE_REGISTER = "registering_endpoint"
STAGE_EVALUATION = "evaluation"
STAGE_BENCHMARK = "benchmark"
STAGE_STOPPING = "stopping"


# ── Helpers ───────────────────────────────────────────────────────────────────


@dataclass
class StudentSnapshot:
    """Plain-scalar snapshot of a StudentModel + its base ModelConfig.

    The orchestrator captures everything once before crossing async
    boundaries so we never carry an ORM row across an await.
    """

    student_model_id: str
    project_id: str
    student_base_model_config_id: str
    nim_checkpoint_ref: str
    quantization_method: str | None
    base_model_name: str
    base_context_window_tokens: int
    base_supports_image_input: bool
    base_local_deploy_metadata: dict[str, Any] | None
    base_structured_generation_support: str
    base_visual_budget_mode: str
    base_visual_budget_support: str
    base_max_images_per_request: int
    dataset_export_ids: list[str]
    guidance_id: str


def _load_student_snapshot(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
) -> StudentSnapshot | None:
    """Load StudentModel + base ModelConfig as plain scalars."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None
    with Session(engine) as session:
        student = session.get(StudentModel, student_model_id)
        if student is None or student.project_id != project_id:
            return None
        base = session.get(ModelConfig, student.student_base_model_config_id)
        if base is None:
            return None
        # dataset_export_ids is JSON-stored; treat as list[str]
        export_ids_raw: Any = student.dataset_export_ids or []
        export_ids: list[str]
        if isinstance(export_ids_raw, dict):
            export_ids = (
                [str(v) for v in cast("dict[str, Any]", export_ids_raw).values()]
                if export_ids_raw
                else []
            )
        elif isinstance(export_ids_raw, list):
            export_ids = [str(v) for v in cast("list[Any]", export_ids_raw)]
        else:
            export_ids = []
        return StudentSnapshot(
            student_model_id=student.student_model_id,
            project_id=student.project_id,
            student_base_model_config_id=student.student_base_model_config_id,
            nim_checkpoint_ref=student.nim_checkpoint_ref or "",
            quantization_method=student.quantization_method,
            base_model_name=base.model_name,
            base_context_window_tokens=base.context_window_tokens,
            base_supports_image_input=base.supports_image_input,
            base_local_deploy_metadata=base.local_deploy_metadata,
            base_structured_generation_support=base.structured_generation_support
            or "unknown",
            base_visual_budget_mode=base.visual_budget_mode or "none",
            base_visual_budget_support=base.visual_budget_support or "unknown",
            base_max_images_per_request=base.max_images_per_request or 5,
            dataset_export_ids=list(export_ids),
            guidance_id=student.guidance_id,
        )


def _resolve_gpu_memory_minimum(
    snapshot: StudentSnapshot,
    settings: Settings,
) -> int:
    """Resolve the GPU memory floor for preflight.

    Thin adapter over the shared policy in ``gpu_memory_floor`` — the
    handoff generator uses the same function, so the two can't drift.
    """
    return resolve_gpu_memory_floor_gb(
        base_model_name=snapshot.base_model_name,
        quantization_method=snapshot.quantization_method,
        base_local_deploy_metadata=snapshot.base_local_deploy_metadata,
        settings=settings,
    )


def _resolve_nim_container_image(
    snapshot: StudentSnapshot,
    explicit: str | None,
) -> str | None:
    """Resolve the NIM container image to pull.

    Explicit caller value wins; otherwise reuse the base ModelConfig's
    seeded ``local_deploy_metadata.nim_container_image`` (the Cosmos
    Reason2 8B / 2B catalog rows ship with it).
    """
    if explicit:
        return explicit
    metadata = snapshot.base_local_deploy_metadata or {}
    image = metadata.get("nim_container_image")
    if isinstance(image, str) and image:
        return image
    return None


def _served_model_name(snapshot: StudentSnapshot) -> str:
    """Stable name clients use as the OpenAI ``model`` parameter."""
    return f"student-{snapshot.student_model_id[:8]}"


def _student_inference_contract(
    snapshot: StudentSnapshot,
    workspace_root: str,
) -> dict[str, Any]:
    """Derive the Student's Inference Contract from its training DatasetExport.

    A Student's ``output_field_mode`` and ``icl_field_mode``
    both default to the training DatasetExport's ``export_field_mode``.

    Delegates to the canonical ``resolve_training_inference_contract``
    helper so registration, deployment-handoff, and lifecycle all share
    one derivation path.
    """
    engine = get_project_engine(snapshot.project_id, workspace_root)
    if engine is None or not snapshot.dataset_export_ids:
        return {
            "output_field_mode": "all",
            "icl_field_mode": "all",
            "icl_max_examples": None,
        }
    with Session(engine) as session:
        return resolve_training_inference_contract(
            session, list(snapshot.dataset_export_ids)
        )


# ── State write helpers ──────────────────────────────────────────────────────


def _write_student_state(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    *,
    fields: dict[str, Any],
    _attempts: int = 4,
    _delay_s: float = 1.5,
) -> None:
    """Apply a partial update to a StudentModel row (commits).

    Retries SQLite lock contention: WAL's ``busy_timeout`` does not cover
    snapshot-upgrade conflicts (read-then-write vs a concurrent writer
    fails instantly with ``database is locked``), and this helper carries
    EVERY lifecycle state write — a preflight persist died on exactly
    that race under FTMS-poller write load, crashing the whole lifecycle
    and stranding ``serving_status="pending"`` (observed
    live 2026-07-15). Each retry opens a fresh session so it re-reads
    current state. The bounded ``time.sleep`` briefly blocks the event
    loop when called from async code — same trade-off as the surrounding
    sync-DB-in-async pattern, worst case ~4.5 s.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return
    for attempt in range(1, _attempts + 1):
        try:
            with Session(engine) as session:
                student = session.get(StudentModel, student_model_id)
                if student is None:
                    return
                for key, value in fields.items():
                    setattr(student, key, value)
                session.commit()
            return
        except OperationalError as exc:
            if "database is locked" not in str(exc) or attempt == _attempts:
                raise
            logger.warning(
                "SQLite lock contention writing StudentModel %s state "
                "(attempt %d/%d) — retrying",
                student_model_id,
                attempt,
                _attempts,
            )
            time.sleep(_delay_s)


def _promote_quality_from_nim_eval(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    *,
    nim_run_id: str,
) -> bool:
    """Promote ``quality_status`` to ``"validated"`` when a NIM-source eval succeeds.

    TAO `evaluate` is the preferred
    quality gate. The NIM-source eval that already runs for serving validation
    MAY also satisfy the quality gate, but **only as a narrow,
    signature-gated fallback** — not a generic "TAO failed → use NIM" rescue.

    Gate logic:

    1. ``quality_status == "validated"`` → no-op. TAO eval already won;
       preserve its ``quality_evaluation_run_id`` audit pointer.
    2. ``quality_status == "pending"`` → promote. No prior TAO eval has
       terminated (cold-start path or operator-driven NIM-only validation) —
       the registry default state.
    3. ``quality_status == "failed"`` → promote **only if** the prior failed
       TAO eval's failure signature matches a known upstream model-loader gap
       (see ``services.tao_failure_classifier.MODEL_LOADER_FAILURE_PATTERNS``).
       Other failure classes (dataset shape, OOM, transient infra) leave
       ``quality_status="failed"`` — NIM eval is not a generic rescue.

    Returns ``True`` if the promotion fired, ``False`` otherwise. Logs the
    decision at ``info`` level so operators can audit which path was taken.
    """
    from vlm_feedback_loop.services import tao_failure_classifier

    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return False
    with Session(engine) as session:
        student = session.get(StudentModel, student_model_id)
        if student is None:
            return False
        if student.quality_status == "validated":
            return False

        prior_status = student.quality_status

        # Conservative gate when prior TAO eval failed — only
        # promote on a known upstream-loader signature match. Other TAO
        # failures stay as ``quality_status="failed"`` so a bad-data /
        # OOM / config-error TAO failure cannot be silently papered over
        # by a NIM eval that just happened to complete.
        if prior_status == "failed":
            train_tao_job_id = student.tao_job_id
            if not train_tao_job_id:
                logger.info(
                    "NIM eval succeeded for Student %s but tao_job_id is unset; "
                    "cannot classify prior failure — leaving quality_status=failed.",
                    student_model_id,
                )
                return False
            matched, signature = tao_failure_classifier.matches_known_loader_gap(
                session,
                project_id=project_id,
                student_train_tao_job_id=train_tao_job_id,
            )
            if not matched:
                logger.info(
                    "NIM eval succeeded for Student %s (run=%s) but the prior "
                    "TAO failure does NOT match a known upstream-loader gap; "
                    "leaving quality_status=failed (conservative gate).",
                    student_model_id,
                    nim_run_id,
                )
                return False
            logger.info(
                "Prior TAO failure for Student %s matched upstream-loader "
                "signature %r; promoting via NIM eval (upstream-loader fallback).",
                student_model_id,
                signature,
            )

        student.quality_status = "validated"
        student.quality_evaluation_run_id = nim_run_id
        session.commit()
    logger.info(
        "Promoted quality_status %s -> validated via NIM eval (run=%s) for Student %s",
        prior_status,
        nim_run_id,
        student_model_id,
    )
    return True


def _promote_quality_to_partial(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    *,
    nim_run_id: str,
    parseable_rate: float,
    threshold: float,
) -> bool:
    """Promote ``quality_status`` to ``"partial"`` when a NIM eval finishes ``incomplete``
    with parseable rate ≥ threshold.

    ``partial`` is a third quality_status value for NIM
    evals that finished ``incomplete`` but produced
    parseable output on at least ``STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD``
    of examples (default 0.90). ``partial`` is **informational, not
    gate-passing** — the deployment_handoff STILL requires
    ``quality_status="validated"``. This helper exists to give operators a
    visible signal that "the model serves and is mostly correct" without
    flipping the production-handoff bar.

    Audit invariants (mandatory):

    * ``validated → partial`` is **never** a legal transition. If the
      paired Student already reached ``validated`` (typically via TAO
      eval rescoring), the helper is a no-op and preserves the existing
      ``quality_evaluation_run_id`` audit pointer.
    * ``partial → partial`` is idempotent (no-op on re-call).
    * ``pending → partial`` and ``failed → partial`` are the only valid
      transitions; both rewrite ``quality_evaluation_run_id`` to the
      passed-in ``nim_run_id``.

    Returns ``True`` if the promotion fired, ``False`` otherwise. Logs the
    decision at ``info`` level so operators can audit which path was taken.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return False
    with Session(engine) as session:
        student = session.get(StudentModel, student_model_id)
        if student is None:
            return False
        if student.quality_status == "validated":
            # Audit invariant: validated is never demoted to partial.
            return False
        if student.quality_status == "partial":
            # Idempotent — already in the partial state from a prior eval.
            return False

        prior_status = student.quality_status
        student.quality_status = "partial"
        student.quality_evaluation_run_id = nim_run_id
        session.commit()
    logger.info(
        "Promoted quality_status %s -> partial via NIM eval (run=%s, "
        "parseable_rate=%.3f, threshold=%.3f) for Student %s",
        prior_status,
        nim_run_id,
        parseable_rate,
        threshold,
        student_model_id,
    )
    return True


def _compute_parseable_rate(run_record: RunRecord | None) -> float:
    """Return the fraction of examples that produced parseable output.

    Used by the partial-quality gate to decide between ``validated`` /
    ``partial`` / no-promotion. Returns ``0.0`` when total is zero or
    counters are unset, which deterministically misses the partial
    threshold so a degenerate run can never land in ``partial``.
    """
    if run_record is None:
        return 0.0
    total = run_record.examples_total or 0
    succeeded = run_record.examples_succeeded or 0
    if total <= 0:
        return 0.0
    return succeeded / total


# Serving validation accepts BOTH terminal states — the model
# deployed, served HTTP responses, and produced parseable JSON for at
# least some invocations. The strict completed-only gate applies to
# QUALITY promotion, not serving.
_SERVING_ACCEPTABLE_STATUSES = ("completed", "incomplete")


def _apply_serving_quality_gate(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    settings: Settings,
    *,
    run_id: str,
    run_record: RunRecord,
) -> None:
    """Unified serving/quality gate — the ONE implementation for
    external and local modes (they drifted apart when each carried a copy:
    the external copy wrongly failed serving on ``incomplete`` runs below
    the partial threshold).

    Caller must have verified ``run_record.status`` is in
    ``_SERVING_ACCEPTABLE_STATUSES``. Serving validates unconditionally;
    quality routes by run status: full promotion on ``completed``, ``partial``
    on ``incomplete`` with ``parseable_rate >= threshold``, otherwise left
    at its prior value (typically ``pending`` or ``failed``).
    """
    _write_student_state(
        project_id,
        student_model_id,
        workspace_root,
        fields={
            "serving_status": "validated",
            "serving_evaluation_run_id": run_id,
        },
    )
    parseable_rate = _compute_parseable_rate(run_record)
    partial_threshold = settings.STUDENT_QUALITY_PARTIAL_PARSEABLE_THRESHOLD
    if run_record.status == "completed":
        _promote_quality_from_nim_eval(
            project_id,
            student_model_id,
            workspace_root,
            nim_run_id=run_id,
        )
    elif parseable_rate >= partial_threshold:
        _promote_quality_to_partial(
            project_id,
            student_model_id,
            workspace_root,
            nim_run_id=run_id,
            parseable_rate=parseable_rate,
            threshold=partial_threshold,
        )
    # else: incomplete below the threshold — serving stays validated;
    # quality keeps its prior value.


def _persist_preflight(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    *,
    passed: bool,
    checks: list[Any],
    failure_stage: str | None = None,
) -> None:
    """Persist preflight status + diagnostics on the StudentModel."""
    details: dict[str, Any] = {
        "checks": [
            {
                "check_name": c.check_name,
                "passed": c.passed,
                "diagnostic": c.diagnostic,
            }
            for c in checks
        ],
    }
    if failure_stage:
        details["failure_stage"] = failure_stage
    _write_student_state(
        project_id,
        student_model_id,
        workspace_root,
        fields={
            "nim_preflight_status": "passed" if passed else "failed",
            "nim_preflight_details": details,
            "nim_preflight_at": utc_now(),
        },
    )


def _record_failure(
    project_id: str,
    student_model_id: str,
    workspace_root: str,
    *,
    failure_stage: str,
    error_detail: str | None = None,
) -> None:
    """Persist serving_status="failed" with failure_stage detail.

    Build a fresh dict for ``nim_preflight_details`` (rather than mutating
    in place) so SQLAlchemy's dirty-tracker fires for the JSON column.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return
    with Session(engine) as session:
        student = session.get(StudentModel, student_model_id)
        if student is None:
            return
        student.serving_status = "failed"
        details = dict(student.nim_preflight_details or {})
        details["failure_stage"] = failure_stage
        if error_detail is not None:
            details["error_detail"] = error_detail
        student.nim_preflight_details = details
        session.commit()


# ── SSE emission helpers ─────────────────────────────────────────────────────


async def _emit_progress(
    project_id: str,
    student_model_id: str,
    stage: str,
    started_at: float,
    *,
    concurrency: int | None = None,
    deployment_id: str | None = None,
) -> None:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    payload: dict[str, Any] = {
        "student_model_id": student_model_id,
        "stage": stage,
        "elapsed_ms": elapsed_ms,
    }
    if concurrency is not None:
        payload["concurrency"] = concurrency
    if deployment_id is not None:
        payload["deployment_id"] = deployment_id
    await sse_manager.emit(project_id, "nim_benchmark_progress", payload)


async def _emit_completed(
    project_id: str,
    student_model_id: str,
    started_at: float,
    *,
    evaluation_run_id: str,
    rescored_metrics: dict[str, Any] | None,
    benchmarks: list[dict[str, Any]],
    skipped_concurrencies: list[int],
) -> None:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    metrics_dict: dict[str, Any] = rescored_metrics or {}
    overall_raw: Any = metrics_dict.get("overall") or {}
    overall: dict[str, Any] = (
        cast("dict[str, Any]", overall_raw) if isinstance(overall_raw, dict) else {}
    )
    # per_field_match_rates lives inside the overall bucket as
    # {field_name: rate} (evaluation_service._agg_to_dict). An earlier read
    # of a top-level "per_field" map with "match_rate" sub-keys matched no
    # serializer shape and always emitted {}.
    per_field_raw: Any = overall.get("per_field_match_rates") or {}
    per_field: dict[str, Any] = (
        cast("dict[str, Any]", per_field_raw) if isinstance(per_field_raw, dict) else {}
    )
    await sse_manager.emit(
        project_id,
        "nim_benchmark_completed",
        {
            "student_model_id": student_model_id,
            "evaluation_run_id": evaluation_run_id,
            "exact_match": overall.get("exact_match_rate", 0.0),
            "per_field_match_rates": {
                name: rate if isinstance(rate, (int, float)) else 0.0
                for name, rate in per_field.items()
            },
            "benchmarks": benchmarks,
            "skipped_concurrencies": skipped_concurrencies,
            "serving_status": "validated",
            "elapsed_ms": elapsed_ms,
        },
    )


async def _emit_failed(
    project_id: str,
    student_model_id: str,
    started_at: float,
    *,
    failure_stage: str,
    error_ref: str | None = None,
    deployment_id: str | None = None,
) -> None:
    elapsed_ms = int((time.monotonic() - started_at) * 1000)
    await sse_manager.emit(
        project_id,
        "run_failed",
        {
            "student_model_id": student_model_id,
            "run_type": "student_nim_deploy",
            "failure_stage": failure_stage,
            "error_ref": error_ref,
            "deployment_id": deployment_id,
            "elapsed_ms": elapsed_ms,
        },
    )


# ── Endpoint + ModelConfig provisioning ───────────────────────────────────────


async def _register_temp_endpoint(
    snapshot: StudentSnapshot,
    workspace_root: str,
    settings: Settings,
    *,
    base_url: str,
    mode: str,
    auth_mode: str,
    local_nim_deployment_id: str | None,
) -> tuple[str, str]:
    """Create the temporary NimEndpoint + ModelConfig pair.

    Returns ``(endpoint_id, model_config_id)``. Both rows are flagged so
    the Teacher / student_base role filters skip them:

      - NimEndpoint: ``is_enabled=False`` post-creation, ``source_kind=
        "auto_registered_student"``.
      - ModelConfig: ``eligible_roles=[STUDENT_INFERENCE_ROLE]`` (a
        sentinel value not in the standard set).
    """
    endpoint_mode = "local_system_managed" if mode == "local" else "self_hosted"
    source_kind = (
        "auto_registered_student" if mode == "local" else "user_configured_student"
    )

    endpoint = await create_nim_endpoint(
        project_id=snapshot.project_id,
        data={
            "display_name": f"Student {snapshot.student_model_id[:8]} ({base_url})",
            "endpoint_mode": endpoint_mode,
            "base_url": base_url,
            "auth_mode": auth_mode,
            "source_kind": source_kind,
        },
        workspace_root=workspace_root,
        settings=settings,
    )
    if endpoint is None:
        raise RuntimeError(
            f"Failed to create temporary NimEndpoint for Student "
            f"{snapshot.student_model_id}"
        )

    engine = get_project_engine(snapshot.project_id, workspace_root)
    if engine is None:
        raise RuntimeError("Project DB unavailable")

    model_config_id = generate_uuid4()
    served_name = _served_model_name(snapshot)

    with Session(engine) as session:
        # Mark the endpoint as not-enabled and link it to the deployment.
        ep_row = session.get(NimEndpoint, endpoint.endpoint_id)
        if ep_row is not None:
            ep_row.is_enabled = False
            if local_nim_deployment_id:
                ep_row.local_nim_deployment_id = local_nim_deployment_id

        # Create the temp ModelConfig. Capability inheritance from the
        # base ModelConfig — the deployed Student NIM
        # serves the SAME bundled vLLM as the base, so probe-status from
        # the base is reliable. Inheriting avoids the schema-invalid
        # storm that surfaces when a fine-tune's reasoning model isn't
        # constrained by ``response_format=json_schema``. Thinking is
        # always ``"none"`` on a Student (base reasoning behavior is
        # baked-in via the fine-tune; toggle is moot).
        temp_mc = ModelConfig(
            model_config_id=model_config_id,
            project_id=snapshot.project_id,
            endpoint_id=endpoint.endpoint_id,
            model_name=served_name,
            context_window_tokens=snapshot.base_context_window_tokens,
            eligible_roles=[STUDENT_INFERENCE_ROLE],
            supports_image_input=snapshot.base_supports_image_input,
            structured_generation_support=snapshot.base_structured_generation_support,
            thinking_toggle_mode="none",
            thinking_toggle_support="unsupported",
            visual_budget_mode=snapshot.base_visual_budget_mode,
            visual_budget_support=snapshot.base_visual_budget_support,
            max_images_per_request=snapshot.base_max_images_per_request,
            image_cap_support="unknown",
            model_quantization=snapshot.quantization_method,
        )
        session.add(temp_mc)
        session.commit()

    return endpoint.endpoint_id, model_config_id


# ── Smoke + health helpers ────────────────────────────────────────────────────


async def _stop_deployment_quiet(
    deployment_id: str | None,
    project_id: str,
    workspace_root: str,
) -> None:
    """Best-effort
    ``stop_local_nim`` for failure-path cleanup. Each ``return`` in
    stages 3-6 of ``run_student_deployment_lifecycle`` would otherwise
    leak the running NIM container, blocking the next variant's deploy
    on a single-GPU host. The catch-all ``except`` block at the end of
    the lifecycle handles unexpected exceptions but NOT the explicit
    ``return`` paths in each stage's failure handler. This helper is
    called inline before each ``return`` so the container is stopped
    regardless of which stage failed.

    Errors during cleanup are logged but never raised — they MUST NOT
    mask the original failure.
    """
    if not deployment_id:
        return
    try:
        await local_nim_service.stop_local_nim(
            deployment_id=deployment_id,
            project_id=project_id,
            workspace_root=workspace_root,
        )
    except Exception:
        logger.exception(
            "stop_local_nim raised during failure cleanup for %s; "
            "container may be orphaned and block the next deploy",
            deployment_id,
        )


async def _restore_displaced_deployment(
    *,
    workspace_root: str,
    settings: Settings,
    displaced: LocalNimDeployment,
) -> None:
    """Best-effort re-deploy a Teacher / embedding NIM that the Student
    lifecycle displaced in step 0. Reuses the same role, model_config,
    container image, and gpu_assignment as the displaced row. The
    fresh deployment gets a NEW ``local_nim_deployment_id`` (the
    displaced row stays in ``status="stopped"`` with its
    ``displaced_by_deployment_id`` audit link intact); the row is
    re-created so the container lifecycle is clean.

    Reads ``gpu_memory_minimum_gb`` per role:
    * ``teacher`` — from the bound ``ModelConfig.local_deploy_metadata``.
    * ``embedding`` — from the deployment-scoped
      ``EmbeddingDeploymentConfig`` (matches the router's
      ``local_nim_service.resolve_deploy_params`` logic).

    Health-polls the restored deployment up to
    ``NIM_STARTUP_TIMEOUT_S``; on timeout, logs a warning but does NOT
    raise. The caller (lifecycle stopping stage) catches any exception
    and continues — the benchmark already succeeded.
    """
    from pathlib import Path

    from sqlalchemy import select

    from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
    from vlm_feedback_loop.db.engine import init_deployment_db

    role = str(displaced.role)
    project_id = str(displaced.project_id)
    model_config_id = str(displaced.model_config_id)
    nim_container_image = str(displaced.nim_container_image)
    gpu_assignment = str(displaced.gpu_assignment)
    preferred_port = int(displaced.host_port)

    # Resolve gpu_memory_minimum_gb per role.
    gpu_min_gb = 0
    if role == "teacher":
        engine = get_project_engine(project_id, workspace_root)
        if engine is not None:
            with Session(engine) as session:
                mc = session.get(ModelConfig, model_config_id)
                if mc is not None and mc.local_deploy_metadata:
                    gpu_min_gb = int(
                        mc.local_deploy_metadata.get("nim_gpu_memory_minimum_gb", 0)
                    )
    elif role == "embedding":
        deploy_engine = init_deployment_db(Path(workspace_root))
        with Session(deploy_engine) as session:
            stmt = select(EmbeddingDeploymentConfig).limit(1)
            edc = session.execute(stmt).scalar_one_or_none()
            if edc is not None:
                gpu_min_gb = int(edc.gpu_memory_minimum_gb)
    # role == "student" cannot land here (Student deploys are never
    # auto-displaced) — defensive default keeps gpu_min_gb=0.

    logger.info(
        "auto-restore: re-deploying displaced %s/%s (role=%s, gpu=%s)",
        project_id,
        displaced.local_nim_deployment_id,
        role,
        gpu_assignment,
    )

    result = await local_nim_service.deploy_local_nim(
        project_id=project_id,
        model_config_id=model_config_id,
        role=role,
        nim_container_image=nim_container_image,
        gpu_assignment=gpu_assignment,
        gpu_memory_minimum_gb=gpu_min_gb,
        preferred_port=preferred_port,
        settings=settings,
        workspace_root=workspace_root,
        # replace_resident=False: the GPU is now free (Student stopped
        # in stage 8). If something else grabbed it in the meantime,
        # the standard 409 path applies — auto-restore stays best-effort.
        replace_resident=False,
    )
    new_deployment = result["deployment"]
    new_deployment_id = str(new_deployment.local_nim_deployment_id)

    # Health-poll the restoration. Use the same budget as the original
    # deploy; failure here surfaces as a warning, not a hard error.
    ok = await _wait_for_deployment_running(
        project_id=project_id,
        deployment_id=new_deployment_id,
        workspace_root=workspace_root,
        deadline_s=settings.NIM_STARTUP_TIMEOUT_S,
    )
    if not ok:
        logger.warning(
            "auto-restore: %s/%s did not reach status=running within "
            "NIM_STARTUP_TIMEOUT_S=%ss; operator can redeploy via NIM Configuration",
            project_id,
            new_deployment_id,
            settings.NIM_STARTUP_TIMEOUT_S,
        )
    else:
        logger.info(
            "auto-restore: %s/%s back to status=running on %s",
            project_id,
            new_deployment_id,
            gpu_assignment,
        )


async def _wait_for_deployment_running(
    project_id: str,
    deployment_id: str,
    workspace_root: str,
    deadline_s: float,
) -> bool:
    """Poll ``LocalNimDeployment.status`` until ``running`` or the deadline.

    The actual Docker health-poll runs in
    ``local_nim_service._poll_health`` (kicked off by ``deploy_local_nim``).
    We just wait for that background task to flip the persisted status.
    """
    deadline = asyncio.get_event_loop().time() + deadline_s
    poll_interval = 5.0
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return False
    while asyncio.get_event_loop().time() < deadline:
        with Session(engine) as session:
            dep = session.get(LocalNimDeployment, deployment_id)
            if dep is None:
                return False
            if dep.status == "running":
                return True
            if dep.status == "failed":
                return False
        await asyncio.sleep(poll_interval)
    return False


async def _smoke_inference(base_url: str, served_model: str) -> bool:
    """Single chat-completions request as a smoke test.

    Returns True when the endpoint answers with a parseable OpenAI-style
    completion. The model output content does not matter — we just verify
    the endpoint speaks the OpenAI-compatible API.
    """
    result = await nim_client.chat_completions(
        base_url,
        {},  # local Student NIM: no auth
        served_model,
        [{"role": "user", "content": 'Say {"ok": true} and nothing else.'}],
        deadline_s=60.0,
        max_retries=1,
        max_tokens=16,
        temperature=0.0,
    )
    if not result.success:
        logger.warning("Student NIM smoke test failed: %s", result.error)
    return result.success


# ── Evaluation phase ─────────────────────────────────────────────────────────


async def _wait_for_run_terminal(
    project_id: str,
    run_id: str,
    workspace_root: str,
    deadline_s: float,
) -> RunRecord | None:
    """Poll the eval RunRecord until it reaches a terminal status."""
    deadline = asyncio.get_event_loop().time() + deadline_s
    poll_interval = 2.0
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return None
    while asyncio.get_event_loop().time() < deadline:
        with Session(engine) as session:
            run = session.get(RunRecord, run_id)
            if run is None:
                return None
            if run.status in TERMINAL_STATUSES:
                session.expunge(run)
                return run
        await asyncio.sleep(poll_interval)
    return None


async def _run_evaluation_phase(
    snapshot: StudentSnapshot,
    workspace_root: str,
    settings: Settings,
    target_model_config_id: str,
) -> tuple[str | None, RunRecord | None, str | None]:
    """Start a NIM-source evaluation against the Student endpoint and wait.

    Returns ``(run_id, run_record, error_detail)``. ``run_record`` is None
    if the run never terminalised within the deadline.

    Also forwards benchmark provenance (NIM profile metadata,
    GPU type/count, Test Pool dataset SHA-256) by re-reading the live
    StudentModel row at eval-phase entry. Profile/GPU fields are written
    between docker-run and eval, so the original ``StudentSnapshot`` taken
    at lifecycle start does not carry them. Re-reading is a single short
    transaction; the rest of the eval pipeline still operates on plain
    scalars.
    """
    contract = _student_inference_contract(snapshot, workspace_root)

    # ── Refresh provenance from live StudentModel + Test Pool DatasetExport ─
    nim_profile_requested: str | None = None
    nim_profile_selected: str | None = None
    nim_profile_metadata: dict[str, Any] | None = None
    gpu_type: str | None = None
    gpu_count: int | None = None
    dataset_manifest_sha256: str | None = None

    engine = get_project_engine(snapshot.project_id, workspace_root)
    if engine is not None:
        with Session(engine) as session:
            student = session.get(StudentModel, snapshot.student_model_id)
            if student is not None:
                nim_profile_requested = student.nim_model_profile_requested
                nim_profile_selected = student.nim_model_profile_selected
                nim_profile_metadata = student.nim_profile_metadata
                gpu_type = student.gpu_type
                gpu_count = student.gpu_count
            # Test Pool DatasetExport (dataset_intent="testing") archive SHA-256.
            dataset_manifest_sha256 = resolve_test_pool_dataset_sha(
                session, snapshot.dataset_export_ids
            )

    response = await start_evaluation_run(
        snapshot.project_id,
        icl_mode="disabled",  # the Student baseline runs without ICL
        # Students are evaluated (and must be served) at native
        # image resolution — §9.3 training consumes native-size images,
        # so inheriting the project's Teacher-oriented visual budget
        # (default high_detail: shortest_edge 1568) upscales the inputs
        # far off the training distribution. Measured on the same
        # checkpoint / same 120 keys: EM 0.95 native vs 0.367 upscaled.
        visual_budget_preset_key="native",
        target_model_config_id=target_model_config_id,
        target_inference_contract=contract,
        settings=settings,
        # Benchmark provenance
        student_model_config_id=snapshot.student_model_id,
        nim_model_profile_requested=nim_profile_requested,
        nim_model_profile_selected=nim_profile_selected,
        nim_profile_metadata=nim_profile_metadata,
        quantization_method=snapshot.quantization_method,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    if isinstance(response, str):
        return None, None, response

    run_id = response["run_id"]
    run = await _wait_for_run_terminal(
        snapshot.project_id,
        run_id,
        workspace_root,
        deadline_s=float(settings.NIM_BENCHMARK_TIMEOUT_S),
    )
    return run_id, run, None


# ── Benchmark phase ──────────────────────────────────────────────────────────


async def _run_benchmark_sweep(
    project_id: str,
    snapshot: StudentSnapshot,
    workspace_root: str,
    settings: Settings,
    *,
    base_url: str,
    served_model: str,
    started_at: float,
    deployment_id: str | None,
    adapter: BenchmarkAdapter | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Run the latency sweep at STUDENT_LATENCY_TEST_CONCURRENCIES.

    Returns ``(benchmarks, skipped_concurrencies)``. On per-concurrency
    timeout the value is appended to ``skipped_concurrencies`` and the
    sweep breaks immediately (the container will be stopped by the
    caller; the queue continues with the next Student variant).
    """
    bench_adapter = adapter or select_adapter()
    project_dir = str(project_dir_path(workspace_root, project_id))
    benchmarks: list[dict[str, Any]] = []
    skipped: list[int] = []

    for concurrency in settings.STUDENT_LATENCY_TEST_CONCURRENCIES:
        await _emit_progress(
            project_id,
            snapshot.student_model_id,
            STAGE_BENCHMARK,
            started_at,
            concurrency=concurrency,
            deployment_id=deployment_id,
        )
        try:
            result: BenchmarkResult = await asyncio.wait_for(
                bench_adapter.run(
                    base_url=base_url,
                    model=served_model,
                    concurrency=concurrency,
                    project_dir=project_dir,
                    student_model_id=snapshot.student_model_id,
                    deadline_s=float(settings.NIM_BENCHMARK_TIMEOUT_S),
                ),
                timeout=float(settings.NIM_BENCHMARK_TIMEOUT_S),
            )
        except TimeoutError:
            logger.warning(
                "Benchmark timeout at concurrency=%d for Student %s",
                concurrency,
                snapshot.student_model_id,
            )
            skipped.append(concurrency)
            break

        if result.failed or result.request_count == 0:
            # The driver produced no real measurement (process failure /
            # missing output / every request errored). Guarding on
            # request_count too catches any driver that forgets to set the
            # failed flag. Record it as skipped rather than persisting a fake
            # zero-latency row the ServingMatrix would render as valid.
            logger.warning(
                "Benchmark driver failed at concurrency=%d for Student %s — "
                "recording as skipped",
                concurrency,
                snapshot.student_model_id,
            )
            skipped.append(concurrency)
            continue

        # Scrape Prometheus metrics post-run; merge into the BenchmarkResult.
        try:
            result.prometheus = await scrape_prometheus(base_url)
        except Exception as exc:
            logger.debug("Prometheus scrape skipped: %s", exc)
            result.prometheus = {}

        benchmarks.append(result.to_json())

    return benchmarks, skipped


def _persist_benchmarks(
    project_id: str,
    workspace_root: str,
    *,
    run_id: str,
    benchmarks: list[dict[str, Any]],
) -> None:
    """Attach the sweep results to the serving evaluation run's metrics.

    ``metrics["benchmarks"]`` is the durable home of measured serving
    data: ``GET /evaluation_runs/{id}`` exposes it, and the Compare page's
    ServingMatrix renders it after the post-benchmark refetch.
    """
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return
    with Session(engine) as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            logger.warning(
                "benchmark persistence: run %s not found in project %s",
                run_id,
                project_id,
            )
            return
        # Reassign (not mutate) so the JSON column change is tracked.
        run.metrics = {**(run.metrics or {}), "benchmarks": benchmarks}
        session.commit()


# ── Main lifecycle entry point ───────────────────────────────────────────────


async def _restore_all_displaced(
    displaced: list[LocalNimDeployment],
    *,
    workspace_root: str,
    settings: Settings,
) -> None:
    """Best-effort re-deploy every resident the Student displaced.

    Runs on ALL lifecycle exits (success or failure) via the lifecycle's
    ``finally`` — on a single-GPU host the Student stopped the resident Teacher
    to take the GPU, and if the deploy or benchmark then failed the SME would
    otherwise be left with no Teacher. Each failure surfaces as a warning only.
    """
    for row in displaced:
        try:
            await _restore_displaced_deployment(
                workspace_root=workspace_root,
                settings=settings,
                displaced=row,
            )
        except Exception as exc:
            logger.warning(
                "auto-restore of displaced %s/%s failed (%s: %s)",
                row.project_id,
                row.local_nim_deployment_id,
                type(exc).__name__,
                str(exc) or "(no message)",
            )


async def run_student_deployment_lifecycle(
    project_id: str,
    student_model_id: str,
    *,
    mode: str,
    nim_endpoint_url: str | None,
    nim_container_image: str | None,
    nim_release_version: str | None,
    gpu_assignment: str | None,
    auth_mode: str,
    settings: Settings,
    workspace_root: str,
    benchmark_adapter: BenchmarkAdapter | None = None,
) -> None:
    """Top-level lifecycle coroutine.

    Always returns None. Side effects: durable state writes on
    StudentModel + LocalNimDeployment + RunRecord + Operation Records,
    plus SSE emissions throughout the lifecycle.
    """
    started_at = time.monotonic()

    async def _fail(
        stage: str,
        detail: str | None = None,
        *,
        ref: str | None = None,
        deployment_id: str | None = None,
    ) -> None:
        """Persist the serving failure and emit run_failed for this lifecycle.

        ``ref`` defaults to ``detail``; pass it explicitly only when the SSE
        wants a short token while the DB keeps the full diagnostic.
        """
        _record_failure(
            project_id,
            student_model_id,
            workspace_root,
            failure_stage=stage,
            error_detail=detail,
        )
        await _emit_failed(
            project_id,
            student_model_id,
            started_at,
            failure_stage=stage,
            error_ref=ref if ref is not None else detail,
            deployment_id=deployment_id,
        )

    snapshot = _load_student_snapshot(project_id, student_model_id, workspace_root)
    if snapshot is None:
        await _emit_failed(
            project_id,
            student_model_id,
            started_at,
            failure_stage="internal_error",
            error_ref="student_model_not_found_post_dispatch",
        )
        return

    if nim_release_version is not None:
        _write_student_state(
            project_id,
            student_model_id,
            workspace_root,
            fields={"nim_vlm_release_version": nim_release_version},
        )

    # ── External mode: skip docker, register endpoint, run eval ────────
    if mode == "external" and nim_endpoint_url:
        try:
            await _emit_progress(
                project_id, student_model_id, STAGE_REGISTER, started_at
            )
            _ep_id, mc_id = await _register_temp_endpoint(
                snapshot,
                workspace_root,
                settings,
                base_url=nim_endpoint_url,
                mode="external",
                auth_mode=auth_mode,
                local_nim_deployment_id=None,
            )
            _write_student_state(
                project_id,
                student_model_id,
                workspace_root,
                fields={"nim_endpoint_url": nim_endpoint_url},
            )

            await _emit_progress(
                project_id, student_model_id, STAGE_EVALUATION, started_at
            )
            run_id, run_record, err = await _run_evaluation_phase(
                snapshot, workspace_root, settings, mc_id
            )
            if err is not None or run_id is None or run_record is None:
                await _fail("eval_failed", err)
                return

            # Unified serving/quality gate (shared with local mode): serving
            # accepts (completed, incomplete); quality routes to
            # validated / partial / no-promotion. See
            # _apply_serving_quality_gate + _promote_quality_to_partial
            # for the audit invariants.
            if run_record.status not in _SERVING_ACCEPTABLE_STATUSES:
                await _fail("eval_failed", f"run_status={run_record.status}")
                return
            _apply_serving_quality_gate(
                project_id,
                student_model_id,
                workspace_root,
                settings,
                run_id=run_id,
                run_record=run_record,
            )

            await _emit_completed(
                project_id,
                student_model_id,
                started_at,
                evaluation_run_id=run_id,
                rescored_metrics=run_record.rescored_metrics,
                benchmarks=[],
                skipped_concurrencies=[],
            )
            return
        except Exception as exc:
            logger.exception(
                "Unhandled external-mode failure for Student %s",
                student_model_id,
            )
            await _fail("internal_error", str(exc))
            return

    # ── Local mode ─────────────────────────────────────────────────────
    image = _resolve_nim_container_image(snapshot, nim_container_image)
    if image is None:
        await _fail(
            "preflight_failed",
            "No NIM container image resolved (base ModelConfig "
            "lacks local_deploy_metadata.nim_container_image and no override "
            "was provided)",
            ref="missing_nim_container_image",
        )
        return

    gpu_min_gb = _resolve_gpu_memory_minimum(snapshot, settings)
    served_model = _served_model_name(snapshot)
    deployment_id: str | None = None
    deployment_endpoint_url: str | None = None
    eval_run_id: str | None = None
    # Residents this Student displaces (single-GPU host). Restored on EVERY
    # exit path by the finally below — populated once the deploy reports what
    # it stopped.
    displaced_for_restore: list[LocalNimDeployment] = []

    try:
        # Resolve GPU placement (one-NIM-per-GPU
        # invariant): on a single-GPU host, the
        # auto-placer raises GpuExhaustedError because the resident
        # Teacher (or any other NIM) holds device=0. The Student
        # lifecycle's invocation IS the implicit operator opt-in for
        # replace semantics — fall back to device=0 and let
        # deploy_local_nim(replace_resident=True) stop the resident
        # before the Student container starts.
        try:
            gpu = await local_nim_service.resolve_gpu_placement(
                role="student",
                explicit_gpu=gpu_assignment,
                workspace_root=workspace_root,
            )
        except local_nim_service.GpuExhaustedError:
            gpu = gpu_assignment or "device=0"
            logger.info(
                "all GPUs occupied; Student lifecycle will displace "
                "the resident on %s before docker_run (replace semantics).",
                gpu,
            )
        except ValueError as exc:
            await _fail("preflight_failed", str(exc))
            return

        # ── Stage 1: preflight ─────────────────────────────────────────
        await _emit_progress(project_id, student_model_id, STAGE_PREFLIGHT, started_at)
        preflight = await local_nim_service.run_preflight_checks(
            nim_container_image=image,
            gpu_memory_minimum_gb=gpu_min_gb,
            gpu_assignment=gpu,
            role="student",
            settings=settings,
        )
        _persist_preflight(
            project_id,
            student_model_id,
            workspace_root,
            passed=preflight.all_passed,
            checks=preflight.checks,
            failure_stage=None if preflight.all_passed else "preflight_failed",
        )
        if not preflight.all_passed:
            # Generate the student_nim_deploy Action Request with full
            # Student context for the SME to copy + hand to infrastructure.
            try:
                student_docker = local_nim_service.docker_run_command_display(
                    nim_container_image=image,
                    container_name=local_nim_service.build_container_name(
                        "student", project_id, student_model_id
                    ),
                    gpu_assignment=gpu,
                    host_port=settings.NIM_STUDENT_PORT,
                    role="student",
                    checkpoint_mount=snapshot.nim_checkpoint_ref,
                    nim_model_name_path=STUDENT_CHECKPOINT_IN_CONTAINER,
                    nim_served_model_name=served_model,
                )
                _generate_preflight_action_request(
                    project_id=project_id,
                    snapshot=snapshot,
                    workspace_root=workspace_root,
                    docker_run_command=student_docker,
                    preflight_checks=preflight.checks,
                    image=image,
                    gpu_assignment=gpu,
                    gpu_memory_minimum_gb=gpu_min_gb,
                    served_model=served_model,
                    nim_release_version=nim_release_version,
                    settings=settings,
                )
            except Exception as exc:
                logger.warning("Action Request generation failed: %s", exc)

            await _fail(
                "preflight_failed",
                "; ".join(c.diagnostic for c in preflight.checks if not c.passed),
                ref="preflight_failed",
            )
            return

        # ── Stage 2: docker run ────────────────────────────────────────
        # replace_resident=True invokes the one-NIM-per-GPU acquire-GPU
        # path. On multi-GPU hosts where step 1 found a free device,
        # this is a no-op (no residents on that GPU). On single-GPU
        # hosts the displaced Teacher (or embedding) is stopped before
        # the Student docker_run; the displaced rows are returned for
        # auto-restore in stage 8 step 9.
        await _emit_progress(project_id, student_model_id, STAGE_DOCKER_RUN, started_at)
        deploy_result = await local_nim_service.deploy_local_nim(
            project_id=project_id,
            model_config_id=snapshot.student_base_model_config_id,
            role="student",
            nim_container_image=image,
            gpu_assignment=gpu,
            gpu_memory_minimum_gb=gpu_min_gb,
            preferred_port=settings.NIM_STUDENT_PORT,
            settings=settings,
            workspace_root=workspace_root,
            student_model_id=student_model_id,
            checkpoint_mount=snapshot.nim_checkpoint_ref,
            nim_model_name_path=STUDENT_CHECKPOINT_IN_CONTAINER,
            nim_served_model_name=served_model,
            precision_method=snapshot.quantization_method,
            replace_resident=True,
        )
        deployment = deploy_result["deployment"]
        # Capture displaced rows so the finally can auto-restore the prior
        # Teacher / embedding after the Student stops — on success AND on any
        # failure path below.
        displaced_for_restore = list(deploy_result.get("displaced", []) or [])
        if deployment.status == "failed":
            await _fail(
                "docker_run_failed",
                deployment.status_reason,
                deployment_id=deployment.local_nim_deployment_id,
            )
            return
        # Past the failure-return above, deployment is the running record.
        # Both fields are mapped non-null on LocalNimDeployment.
        deployment_id = str(deployment.local_nim_deployment_id)
        deployment_endpoint_url = str(deployment.endpoint_url)

        # Persist the real GPU MODEL name (e.g. "NVIDIA A100-SXM4-80GB") as the
        # eval provenance gpu_type — ``gpu`` here is just the placement string
        # ("device=0"), which is not reproducibility metadata. Best-effort:
        # fall back to the device string if the inventory can't be read.
        gpu_type_value = gpu
        try:
            from vlm_feedback_loop.services.environment import probe_gpu_inventory

            device_idx = int(local_nim_service.extract_device_index(gpu))
            inventory = await probe_gpu_inventory()
            if 0 <= device_idx < len(inventory):
                gpu_type_value = inventory[device_idx].name
        except Exception as exc:
            logger.debug("Could not resolve GPU model name for %s: %s", gpu, exc)

        _write_student_state(
            project_id,
            student_model_id,
            workspace_root,
            fields={
                "nim_container_id": deployment.container_id,
                "nim_endpoint_url": deployment_endpoint_url,
                "gpu_type": gpu_type_value,
                "gpu_count": 1,
            },
        )

        # ── Stage 3: health poll ────────────────────────────────────────
        await _emit_progress(
            project_id,
            student_model_id,
            STAGE_HEALTH_POLL,
            started_at,
            deployment_id=deployment_id,
        )
        healthy = await _wait_for_deployment_running(
            project_id,
            deployment_id,
            workspace_root,
            float(settings.NIM_STARTUP_TIMEOUT_S),
        )
        if not healthy:
            await _fail("health_timeout", deployment_id=deployment_id)
            # Stop the orphaned container before returning so the
            # next variant's deploy isn't blocked on a single-GPU host.
            await _stop_deployment_quiet(deployment_id, project_id, workspace_root)
            return

        # ── Stage 4: smoke ─────────────────────────────────────────────
        await _emit_progress(
            project_id,
            student_model_id,
            STAGE_SMOKE,
            started_at,
            deployment_id=deployment_id,
        )
        smoke_ok = await _smoke_inference(deployment_endpoint_url, served_model)
        if not smoke_ok:
            await _fail("smoke_failed", deployment_id=deployment_id)
            await _stop_deployment_quiet(deployment_id, project_id, workspace_root)
            return

        # ── Stage 5: register temp endpoint + ModelConfig ──────────────
        await _emit_progress(
            project_id,
            student_model_id,
            STAGE_REGISTER,
            started_at,
            deployment_id=deployment_id,
        )
        try:
            _, temp_mc_id = await _register_temp_endpoint(
                snapshot,
                workspace_root,
                settings,
                base_url=deployment_endpoint_url,
                mode="local",
                auth_mode="none",
                local_nim_deployment_id=deployment_id,
            )
        except Exception as exc:
            await _fail(
                "register_endpoint_failed", str(exc), deployment_id=deployment_id
            )
            await _stop_deployment_quiet(deployment_id, project_id, workspace_root)
            return

        # ── Stage 6: evaluation against Test Pool via NIM ──────────────
        await _emit_progress(
            project_id,
            student_model_id,
            STAGE_EVALUATION,
            started_at,
            deployment_id=deployment_id,
        )
        eval_run_id, run_record, err = await _run_evaluation_phase(
            snapshot, workspace_root, settings, temp_mc_id
        )
        # The lifecycle's eval gate
        # exists to validate that the SERVING container produces parseable
        # output, NOT to validate model accuracy. A run finalizing as
        # ``incomplete`` means at least one example failed
        # schema-validation after the retry pass — typically because the
        # fine-tuned model emits some examples with the wrong field shape
        # (a model-quality issue, not a serving issue). For serving validation, that is
        # acceptable: the container deployed, served HTTP responses, and
        # produced parseable JSON for SOME invocations. Quality
        # validation (the strict ``completed`` gate) lives in
        # ``_promote_quality_from_nim_eval`` further down — that path
        # remains conservative and only flips ``quality_status`` on a
        # clean ``completed`` run + matching loader-gap signature.
        eval_acceptable_for_serving = (
            err is None
            and eval_run_id is not None
            and run_record is not None
            and run_record.status in _SERVING_ACCEPTABLE_STATUSES
        )
        if not eval_acceptable_for_serving:
            detail = err or (
                f"run_status={run_record.status}" if run_record is not None else None
            )
            await _fail("eval_failed", detail, deployment_id=deployment_id)
            await _stop_deployment_quiet(deployment_id, project_id, workspace_root)
            return
        # eval_acceptable_for_serving narrows both to non-None.
        assert eval_run_id is not None
        assert run_record is not None

        # ── Stage 7: benchmark sweep ───────────────────────────────────
        benchmarks, skipped = await _run_benchmark_sweep(
            project_id,
            snapshot,
            workspace_root,
            settings,
            base_url=deployment_endpoint_url,
            served_model=served_model,
            started_at=started_at,
            deployment_id=deployment_id,
            adapter=benchmark_adapter,
        )
        # Persist the sweep onto the serving evaluation run. The SSE
        # payload below is transient — the Compare page responds to
        # ``nim_benchmark_completed`` by refetching the suite/student
        # queries and reads ``metrics.benchmarks`` from run records, so
        # without this write the ServingMatrix renders em-dashes after
        # any reload.
        if benchmarks:
            _persist_benchmarks(
                project_id,
                workspace_root,
                run_id=eval_run_id,
                benchmarks=benchmarks,
            )

        # ── Stage 8: stopping ──────────────────────────────────────────
        await _emit_progress(
            project_id,
            student_model_id,
            STAGE_STOPPING,
            started_at,
            deployment_id=deployment_id,
        )
        await local_nim_service.stop_local_nim(
            deployment_id=deployment_id,
            project_id=project_id,
            workspace_root=workspace_root,
        )

        # Displaced residents are auto-restored in the finally (below), which
        # covers this success path AND every failure return.

        # Unified serving/quality gate (shared with external mode): serving
        # validates, quality routes per run status + parseable rate.
        _apply_serving_quality_gate(
            project_id,
            student_model_id,
            workspace_root,
            settings,
            run_id=eval_run_id,
            run_record=run_record,
        )
        await _emit_completed(
            project_id,
            student_model_id,
            started_at,
            evaluation_run_id=eval_run_id,
            rescored_metrics=run_record.rescored_metrics,
            benchmarks=benchmarks,
            skipped_concurrencies=skipped,
        )

    except Exception as exc:  # final safety net
        logger.exception("Unhandled lifecycle failure for Student %s", student_model_id)
        await _fail("internal_error", str(exc), deployment_id=deployment_id)
        # Best-effort container teardown after any unhandled failure.
        if deployment_id:
            with contextlib.suppress(Exception):
                await local_nim_service.stop_local_nim(
                    deployment_id=deployment_id,
                    project_id=project_id,
                    workspace_root=workspace_root,
                )
    finally:
        # Restore any resident this Student displaced — on success (Student
        # already stopped at stage 8) and on every failure path (the Student
        # container was stopped by the stage's failure handler). Without this,
        # a failed deploy/benchmark on a single-GPU host leaves the SME with no
        # Teacher. Idempotent + best-effort; empty when nothing
        # was displaced.
        await _restore_all_displaced(
            displaced_for_restore,
            workspace_root=workspace_root,
            settings=settings,
        )


def _generate_preflight_action_request(
    *,
    project_id: str,
    snapshot: StudentSnapshot,
    workspace_root: str,
    docker_run_command: str,
    preflight_checks: list[Any],
    image: str,
    gpu_assignment: str,
    gpu_memory_minimum_gb: int,
    served_model: str,
    nim_release_version: str | None,
    settings: Settings,
) -> None:
    """Generate a ``student_nim_deploy`` Action Request on preflight failure.

    Only fires for preflight failures. Later lifecycle
    failures emit ``run_failed`` SSE only — those are operational
    diagnostics, not handoff requests.
    """
    from vlm_feedback_loop.db.models.project import Project

    # Resolve project name for the rendered text.
    project_name = project_id
    engine = get_project_engine(project_id, workspace_root)
    if engine is not None:
        with Session(engine) as session:
            project = session.get(Project, project_id)
            if project is not None:
                project_name = project.name

    context = {
        "docker_run_command": docker_run_command,
        "preflight_checks": [
            {
                "check_name": c.check_name,
                "passed": c.passed,
                "diagnostic": c.diagnostic,
            }
            for c in preflight_checks
        ],
        "role": "student",
        "nim_container_image": image,
        "gpu_assignment": gpu_assignment,
        "gpu_memory_minimum_gb": gpu_memory_minimum_gb,
        "host_port": settings.NIM_STUDENT_PORT,
        "student_model_id": snapshot.student_model_id,
        "nim_checkpoint_ref": snapshot.nim_checkpoint_ref,
        "quantization_method": snapshot.quantization_method or "bf16",
        "nim_served_model_name": served_model,
        "nim_model_name_path": STUDENT_CHECKPOINT_IN_CONTAINER,
        "checkpoint_directory_structure": [
            "config.json",
            "tokenizer.json|tokenizer.model",
            "*.safetensors|pytorch_model*.bin",
        ],
        "nim_release_version": nim_release_version or "(not specified)",
    }

    generate_action_request(
        request_type="student_nim_deploy",
        project_name=project_name,
        project_id=project_id,
        context=context,
    )
