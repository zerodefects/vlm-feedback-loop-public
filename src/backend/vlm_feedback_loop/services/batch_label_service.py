# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batch labeling run orchestration service.

Implements:
  - Batch labeling run lifecycle: create, execute (background), resume, cancel,
    get, list.
  - 7-status state machine: queued/running/paused/canceling/completed/
    canceled/failed.
  - Provider-aware concurrent Teacher inference (hosted → serial, self-hosted
    → parallel lanes) with circuit breaker protection.
  - SSE event production: batch_label_progress, batch_label_completed, run_failed.
  - Scale-Up Readiness Gate verification before start.
  - Restart recovery: queued/running → queued+recovered then auto-resumed
    (idempotent per-example skip); canceling → canceled; paused stays.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import ACTIVE_RUN_STATUSES, RunRecord
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.exact_match_evaluator import (
    SchemaValidationReport,
    validate_proposal,
)
from vlm_feedback_loop.services.icl_service import query_icl_candidates
from vlm_feedback_loop.services.invocation_outcome import (
    InvocationArtifactRefs,
    apply_invocation_outcome,
    classify_invocation_status,
    write_invocation_artifacts,
)
from vlm_feedback_loop.services.nim_client import build_endpoint_auth_headers
from vlm_feedback_loop.services.priority import priority_dispatch
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.prompt_service import (
    ModelConfigInput,
    TeacherInvocationResult,
    invoke_teacher,
)
from vlm_feedback_loop.services.run_config import (
    create_runtime_config_snapshot,
    snapshot_run_config,
)
from vlm_feedback_loop.services.run_queries import (
    find_run,
    list_runs_page,
    update_run_if_not_terminal,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.sse import sse_manager
from vlm_feedback_loop.services.teacher_rejection import (
    RunExampleCounts,
    clear_runtime_rejections,
    finalize_canceled,
    finalize_runtime_rejection,
    finalize_unhandled_exception,
    mark_operation_ignored_if_canceling,
    record_runtime_rejections,
)

logger = logging.getLogger("vlm_feedback_loop.batch_label_service")

# ── Module-level state ──────────────────────────────────────────────────────

# Per-run cancellation events.  Populated when a run enters ``running``,
# cleared on terminal state.
_cancel_events: dict[str, asyncio.Event] = {}

# The per-run runtime-rejection registries and their record/finalize
# lifecycle live in ``services.teacher_rejection`` — one implementation
# shared with ``evaluation_service``.

TERMINAL_STATUSES = frozenset({"completed", "canceled", "failed"})
BATCH_INVOCATION_STATUSES = frozenset(
    {
        "pending",
        "success",
        "schema_invalid",
        "timeout",
        "endpoint_error",
        "rate_limited",
    }
)


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class BatchExampleResult:
    """Result of batch labeling a single example."""

    example_key: str
    invocation_id: str
    invocation_status: str  # success | schema_invalid | timeout | endpoint_error
    proposal_json: dict[str, Any] | None
    schema_valid_core: bool
    persistence: _BatchInvocationPersistence | None = None


@dataclass(frozen=True)
class _BatchInvocationPersistence:
    """Invocation outcome fields committed with the label domain state."""

    teacher_result: TeacherInvocationResult
    validation_report: SchemaValidationReport
    t_validation_ms: int
    artifact_refs: InvocationArtifactRefs
    structured_generation_mode_effective: str


def _resolve_batch_concurrency(
    endpoint_mode: str | None,
    settings: Settings,
    *,
    explicit_override: int | None = None,
) -> int:
    """Pick the effective dispatch width, mirroring the evaluation worker.

    - ``explicit_override`` (per-run ``concurrency`` body field, persisted in
      run metrics so restart recovery resumes at the same width) wins when
      provided.
    - Hosted endpoints (build.nvidia.com) → ``BATCH_LABEL_CONCURRENCY_HOSTED``
      (default 1). Stays polite under shared per-account RPM caps.
    - Self-hosted / local NIMs → ``BATCH_LABEL_CONCURRENCY_SELF_HOSTED``
      (default 8). No shared rate limit; saturate the GPU pipeline instead of
      leaving it idle between sequential requests.
    """
    if explicit_override is not None and explicit_override > 0:
        return explicit_override
    if endpoint_mode == "hosted":
        return settings.BATCH_LABEL_CONCURRENCY_HOSTED
    return settings.BATCH_LABEL_CONCURRENCY_SELF_HOSTED


def _next_circuit_breaker_streak(
    current: int,
    result: BatchExampleResult,
) -> int:
    """Advance the endpoint-failure streak for one authoritative outcome."""
    if result.invocation_status == "success" and result.schema_valid_core:
        return 0
    if result.invocation_status in ("timeout", "endpoint_error", "rate_limited"):
        return current + 1
    return current


# ── Inference wrapper (mockable injection point for tests) ──────────────────


def _validate_batch_operation(
    record: OperationRecord,
    project_id: str,
    run_id: str,
    example_key: str,
    *,
    run_config: dict[str, Any],
) -> None:
    """Require one operation to belong exactly to its durable batch item."""
    expected: dict[str, Any] = {
        "project_id": project_id,
        "purpose": "batch_label",
        "batch_label_run_id": run_id,
        "example_key": example_key,
        "guidance_id": run_config["guidance_id"],
        "model_config_id": run_config["model_config_id"],
        "endpoint_id": run_config["endpoint_id"],
        "model_name": run_config["model_name"],
        "label_tier": "auto_labeled",
    }
    for field, value in expected.items():
        if getattr(record, field) != value:
            raise RuntimeError(
                "Batch operation does not match its run item: "
                f"run_id={run_id}, example_key={example_key}, field={field}"
            )


def _claim_batch_operation(
    project_id: str,
    run_id: str,
    example_key: str,
    *,
    run_config: dict[str, Any],
    engine: Any,
) -> str:
    """Create or reuse the one pending operation for a batch item."""
    with Session(engine) as session:
        records = (
            session.query(OperationRecord)
            .filter_by(batch_label_run_id=run_id, example_key=example_key)
            .all()
        )
        if len(records) > 1:
            raise RuntimeError(
                "Batch item has multiple OperationRecords: "
                f"run_id={run_id}, example_key={example_key}"
            )

        if records:
            record = records[0]
            if record.invocation_status != "pending":
                raise RuntimeError(
                    "Batch executor reached an already-terminal item: "
                    f"run_id={run_id}, example_key={example_key}"
                )
            _validate_batch_operation(
                record,
                project_id,
                run_id,
                example_key,
                run_config=run_config,
            )
            return record.inference_invocation_id

        inference_invocation_id = generate_uuid4()
        session.add(
            OperationRecord(
                inference_invocation_id=inference_invocation_id,
                project_id=project_id,
                purpose="batch_label",
                example_key=example_key,
                guidance_id=run_config["guidance_id"],
                model_config_id=run_config["model_config_id"],
                endpoint_id=run_config["endpoint_id"],
                model_name=run_config["model_name"],
                invocation_status="pending",
                label_tier="auto_labeled",
                batch_label_run_id=run_id,
            )
        )
        session.commit()
        return inference_invocation_id


async def _invoke_for_batch_label(
    project_id: str,
    run_id: str,
    example_key: str,
    *,
    run_config: dict[str, Any],
    engine: Any,
    settings: Settings,
) -> BatchExampleResult:
    """Invoke the Teacher for one example and validate the result.

    Mirrors ``evaluation_service._invoke_for_evaluation`` minus the Exact
    Match step — batch labeling has no ground truth; the validated output
    IS the Auto-Labeled label. The outer executor handles Label creation
    and Example state transition based on the ``BatchExampleResult`` this
    function returns.

    Differences vs evaluation:
    - ``purpose="batch_label"``; OperationRecord links via
      ``batch_label_run_id`` rather than ``evaluation_run_id``.
    - ``label_tier="auto_labeled"`` (evaluation uses ``"proposal"``).
    - ``run_config["icl_mode"]`` mirrors the evaluation API's field:
      ``"disabled"`` skips the ICL candidate query entirely, so the run
      labels at the Teacher's zero-shot form. Default ``"enabled"`` — the
      §8.3 shipped behavior — but ICL-negative teachers demonstrably exist
      (freiburg/Omni: zero-shot 0.525 vs depth-1 0.217), and the only
      prior lever was the process-global ``ICL_MAX_EXAMPLES=0`` hack that
      poisoned every other project on the backend.
    - No ``match_fields`` / ``exact_match_pass`` — there is no ground truth.

    Unit tests monkeypatch this function directly, so the signature and
    return dataclass are a stable seam. The wire-mock end-to-end tests
    (``TestProfileBProductionPipeline``) mock
    ``nim_client.chat_completions`` at the wire level and exercise this
    pipeline end-to-end.
    """
    guidance_id: str = run_config["guidance_id"]
    guidance_fields: list[dict[str, Any]] = run_config["guidance_fields"]
    model_name: str = run_config["model_name"]
    mc_input: ModelConfigInput = run_config["mc_input"]
    project_dir: str = run_config["project_dir"]
    storage_refs: dict[str, str] = run_config["storage_refs"]
    storage_ref = storage_refs.get(example_key)

    # ── 1. Persist or reuse the pending operation BEFORE the NIM call ────
    inference_invocation_id = _claim_batch_operation(
        project_id,
        run_id,
        example_key,
        run_config=run_config,
        engine=engine,
    )

    # ── 2. Query ICL candidates (unless the run disables ICL) ────────────
    icl_mode: str = run_config.get("icl_mode", "enabled")
    icl_candidates: list[Any] = []
    if icl_mode != "disabled":
        with Session(engine) as session:
            icl_candidates = query_icl_candidates(session, project_id, guidance_id)

    auth_headers = build_endpoint_auth_headers(
        run_config["endpoint_auth_mode"],
        get_effective_secret("NVIDIA_API_KEY", settings),
    )

    # ── 3. Call invoke_teacher ───────────────────────────────────────────
    # Run-level structured-gen mode is baked into ``mc_input`` at snapshot
    # time (prompt_only → structured_generation_support="unsupported"). If
    # this call hits a mid-run response_format rejection under ``auto``
    # mode, the whole run must fail — handled after the call
    # via ``_structured_gen_rejected`` + cancel_event signaling. The
    # batch-label loop picks up the cancel at the top of the next
    # iteration; Phase C then finalizes ``failed`` with
    # ``status_reason="structured_generation_rejected"``.
    teacher_result = await invoke_teacher(
        project_id=project_id,
        example_key=example_key,
        purpose="batch_label",
        inference_invocation_id=inference_invocation_id,
        guidance_description=run_config["guidance_description"],
        guidance_rules=run_config["guidance_rules"],
        guidance_fields=guidance_fields,
        generation_order=run_config["generation_order"],
        derived_json_schema=run_config["derived_json_schema"],
        output_field_mode=run_config["inference_contract"].get(
            "output_field_mode", "all"
        ),
        icl_field_mode=run_config["inference_contract"].get(
            "icl_field_mode", "core_only"
        ),
        model_name=model_name,
        model_config=mc_input,
        endpoint_base_url=run_config["endpoint_base_url"],
        auth_headers=auth_headers,
        icl_candidates=icl_candidates,
        generation_preset_key=run_config["gen_preset_key"],
        thinking_on=run_config["thinking_on"],
        visual_budget_preset_key=run_config["vb_preset_key"],
        **run_config["invoke_settings"],
        icl_max_examples=run_config["icl_max_examples"],
        icl_sim_gap=run_config["icl_sim_gap"],
        icl_abs_threshold=run_config["icl_abs_threshold"],
        scope_id=run_id,  # deterministic per-example seed
        deadline_s=float(settings.HTTP_DEADLINE_BACKGROUND_S),
        max_retries=settings.HTTP_MAX_RETRIES,
        query_storage_ref=storage_ref,
        image_transport_max_longest_edge=run_config["image_transport_max_longest_edge"],
        settings=settings,
    )

    # ── 4. Validate the response against SchemaCore ──────────────────────
    # Stage 4 timing — schema validation. Persisted on OperationRecord
    # alongside the other stage timings.
    _validation_t0 = time.monotonic()
    validation_report = validate_proposal(
        teacher_result.content,
        guidance_fields,
        teacher_result.finish_reason,
    )
    t_validation_ms = int((time.monotonic() - _validation_t0) * 1000)

    # ── 5. Classify final invocation_status ──────────────────────────────
    final_status = classify_invocation_status(teacher_result, validation_report)

    # ── 6. Write artifact files ──────────────────────────────────────────
    artifacts_dir = Path(project_dir) / "artifacts"
    artifact_refs = write_invocation_artifacts(
        artifacts_dir,
        inference_invocation_id,
        teacher_result=teacher_result,
        validation_report=validation_report,
    )

    return BatchExampleResult(
        example_key=example_key,
        invocation_id=inference_invocation_id,
        invocation_status=final_status,
        proposal_json=validation_report.normalized_json,
        schema_valid_core=validation_report.schema_valid_core,
        persistence=_BatchInvocationPersistence(
            teacher_result=teacher_result,
            validation_report=validation_report,
            t_validation_ms=t_validation_ms,
            artifact_refs=artifact_refs,
            structured_generation_mode_effective=run_config["sgm_effective"],
        ),
    )


def _commit_batch_result(
    project_id: str,
    run_id: str,
    guidance_id: str,
    result: BatchExampleResult,
    *,
    engine: Any,
    cancel_event: asyncio.Event,
    circuit_breaker_consecutive: int,
    circuit_breaker_tripped: bool,
) -> Literal["persisted", "sme_superseded", "terminal", "inactive"]:
    """Commit one invocation outcome and its domain effect as one unit."""
    with Session(engine) as session:
        if result.persistence is not None:
            record = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.invocation_id,
                    project_id=project_id,
                    purpose="batch_label",
                    batch_label_run_id=run_id,
                    example_key=result.example_key,
                )
                .one_or_none()
            )
            if record is None or record.invocation_status != "pending":
                raise RuntimeError(
                    "Batch outcome has no matching pending operation: "
                    f"run_id={run_id}, example_key={result.example_key}"
                )
            persistence = result.persistence
            apply_invocation_outcome(
                record,
                final_status=result.invocation_status,
                teacher_result=persistence.teacher_result,
                validation_report=persistence.validation_report,
                t_validation_ms=persistence.t_validation_ms,
                artifact_refs=persistence.artifact_refs,
                structured_generation_mode_effective=(
                    persistence.structured_generation_mode_effective
                ),
                structured_generation_fallback_used=(
                    persistence.teacher_result.structured_generation_fallback_used
                ),
                thinking_fallback_used=False,
                visual_budget_fallback_used=False,
            )
            if mark_operation_ignored_if_canceling(session, record, run_id):
                session.commit()
                return "inactive"

        # A runtime capability rejection sets the in-memory event before its
        # run-level failure is persisted. Retain the invocation audit outcome,
        # but do not let it affect labels, progress, or breaker state.
        if cancel_event.is_set():
            session.commit()
            return "inactive"

        # Every authoritative outcome, not only a valid label, must prove the
        # run is still running. This write also takes SQLite's writer lock, so
        # guidance edits and cancellation order wholly before or after it.
        if not update_run_if_not_terminal(
            session,
            run_id,
            {"status": RunRecord.status},
            only_status="running",
        ):
            session.commit()
            return "inactive"

        run = session.query(RunRecord).filter_by(run_id=run_id).one()
        if not isinstance(run.metrics, dict):
            raise RuntimeError(f"Batch run {run_id} has no durable metadata")
        metrics = dict(run.metrics)
        metrics["circuit_breaker_consecutive"] = circuit_breaker_consecutive
        metrics["circuit_breaker_tripped"] = circuit_breaker_tripped
        run.metrics = metrics

        valid_success = (
            result.invocation_status == "success" and result.schema_valid_core
        )
        if not valid_success:
            session.commit()
            return "terminal"

        example = (
            session.query(Example)
            .filter_by(project_id=project_id, example_key=result.example_key)
            .one_or_none()
        )
        if example is None:
            raise RuntimeError(
                "Batch item Example vanished before outcome commit: "
                f"run_id={run_id}, example_key={result.example_key}"
            )
        if example.state in ("Verified", "Omitted"):
            session.commit()
            return "sme_superseded"
        if result.proposal_json is None:
            raise RuntimeError(
                "Schema-valid batch result has no normalized label: "
                f"run_id={run_id}, example_key={result.example_key}"
            )

        session.query(Label).filter(
            Label.project_id == project_id,
            Label.example_key == result.example_key,
            Label.label_status == "auto_labeled",
        ).delete()
        session.add(
            Label(
                label_id=generate_uuid4(),
                project_id=project_id,
                example_key=result.example_key,
                label_status="auto_labeled",
                guidance_id=guidance_id,
                inference_invocation_id=result.invocation_id,
                label_json=result.proposal_json,
                labeled_at=utc_now(),
                batch_label_run_id=run_id,
            )
        )
        example.state = "Auto-Labeled"
        session.commit()
        return "persisted"


def reconcile_batch_items(
    session: Session,
    project_id: str,
    run_id: str,
    input_keys: list[str],
    *,
    run_config: dict[str, Any],
    mode: Literal["active_resume", "canceled_recovery"] = "active_resume",
) -> dict[str, tuple[str, bool]]:
    """Classify exact durable items for resume or canceled-run recovery.

    Active resume repairs a torn success to pending and rejects ignored
    outcomes. Canceled recovery skips both: neither is authoritative evidence,
    and a terminal run must preserve its audit rows unchanged.
    """
    input_key_set = set(input_keys)
    if len(input_keys) != len(input_key_set):
        raise RuntimeError(f"Batch run {run_id} contains duplicate input keys")

    operations = (
        session.query(OperationRecord)
        .filter(OperationRecord.batch_label_run_id == run_id)
        .all()
    )
    operations_by_key: dict[str, list[OperationRecord]] = {}
    for operation in operations:
        if operation.example_key is None or operation.example_key not in input_key_set:
            raise RuntimeError(
                "Batch operation does not belong to the run input snapshot: "
                f"run_id={run_id}, example_key={operation.example_key}"
            )
        _validate_batch_operation(
            operation,
            project_id,
            run_id,
            operation.example_key,
            run_config=run_config,
        )
        operations_by_key.setdefault(operation.example_key, []).append(operation)

    duplicate_keys = [key for key, rows in operations_by_key.items() if len(rows) > 1]
    if duplicate_keys:
        raise RuntimeError(
            "Batch run contains multiple OperationRecords for one item: "
            f"run_id={run_id}, example_key={sorted(duplicate_keys)[0]}"
        )

    states = {
        str(example_key): str(state)
        for example_key, state in session.execute(
            select(Example.example_key, Example.state).where(
                Example.project_id == project_id,
                Example.example_key.in_(input_keys),
            )
        ).all()
    }
    missing_examples = sorted(input_key_set - set(states))
    if missing_examples:
        raise RuntimeError(
            "Batch run input snapshot references a missing Example: "
            f"run_id={run_id}, example_key={missing_examples[0]}"
        )
    auto_labels = (
        session.query(Label)
        .filter(
            Label.project_id == project_id,
            Label.example_key.in_(input_keys),
            Label.label_status == "auto_labeled",
        )
        .all()
    )
    labels_by_key: dict[str, Label] = {}
    for label in auto_labels:
        if label.example_key in labels_by_key:
            raise RuntimeError(
                "Batch item has multiple live Auto-Labeled Labels: "
                f"run_id={run_id}, example_key={label.example_key}"
            )
        labels_by_key[label.example_key] = label

    done: dict[str, tuple[str, bool]] = {}
    repaired = False
    for example_key, rows in operations_by_key.items():
        operation = rows[0]
        if operation.invocation_status not in BATCH_INVOCATION_STATUSES:
            raise RuntimeError(
                "Batch operation has an invalid invocation status: "
                f"run_id={run_id}, example_key={example_key}, "
                f"status={operation.invocation_status}"
            )
        if operation.ignored_due_to_run_cancellation:
            if mode == "canceled_recovery":
                continue
            raise RuntimeError(
                "Active batch run contains a cancellation-ignored operation: "
                f"run_id={run_id}, example_key={example_key}"
            )
        if operation.invocation_status == "pending":
            continue

        valid_success = operation.invocation_status == "success" and bool(
            operation.schema_valid_core
        )
        if not valid_success:
            done[example_key] = (
                operation.invocation_status,
                bool(operation.schema_valid_core),
            )
            continue

        state = states.get(example_key)
        label = labels_by_key.get(example_key)
        coherent_machine_label = (
            state == "Auto-Labeled"
            and label is not None
            and label.batch_label_run_id == run_id
            and label.inference_invocation_id == operation.inference_invocation_id
            and label.guidance_id == run_config["guidance_id"]
        )
        if coherent_machine_label or state in ("Verified", "Omitted"):
            done[example_key] = ("success", True)
            continue

        if mode == "canceled_recovery":
            continue
        operation.invocation_status = "pending"
        repaired = True

    if repaired:
        session.commit()
    return done


def summarize_batch_outcomes(
    outcomes: dict[str, tuple[str, bool]],
) -> RunExampleCounts:
    """Summarize one terminal outcome per logical batch item."""
    succeeded = schema_invalid = timed_out = endpoint_error = 0
    for status, valid in outcomes.values():
        if status == "success" and valid:
            succeeded += 1
        elif status == "timeout":
            timed_out += 1
        elif status in ("endpoint_error", "rate_limited"):
            endpoint_error += 1
        else:
            schema_invalid += 1
    return RunExampleCounts(
        succeeded=succeeded,
        schema_invalid=schema_invalid,
        timeout=timed_out,
        endpoint_error=endpoint_error,
    )


# ── Start batch label run ──────────────────────────────────────────────────


async def start_batch_label_run(
    project_id: str,
    *,
    include_auto_labeled: bool = False,
    run_limit: int | None = None,
    structured_generation_mode: str | None = None,
    ingested_after: str | None = None,
    ingested_before: str | None = None,
    concurrency: int | None = None,
    icl_mode: str = "enabled",
    settings: Settings,
) -> dict[str, Any] | str:
    """Create a new batch label run, snapshot config, and register background task.

    Returns a response dict on success or an error string for HTTP mapping.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"

        if not project.active_guidance_id:
            return "No active Guidance configured"
        if not project.teacher_model_config_id:
            return "No Teacher model configured"
        snap_model_config_id = project.teacher_model_config_id
        snap_gen_preset = project.labeling_generation_preset_key
        snap_vb_preset = project.visual_budget_preset_key
        if snap_gen_preset not in settings.LABELING_PRESETS:
            return (
                f"invalid generation_preset_key {snap_gen_preset!r}: "
                f"not in {sorted(settings.LABELING_PRESETS)}"
            )
        if snap_vb_preset not in settings.VISUAL_BUDGET_PRESETS:
            return (
                f"invalid visual_budget_preset_key {snap_vb_preset!r}: "
                f"not in {sorted(settings.VISUAL_BUDGET_PRESETS)}"
            )
        runtime_config_snapshot = create_runtime_config_snapshot(
            session,
            project_id,
            snap_model_config_id,
            settings=settings,
            generation_preset_key=snap_gen_preset,
            visual_budget_preset_key=snap_vb_preset,
            icl_max_examples=settings.ICL_MAX_EXAMPLES,
            icl_candidate_limit=None,
            icl_sim_gap=settings.ICL_SIM_GAP,
            icl_abs_threshold=settings.ICL_ABS_THRESHOLD,
        )

        # ── Verify Scale-Up Readiness Gate ────────────────────────────
        from vlm_feedback_loop.services.evaluation_service import (
            compute_scaleup_gate,
        )

        gate_result = compute_scaleup_gate(project_id, settings=settings)
        if isinstance(gate_result, str):
            return f"conflict: Scale-Up Readiness Gate check failed: {gate_result}"
        if gate_result.get("gate_status") != "ready":
            return "conflict: Scale-Up Readiness Gate not ready"

        # ── Reject a second concurrent batch run ──────────────────────
        # One batch run per project at a time. Two overlapping runs share the
        # same start-time Unlabeled snapshot and would race to write duplicate
        # auto_labeled Labels for the same key (a double-clicked "Start Batch
        # Labeling" is the common trigger). A queued/running/canceling run must
        # reach a terminal state (or be paused) before another starts.
        active_batch = (
            session.query(RunRecord)
            .filter(
                RunRecord.project_id == project_id,
                RunRecord.run_type == "batch_label_run",
                RunRecord.status.in_(ACTIVE_RUN_STATUSES),
            )
            .first()
        )
        if active_batch is not None:
            return (
                "conflict: A batch labeling run is already in progress for this "
                "project. Wait for it to finish, pause, or cancel it first."
            )

        # ── Select input examples ─────────────────────────────────────
        states = ["Unlabeled"]
        if include_auto_labeled:
            states.append("Auto-Labeled")

        stmt = (
            select(Example.example_key)
            .where(
                Example.project_id == project_id,
                Example.state.in_(states),
            )
            .order_by(Example.ingested_at.asc(), Example.example_key.asc())
        )
        if ingested_after:
            stmt = stmt.where(Example.ingested_at >= ingested_after)
        if ingested_before:
            stmt = stmt.where(Example.ingested_at <= ingested_before)

        all_keys = list(session.execute(stmt).scalars().all())

        # Apply run limit: take the smaller of request limit, settings limit,
        # and total available.
        effective_limit = len(all_keys)
        if run_limit is not None and run_limit < effective_limit:
            effective_limit = run_limit
        if (
            settings.BATCH_LABEL_RUN_LIMIT is not None
            and effective_limit > settings.BATCH_LABEL_RUN_LIMIT
        ):
            effective_limit = settings.BATCH_LABEL_RUN_LIMIT
        input_keys = all_keys[:effective_limit]

        # ── Snapshot configuration ────────────────────────────────────
        effective_sgm = (
            structured_generation_mode or project.structured_generation_mode_default
        )
        thinking_mode = "on" if project.thinking_default_on else "off"

        snap_guidance_id = project.active_guidance_id

        run_id = generate_uuid4()
        now = utc_now()

        run_record = RunRecord(
            run_id=run_id,
            project_id=project_id,
            run_type="batch_label_run",
            status="queued",
            guidance_id=snap_guidance_id,
            model_config_id=snap_model_config_id,
            generation_preset_key=snap_gen_preset,
            thinking_mode_effective=thinking_mode,
            visual_budget_preset_key=snap_vb_preset,
            structured_generation_mode_effective=effective_sgm,
            runtime_config_snapshot=runtime_config_snapshot,
            # Persisted (same column the eval worker uses) so restart
            # recovery resumes an ICL-off run as ICL-off.
            icl_mode=icl_mode,
            examples_total=len(input_keys),
            # Store input keys for the background executor to iterate over.
            # On completion, metrics is updated with final counters. The
            # concurrency override is persisted (not runtime-only like the
            # eval worker's) because batch runs are restart-recoverable and
            # must resume at the width the operator chose.
            metrics={
                "input_keys": input_keys,
                "include_auto_labeled": include_auto_labeled,
                # The UI needs the configured consecutive threshold if the
                # run pauses. Aggregate lifetime errors cannot reconstruct it
                # after an earlier success reset the breaker counter.
                "circuit_breaker_threshold": (
                    settings.BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD
                ),
                "circuit_breaker_consecutive": 0,
                "circuit_breaker_tripped": False,
                **(
                    {"concurrency_override": concurrency}
                    if concurrency is not None
                    else {}
                ),
            },
        )
        session.add(run_record)
        session.commit()

    # ── Register background task ────────────────────────────────────────
    dispatch_batch_label_run(project_id, run_id, settings)

    return {
        "run_id": run_id,
        "run_type": "batch_label_run",
        "status": "queued",
        "guidance_id": snap_guidance_id,
        "model_config_id": snap_model_config_id,
        "generation_preset_key": snap_gen_preset,
        "thinking_mode_effective": thinking_mode,
        "visual_budget_preset_key": snap_vb_preset,
        "structured_generation_mode_effective": effective_sgm,
        "icl_mode": icl_mode,
        "examples_total": len(input_keys),
        "created_at": now,
    }


# ── Background execution ───────────────────────────────────────────────────


def dispatch_batch_label_run(project_id: str, run_id: str, settings: Settings) -> None:
    """Register the batch-label executor as a background task.

    The single dispatch point for both the initial start (``start_batch_label_
    run``) and startup auto-resume (``main.lifespan``); the executor is
    idempotent, so resuming a run that already processed some examples skips
    them via their OperationRecord.
    """
    executor = _execute_batch_label(project_id, run_id, settings)
    try:
        background_manager.register(
            task_id=f"batch-label-{run_id}",
            coro=executor,
        )
    except Exception:
        executor.close()
        raise


async def _execute_batch_label(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> None:
    """Background coroutine that runs the batch labeling pipeline."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return  # project gone; nothing to do

    cancel_event = asyncio.Event()
    if run_id in _cancel_events:
        return
    _cancel_events[run_id] = cancel_event

    try:
        # ── Phase A: queued → running ───────────────────────────────────
        with Session(engine) as session:
            run = session.query(RunRecord).filter_by(run_id=run_id).first()
            if run is None or run.status in TERMINAL_STATUSES:
                return
            # May already be canceling (requested before execution starts):
            # atomically flip only if still 'canceling'.
            if update_run_if_not_terminal(
                session,
                run_id,
                {"status": "canceled", "completed_at": utc_now()},
                only_status="canceling",
            ):
                session.commit()
                await sse_manager.emit(
                    project_id,
                    "batch_label_completed",
                    {"run_id": run_id, "status": "canceled"},
                )
                return

            if not isinstance(run.metrics, dict):
                raise RuntimeError(f"Batch run {run_id} has no input snapshot")
            meta = dict(run.metrics)
            raw_input_keys = meta.get("input_keys")
            if not isinstance(raw_input_keys, list):
                raise RuntimeError(f"Batch run {run_id} has invalid input keys")
            untyped_input_keys = cast("list[Any]", raw_input_keys)
            if not all(isinstance(key, str) for key in untyped_input_keys):
                raise RuntimeError(f"Batch run {run_id} has invalid input keys")
            input_keys = list(cast("list[str]", untyped_input_keys))
            if len(input_keys) != len(set(input_keys)):
                raise RuntimeError(f"Batch run {run_id} contains duplicate input keys")
            if run.examples_total != len(input_keys):
                raise RuntimeError(
                    "Batch run input snapshot disagrees with its frozen total: "
                    f"run_id={run_id}, total={run.examples_total}, "
                    f"input_keys={len(input_keys)}"
                )
            existing_keys = set(
                session.execute(
                    select(Example.example_key).where(
                        Example.project_id == project_id,
                        Example.example_key.in_(input_keys),
                    )
                ).scalars()
            )
            missing_keys = sorted(set(input_keys) - existing_keys)
            if missing_keys:
                raise RuntimeError(
                    "Batch run input snapshot references a missing Example: "
                    f"run_id={run_id}, example_key={missing_keys[0]}"
                )
            concurrency_override: int | None = meta.get("concurrency_override")
            threshold_snapshot = meta.get("circuit_breaker_threshold")
            if threshold_snapshot is None:
                circuit_breaker_threshold = (
                    settings.BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD
                )
                meta["circuit_breaker_threshold"] = circuit_breaker_threshold
                run.metrics = meta
            elif (
                isinstance(threshold_snapshot, bool)
                or not isinstance(threshold_snapshot, int)
                or threshold_snapshot < 1
            ):
                raise RuntimeError(
                    f"Batch run {run_id} has an invalid circuit-breaker threshold"
                )
            else:
                circuit_breaker_threshold = threshold_snapshot
            streak_snapshot = meta.get("circuit_breaker_consecutive")
            if streak_snapshot is not None and (
                isinstance(streak_snapshot, bool)
                or not isinstance(streak_snapshot, int)
                or streak_snapshot < 0
            ):
                raise RuntimeError(
                    f"Batch run {run_id} has an invalid circuit-breaker streak"
                )
            tripped_snapshot = meta.get("circuit_breaker_tripped")
            if tripped_snapshot is not None and not isinstance(tripped_snapshot, bool):
                raise RuntimeError(
                    f"Batch run {run_id} has an invalid circuit-breaker latch"
                )
            total = run.examples_total
            guidance_id = run.guidance_id
            if guidance_id is None:
                raise RuntimeError(f"Batch run {run_id} has no guidance snapshot")

            # Snapshot config as plain scalars — see services/run_config.py
            run_config: dict[str, Any] = snapshot_run_config(
                session,
                project_id,
                run,
                example_keys=input_keys,
                settings=settings,
            )
            # ICL mode rides on the run row so a restart-recovered
            # run resumes with the mode the operator chose at start.
            run_config["icl_mode"] = run.icl_mode or "enabled"
            run_config["circuit_breaker_threshold"] = circuit_breaker_threshold
            run_config["circuit_breaker_consecutive"] = streak_snapshot
            run_config["circuit_breaker_tripped"] = tripped_snapshot

            # Every dispatch path first places the run in queued. Requiring
            # that exact source state makes one executor the sole owner.
            if not update_run_if_not_terminal(
                session,
                run_id,
                {"status": "running", "started_at": utc_now()},
                only_status="queued",
            ):
                return
            # A run created concurrently with a guidance edit can commit
            # after the edit's cancel sweep ran — the one writer the sweep
            # cannot see. The claim UPDATE holds the write lock, so this
            # read is post-edit truth: a mismatch means the run's snapshot
            # belongs to a retired version and it fails exactly as the
            # sweep would have failed it.
            active_gid_now = (
                session.query(Project.active_guidance_id)
                .filter_by(project_id=project_id)
                .scalar()
            )
            if active_gid_now != guidance_id:
                update_run_if_not_terminal(
                    session,
                    run_id,
                    {
                        "status": "failed",
                        "status_reason": "guidance_edited_during_run",
                        "completed_at": utc_now(),
                    },
                    only_status="running",
                )
                session.commit()
                return
            session.commit()

        # ── Phase B: provider-aware concurrent processing ───────────────
        # Dispatch width mirrors the evaluation worker: hosted endpoints stay
        # serial (rate-limit politeness); self-hosted NIMs run N parallel
        # lanes pulling from one shared iterator. At width 1 this is exactly
        # the sequential loop the run state machine was written for.
        effective_concurrency = _resolve_batch_concurrency(
            run_config.get("endpoint_mode"),
            settings,
            explicit_override=concurrency_override,
        )
        logger.info(
            "batch_label_concurrency run_id=%s endpoint_mode=%s "
            "effective_concurrency=%d explicit_override=%s",
            run_id,
            run_config.get("endpoint_mode"),
            effective_concurrency,
            concurrency_override,
            extra={"component": "batch_label_service", "project_id": project_id},
        )

        succeeded = 0
        schema_invalid = 0
        timed_out = 0
        endpoint_err = 0
        clobber_skipped = 0  # invocations whose auto-label we dropped because
        # the SME had Verified/Omitted the example mid-run

        # Resume only after proving terminal outcomes and their domain lineage
        # agree. A historical split write is reset to pending and reuses its
        # durable invocation identity.
        with Session(engine) as session:
            done_statuses = reconcile_batch_items(
                session,
                project_id,
                run_id,
                input_keys,
                run_config=run_config,
            )
            stored_streak = run_config["circuit_breaker_consecutive"]
            stored_tripped = run_config["circuit_breaker_tripped"]
            if stored_streak is None:
                # Pre-fix runs did not persist completion order. Counting all
                # durable endpoint failures is deliberately conservative: an
                # interrupted legacy outage may pause early, never spend past
                # the configured safety threshold.
                circuit_breaker_consecutive = sum(
                    status in ("timeout", "endpoint_error", "rate_limited")
                    for status, _valid in done_statuses.values()
                )
            else:
                circuit_breaker_consecutive = stored_streak
            circuit_breaker_was_tripped = bool(stored_tripped) or (
                circuit_breaker_consecutive >= circuit_breaker_threshold
            )
            if stored_streak is None or stored_tripped is None:
                run = session.query(RunRecord).filter_by(run_id=run_id).one()
                metrics = dict(run.metrics or {})
                metrics["circuit_breaker_consecutive"] = circuit_breaker_consecutive
                metrics["circuit_breaker_tripped"] = circuit_breaker_was_tripped
                run.metrics = metrics
                session.commit()
        prior_counts = summarize_batch_outcomes(done_statuses)
        succeeded = prior_counts.succeeded
        schema_invalid = prior_counts.schema_invalid
        timed_out = prior_counts.timeout
        endpoint_err = prior_counts.endpoint_error
        pending_keys = [k for k in input_keys if k not in done_statuses]

        # Set when the breaker trips: lanes stop pulling new work, while
        # already-in-flight requests complete and are recorded — a dispatch
        # stop, not preemption. The run pauses only after lanes drain, so
        # counters and OperationRecords stay consistent for resume.
        breaker_tripped = asyncio.Event()
        if circuit_breaker_was_tripped:
            breaker_tripped.set()
        # Set when the durable run is no longer authoritative. Unlike user
        # cancellation, this can represent an externally failed run and must
        # stop dispatch without publishing a false canceled event.
        dispatch_stopped = asyncio.Event()
        # First unhandled exception raised by a lane. Re-raised after the
        # drain so the run fails with status_reason="unhandled_exception",
        # preserving the sequential loop's abort-the-run contract.
        lane_failure: list[BaseException] = []

        # Progress cadence: every example for runs up to 100 (the shipped
        # interactive behavior), ~1% increments beyond that so a
        # multi-thousand-example run doesn't produce one RunRecord write
        # transaction plus one SSE event per example.
        progress_interval = max(1, total // 100)

        def _processed() -> int:
            return succeeded + schema_invalid + timed_out + endpoint_err

        def _persist_counters(
            *, status: str | None = None, paused_reason: str | None = None
        ) -> bool:
            values: dict[str, Any] = {
                "examples_succeeded": succeeded,
                "examples_schema_invalid": schema_invalid,
                "examples_timeout": timed_out,
                "examples_endpoint_error": endpoint_err,
            }
            if status is not None:
                values["status"] = status
                values["paused_reason"] = paused_reason
            with Session(engine) as session:
                # A status transition is stricter than a counter checkpoint:
                # only the running owner may pause. Counter-only writes can
                # still land during canceling so its final partial totals are
                # exact, but neither path can resurrect a terminal row.
                if status is not None:
                    applied = update_run_if_not_terminal(
                        session,
                        run_id,
                        values,
                        only_status="running",
                    )
                else:
                    applied = update_run_if_not_terminal(
                        session,
                        run_id,
                        values,
                        terminal_statuses=TERMINAL_STATUSES,
                    )
                session.commit()
                return applied

        async def _emit_progress(**extra_fields: Any) -> None:
            await sse_manager.emit(
                project_id,
                "batch_label_progress",
                {
                    "run_id": run_id,
                    "processed": _processed(),
                    "total": total,
                    "examples_succeeded": succeeded,
                    "examples_schema_invalid": schema_invalid,
                    "examples_timeout": timed_out,
                    "examples_endpoint_error": endpoint_err,
                    **extra_fields,
                },
            )

        # Recovery reconstructed these counters from the durable item ledger.
        # Publish the same truth to REST before a slow pending invocation can
        # leave clients observing stale pre-crash columns behind a fresh SSE.
        _persist_counters()
        await _emit_progress()

        async def _label_one(example_key: str) -> None:
            """Invoke the Teacher for one example and fold in the outcome.

            Counter and breaker mutations happen in await-free blocks after
            the invocation returns, so lanes (asyncio tasks on one event
            loop) never interleave inside them — no locking needed.
            """
            nonlocal succeeded, schema_invalid, timed_out, endpoint_err
            nonlocal clobber_skipped, circuit_breaker_consecutive

            result = await _invoke_for_batch_label(
                project_id,
                run_id,
                example_key,
                run_config=run_config,
                engine=engine,
                settings=settings,
            )

            next_breaker_streak = _next_circuit_breaker_streak(
                circuit_breaker_consecutive, result
            )
            next_breaker_tripped = breaker_tripped.is_set() or (
                next_breaker_streak >= circuit_breaker_threshold
            )
            disposition = _commit_batch_result(
                project_id,
                run_id,
                guidance_id,
                result,
                engine=engine,
                cancel_event=cancel_event,
                circuit_breaker_consecutive=next_breaker_streak,
                circuit_breaker_tripped=next_breaker_tripped,
            )
            if disposition == "inactive":
                dispatch_stopped.set()
                return
            circuit_breaker_consecutive = next_breaker_streak

            # Only an outcome that committed while the run was authoritative
            # may choose the run's terminal cause. A capability rejection
            # crossing a previously committed user-cancel boundary is an
            # ignored audit record and must not turn ``canceling`` into
            # ``failed``.
            if result.persistence is not None:
                record_runtime_rejections(
                    run_id,
                    result.persistence.teacher_result,
                    sgm_effective=run_config["sgm_effective"],
                    settings=settings,
                    cancel_event=cancel_event,
                )

            if result.invocation_status == "success" and result.schema_valid_core:
                if disposition == "sme_superseded":
                    clobber_skipped += 1
                succeeded += 1

            elif result.invocation_status == "timeout":
                timed_out += 1

            elif result.invocation_status in ("endpoint_error", "rate_limited"):
                # ``rate_limited`` is the 429-exhausted variant of
                # ``endpoint_error`` — same operational meaning for the
                # circuit breaker (network is unreachable), more specific
                # status string for downstream UI copy. Both increment the
                # counter so a sustained 429 storm pauses the run via the
                # circuit breaker rather than burning every example.
                endpoint_err += 1

            else:
                # schema_invalid (invocation_status may be "success" but
                # schema_valid_core is False, or invocation_status is
                # "schema_invalid").  Does NOT affect circuit breaker.
                schema_invalid += 1

            # Circuit breaker check (§8.2 step 8). "Consecutive" is counted
            # in completion order; lanes stop pulling new work as soon as
            # the counter trips, and the run pauses after the drain.
            if next_breaker_tripped:
                breaker_tripped.set()
                return

            processed = _processed()
            if processed % progress_interval == 0 or processed == total:
                _persist_counters()
                await _emit_progress()

        key_iter = iter(pending_keys)

        async def _lane() -> None:
            for example_key in key_iter:
                if (
                    cancel_event.is_set()
                    or breaker_tripped.is_set()
                    or dispatch_stopped.is_set()
                    or lane_failure
                ):
                    return
                # Priority dispatch: hold if foreground active
                await priority_dispatch.wait_for_background()
                if (
                    cancel_event.is_set()
                    or breaker_tripped.is_set()
                    or dispatch_stopped.is_set()
                    or lane_failure
                ):
                    return
                try:
                    await _label_one(example_key)
                except BaseException as exc:
                    lane_failure.append(exc)
                    return

        lanes = [
            asyncio.create_task(_lane()) for _ in range(max(1, effective_concurrency))
        ]
        await asyncio.gather(*lanes)

        # Persist whatever the lanes accumulated since the last interval
        # write before resolving competing terminal causes. Capability
        # rejection wins, then durable user cancellation, then an unhandled
        # lane error. This keeps cancellation intent independent of which
        # in-flight request happened to finish or raise first.
        _persist_counters()
        if await finalize_runtime_rejection(
            engine,
            project_id,
            run_id,
            run_type="batch_label_run",
            terminal_statuses=TERMINAL_STATUSES,
            example_counts=RunExampleCounts(
                succeeded=succeeded,
                schema_invalid=schema_invalid,
                timeout=timed_out,
                endpoint_error=endpoint_err,
            ),
        ):
            return

        if cancel_event.is_set() or dispatch_stopped.is_set() or lane_failure:
            if await _finalize_canceled(engine, project_id, run_id):
                return
            with Session(engine) as session:
                durable_status = (
                    session.query(RunRecord.status).filter_by(run_id=run_id).scalar()
                )
            if durable_status != "running":
                return
            if lane_failure:
                raise lane_failure[0]
            raise RuntimeError(
                f"Batch run {run_id} stopped dispatch without a durable cause"
            )

        if breaker_tripped.is_set() and not cancel_event.is_set():
            paused = _persist_counters(
                status="paused",
                paused_reason="circuit_breaker_threshold_reached",
            )
            if not paused:
                await _finalize_canceled(engine, project_id, run_id)
                return
            await _emit_progress(paused=True)
            logger.info(
                "Batch run %s paused: circuit breaker (consecutive=%d)",
                run_id,
                circuit_breaker_consecutive,
                extra={
                    "component": "batch_label_service",
                    "project_id": project_id,
                },
            )
            return  # paused — not terminal, no completed_at

        # ── Phase C: finalize ───────────────────────────────────────────

        if clobber_skipped:
            logger.info(
                "Batch run %s: %d successful invocation(s) did not persist an "
                "auto-label because the SME labeled those examples mid-run",
                run_id,
                clobber_skipped,
                extra={"component": "batch_label_service", "project_id": project_id},
            )

        # All examples processed — determine terminal status.
        # "completed" means all examples reached a terminal per-example outcome.
        terminal_status = "completed"

        now = utc_now()
        with Session(engine) as session:
            # A cancel requested while the last few examples processed:
            # atomically flip only if still 'canceling'.
            if update_run_if_not_terminal(
                session,
                run_id,
                {"status": "canceled", "completed_at": now},
                only_status="canceling",
            ):
                session.commit()
                await sse_manager.emit(
                    project_id,
                    "batch_label_completed",
                    {"run_id": run_id, "status": "canceled"},
                )
                return
            # Otherwise complete — but a concurrent schema-evolution wipe
            # may have terminalized the run; the conditional update makes
            # the completion write atomic with the not-terminal check so a
            # failed row is never resurrected to 'completed'.
            if not update_run_if_not_terminal(
                session,
                run_id,
                {
                    "status": terminal_status,
                    "completed_at": now,
                    "examples_succeeded": succeeded,
                    "examples_schema_invalid": schema_invalid,
                    "examples_timeout": timed_out,
                    "examples_endpoint_error": endpoint_err,
                },
                terminal_statuses=TERMINAL_STATUSES,
            ):
                return
            session.commit()

        await sse_manager.emit(
            project_id,
            "batch_label_completed",
            {
                "run_id": run_id,
                "status": terminal_status,
                "examples_succeeded": succeeded,
                "examples_schema_invalid": schema_invalid,
                "examples_timeout": timed_out,
                "examples_endpoint_error": endpoint_err,
            },
        )

    except Exception:
        logger.exception(
            "Batch label run %s failed with unhandled exception",
            run_id,
            extra={"component": "batch_label_service", "project_id": project_id},
        )
        await finalize_unhandled_exception(
            engine,
            project_id,
            run_id,
            run_type="batch_label_run",
            error_summary="Unhandled exception during batch labeling",
            terminal_statuses=TERMINAL_STATUSES,
        )
    finally:
        if _cancel_events.get(run_id) is cancel_event:
            _cancel_events.pop(run_id, None)
        clear_runtime_rejections(run_id)


async def _finalize_canceled(
    engine: Any,
    project_id: str,
    run_id: str,
) -> bool:
    """Transition a canceling batch run to canceled and emit SSE.

    Thin wrapper over ``teacher_rejection.finalize_canceled`` (the one
    implementation shared with ``evaluation_service``) with the
    batch-label parameters bound, then pops this module's cancel-event
    registry — the shared helper never touches it.
    """
    applied = await finalize_canceled(
        engine,
        project_id,
        run_id,
        event_name="batch_label_completed",
    )
    _cancel_events.pop(run_id, None)
    return applied


# ── Resume ──────────────────────────────────────────────────────────────────


async def resume_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Resume a paused batch label run.

    Transitions paused → queued and registers a new background task.
    The executor's idempotency check skips already-processed examples.
    The circuit breaker counter restarts at 0.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    task_id = f"batch-label-{run_id}"
    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="batch_label_run")
        if isinstance(run, str):
            return run
        if run.status != "paused":
            return f"conflict: Run {run_id} is not paused (status={run.status})"

        # Same one-active-run invariant as start_batch_label_run: another run
        # could have been started (start only blocks queued/running/canceling,
        # not paused) while this one was paused. Resuming both would race to
        # write duplicate auto_labeled Labels over the overlapping snapshot.
        other_active = (
            session.query(RunRecord)
            .filter(
                RunRecord.project_id == project_id,
                RunRecord.run_type == "batch_label_run",
                RunRecord.run_id != run_id,
                RunRecord.status.in_(ACTIVE_RUN_STATUSES),
            )
            .first()
        )
        if other_active is not None:
            return (
                "conflict: Another batch labeling run is already in progress "
                "for this project. Wait for it to finish, pause, or cancel it "
                "before resuming this one."
            )

    # A circuit-breaker owner persists ``paused`` before emitting its final
    # progress event. Join that same-ID task before registering the resumed
    # executor; otherwise the manager correctly rejects the duplicate and the
    # already-committed queued transition becomes stranded until restart.
    await background_manager.cancel_task(task_id)

    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="batch_label_run")
        if isinstance(run, str):
            return run
        if run.status != "paused":
            return f"conflict: Run {run_id} is no longer paused (status={run.status})"
        other_active = (
            session.query(RunRecord)
            .filter(
                RunRecord.project_id == project_id,
                RunRecord.run_type == "batch_label_run",
                RunRecord.run_id != run_id,
                RunRecord.status.in_(ACTIVE_RUN_STATUSES),
            )
            .first()
        )
        if other_active is not None:
            return (
                "conflict: Another batch labeling run is already in progress "
                "for this project. Wait for it to finish, pause, or cancel it "
                "before resuming this one."
            )
        if not isinstance(run.metrics, dict):
            return f"conflict: Run {run_id} has no durable metadata"
        paused_reason = run.paused_reason
        previous_metrics = dict(run.metrics)
        resumed_metrics = dict(previous_metrics)
        resumed_metrics["circuit_breaker_consecutive"] = 0
        resumed_metrics["circuit_breaker_tripped"] = False

        # Atomic paused → queued: a concurrent schema-evolution wipe may
        # have failed this paused run in the window since the read above;
        # only re-dispatch if the transition actually applied.
        resumed = update_run_if_not_terminal(
            session,
            run_id,
            {
                "status": "queued",
                "paused_reason": None,
                "metrics": resumed_metrics,
            },
            only_status="paused",
        )
        session.commit()
    if not resumed:
        return f"conflict: Run {run_id} is no longer paused"

    try:
        dispatch_batch_label_run(project_id, run_id, settings)
    except Exception:
        # Registration failed before a task was created. Put the durable run
        # back into its resumable state rather than leaving queued work with no
        # owner; preserve the original breaker evidence for the next attempt.
        with Session(engine) as session:
            update_run_if_not_terminal(
                session,
                run_id,
                {
                    "status": "paused",
                    "paused_reason": paused_reason,
                    "metrics": previous_metrics,
                },
                only_status="queued",
            )
            session.commit()
        raise

    return {"run_id": run_id, "status": "queued"}


# ── Cancel ──────────────────────────────────────────────────────────────────


def signal_run_cancellation(run_id: str) -> None:
    """Set the in-process cancel event for a live batch task.

    Single owner of the event-set, for the in-file cancel path and
    cross-service cancellation alike (schema evolution fails active
    runs in the DB, then calls this after commit so the executor stops
    invoking the Teacher). The finalizer independently refuses to
    overwrite terminal rows, so a missed signal degrades to wasted
    work, never to a resurrected run.
    """
    evt = _cancel_events.get(run_id)
    if evt is not None:
        evt.set()


async def cancel_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Request cancellation of a batch label run."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="batch_label_run")
        if isinstance(run, str):
            return run
        if run.status in TERMINAL_STATUSES:
            return f"conflict: Run {run_id} already in terminal state ({run.status})"

        now = utc_now()
        # Both transitions below are atomic: a schema-evolution wipe may
        # have failed this run in the window since the read; the
        # conditional update makes the not-terminal check and the write one
        # locked statement, so a terminalized run is never resurrected to
        # canceling/canceled.
        if run.status == "paused":
            # Paused runs have no background task — go straight to canceled.
            applied = update_run_if_not_terminal(
                session,
                run_id,
                {
                    "status": "canceled",
                    "completed_at": now,
                    "cancel_requested_at": now,
                },
                only_status="paused",
            )
            if applied:
                session.commit()
                return {
                    "run_id": run_id,
                    "status": "canceled",
                    "cancel_requested_at": now,
                }
            # Not paused anymore: either resumed (live again — fall through
            # to the canceling transition, which its executor honors) or
            # terminalized (the fall-through refuses and reports it).

        applied = update_run_if_not_terminal(
            session,
            run_id,
            {"status": "canceling", "cancel_requested_at": now},
            terminal_statuses=TERMINAL_STATUSES,
        )
        session.commit()
        if not applied:
            return f"conflict: Run {run_id} is already in a terminal state"

    signal_run_cancellation(run_id)

    return {"run_id": run_id, "status": "canceling", "cancel_requested_at": now}


# ── Get / List ──────────────────────────────────────────────────────────────


def get_batch_label_run(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Load a single batch label run with full detail."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="batch_label_run")
        if isinstance(run, str):
            return run
        return _run_to_dict(run, session=session)


def list_batch_label_runs(
    project_id: str,
    *,
    status_filter: str | None = None,
    cursor: str | None = None,
    limit: int = 20,
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None]:
    """List batch label runs with cursor pagination, newest-first."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return [], None

    with Session(engine) as session:
        rows, next_cursor = list_runs_page(
            session,
            project_id=project_id,
            run_type="batch_label_run",
            status_filter=status_filter,
            cursor=cursor,
            limit=limit,
        )
        dicts = [_run_to_dict(r, session=session) for r in rows]
    return dicts, next_cursor


def _aggregate_common_errors(
    session: Session, run_id: str, top_n: int = 3
) -> list[dict[str, Any]]:
    """Compute top-N common error signatures for a batch labeling run.

    Groups OperationRecord failures by error signature and returns the most
    frequent ones for the Batch Labeling screen's "Common errors:" display. Returns ``[]`` on
    any unexpected shape in historical records (defensive: a malformed old run
    should never prevent status from loading).
    """
    try:
        failed_ops = (
            session.query(OperationRecord)
            .filter(
                OperationRecord.batch_label_run_id == run_id,
                OperationRecord.invocation_status != "success",
                OperationRecord.ignored_due_to_run_cancellation == False,  # noqa: E712
            )
            .all()
        )
        buckets: dict[str, dict[str, Any]] = {}
        for op in failed_ops:
            code, sample = _classify_error_signature(op)
            if code is None:
                continue
            if code not in buckets:
                buckets[code] = {"code": code, "count": 0, "sample": sample}
            buckets[code]["count"] += 1

        ranked = sorted(buckets.values(), key=lambda b: b["count"], reverse=True)
        return ranked[:top_n]
    except Exception:  # aggregation must never break status load
        logger.exception("common_errors aggregation failed for run %s", run_id)
        return []


def _classify_error_signature(
    op: OperationRecord,
) -> tuple[str | None, str | None]:
    """Return a (code, sample_message) tuple describing an error signature."""
    status = op.invocation_status
    if status == "schema_invalid":
        errors = op.validation_errors_core
        first_err: str | None = None
        if isinstance(errors, list) and errors:
            first_err = errors[0]
        elif isinstance(errors, dict) and errors:
            # Legacy shape safety — take first JSON-serializable value.
            errors_dict = cast("dict[str, Any]", errors)
            first_val: Any = next(iter(errors_dict.values()), None)
            first_err = first_val if isinstance(first_val, str) else None
        # Group by field name if the error starts with "<field>:"
        field_key = "schema_invalid"
        if first_err and ":" in first_err:
            field_key = f"schema_invalid:{first_err.split(':', 1)[0].strip()}"
        return field_key, first_err
    if status == "timeout":
        return "timeout", "Request timed out before the model responded."
    if status == "endpoint_error":
        return "endpoint_error", "Could not reach the NIM endpoint."
    if status == "structured_generation_rejected":
        return (
            "structured_generation_rejected",
            "The model rejected json_schema output for this run.",
        )
    if status in ("error", "pending", "canceled"):
        return None, None  # Don't surface these as user-facing failures
    return status, None


def _resolve_model_name(session: Session, model_config_id: str | None) -> str | None:
    """Look up the displayable `model_name` for a batch run's snapshotted config."""
    if not model_config_id:
        return None
    mc = session.get(ModelConfig, model_config_id)
    return mc.model_name if mc is not None else None


def _resolve_guidance_version(session: Session, guidance_id: str | None) -> int | None:
    """Look up the integer `version_number` for a batch run's snapshotted guidance."""
    if not guidance_id:
        return None
    g = session.get(Guidance, guidance_id)
    return g.version_number if g is not None else None


def _run_to_dict(run: RunRecord, *, session: Session) -> dict[str, Any]:
    """Convert a batch label RunRecord to the API response dict.

    Resolves ``model_config_id`` → ``model_name`` and ``guidance_id`` →
    ``guidance_version_number`` so the Batch Labeling screen's "Config:" line
    renders human-readable values instead of raw UUIDs.
    """
    processed = (
        (run.examples_succeeded or 0)
        + (run.examples_schema_invalid or 0)
        + (run.examples_timeout or 0)
        + (run.examples_endpoint_error or 0)
    )
    progress: dict[str, int] | None = None
    if run.status in ("queued", "running", "paused") and run.examples_total:
        progress = {"processed": processed, "total": run.examples_total}

    return {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "status": run.status,
        "status_reason": run.status_reason,
        "paused_reason": run.paused_reason,
        "circuit_breaker_threshold": (
            run.metrics.get("circuit_breaker_threshold")
            if isinstance(run.metrics, dict)
            else None
        ),
        "guidance_id": run.guidance_id,
        "guidance_version_number": _resolve_guidance_version(session, run.guidance_id),
        "model_config_id": run.model_config_id,
        "model_name": _resolve_model_name(session, run.model_config_id),
        "generation_preset_key": run.generation_preset_key,
        "thinking_mode_effective": run.thinking_mode_effective,
        "visual_budget_preset_key": run.visual_budget_preset_key,
        "structured_generation_mode_effective": run.structured_generation_mode_effective,
        "icl_mode": run.icl_mode,
        "progress": progress,
        "examples_succeeded": run.examples_succeeded or 0,
        "examples_schema_invalid": run.examples_schema_invalid or 0,
        "examples_timeout": run.examples_timeout or 0,
        "examples_endpoint_error": run.examples_endpoint_error or 0,
        "examples_total": run.examples_total or 0,
        "common_errors": _aggregate_common_errors(session, run.run_id),
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "cancel_requested_at": run.cancel_requested_at,
        "recovered_from_restart": run.recovered_from_restart,
    }
