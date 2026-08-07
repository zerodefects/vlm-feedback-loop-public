# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation run orchestration service.

Implements:
  - Evaluation run lifecycle: create, execute (background), cancel, get, list.
  - 7-status state machine: queued/running/canceling/completed/
    incomplete/canceled/failed.
  - Concurrent inference against the Test Pool, provider-aware:
    EVAL_CONCURRENCY_HOSTED / EVAL_CONCURRENCY_SELF_HOSTED.
  - Sequential retry pass for failed examples.
  - Aggregate metrics via the canonical Exact Match evaluator.
  - Returning vs New pool comparison.
  - Coverage gap detection.
  - Trigger status computation and dismissal.
  - SSE event production: evaluation_started, evaluation_progress,
    evaluation_completed, run_failed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.schemas.inference_contract import TEACHER_CONTRACT
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.exact_match_evaluator import (
    AggregateMetrics,
    FieldMatchResult,
    compute_aggregate_metrics,
    match_fields,
    normalize_ground_truth,
    validate_proposal,
)
from vlm_feedback_loop.services.icl_service import (
    count_icl_eligible_edits,
    query_icl_candidates,
)
from vlm_feedback_loop.services.invocation_outcome import (
    apply_invocation_outcome,
    classify_invocation_status,
    write_invocation_artifacts,
)
from vlm_feedback_loop.services.nim_client import build_endpoint_auth_headers
from vlm_feedback_loop.services.pool_service import (
    assess_pool_class_coverage,
    create_pool_snapshot,
)
from vlm_feedback_loop.services.priority import priority_dispatch
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.prompt_service import (
    ModelConfigInput,
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
    clear_runtime_rejections,
    finalize_canceled,
    finalize_runtime_rejection,
    finalize_unhandled_exception,
    mark_operation_ignored_if_canceling,
    record_runtime_rejections,
)

logger = logging.getLogger("vlm_feedback_loop.evaluation_service")

# ── Module-level state ──────────────────────────────────────────────────────

# Per-run cancellation events.  Populated when a run enters ``running``,
# cleared on terminal state.
_cancel_events: dict[str, asyncio.Event] = {}

# Auto-Evaluate may be considered by several short mutation paths (label save,
# project settings, and Guidance edits). Serialize its read-trigger/start
# sequence per project so two near-simultaneous mutations cannot create two
# automatic runs from the same persisted trigger.
_auto_start_locks: dict[str, asyncio.Lock] = {}

# The per-run runtime-rejection registries and their record/finalize
# lifecycle live in ``services.teacher_rejection`` — one implementation
# shared with ``batch_label_service``.

TERMINAL_STATUSES = frozenset({"completed", "incomplete", "canceled", "failed"})


# ── Data classes ────────────────────────────────────────────────────────────


@dataclass
class EvalExampleResult:
    """Result of evaluating a single pool example."""

    example_key: str
    invocation_id: str
    invocation_status: str  # success | schema_invalid | timeout | endpoint_error
    proposal_json: dict[str, Any] | None
    schema_valid_core: bool
    field_matches: list[FieldMatchResult] | None
    exact_match_pass: bool | None
    ignored_due_to_run_cancellation: bool = False


def _resolve_eval_concurrency(
    endpoint_mode: str | None,
    settings: Settings,
    *,
    explicit_override: int | None = None,
) -> int:
    """Pick effective Phase B semaphore size, mirroring the CLIP embedding worker's pattern.

    - ``explicit_override`` (per-run body field) wins when provided. Used by
      the ICL diag harness for rate-limit gating.
    - Hosted endpoints (build.nvidia.com) → ``EVAL_CONCURRENCY_HOSTED``
      (default 1). Stays polite under shared per-account RPM caps.
    - Self-hosted / local NIMs → ``EVAL_CONCURRENCY_SELF_HOSTED`` (default 8).
      No shared rate limit; saturate the GPU pipeline.
    """
    if explicit_override is not None and explicit_override > 0:
        return explicit_override
    if endpoint_mode == "hosted":
        return settings.EVAL_CONCURRENCY_HOSTED
    # ``self_hosted`` and ``local_system_managed`` both treated as self-hosted
    # for rate-limit purposes (no per-account RPM cap).
    return settings.EVAL_CONCURRENCY_SELF_HOSTED


def _limit_icl_candidate_prefix(candidates: list[Any], limit: int | None) -> list[Any]:
    """Keep the first ``limit`` corrections in deterministic loop order.

    Candidate timestamps have one-second precision, so ``example_key`` is the
    canonical tie-break. The returned list preserves the selector contract of
    newest timestamp first and ascending key within equal timestamps.
    """
    if limit is None or len(candidates) <= limit:
        return candidates
    oldest_first = sorted(
        candidates, key=lambda candidate: (candidate.labeled_at, candidate.example_key)
    )
    first_n = oldest_first[:limit]
    # Python sorting is stable: establish ascending key order first, then put
    # timestamps newest-first without reversing equal-timestamp keys.
    first_n = sorted(first_n, key=lambda candidate: candidate.example_key)
    return sorted(first_n, key=lambda candidate: candidate.labeled_at, reverse=True)


async def _invoke_for_evaluation(
    project_id: str,
    run_id: str,
    example_key: str,
    *,
    run_config: dict[str, Any],
    engine: Any,
    settings: Settings,
    retry_of_inference_invocation_id: str | None = None,
) -> EvalExampleResult:
    """Invoke the Teacher for one pool example and score the result.

    Mirrors ``proposal_service.create_proposal``'s invocation pipeline, with
    four evaluation-specific differences:

    - ``purpose="evaluation"`` and ``scope_id=run_id`` so ``invoke_teacher``
      injects a deterministic per-example seed.
    - Uses ``HTTP_DEADLINE_BACKGROUND_S`` (not interactive).
    - The persisted ``OperationRecord`` carries ``evaluation_run_id=run_id``.
    - After schema validation, the normalized JSON is compared against
      ``ground_truth[example_key]`` via :func:`match_fields` so the
      record's ``exact_match_pass`` and ``field_matches`` are authoritative.

    Seam-level unit tests monkeypatch this function entirely; the signature +
    return type are their contract. The wire-mock end-to-end tests
    (``TestProfileBProductionPipeline``) mock
    ``nim_client.chat_completions`` at the lowest level and let this
    pipeline run end-to-end against the real NIM call path.
    """
    inference_invocation_id = generate_uuid4()

    guidance_id: str = run_config["guidance_id"]
    guidance_fields: list[dict[str, Any]] = run_config["guidance_fields"]
    model_config_id: str = run_config["model_config_id"]
    endpoint_id: str = run_config["endpoint_id"]
    model_name: str = run_config["model_name"]
    mc_input: ModelConfigInput = run_config["mc_input"]
    project_dir: str = run_config["project_dir"]
    ground_truth: dict[str, dict[str, Any]] = run_config["ground_truth"]
    storage_refs: dict[str, str] = run_config["storage_refs"]
    icl_mode: str = run_config["icl_mode"]
    storage_ref = storage_refs.get(example_key)

    # ── 1. Persist pending OperationRecord BEFORE the NIM call ──────────
    ignored_due_to_run_cancellation = False
    with Session(engine) as session:
        session.add(
            OperationRecord(
                inference_invocation_id=inference_invocation_id,
                project_id=project_id,
                purpose="evaluation",
                example_key=example_key,
                guidance_id=guidance_id,
                model_config_id=model_config_id,
                endpoint_id=endpoint_id,
                model_name=model_name,
                invocation_status="pending",
                label_tier="proposal",
                evaluation_run_id=run_id,
                retry_of_inference_invocation_id=retry_of_inference_invocation_id,
            )
        )
        session.commit()

    # ── 2. Query ICL candidates (empty when icl_mode="disabled") ─────────
    icl_candidates: list[Any] = []
    if icl_mode != "disabled":
        with Session(engine) as session:
            icl_candidates = query_icl_candidates(session, project_id, guidance_id)
        # Pool-growth simulation (diagnostic): restrict the candidate POOL to
        # the first ``icl_candidate_limit`` Edits in stable TEMPORAL (loop)
        # order = the first N Edits the SME verified, by ``Label.labeled_at``
        # ascending (the canonical creation-order column; NOT NULL, sortable
        # ISO-8601 string, and the same column query_icl_candidates already
        # orders by). This caps the POOL upstream of selection, so the
        # downstream relevance/diversity top-K in invoke_teacher picks from
        # only those N. We then restore newest-first order (labeled_at DESC)
        # to honor select_icl_examples' "already sorted newest-first" contract.
        # None (default) leaves the full pool untouched.
        icl_candidates = _limit_icl_candidate_prefix(
            icl_candidates, run_config.get("icl_candidate_limit")
        )

    auth_headers = build_endpoint_auth_headers(
        run_config["endpoint_auth_mode"],
        get_effective_secret("NVIDIA_API_KEY", settings),
    )

    # ── 3. Call invoke_teacher ───────────────────────────────────────────
    # Run-level structured-gen mode is baked into ``mc_input`` at snapshot
    # time (prompt_only → structured_generation_support="unsupported"). If
    # this invocation hits a mid-run response_format rejection under
    # ``auto`` mode, the whole run must fail — handled after
    # the call via ``_structured_gen_rejected`` + cancel_event signaling.
    teacher_result = await invoke_teacher(
        project_id=project_id,
        example_key=example_key,
        purpose="evaluation",
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
    # Stage 4 timing — schema validation + normalization + field
    # matching. Persisted alongside the other stage timings on the
    # OperationRecord.
    _validation_t0 = time.monotonic()
    validation_report = validate_proposal(
        teacher_result.content,
        guidance_fields,
        teacher_result.finish_reason,
    )

    # ── 5. Score against ground truth when schema-valid ──────────────────
    truth = ground_truth.get(example_key)
    field_matches: list[FieldMatchResult] | None = None
    exact_match_pass: bool | None = None
    if (
        validation_report.schema_valid_core
        and validation_report.normalized_json is not None
        and truth is not None
    ):
        field_matches = match_fields(
            validation_report.normalized_json,
            truth,
            guidance_fields,
        )
        exact_match_pass = all(r.matched for r in field_matches)
    t_validation_ms = int((time.monotonic() - _validation_t0) * 1000)

    # ── 6. Final invocation_status classification ────────────────────────
    final_status = classify_invocation_status(teacher_result, validation_report)

    # ── 7. Write artifact files ──────────────────────────────────────────
    artifacts_dir = Path(project_dir) / "artifacts"
    artifact_refs = write_invocation_artifacts(
        artifacts_dir,
        inference_invocation_id,
        teacher_result=teacher_result,
        validation_report=validation_report,
    )

    # ── 8. Update the OperationRecord with all outcome fields ────────────
    with Session(engine) as session:
        record = (
            session.query(OperationRecord)
            .filter_by(inference_invocation_id=inference_invocation_id)
            .first()
        )
        if record is not None:
            apply_invocation_outcome(
                record,
                final_status=final_status,
                teacher_result=teacher_result,
                validation_report=validation_report,
                t_validation_ms=t_validation_ms,
                artifact_refs=artifact_refs,
                structured_generation_mode_effective=run_config["sgm_effective"],
                structured_generation_fallback_used=(
                    teacher_result.structured_generation_fallback_used
                ),
                # Eval never auto-retries a rejected capability mid-run
                # (mode flips break reproducibility) — run-level
                # finalization via ``_maybe_finalize_runtime_rejection``
                # handles the rejected run instead, so the per-invocation
                # fallback flags are always False here.
                thinking_fallback_used=False,
                visual_budget_fallback_used=False,
                exact_match_pass=exact_match_pass,
            )
            ignored_due_to_run_cancellation = mark_operation_ignored_if_canceling(
                session, record, run_id
            )
            session.commit()

    # The outcome transaction orders the invocation against a durable user
    # cancellation. Only authoritative outcomes may fail the whole run for a
    # capability rejection; ignored audit outcomes leave cancellation in
    # control of the terminal state.
    if not ignored_due_to_run_cancellation:
        record_runtime_rejections(
            run_id,
            teacher_result,
            sgm_effective=run_config["sgm_effective"],
            settings=settings,
            cancel_event=_cancel_events.get(run_id),
        )

    return EvalExampleResult(
        example_key=example_key,
        invocation_id=inference_invocation_id,
        invocation_status=final_status,
        proposal_json=validation_report.normalized_json,
        schema_valid_core=validation_report.schema_valid_core,
        field_matches=field_matches,
        exact_match_pass=exact_match_pass,
        ignored_due_to_run_cancellation=ignored_due_to_run_cancellation,
    )


# ── Start evaluation run ────────────────────────────────────────────────────


async def start_evaluation_run(
    project_id: str,
    *,
    icl_mode: str = "enabled",
    structured_generation_mode: str | None = None,
    icl_max_examples: int | None = None,
    icl_candidate_limit: int | None = None,
    eval_concurrency: int | None = None,
    icl_sim_gap: float | None = None,
    icl_abs_threshold: float | None = None,
    generation_preset_key: str | None = None,
    thinking_on: bool | None = None,
    visual_budget_preset_key: str | None = None,
    target_model_config_id: str | None = None,
    target_inference_contract: dict[str, Any] | None = None,
    settings: Settings,
    # ── Provenance kwargs ─────────────────────────────────────────────────
    student_model_config_id: str | None = None,
    nim_model_profile_requested: str | None = None,
    nim_model_profile_selected: str | None = None,
    nim_profile_metadata: dict[str, Any] | None = None,
    quantization_method: str | None = None,
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    dataset_manifest_sha256: str | None = None,
) -> dict[str, Any] | str:
    """Create a new evaluation run, snapshot config, and register background task.

    By default the run targets the project's active Teacher model with the
    fixed Teacher Inference Contract (active output schema, Core-only ICL).
    The output includes ``rationale_note`` only when active Guidance enables it.
    Two optional overrides are supported:

      - ``target_model_config_id``: project-scoped ModelConfig the run should
        invoke instead of the active Teacher.  Used by the Student NIM
        deployment lifecycle (services/student_nim_lifecycle.py) so the
        same evaluation pipeline can be pointed at a temporary Student
        endpoint without flipping ``project.teacher_model_config_id`` and
        breaking concurrent Interactive Labeling.
      - ``target_inference_contract``: serialized Inference Contract dict
        snapshotted onto the RunRecord instead of TEACHER_CONTRACT.
        Lets a ``core_only``-trained Student be evaluated under the same
        field-mode contract it was trained against.

    Both overrides are project-scoped and validated.  When both are None,
    the existing Teacher behavior is preserved exactly.

    Optional provenance kwargs are persisted
    verbatim onto the RunRecord (suite linkage + Student-variant identity +
    NIM profile + quantization + GPU + dataset manifest SHA-256). They are
    inputs only — no validation here beyond accepting the values; call sites
    (suite service, NIM lifecycle) are responsible for sourcing them from
    the StudentModel snapshot at fan-out time.

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
        # When a target_model_config_id is provided, validate it exists in
        # this project — that's the model the run will use.  Otherwise
        # fall back to the active Teacher.
        if target_model_config_id is not None:
            target_mc = (
                session.query(ModelConfig)
                .filter_by(
                    project_id=project_id, model_config_id=target_model_config_id
                )
                .first()
            )
            if target_mc is None:
                return f"not found: ModelConfig {target_model_config_id}"
        elif not project.teacher_model_config_id:
            return "No Teacher model configured"
        snap_model_config_id = target_model_config_id or project.teacher_model_config_id
        assert snap_model_config_id is not None

        snap_gen_preset = (
            generation_preset_key or project.labeling_generation_preset_key
        )
        snap_vb_preset = visual_budget_preset_key or project.visual_budget_preset_key

        # ── Validate request BEFORE any side effects ────────────────────
        # The supersede step below sets an in-flight run's in-memory cancel
        # event, which a DB rollback cannot undo. Reject an invalid request
        # here so a bad preset key can never cancel a running evaluation
        # without starting a replacement.
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

        contract_dict = dict(target_inference_contract or TEACHER_CONTRACT.model_dump())
        contract_icl_max = contract_dict.get("icl_max_examples")
        effective_icl_max = (
            icl_max_examples
            if icl_max_examples is not None
            else (
                contract_icl_max
                if contract_icl_max is not None
                else settings.ICL_MAX_EXAMPLES
            )
        )
        contract_dict["icl_max_examples"] = effective_icl_max
        effective_icl_sim_gap = (
            icl_sim_gap if icl_sim_gap is not None else settings.ICL_SIM_GAP
        )
        effective_icl_abs_threshold = (
            icl_abs_threshold
            if icl_abs_threshold is not None
            else settings.ICL_ABS_THRESHOLD
        )
        runtime_config_snapshot = create_runtime_config_snapshot(
            session,
            project_id,
            snap_model_config_id,
            settings=settings,
            generation_preset_key=snap_gen_preset,
            visual_budget_preset_key=snap_vb_preset,
            icl_max_examples=effective_icl_max,
            icl_candidate_limit=icl_candidate_limit,
            icl_sim_gap=effective_icl_sim_gap,
            icl_abs_threshold=effective_icl_abs_threshold,
        )

        # ── Pool snapshot ───────────────────────────────────────────────
        pool = create_pool_snapshot(
            session,
            project_id,
            project.active_guidance_id,
        )
        if pool is None:
            return "Pool is empty; cannot start evaluation"

        # ── ICL eligible count at start ─────────────────────────────────
        icl_count = count_icl_eligible_edits(
            session, project_id, project.active_guidance_id
        )

        # ── Supersede any running gate-basis evaluation ─────────────────
        # Newest-config-wins (§7.1) exists so a gate-basis run can never
        # certify a configuration the SME has since changed. Student
        # benchmark runs (student_model_config_id set, §9.5.2) snapshot
        # immutable configs at creation, so staleness cannot apply to
        # them: they neither fire the supersede nor receive it. Before
        # this scoping, a Student rotation silently killed a running
        # Teacher baseline (and vice versa), wasting the whole run —
        # canceled runs never produce authoritative metrics (§13.2).
        superseded_run_id = None
        if student_model_config_id is None:
            running_runs = (
                session.query(RunRecord)
                .filter(
                    RunRecord.project_id == project_id,
                    RunRecord.run_type == "evaluation_run",
                    RunRecord.student_model_config_id.is_(None),
                    # Deliberately narrower than ACTIVE_RUN_STATUSES: a run
                    # already "canceling" needs no re-cancel.
                    RunRecord.status.in_(["queued", "running"]),
                )
                .all()
            )
            for rr in running_runs:
                # Atomic supersede flip: skip a run another writer terminalized.
                if not update_run_if_not_terminal(
                    session,
                    rr.run_id,
                    {
                        "status": "canceling",
                        "status_reason": "superseded_by_newer_evaluation",
                        "cancel_requested_at": utc_now(),
                    },
                    terminal_statuses=TERMINAL_STATUSES,
                ):
                    continue
                superseded_run_id = rr.run_id
                # Signal the cancel event if the run is in-flight
                signal_run_cancellation(rr.run_id)

        # ── Snapshot config ─────────────────────────────────────────────
        effective_sgm = (
            structured_generation_mode or project.structured_generation_mode_default
        )
        # Thinking is snapshotted; a per-run override wins over the project
        # default. Incapable models (thinking_toggle_mode none/always_on) no-op
        # this at resolve time (prompt_service.resolve_thinking_fields).
        if thinking_on is not None:
            thinking_mode = "on" if thinking_on else "off"
        else:
            thinking_mode = "on" if project.thinking_default_on else "off"

        run_id = generate_uuid4()
        now = utc_now()

        # Capture scalars before session closes (avoid DetachedInstanceError)
        pool_version = pool.pool_version
        pool_id = pool.pool_id
        pool_member_count = pool.member_count
        snap_guidance_id = project.active_guidance_id
        run_record = RunRecord(
            run_id=run_id,
            project_id=project_id,
            run_type="evaluation_run",
            status="queued",
            pool_version_id=pool_id,
            guidance_id=snap_guidance_id,
            model_config_id=snap_model_config_id,
            icl_mode=icl_mode,
            evaluation_source="nim",
            generation_preset_key=snap_gen_preset,
            thinking_mode_effective=thinking_mode,
            visual_budget_preset_key=snap_vb_preset,
            structured_generation_mode_effective=effective_sgm,
            inference_contract=contract_dict,
            runtime_config_snapshot=runtime_config_snapshot,
            icl_eligible_count_at_start=icl_count,
            examples_total=pool_member_count,
            # Evaluation provenance
            student_model_config_id=student_model_config_id,
            nim_model_profile_requested=nim_model_profile_requested,
            nim_model_profile_selected=nim_model_profile_selected,
            nim_profile_metadata=nim_profile_metadata,
            quantization_method=quantization_method,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            dataset_manifest_sha256=dataset_manifest_sha256,
        )
        session.add(run_record)
        session.commit()

    # ── Register background task ────────────────────────────────────────
    background_manager.register(
        task_id=f"eval-run-{run_id}",
        coro=_execute_evaluation(
            project_id,
            run_id,
            settings,
            eval_concurrency=eval_concurrency,
        ),
    )

    return {
        "run_id": run_id,
        "run_type": "evaluation_run",
        "status": "queued",
        "pool_version": pool_version,
        "guidance_id": snap_guidance_id,
        "model_config_id": snap_model_config_id,
        "generation_preset_key": snap_gen_preset,
        "thinking_mode_effective": thinking_mode,
        "visual_budget_preset_key": snap_vb_preset,
        "structured_generation_mode_effective": effective_sgm,
        "evaluation_source": "nim",
        "icl_mode": icl_mode,
        "created_at": now,
        "superseded_run_id": superseded_run_id,
    }


# ── Background evaluation execution ────────────────────────────────────────


def _persist_progress_counters(
    engine: Engine, run_id: str, results: dict[str, EvalExampleResult]
) -> None:
    """Persist per-status example counters at the SSE progress cadence.

    REST is authoritative: ``_run_to_dict`` derives ``progress.processed``
    from these RunRecord columns, so without mid-run writes every REST
    read (page load, SSE reconnect, polling scripts) reports 0 until the
    run terminalizes. Phase G uses the same per-example result map at
    finalization; retried examples simply overwrite their bucket here.
    Outcomes classified as cancellation-ignored never enter this map.
    """
    succeeded = schema_invalid = timed_out = endpoint_err = 0
    for r in results.values():
        if r.invocation_status == "success":
            succeeded += 1
        elif r.invocation_status == "schema_invalid":
            schema_invalid += 1
        elif r.invocation_status == "timeout":
            timed_out += 1
        else:
            # endpoint_error and rate_limited share a bucket, mirroring
            # the batch-label loop's counter reconstruction.
            endpoint_err += 1
    with Session(engine) as session:
        # Never write counters onto a terminalized run (a schema-evolution
        # cancel), consistent with the finalizer's terminal guard.
        update_run_if_not_terminal(
            session,
            run_id,
            {
                "examples_succeeded": succeeded,
                "examples_schema_invalid": schema_invalid,
                "examples_timeout": timed_out,
                "examples_endpoint_error": endpoint_err,
            },
            terminal_statuses=TERMINAL_STATUSES,
        )
        session.commit()


async def _execute_evaluation(
    project_id: str,
    run_id: str,
    settings: Settings,
    *,
    eval_concurrency: int | None = None,
) -> None:
    """Background coroutine that runs the evaluation pipeline.

    The semantic diagnostic controls are persisted in the run's runtime
    snapshot. The sole runtime-only override is execution width; evaluation
    runs fail rather than resume after a backend restart.

    - ``eval_concurrency``: explicit override of the provider-aware
      ``EVAL_CONCURRENCY_HOSTED`` / ``EVAL_CONCURRENCY_SELF_HOSTED`` default.
      Hosted endpoints already default to 1 (under the shared per-account
      RPM cap); this is for advanced use only.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return  # project gone; nothing to do

    cancel_event = asyncio.Event()
    _cancel_events[run_id] = cancel_event

    try:
        # ── Phase A: queued → running ───────────────────────────────────
        with Session(engine) as session:
            run = session.query(RunRecord).filter_by(run_id=run_id).first()
            if run is None or run.status in TERMINAL_STATUSES:
                return
            # May already be canceling (superseded before execution starts):
            # atomically flip only if still 'canceling'.
            if update_run_if_not_terminal(
                session,
                run_id,
                {"status": "canceled", "completed_at": utc_now()},
                only_status="canceling",
            ):
                session.commit()
                return
            pool_version_id = run.pool_version_id
            total = run.examples_total
            # Atomic queued → running claim: the UPDATE re-checks the
            # status under the write lock, so a schema-evolution wipe that
            # terminalized the run cannot be resurrected into a live run.
            if not update_run_if_not_terminal(
                session,
                run_id,
                {"status": "running", "started_at": utc_now()},
                terminal_statuses=TERMINAL_STATUSES | {"canceling"},
            ):
                return
            # A run created concurrently with a guidance edit can commit
            # after the edit's cancel sweep ran — the one writer the sweep
            # cannot see. The claim UPDATE holds the write lock, so this
            # read is post-edit truth: a mismatch means the run would score
            # a retired schema and it fails exactly as the sweep would
            # have failed it.
            active_gid_now = (
                session.query(Project.active_guidance_id)
                .filter_by(project_id=project_id)
                .scalar()
            )
            if active_gid_now != run.guidance_id:
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

        await sse_manager.emit(
            project_id,
            "evaluation_started",
            {
                "run_id": run_id,
                "pool_version_id": pool_version_id,
                "total": total,
            },
        )

        # ── Load pool snapshot + ground truth + schema ──────────────────
        # We capture *everything* that _invoke_for_evaluation needs as plain
        # scalars before Phase B so the async tasks never touch ORM objects
        # across the session boundary. The run's snapshotted configuration
        # (generation_preset_key, thinking_mode_effective, etc.) is the
        # source of truth: mid-run project edits MUST NOT affect an
        # in-flight evaluation.
        with Session(engine) as session:
            pool = session.query(Pool).filter_by(pool_id=pool_version_id).first()
            if pool is None:
                raise RuntimeError(f"Pool snapshot {pool_version_id} not found")
            member_keys = list(pool.member_example_keys)

            # Ground truth labels for the pool members
            ground_truth: dict[str, dict[str, Any]] = {}
            labels = (
                session.query(Label)
                .filter(
                    Label.project_id == project_id,
                    Label.label_status == "verified",
                    Label.example_key.in_(member_keys),
                )
                .all()
            )
            for lbl in labels:
                ground_truth[lbl.example_key] = lbl.label_json

            # Re-read the run record for its snapshotted config fields
            run = session.query(RunRecord).filter_by(run_id=run_id).first()
            if run is None:
                raise RuntimeError(f"Run {run_id} vanished during execute")
            icl_mode = run.icl_mode or "enabled"

            run_config: dict[str, Any] = snapshot_run_config(
                session,
                project_id,
                run,
                example_keys=member_keys,
                settings=settings,
            )
            run_config.update(
                {
                    "icl_mode": icl_mode,
                    "ground_truth": ground_truth,
                }
            )
            guidance_fields: list[dict[str, Any]] = run_config["guidance_fields"]
            # Legacy Verified rows may predate the save-time normalization
            # boundary; match_fields assumes canonical values on both
            # sides, so canonicalize the ground truth once per run (the
            # dict is shared with run_config["ground_truth"]).
            for gt_key in list(ground_truth):
                ground_truth[gt_key] = normalize_ground_truth(
                    ground_truth[gt_key], guidance_fields
                )

        # ── Phase B: concurrent inference ───────────────────────────────
        # Provider-aware concurrency (mirrors the CLIP worker). Per-run override
        # via ``eval_concurrency`` body field wins when set; otherwise pick
        # by the Teacher endpoint's mode (hosted → 1, self-hosted → 8).
        effective_concurrency = _resolve_eval_concurrency(
            run_config.get("endpoint_mode"),
            settings,
            explicit_override=eval_concurrency,
        )
        logger.info(
            "eval_phase_b_concurrency run_id=%s endpoint_mode=%s "
            "effective_concurrency=%d explicit_override=%s",
            run_id,
            run_config.get("endpoint_mode"),
            effective_concurrency,
            eval_concurrency,
        )
        semaphore = asyncio.Semaphore(effective_concurrency)
        results: dict[str, EvalExampleResult] = {}
        progress_interval = max(1, total // 20)  # ~5% increments

        async def _eval_one(example_key: str) -> None:
            if cancel_event.is_set():
                return
            await priority_dispatch.wait_for_background()
            async with semaphore:
                if cancel_event.is_set():
                    return
                result = await _invoke_for_evaluation(
                    project_id,
                    run_id,
                    example_key,
                    run_config=run_config,
                    engine=engine,
                    settings=settings,
                )
                if result.ignored_due_to_run_cancellation:
                    return
                results[example_key] = result
                processed = len(results)

                if processed % progress_interval == 0 or processed == total:
                    _persist_progress_counters(engine, run_id, results)
                    await sse_manager.emit(
                        project_id,
                        "evaluation_progress",
                        {
                            "run_id": run_id,
                            "processed": processed,
                            "total": total,
                        },
                    )

        tasks = [asyncio.create_task(_eval_one(key)) for key in member_keys]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        # Inspect ``gather`` return value for unhandled exceptions raised
        # inside ``_eval_one`` that DIDN'T route through the side-channels
        # (``_structured_gen_rejected`` registry, ``cancel_event``).
        # Without this inspection, an exception raised before
        # ``results[example_key] = result`` would silently drop the
        # example from ``results``, and Phase C retry wouldn't pick it up
        # because Phase C iterates ``results.items()``. Synthesize a
        # per-example ``endpoint_error`` outcome so the example is
        # retried sequentially in Phase C.
        for key, outcome in zip(member_keys, gathered, strict=True):
            if (
                isinstance(outcome, BaseException)
                and key not in results
                and not cancel_event.is_set()
            ):
                logger.error(
                    "Eval task raised unhandled %s for %s in run %s — "
                    "synthesizing endpoint_error so Phase C retries it",
                    type(outcome).__name__,
                    key,
                    run_id,
                )
                results[key] = EvalExampleResult(
                    example_key=key,
                    invocation_id="",
                    invocation_status="endpoint_error",
                    proposal_json=None,
                    schema_valid_core=False,
                    field_matches=None,
                    exact_match_pass=None,
                )

        # Check cancellation after concurrent burst. The runtime-rejection
        # finalizer (structured-gen, thinking, visual-budget) takes
        # precedence — those terminal states are ``failed`` with a
        # specific status_reason, not ``canceled``.
        if await _maybe_finalize_runtime_rejection(engine, project_id, run_id):
            return
        if cancel_event.is_set():
            _persist_progress_counters(engine, run_id, results)
            await _finalize_canceled(engine, project_id, run_id)
            return

        # ── Phase C: sequential retry ───────────────────────────────────
        # ``rate_limited`` is the 429-exhausted variant of ``endpoint_error``;
        # both are retryable in Phase C (concurrency=1 mitigates rate-limit
        # pressure naturally compared to the concurrent burst).
        async def _retry_invoke(
            key: str,
            retry_of_inference_invocation_id: str | None,
        ) -> EvalExampleResult:
            """One sequential retry of an example, with the same protection as
            the Phase B ``gather`` inspection: a retry that raises generically
            MUST mark the example as persistently failed (so the run finalizes
            ``incomplete``), not propagate up and finalize ``failed``. Phase D's
            persistent-failure list handles endpoint_error/timeout/rate_limited
            as "could not be evaluated" examples."""
            try:
                return await _invoke_for_evaluation(
                    project_id,
                    run_id,
                    key,
                    run_config=run_config,
                    engine=engine,
                    settings=settings,
                    retry_of_inference_invocation_id=(
                        retry_of_inference_invocation_id or None
                    ),
                )
            except BaseException as exc:
                logger.error(
                    "Eval Phase C retry raised unhandled %s for %s in run %s — "
                    "marking endpoint_error so run finalizes incomplete",
                    type(exc).__name__,
                    key,
                    run_id,
                )
                return EvalExampleResult(
                    example_key=key,
                    invocation_id="",
                    invocation_status="endpoint_error",
                    proposal_json=None,
                    schema_valid_core=False,
                    field_matches=None,
                    exact_match_pass=None,
                )

        failed_keys = [
            key
            for key, r in results.items()
            if r.invocation_status in ("timeout", "endpoint_error", "rate_limited")
        ]
        for key in failed_keys:
            if cancel_event.is_set():
                break
            await priority_dispatch.wait_for_background()
            if cancel_event.is_set():
                break
            prior_invocation_id = results[key].invocation_id
            retry_result = await _retry_invoke(key, prior_invocation_id)
            if cancel_event.is_set() or retry_result.ignored_due_to_run_cancellation:
                break
            results[key] = retry_result
            _persist_progress_counters(engine, run_id, results)
            # A retry replaces an existing outcome; len(results) is the
            # number of examples with an outcome.
            await sse_manager.emit(
                project_id,
                "evaluation_progress",
                {
                    "run_id": run_id,
                    "processed": len(results),
                    "total": total,
                },
            )

        # ── Phase C2: bounded multi-pass retry for rate-limited examples ──
        # A hosted 429 (``rate_limited``) is transient — the per-model quota
        # recovers — unlike ``schema_invalid`` (a real model failure) or a
        # ``timeout``/``endpoint_error`` that may be a dead endpoint. The single
        # pass above is often not enough under sustained quota pressure, leaving
        # the example persistently failed and the eval ``incomplete``. That
        # corrupts the Returning-vs-New regression signal, which depends on the
        # SAME pool images being scored each run. So retry the
        # still-rate-limited subset a bounded number of extra passes, each after
        # a backoff (letting the adaptive throttle/quota recover). Bounded so a
        # true sustained outage still finalizes ``incomplete``.
        max_passes = max(0, settings.EVAL_RATE_LIMIT_RETRY_MAX_PASSES)
        backoff_s = max(0.0, float(settings.EVAL_RATE_LIMIT_RETRY_BACKOFF_S))
        for pass_idx in range(max_passes):
            if cancel_event.is_set():
                break
            still_rate_limited = [
                key
                for key, r in results.items()
                if r.invocation_status == "rate_limited"
            ]
            if not still_rate_limited:
                break
            logger.info(
                "Eval run %s: %d example(s) still rate-limited after retry; "
                "backing off %.0fs then retrying (pass %d/%d)",
                run_id,
                len(still_rate_limited),
                backoff_s,
                pass_idx + 1,
                max_passes,
                extra={"component": "evaluation_service", "project_id": project_id},
            )
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=backoff_s)
                break
            except TimeoutError:
                pass
            for key in still_rate_limited:
                if cancel_event.is_set():
                    break
                await priority_dispatch.wait_for_background()
                if cancel_event.is_set():
                    break
                prior_invocation_id = results[key].invocation_id
                retry_result = await _retry_invoke(key, prior_invocation_id)
                if (
                    cancel_event.is_set()
                    or retry_result.ignored_due_to_run_cancellation
                ):
                    break
                results[key] = retry_result

        if await _maybe_finalize_runtime_rejection(engine, project_id, run_id):
            return
        if cancel_event.is_set():
            _persist_progress_counters(engine, run_id, results)
            await _finalize_canceled(engine, project_id, run_id)
            return

        # ── Phase D: metrics aggregation ────────────────────────────────
        core_fields = [f for f in guidance_fields if f.get("role") == "core"]
        results_by_key: dict[str, list[FieldMatchResult]] = {}
        persistently_failed_keys: list[str] = []

        for key in member_keys:
            r = results.get(key)
            if r is None or r.field_matches is None:
                persistently_failed_keys.append(key)
                # Alignment with the TAO rescoring path: a persistent
                # schema-invalid response IS the model's answer — a scored
                # miss on every core field — so it stays in the metric
                # denominator (match_fields({}, truth) mirrors the rescoring
                # service's normalized_pred={} treatment). Timeouts and
                # endpoint errors carry no answer to score and stay
                # excluded. Either way the run still finalizes
                # ``incomplete``, keeping the completed-vs-incomplete
                # strictness unchanged.
                if r is not None and r.invocation_status == "schema_invalid":
                    truth = ground_truth.get(key)
                    if truth is not None:
                        results_by_key[key] = match_fields({}, truth, guidance_fields)
                continue
            results_by_key[key] = r.field_matches

        all_field_results = list(results_by_key.values())
        agg_overall = compute_aggregate_metrics(all_field_results, core_fields)

        # ── Phase E: Returning vs New ───────────────────────────────────
        prev_run, prev_pool = _find_previous_completed_eval(
            engine,
            project_id,
            run_id,
        )
        returning_keys: list[str] | None = None
        new_keys: list[str] | None = None
        prev_overall: float | None = None
        prev_pool_version: int | None = None
        agg_returning: AggregateMetrics | None = None
        agg_new: AggregateMetrics | None = None

        if prev_run is not None and prev_pool is not None:
            prev_members = set(prev_pool.member_example_keys)
            current_members = set(member_keys)
            returning_keys = sorted(current_members & prev_members)
            new_keys = sorted(current_members - prev_members)
            prev_metrics: dict[str, Any] = prev_run.metrics or {}
            prev_overall_raw: Any = prev_metrics.get("overall", {})
            prev_overall_dict: dict[str, Any] = (
                cast("dict[str, Any]", prev_overall_raw)
                if isinstance(prev_overall_raw, dict)
                else {}
            )
            prev_overall_value: Any = prev_overall_dict.get("exact_match_rate")
            prev_overall = (
                float(prev_overall_value)
                if isinstance(prev_overall_value, (int, float))
                else None
            )
            prev_pool_version = prev_pool.pool_version

            # Per-bucket metrics
            ret_results = [
                results_by_key[k] for k in returning_keys if k in results_by_key
            ]
            new_results = [results_by_key[k] for k in new_keys if k in results_by_key]
            if ret_results:
                agg_returning = compute_aggregate_metrics(
                    ret_results,
                    core_fields,
                )
            if new_results:
                agg_new = compute_aggregate_metrics(
                    new_results,
                    core_fields,
                )

        metrics_dict = _serialize_metrics_with_buckets(
            agg_overall,
            agg_returning,
            agg_new,
        )

        # ── Phase F: coverage gaps ──────────────────────────────────────
        gaps = _compute_coverage_gaps(core_fields, ground_truth, member_keys)

        # ── Phase G: finalize ───────────────────────────────────────────
        # Late runtime rejection wins over all other terminals. Phases D–F
        # make no invocations, so a rejection landing here should be
        # unreachable (the post-Phase-C check catches everything) — kept as
        # defense-in-depth, routed through the same single finalizer so the
        # terminal state and SSE event cannot diverge from the live path.
        if await _maybe_finalize_runtime_rejection(engine, project_id, run_id):
            return
        terminal_status = "completed" if not persistently_failed_keys else "incomplete"

        with Session(engine) as session:
            run = session.query(RunRecord).filter_by(run_id=run_id).first()
            if run is None:
                return
            # Might have been canceled while computing metrics — atomically
            # flip only if still 'canceling'.
            if update_run_if_not_terminal(
                session,
                run_id,
                {"status": "canceled", "completed_at": utc_now()},
                only_status="canceling",
            ):
                session.commit()
                return

            # ICL eligible count at completion
            icl_count = count_icl_eligible_edits(session, project_id, run.guidance_id)

            # Persist one final status per pool member. OperationRecord is an
            # attempt ledger, so counting its rows would inflate failures when
            # the same example is retried and could make processed exceed the
            # frozen pool total. ``results`` is keyed by example and each retry
            # replaces the prior outcome.
            counts: dict[str, int] = {}
            for result in results.values():
                status = result.invocation_status
                counts[status] = counts.get(status, 0) + 1

            # One conditional UPDATE for the whole terminal transition: a
            # concurrent schema-evolution wipe may have failed this run
            # (its labels are gone), so the not-terminal check and the
            # write must be atomic — resurrecting a failed run to
            # "completed" would publish metrics for deleted ground truth.
            examples_total = run.examples_total
            if not update_run_if_not_terminal(
                session,
                run_id,
                {
                    "status": terminal_status,
                    "completed_at": utc_now(),
                    "metrics": metrics_dict,
                    "returning_example_keys": returning_keys,
                    "new_example_keys": new_keys,
                    "previous_overall_exact_match": prev_overall,
                    "previous_pool_version": prev_pool_version,
                    "coverage_gaps": gaps,
                    "icl_eligible_count_at_completion": icl_count,
                    "examples_succeeded": counts.get("success", 0),
                    "examples_schema_invalid": counts.get("schema_invalid", 0),
                    "examples_timeout": counts.get("timeout", 0),
                    "examples_endpoint_error": (
                        counts.get("endpoint_error", 0) + counts.get("rate_limited", 0)
                    ),
                },
                terminal_statuses=TERMINAL_STATUSES,
            ):
                return
            session.commit()

            # Single structured finalize line: the per-status breakdown vs
            # pool total is what a post-mortem needs to tell apart "scored
            # fine", "rate-limited / timed out (<50% scored)", and "all
            # endpoint errors" once no in-flight logs survive.
            logger.info(
                "evaluation_finalized run_id=%s status=%s pool_total=%s "
                "succeeded=%s schema_invalid=%s timeout=%s endpoint_error=%s "
                "persistently_failed=%s",
                run_id,
                terminal_status,
                examples_total,
                counts.get("success", 0),
                counts.get("schema_invalid", 0),
                counts.get("timeout", 0),
                counts.get("endpoint_error", 0) + counts.get("rate_limited", 0),
                len(persistently_failed_keys),
                extra={"component": "evaluation_service", "project_id": project_id},
            )

        # Diagnostic guard: when this run used ICL, check whether
        # the order-sensitive Test/Train split degenerated into disjoint class
        # sets. A class-clustered labeling order (manifest-order autorun,
        # class-sorted batch import) can fill the Test Pool with whole classes
        # that have no Train-Pool representation; relevance-ICL then can only
        # retrieve wrong-class exemplars for them, so the metrics just reported
        # understate true accuracy for those classes. The CLIP-diverse review
        # selector (the product default) avoids this. WARNING only — scores and
        # run status are untouched.
        if icl_mode != "disabled":
            try:
                coverage = assess_pool_class_coverage(engine, project_id)
                if coverage["degenerate"]:
                    logger.warning(
                        "Degenerate Test/Train class split: %d test class(es) have "
                        "no Train-pool representation, so relevance-ICL metrics "
                        "understate true accuracy for them. A CLIP-diverse review "
                        "order (the default) keeps the split class-representative.",
                        len(coverage["test_only_classes"]),
                        extra={
                            "component": "evaluation_service",
                            "project_id": project_id,
                            "details": {
                                "run_id": run_id,
                                "class_field": coverage["class_field"],
                                "test_only_classes": coverage["test_only_classes"],
                                "test_classes": coverage["test_classes"],
                                "train_classes": coverage["train_classes"],
                                "overlap_count": coverage["overlap_count"],
                            },
                        },
                    )
            except Exception:
                # Diagnostic must never break finalization.
                logger.debug(
                    "Pool class-coverage diagnostic failed for run %s",
                    run_id,
                    exc_info=True,
                    extra={"component": "evaluation_service", "project_id": project_id},
                )

        await sse_manager.emit(
            project_id,
            "evaluation_completed",
            {
                "run_id": run_id,
                "status": terminal_status,
                "metrics": metrics_dict,
            },
        )

    except Exception:
        logger.exception(
            "Evaluation run %s failed with unhandled exception",
            run_id,
            extra={"component": "evaluation_service", "project_id": project_id},
        )
        await finalize_unhandled_exception(
            engine,
            project_id,
            run_id,
            run_type="evaluation_run",
            error_summary="Unhandled exception during evaluation",
            terminal_statuses=TERMINAL_STATUSES,
        )
    finally:
        _cancel_events.pop(run_id, None)
        clear_runtime_rejections(run_id)


async def _finalize_canceled(
    engine: Any,
    project_id: str,
    run_id: str,
) -> bool:
    """Transition a canceling run to canceled and emit SSE.

    Thin wrapper over ``teacher_rejection.finalize_canceled`` (the one
    implementation shared with ``batch_label_service``) with the
    evaluation-specific parameters bound, then pops this module's cancel-event
    registry, which the shared helper never touches.
    """
    applied = await finalize_canceled(
        engine,
        project_id,
        run_id,
        event_name="evaluation_completed",
    )
    _cancel_events.pop(run_id, None)
    return applied


async def _maybe_finalize_runtime_rejection(
    engine: Any,
    project_id: str,
    run_id: str,
) -> bool:
    """Finalize the run as ``failed`` if a runtime rejection was recorded.

    Thin wrapper over ``teacher_rejection.finalize_runtime_rejection`` (the
    one implementation shared with ``batch_label_service``) with the
    evaluation-specific parameters bound. Returns True when it finalized. The
    per-run cancel event is cleaned up by the executor's ``finally`` block,
    which every call site returns into.
    """
    return await finalize_runtime_rejection(
        engine,
        project_id,
        run_id,
        run_type="evaluation_run",
        terminal_statuses=TERMINAL_STATUSES,
    )


# ── Cancel ──────────────────────────────────────────────────────────────────


def signal_run_cancellation(run_id: str) -> None:
    """Set the in-process cancel event for a live evaluation task.

    Single owner of the event-set, for in-file cancel paths and
    cross-service cancellation alike (schema evolution fails active
    runs in the DB, then calls this after commit so the executor stops
    invoking the Teacher instead of running the remaining pool to
    completion). The executor's finalizer independently refuses to
    overwrite terminal rows, so a missed signal (e.g. backend restart)
    degrades to wasted work, never to a resurrected run.
    """
    evt = _cancel_events.get(run_id)
    if evt is not None:
        evt.set()


async def cancel_evaluation_run(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Request cancellation of an evaluation run."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="evaluation_run")
        if isinstance(run, str):
            return run
        if run.status in TERMINAL_STATUSES:
            return f"conflict: Run {run_id} already in terminal state ({run.status})"

        now = utc_now()
        # Atomic: a schema-evolution wipe may have terminalized the run
        # since the read above; never resurrect it to canceling.
        applied = update_run_if_not_terminal(
            session,
            run_id,
            {"status": "canceling", "cancel_requested_at": now},
            terminal_statuses=TERMINAL_STATUSES,
        )
        session.commit()
        if not applied:
            return f"conflict: Run {run_id} already in terminal state"

    # Signal the background task
    signal_run_cancellation(run_id)

    return {"run_id": run_id, "status": "canceling", "cancel_requested_at": now}


# ── Get / List ──────────────────────────────────────────────────────────────


def get_evaluation_run(
    project_id: str,
    run_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Load a single evaluation run with full detail."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        run = find_run(session, project_id, run_id, run_type="evaluation_run")
        if isinstance(run, str):
            return run
        return _run_to_dict(run)


def list_evaluation_runs(
    project_id: str,
    *,
    status_filter: str | None = None,
    basis: Literal["gate", "benchmark"] | None = None,
    cursor: str | None = None,
    limit: int = 20,
    settings: Settings,
) -> tuple[list[dict[str, Any]], str | None]:
    """List evaluation runs with cursor pagination, newest-first.

    ``basis`` optionally scopes by provenance: ``"gate"`` = gate-basis
    Teacher runs only, ``"benchmark"`` = Student benchmark runs only.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return [], None

    with Session(engine) as session:
        rows, next_cursor = list_runs_page(
            session,
            project_id=project_id,
            run_type="evaluation_run",
            status_filter=status_filter,
            cursor=cursor,
            limit=limit,
            basis=basis,
        )
    return [_run_to_dict(r) for r in rows], next_cursor


def _run_to_dict(run: RunRecord) -> dict[str, Any]:
    """Convert a RunRecord to the API response dict."""
    # Keep the frozen denominator visible after completion as well as while
    # the run is active. REST is authoritative after an SSE reconnect, and
    # terminal consumers still need to distinguish (for example) 119/120
    # usable results from a 119-example pool.
    progress: dict[str, int] | None = None
    if run.examples_total:
        processed = min(
            run.examples_total,
            (
                (run.examples_succeeded or 0)
                + (run.examples_schema_invalid or 0)
                + (run.examples_timeout or 0)
                + (run.examples_endpoint_error or 0)
            ),
        )
        progress = {"processed": processed, "total": run.examples_total}

    return {
        "run_id": run.run_id,
        "run_type": run.run_type,
        "status": run.status,
        "status_reason": run.status_reason,
        "pool_version_id": run.pool_version_id,
        "guidance_id": run.guidance_id,
        "model_config_id": run.model_config_id,
        "icl_mode": run.icl_mode,
        "evaluation_source": run.evaluation_source,
        "generation_preset_key": run.generation_preset_key,
        "thinking_mode_effective": run.thinking_mode_effective,
        "visual_budget_preset_key": run.visual_budget_preset_key,
        "structured_generation_mode_effective": run.structured_generation_mode_effective,
        "inference_contract": run.inference_contract,
        "icl_eligible_count_at_start": run.icl_eligible_count_at_start,
        "icl_eligible_count_at_completion": run.icl_eligible_count_at_completion,
        "progress": progress,
        "metrics": run.metrics,
        "previous_pool_version": run.previous_pool_version,
        "returning_example_keys": run.returning_example_keys,
        "new_example_keys": run.new_example_keys,
        "previous_overall_exact_match": run.previous_overall_exact_match,
        "coverage_gaps": run.coverage_gaps,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        # Evaluation provenance
        "student_model_config_id": run.student_model_config_id,
        "nim_model_profile_requested": run.nim_model_profile_requested,
        "nim_model_profile_selected": run.nim_model_profile_selected,
        "nim_profile_metadata": run.nim_profile_metadata,
        "quantization_method": run.quantization_method,
        "gpu_type": run.gpu_type,
        "gpu_count": run.gpu_count,
        "dataset_manifest_sha256": run.dataset_manifest_sha256,
    }


# ── Trigger status ──────────────────────────────────────────────────────────


def compute_trigger_status(
    project_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Derive evaluation trigger state from persisted data.

    Recomputed fresh on each call — never cached.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"

        # ── first_pool_threshold ────────────────────────────────────────
        pool_count = (
            session.execute(
                select(func.count())
                .select_from(Label)
                .where(
                    Label.project_id == project_id,
                    Label.label_status == "verified",
                    Label.pool_assignment == "test_pool",
                )
            ).scalar()
            or 0
        )

        threshold = settings.EVAL_FIRST_POOL_SIZE
        fpt_active = pool_count >= threshold

        # Dismissed once a Teacher-contract evaluation of the current
        # era exists — a TAO/Student run is not the first Teacher
        # evaluation this nudge asks for. The same run doubles as the
        # config-change / icl_growth baseline below.
        last_completed = _era_teacher_evals(
            session, project_id, ["completed", "incomplete"]
        ).first()
        completed_eval_exists = last_completed is not None

        # The active message is fixed product copy. The SME-facing
        # surface MUST NOT expose raw thresholds (no-jargon rule).
        first_pool = {
            "is_active": fpt_active and not completed_eval_exists,
            "dismissed": completed_eval_exists,
            "message": (
                f"{pool_count} images reserved for testing. Run an evaluation to measure quality."
                if fpt_active
                else f"Test pool has {pool_count} images (need {threshold})."
            ),
            "context": {"pool_count": pool_count, "threshold": threshold},
        }

        # ── configuration_change ────────────────────────────────────────
        if last_completed is None:
            config_change: dict[str, Any] = {
                "is_active": False,
                "dismissed": False,
                "message": "No previous evaluation to compare against.",
                "context": {"changed_fields": []},
            }
        else:
            changed = detect_config_changes(project, last_completed)
            config_change = {
                "is_active": len(changed) > 0,
                "dismissed": False,
                "message": (
                    f"Settings changed since last evaluation: {', '.join(changed)}."
                    if changed
                    else "Settings match last evaluation."
                ),
                "context": {"changed_fields": changed},
            }

        # ── icl_growth ──────────────────────────────────────────────────
        icl_count = count_icl_eligible_edits(
            session, project_id, project.active_guidance_id
        )

        last_eval_count = (
            last_completed.icl_eligible_count_at_completion
            if last_completed and last_completed.icl_eligible_count_at_completion
            else 0
        )
        dismissed_count = project.icl_recommendation_dismissed_at_count or 0
        baseline = max(last_eval_count, dismissed_count)

        icl_active = baseline > 0 and icl_count >= 2 * baseline
        icl_growth = {
            "is_active": icl_active,
            "dismissed": False,
            "message": (
                f"{icl_count} edits since baseline of {baseline}. "
                "Run an evaluation to see if accuracy improved."
                if icl_active
                else f"ICL eligible: {icl_count} (baseline: {baseline})."
            ),
            "context": {
                "baseline_count": baseline,
                "current_count": icl_count,
            },
        }

    return {
        "auto_evaluate_enabled": project.auto_evaluate_enabled,
        "first_pool_threshold": first_pool,
        "configuration_change": config_change,
        "icl_growth": icl_growth,
        "updated_at": utc_now(),
    }


async def maybe_start_auto_evaluation(
    project_id: str,
    settings: Settings,
) -> dict[str, Any] | str | None:
    """Start one gate-basis evaluation when an enabled trigger is active.

    This is the backend-authoritative bridge between the persisted
    ``auto_evaluate_enabled`` setting and the three §7.1 trigger families.
    Callers invoke it immediately after a mutation that can activate a
    trigger. It creates only the durable queued RunRecord; inference remains
    in the normal background task, so the originating SME mutation is never
    blocked on a Teacher request.

    An existing queued/running/canceling gate-basis run suppresses another
    automatic start. Manual starts retain their documented newest-config-wins
    behavior in ``start_evaluation_run``.
    """
    lock = _auto_start_locks.setdefault(project_id, asyncio.Lock())
    async with lock:
        trigger_status = compute_trigger_status(project_id, settings)
        if isinstance(trigger_status, str):
            return trigger_status
        if not trigger_status["auto_evaluate_enabled"]:
            return None

        active_triggers = [
            name
            for name in (
                "first_pool_threshold",
                "configuration_change",
                "icl_growth",
            )
            if trigger_status[name]["is_active"]
        ]
        if not active_triggers:
            return None

        engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
        if engine is None:
            return f"not found: Project {project_id}"
        with Session(engine) as session:
            project = session.query(Project).filter_by(project_id=project_id).one()
            active_run = (
                session.query(RunRecord)
                .filter(
                    RunRecord.project_id == project_id,
                    RunRecord.run_type == "evaluation_run",
                    RunRecord.student_model_config_id.is_(None),
                    RunRecord.status.in_(["queued", "running", "canceling"]),
                )
                .order_by(RunRecord.created_at.desc())
                .first()
            )
        # The same persisted trigger remains active until its run completes.
        # Do not restart that run after every subsequent label save. A newly
        # changed Teacher/Guidance/control snapshot is different:
        # newest-config-wins requires the normal start path to supersede the
        # stale queued/running run. A cancel already in progress is never
        # duplicated.
        if active_run is not None and (
            active_run.status == "canceling"
            or not detect_config_changes(project, active_run)
        ):
            return None

        result = await start_evaluation_run(project_id, settings=settings)
        if isinstance(result, str):
            logger.warning(
                "Auto-Evaluate could not start for project %s (%s): %s",
                project_id,
                ",".join(active_triggers),
                result,
            )
            return result

        logger.info(
            "Auto-Evaluate queued run %s for project %s (%s)",
            result["run_id"],
            project_id,
            ",".join(active_triggers),
        )
        return {**result, "auto_trigger_types": active_triggers}


def detect_config_changes(
    project: Project,
    last_run: RunRecord,
) -> list[str]:
    """Compare 5 tracked project fields against the last evaluation snapshot."""
    changed: list[str] = []
    thinking_mode = "on" if project.thinking_default_on else "off"

    if project.teacher_model_config_id != last_run.model_config_id:
        changed.append("teacher_model")
    if project.active_guidance_id != last_run.guidance_id:
        changed.append("guidance")
    if project.labeling_generation_preset_key != last_run.generation_preset_key:
        changed.append("generation_preset")
    if thinking_mode != last_run.thinking_mode_effective:
        changed.append("thinking")
    if project.visual_budget_preset_key != last_run.visual_budget_preset_key:
        changed.append("visual_budget")
    return changed


# ── Trigger dismissal ───────────────────────────────────────────────────────


def dismiss_trigger(
    project_id: str,
    trigger_type: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Update trigger dismissal state."""
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"

        if trigger_type == "icl_growth":
            # Update baseline to current ICL-eligible count
            icl_count = count_icl_eligible_edits(
                session, project_id, project.active_guidance_id
            )
            project.icl_recommendation_dismissed_at_count = icl_count
            session.commit()
        # first_pool_threshold and configuration_change: no persistent state change
        # (first_pool dismissed by running an eval; config_change re-activates on each change)

    return {"trigger_type": trigger_type, "dismissed": True}


# ── Helpers ─────────────────────────────────────────────────────────────────


def _agg_to_dict(agg: AggregateMetrics) -> dict[str, Any]:
    """Convert a single AggregateMetrics to a JSON-serializable dict."""
    pv_metrics: dict[str, Any] = {}
    for field_name, values in agg.per_value_metrics.items():
        pv_metrics[field_name] = {
            val: {
                "precision": round(m.precision, 6),
                "recall": round(m.recall, 6),
                "f1": round(m.f1, 6),
            }
            for val, m in values.items()
        }
    return {
        "exact_match_rate": round(agg.overall_exact_match_rate, 6),
        "example_count": agg.example_count,
        "per_field_match_rates": {
            k: round(v, 6) for k, v in agg.per_core_field_match_rate.items()
        },
        "per_value_metrics": pv_metrics,
        "macro_f1": round(agg.overall_macro_f1, 6),
        "per_field_macro_f1": {
            k: round(v, 6) for k, v in agg.per_field_macro_f1.items()
        },
    }


def serialize_metrics_overall(agg: AggregateMetrics) -> dict[str, Any]:
    """Convert AggregateMetrics to the metrics JSON (overall bucket only).

    The bucket shape must match :func:`_serialize_metrics_with_buckets` —
    ``per_field_match_rates`` / ``per_value_metrics`` nested under
    ``overall`` — because every consumer (scale-up gate criteria 2–3, the
    Compare page's per-field drill-down, the deployment handoff) reads one
    ``EvaluationMetrics`` contract regardless of ``evaluation_source``. A
    flat variant that puts those keys at the top level is silently read
    as ``{}`` by the gate and the UI. Delegating (rather than inlining
    the shape) makes divergence impossible.
    """
    return _serialize_metrics_with_buckets(agg, None, None)


def _serialize_metrics_with_buckets(
    overall: AggregateMetrics,
    returning: AggregateMetrics | None,
    new: AggregateMetrics | None,
) -> dict[str, Any]:
    """Serialize metrics with Returning/New buckets."""
    return {
        "overall": _agg_to_dict(overall),
        "returning": _agg_to_dict(returning) if returning else None,
        "new": _agg_to_dict(new) if new else None,
    }


def _active_schema_has_categorical_core(session: Session, project: Project) -> bool:
    """Whether the active Guidance schema has any categorical Core field.

    Categorical means enum / enum_set / boolean — the only field types
    the evaluator emits per-value P/R/F1 for. Gate criterion 3 applies
    only when at least one exists — the Spec's "every value of every
    categorical Core field" criterion is vacuously true otherwise.
    """
    if not project.active_guidance_id:
        return False
    guidance = (
        session.query(Guidance)
        .filter_by(guidance_id=project.active_guidance_id)
        .first()
    )
    if guidance is None or not guidance.schema:
        return False
    fields = guidance.schema.get("fields", [])
    return any(
        f.get("role") == "core" and f.get("type") in ("enum", "enum_set", "boolean")
        for f in fields
    )


def _era_teacher_evals(
    session: Session,
    project_id: str,
    statuses: list[str],
):
    """Teacher-contract evaluation runs of the current schema era, newest first.

    TAO rescoring runs (evaluation_source="tao") and Student NIM serving
    runs (student_model_config_id set) describe a different model; runs
    recorded under a pre-semantic-Core-change Guidance scored labels the
    system itself deleted. Neither may serve as the gate's quality basis,
    a trigger baseline, or the Returning/New baseline.
    """
    era_floor = _current_era_floor(session, project_id)
    return (
        session.query(RunRecord)
        .join(Guidance, RunRecord.guidance_id == Guidance.guidance_id)
        .filter(
            RunRecord.project_id == project_id,
            RunRecord.run_type == "evaluation_run",
            RunRecord.status.in_(statuses),
            RunRecord.evaluation_source == "nim",
            RunRecord.student_model_config_id.is_(None),
            Guidance.version_number >= era_floor,
        )
        .order_by(RunRecord.created_at.desc())
    )


def _current_era_floor(session: Session, project_id: str) -> int:
    """Guidance-version floor of the current schema era.

    The highest Guidance version born from a semantic Core change marks
    the era floor; evaluation runs recorded under an earlier version are
    audit history, never baselines — a semantic Core change requires the Returning/New
    baseline and the Auto-Evaluate trigger counters (first pool threshold,
    configuration change, ICL growth) to rebuild from zero under the new
    Guidance. No semantic change yet → floor 0 (every version qualifies).
    """
    return (
        session.query(func.max(Guidance.version_number))
        .filter(
            Guidance.project_id == project_id,
            Guidance.semantic_core_change_from_guidance_id.is_not(None),
        )
        .scalar()
        or 0
    )


def _find_previous_completed_eval(
    engine: Any,
    project_id: str,
    current_run_id: str,
) -> tuple[RunRecord | None, Pool | None]:
    """Find the most recent *completed* evaluation before this run.

    The Returning/New comparison baselines against the previous completed
    evaluation's snapshot, and summary metrics are computed only from runs
    whose status is ``completed``.
    Incomplete, canceled, and failed runs carry diagnostic-only aggregates
    and MUST NOT be used as the previous baseline — doing so produces a
    spurious "+50% vs previous" delta when the only prior run is an
    ``incomplete`` with zero usable examples.

    Only Teacher-contract runs qualify as the baseline: a TAO Student
    evaluation (``evaluation_source="tao"``) or a Student NIM serving
    run (``student_model_config_id`` set) describes a different model
    and would corrupt the "same images, then vs now" comparison.
    ``model_config_id`` is deliberately NOT filtered — comparing
    before/after a Teacher or Guidance change is exactly what the
    configuration-change nudge asks the SME to do — but the baseline
    never crosses a semantic Core change: re-labeled examples
    keep their example_keys, so a cross-era "Returning" bucket would
    compare labels produced under two different schemas. The first
    evaluation of a new schema era starts with no baseline.
    """
    with Session(engine) as session:
        prev_run = (
            _era_teacher_evals(session, project_id, ["completed"])
            .filter(RunRecord.run_id != current_run_id)
            .first()
        )
        if prev_run is None or prev_run.pool_version_id is None:
            return None, None

        prev_pool = (
            session.query(Pool)
            .filter_by(
                pool_id=prev_run.pool_version_id,
            )
            .first()
        )
        return prev_run, prev_pool


def _compute_coverage_gaps(
    core_fields: list[dict[str, Any]],
    ground_truth: dict[str, dict[str, Any]],
    member_keys: list[str],
) -> list[dict[str, Any]]:
    """Identify schema values not represented in the pool ground truth.

    Also exported as :func:`compute_coverage_gaps` for reuse by the TAO
    re-scoring service — the semantics are identical for NIM-sourced and
    TAO-sourced evaluations.
    """
    gaps: list[dict[str, Any]] = []

    for f in core_fields:
        ftype = f.get("type")
        fname = f.get("field_name")
        if not fname:
            continue

        # Collect observed values across all pool members
        observed: set[Any] = set()
        for key in member_keys:
            gt: dict[str, Any] = ground_truth.get(key, {})
            val: Any = gt.get(fname)
            if val is None:
                continue
            if ftype == "enum_set" and isinstance(val, list):
                observed.update(cast("list[Any]", val))
            else:
                observed.add(val)

        # Check against allowed values or expected range
        missing: list[str] = []
        if ftype in ("enum", "enum_set"):
            allowed = f.get("allowed_values", [])
            missing = [v for v in allowed if v not in observed]
        elif ftype == "boolean":
            for bv in [True, False]:
                if bv not in observed:
                    missing.append(str(bv).lower())
        elif ftype == "integer":
            # Report observed range vs schema range
            min_v = f.get("minimum")
            max_v = f.get("maximum")
            if min_v is not None and max_v is not None:
                int_observed = {v for v in observed if isinstance(v, int)}
                missing = [
                    str(v) for v in range(min_v, max_v + 1) if v not in int_observed
                ]
        # String: no coverage check

        if missing:
            gaps.append(
                {
                    "field_name": fname,
                    "field_type": ftype,
                    "missing_values": missing,
                }
            )

    return gaps


# ── Scale-Up Readiness Gate ─────────────────────────────────────────────────


def compute_accept_rate(
    session: Session,
    project_id: str,
    window: int,
) -> tuple[float, int]:
    """Compute rolling Accept rate over the last *window* Verified labels.

    Returns ``(rate, denominator)`` where *denominator* is
    ``min(window, total_verified)``.
    """
    labels = list(
        session.execute(
            select(Label.verified_outcome)
            .where(
                Label.project_id == project_id,
                Label.label_status == "verified",
                Label.verified_outcome.in_(["Accept", "Edit"]),
            )
            .order_by(Label.verified_at.desc())
            .limit(window)
        )
        .scalars()
        .all()
    )
    if not labels:
        return 0.0, 0
    accept_count = sum(1 for o in labels if o == "Accept")
    return accept_count / len(labels), len(labels)


def compute_scaleup_gate(
    project_id: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Evaluate the 5-criteria Scale-Up Readiness Gate.

    Returns a structured dict or an error string.
    Lightweight — queries persisted metrics and counts only.
    """
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"not found: Project {project_id}"

    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"not found: Project {project_id}"

        criteria: list[dict[str, Any]] = []

        # ── 1. Overall Exact Match ──────────────────────────────────────
        last_eval = _era_teacher_evals(session, project_id, ["completed"]).first()

        em_threshold = project.scaleup_exact_match_threshold
        if last_eval is None or last_eval.metrics is None:
            criteria.append(
                {
                    "criterion_name": "overall_exact_match",
                    "passed": False,
                    "current_value": 0.0,
                    "threshold": em_threshold,
                    "message": "No completed evaluation run found. Run an evaluation to measure quality.",
                    # Structural discriminator: current_value=0.0 is also a
                    # legitimate 0% Exact-Match result, so the UI keys its
                    # "no evaluation yet" pending state on this flag — the
                    # human-readable message above is NOT a wire contract.
                    "details": {"no_completed_run": True},
                }
            )
            overall_em = 0.0
        else:
            overall_em = last_eval.metrics.get("overall", {}).get(
                "exact_match_rate",
                0.0,
            )
            evaluated_model = session.get(ModelConfig, last_eval.model_config_id)
            changed_fields = detect_config_changes(project, last_eval)
            passed = overall_em >= em_threshold
            criteria.append(
                {
                    "criterion_name": "overall_exact_match",
                    "passed": passed,
                    "current_value": round(overall_em, 4),
                    "threshold": em_threshold,
                    "message": (
                        f"Model accuracy: {overall_em:.0%} overall (need {em_threshold:.0%}). "
                        + (
                            "Passed."
                            if passed
                            else "Continue labeling or refine Guidance."
                        )
                    ),
                    # The gate intentionally uses the most recent completed
                    # Teacher evaluation even after project settings change.
                    # Preserve that historical attribution so the UI never
                    # presents this score as if the mutable current Teacher
                    # produced it.
                    "details": {
                        "evaluation_run_id": last_eval.run_id,
                        "evaluated_model_config_id": last_eval.model_config_id,
                        "evaluated_model_name": (
                            evaluated_model.model_name if evaluated_model else None
                        ),
                        "current_configuration_differs": bool(changed_fields),
                        "changed_fields": changed_fields,
                    },
                }
            )

        # ── 2. Per-core-field match rate ────────────────────────────────
        pf_threshold = project.scaleup_per_field_match_threshold
        if last_eval is None or last_eval.metrics is None:
            # Blocked-by marker lets the UI filter this out of the
            # actionable "next steps" list while keeping the criterion
            # visible in the full gate-details expander. Without this,
            # three separate criteria surface the same "no evaluation"
            # point as next steps, creating noise.
            criteria.append(
                {
                    "criterion_name": "per_field_match",
                    "passed": False,
                    "current_value": 0.0,
                    "threshold": pf_threshold,
                    "message": "Depends on evaluation results.",
                    "details": {
                        "failing_fields": [],
                        "blocked_by": "overall_exact_match",
                    },
                }
            )
        else:
            per_field = last_eval.metrics.get("overall", {}).get(
                "per_field_match_rates",
                {},
            )
            failing = [
                {"field_name": f, "current_rate": round(r, 4)}
                for f, r in per_field.items()
                if r < pf_threshold
            ]
            passed = len(failing) == 0 and len(per_field) > 0
            min_rate = min(per_field.values()) if per_field else 0.0
            if failing:
                msg = (
                    "Per-field quality: "
                    + ", ".join(
                        f"'{ff['field_name']}' at {ff['current_rate']:.0%}"
                        for ff in failing
                    )
                    + f" (need {pf_threshold:.0%}). Continue labeling or refine Guidance."
                )
            else:
                msg = f"All fields at or above {pf_threshold:.0%}. Passed."
            criteria.append(
                {
                    "criterion_name": "per_field_match",
                    "passed": passed,
                    "current_value": round(min_rate, 4),
                    "threshold": pf_threshold,
                    "message": msg,
                    "details": {"failing_fields": failing},
                }
            )

        # ── 3. Minimum per-value F1 ────────────────────────────────────
        # Per-value P/R/F1 exists only for categorical Core fields
        # (enum / enum_set / boolean). Gate criterion 3 quantifies
        # over "every value of every categorical Core field", which is
        # vacuously satisfied when the schema has none — a string/
        # integer-only Core schema (a legal schema shape) must not be
        # permanently gated by a criterion that cannot apply to it.
        f1_threshold = project.scaleup_min_per_value_f1_threshold
        has_categorical_core = _active_schema_has_categorical_core(session, project)
        if last_eval is None or last_eval.metrics is None:
            # See per_field_match above — blocked_by marker.
            criteria.append(
                {
                    "criterion_name": "min_per_value_f1",
                    "passed": False,
                    "current_value": 0.0,
                    "threshold": f1_threshold,
                    "message": "Depends on evaluation results.",
                    "details": {
                        "failing_values": [],
                        "blocked_by": "overall_exact_match",
                    },
                }
            )
        else:
            pv = last_eval.metrics.get("overall", {}).get(
                "per_value_metrics",
                {},
            )
            failing_vals: list[dict[str, Any]] = []
            min_f1 = 1.0
            for field_name, values in pv.items():
                for val_name, val_metrics in values.items():
                    f1 = val_metrics.get("f1", 0.0)
                    if f1 < min_f1:
                        min_f1 = f1
                    if f1 < f1_threshold:
                        failing_vals.append(
                            {
                                "field_name": field_name,
                                "value": val_name,
                                "f1": round(f1, 4),
                                "precision": round(
                                    val_metrics.get("precision", 0.0), 4
                                ),
                                "recall": round(val_metrics.get("recall", 0.0), 4),
                            }
                        )
            if not pv:
                # Vacuous pass when the schema has no categorical Core
                # fields; a categorical schema with empty metrics is a
                # degenerate evaluation and stays failed.
                passed = not has_categorical_core
                min_f1 = 1.0 if passed else 0.0
                msg = (
                    "No categorical Core fields in the schema; "
                    "this criterion does not apply. Passed."
                    if passed
                    else "Evaluation reported no per-value metrics for the "
                    "schema's categorical Core fields. Run a new evaluation."
                )
            elif failing_vals:
                passed = False
                top = failing_vals[0]
                msg = (
                    f"Per-value quality: '{top['value']}' in '{top['field_name']}' "
                    f"has F1 {top['f1']:.0%} (need {f1_threshold:.0%}, "
                    f"precision {top['precision']:.0%}, recall {top['recall']:.0%}). "
                    "Add more examples or refine Guidance."
                )
            else:
                passed = True
                msg = f"All per-value F1 scores at or above {f1_threshold:.0%}. Passed."
            criteria.append(
                {
                    "criterion_name": "min_per_value_f1",
                    "passed": passed,
                    "current_value": round(min_f1, 4),
                    "threshold": f1_threshold,
                    "message": msg,
                    "details": {"failing_values": failing_vals},
                }
            )

        # ── 4. Accept rate ──────────────────────────────────────────────
        ar_threshold = project.scaleup_accept_rate_threshold
        ar_window = project.scaleup_accept_rate_window
        accept_rate, denom = compute_accept_rate(session, project_id, ar_window)
        ar_passed = accept_rate >= ar_threshold and denom > 0
        criteria.append(
            {
                "criterion_name": "accept_rate",
                "passed": ar_passed,
                "current_value": round(accept_rate, 4),
                "threshold": ar_threshold,
                "message": (
                    f"Accept rate: {accept_rate:.0%} over last {denom} labels "
                    f"(need {ar_threshold:.0%}). "
                    + ("Passed." if ar_passed else "Continue Interactive Labeling.")
                ),
                "details": None,
            }
        )

        # ── 5. Minimum Test Pool size ───────────────────────────────────
        pool_threshold = project.scaleup_min_test_pool_size
        pool_count = (
            session.execute(
                select(func.count())
                .select_from(Label)
                .where(
                    Label.project_id == project_id,
                    Label.label_status == "verified",
                    Label.pool_assignment == "test_pool",
                )
            ).scalar()
            or 0
        )
        pool_passed = pool_count >= pool_threshold
        # Growth disclosure: late verification grows the pool toward
        # floor(total_verified × test_pool_fraction) (§4.3.1-2) — by design,
        # but silently from the SME's seat, and at batch scale it re-bases
        # the benchmark between evaluations. Surface the target so the UI
        # can show impending growth; the documented pin lever is lowering
        # ``test_pool_fraction`` so the target stays at/below membership.
        total_verified = (
            session.execute(
                select(func.count())
                .select_from(Label)
                .where(
                    Label.project_id == project_id,
                    Label.label_status == "verified",
                )
            ).scalar()
            or 0
        )
        pool_target = math.floor(total_verified * project.test_pool_fraction)
        criteria.append(
            {
                "criterion_name": "min_test_pool_size",
                "passed": pool_passed,
                "current_value": pool_count,
                "threshold": pool_threshold,
                "message": (
                    f"Test Pool: {pool_count} examples (need {pool_threshold}). "
                    + (
                        "Passed."
                        if pool_passed
                        else "Continue labeling to grow the pool."
                    )
                ),
                "details": {
                    "pool_target": pool_target,
                    "test_pool_fraction": project.test_pool_fraction,
                    "total_verified": total_verified,
                },
            }
        )

    # ── Gate status ─────────────────────────────────────────────────────
    all_pass = all(c["passed"] for c in criteria)
    gate_status = "ready" if all_pass else "not_ready"

    # ── Structured gate_evaluation log line ─────────────────────────────
    _log_gate_evaluation(project_id, gate_status, criteria)

    return {
        "gate_status": gate_status,
        "criteria": criteria,
        "evaluated_at": utc_now(),
    }


# ── Structured gate_evaluation logging ──────────────────────────────────────


def _log_gate_evaluation(
    project_id: str,
    gate_status: str,
    criteria: list[dict[str, Any]],
) -> None:
    """Emit the structured ``gate_evaluation`` log line.

    Default level: INFO.  Logs each criterion's current value, threshold,
    and pass/fail.  Emitted on every re-evaluation.
    """
    details = {
        "gate_status": gate_status,
        "criteria": [
            {
                "criterion_name": c["criterion_name"],
                "passed": c["passed"],
                "current_value": c["current_value"],
                "threshold": c["threshold"],
            }
            for c in criteria
        ],
    }
    logger.log(
        logging.INFO,
        "Gate evaluation: %s (%d/%d criteria pass)",
        gate_status,
        sum(1 for c in criteria if c["passed"]),
        len(criteria),
        extra={
            "component": "gate_evaluation",
            "project_id": project_id,
            "details": details,
        },
    )


# ── Public re-export for the TAO re-scoring service ────────────────────────

# The TAO re-scoring service (tao_rescoring_service.py) consumes this.
# Re-exposing it as a public name (without the leading underscore) keeps
# it from reaching into private surface while avoiding code duplication
# of the gap-computation logic. (``serialize_metrics_overall`` above is
# already public for the same consumer.)
compute_coverage_gaps = _compute_coverage_gaps
