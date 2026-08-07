# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proposal service — orchestrates the full interactive proposal pipeline.

Ties together the review selector, ICL selection + token budget, prompt
rendering + Teacher invocation, and proposal validation + Operation Record
persistence.

Covers the proposal attempt flow, the proposal endpoint contract, the
Operation Record, the persist-before-invocation rule, and structured
generation runtime rejection.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.exact_match_evaluator import (
    SchemaValidationReport,
    validate_proposal,
)
from vlm_feedback_loop.services.icl_service import query_icl_candidates
from vlm_feedback_loop.services.invocation_outcome import (
    apply_invocation_outcome,
    classify_invocation_status,
    write_invocation_artifacts,
)
from vlm_feedback_loop.services.model_config_service import endpoint_is_operational
from vlm_feedback_loop.services.nim_client import build_endpoint_auth_headers
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.prompt_service import (
    ModelConfigInput,
    TeacherInvocationResult,
    invoke_teacher,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.teacher_rejection import (
    is_structured_gen_rejection as _is_structured_gen_rejection,
)
from vlm_feedback_loop.services.teacher_rejection import (
    is_thinking_toggle_rejection as _is_thinking_toggle_rejection,
)
from vlm_feedback_loop.services.teacher_rejection import (
    is_visual_budget_rejection as _is_visual_budget_rejection,
)
from vlm_feedback_loop.services.token_budget_service import (
    token_budget_invoke_kwargs,
)

logger = logging.getLogger("vlm_feedback_loop.proposal_service")


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class ProposalResult:
    """Internal result returned by :func:`create_proposal`."""

    inference_invocation_id: str
    example_key: str
    proposal_json: dict[str, Any] | None
    schema_valid_core: bool
    validation_errors_core: list[str]
    validation_errors_aux: list[str]
    invocation_status: (
        str  # success | schema_invalid | timeout | endpoint_error | rate_limited
    )
    latency_ms_end_to_end: int | None
    icl_images_attached_count: int
    icl_example_keys_used: list[str]
    used_existing_label: bool


# ── Helpers ──────────────────────────────────────────────────────────────────


# Runtime-capability rejection detectors live in services.teacher_rejection —
# one source of truth across proposal / evaluation / batch_label.


def _log_schema_validation(
    project_id: str,
    report: SchemaValidationReport,
    inference_invocation_id: str,
) -> None:
    """Log point 2 (Spec §11): schema validation.

    Always at INFO level.  Details include validity classification,
    error counts, specific errors, and normalization steps.
    """
    normalization_steps: list[str] = []
    for fr in report.field_results:
        if fr.normalization_steps:
            normalization_steps.extend(
                f"{fr.field_name}: {s}" for s in fr.normalization_steps
            )

    details: dict[str, Any] = {
        "schema_valid_core": report.schema_valid_core,
        "core_error_count": len(report.core_errors),
        "core_errors": report.core_errors,
        "aux_error_count": len(report.aux_errors),
        "aux_errors": report.aux_errors,
        "normalization_steps": normalization_steps,
        "parse_error": report.parse_error,
        "truncation_attributed": report.truncation_attributed_schema_invalid,
    }

    logger.info(
        "Schema validation: schema_valid_core=%s, core_errors=%d, aux_errors=%d",
        report.schema_valid_core,
        len(report.core_errors),
        len(report.aux_errors),
        extra={
            "component": "schema_validation",
            "project_id": project_id,
            "correlation_id": inference_invocation_id,
            "details": details,
        },
    )


def _demote_capability_support(engine: Any, model_config_id: str, field: str) -> None:
    """Persist a runtime capability demotion.

    Sets ``field`` (``thinking_toggle_support`` / ``visual_budget_support``)
    to ``"unsupported"`` on the ModelConfig so subsequent invocations skip
    the rejected override automatically. Best-effort: a persistence failure
    logs and continues — the in-flight retry still proceeds.
    """
    try:
        with Session(engine) as session:
            mc_db = (
                session.query(ModelConfig)
                .filter_by(model_config_id=model_config_id)
                .first()
            )
            if mc_db is not None and getattr(mc_db, field) != "unsupported":
                setattr(mc_db, field, "unsupported")
                session.commit()
    except Exception:
        logger.warning(
            "Failed to persist %s=unsupported demotion", field, exc_info=True
        )


# ── Main pipeline ────────────────────────────────────────────────────────────


async def create_proposal(
    project_id: str,
    *,
    example_key: str,
    teacher_model_config_id_override: str | None = None,
    guidance_id_override: str | None = None,
    generation_preset_key_override: str | None = None,
    thinking_mode_override: str | None = None,
    visual_budget_preset_key_override: str | None = None,
    retry_of_inference_invocation_id: str | None = None,
    use_existing_label: bool = False,
    settings: Settings,
) -> ProposalResult | str:
    """Create an interactive proposal for an example.

    Returns
    -------
    ProposalResult on success (including schema_invalid / timeout / endpoint_error).
    str error message for 4xx-level issues (missing guidance, example not found, etc.).

    The caller (router) maps ``str`` to the appropriate HTTP error code.
    """

    # ── 1. Resolve project and engine ────────────────────────────────────
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    if engine is None:
        return f"Project not found: {project_id}"

    # Load project, guidance, model config, endpoint, example in one session
    with Session(engine) as session:
        project = session.query(Project).filter_by(project_id=project_id).first()
        if project is None:
            return f"Project not found: {project_id}"

        # ── 2. Resolve guidance ──────────────────────────────────────────
        effective_guidance_id = guidance_id_override or project.active_guidance_id
        if not effective_guidance_id:
            return "No active guidance configured for this project"

        guidance = (
            session.query(Guidance)
            .filter_by(
                project_id=project_id,
                guidance_id=effective_guidance_id,
            )
            .first()
        )
        if guidance is None:
            return f"Guidance not found: {effective_guidance_id}"

        # ── 3. Resolve model config + endpoint ───────────────────────────
        effective_model_config_id = (
            teacher_model_config_id_override or project.teacher_model_config_id
        )
        if not effective_model_config_id:
            return "No teacher model configured for this project"

        model_config = (
            session.query(ModelConfig)
            .filter_by(
                project_id=project_id,
                model_config_id=effective_model_config_id,
            )
            .first()
        )
        if model_config is None:
            return f"Model config not found: {effective_model_config_id}"

        endpoint = (
            session.query(NimEndpoint)
            .filter_by(
                project_id=project_id,
                endpoint_id=model_config.endpoint_id,
            )
            .first()
        )
        if endpoint is None:
            return (
                f"NIM endpoint not found for model config: {effective_model_config_id}"
            )

        # ── 4. Resolve example ───────────────────────────────────────────
        example = (
            session.query(Example)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
            )
            .first()
        )
        if example is None:
            return f"Example not found: {example_key}"

        # ── 5. Handle use_existing_label ─────────────────────────────────
        if use_existing_label:
            existing_label = (
                session.query(Label)
                .filter_by(
                    project_id=project_id,
                    example_key=example_key,
                    label_status="auto_labeled",
                )
                .first()
            )
            if existing_label is not None:
                return ProposalResult(
                    inference_invocation_id=existing_label.inference_invocation_id,
                    example_key=example_key,
                    proposal_json=existing_label.label_json,
                    schema_valid_core=True,
                    validation_errors_core=[],
                    validation_errors_aux=[],
                    invocation_status="success",
                    latency_ms_end_to_end=None,
                    icl_images_attached_count=0,
                    icl_example_keys_used=[],
                    used_existing_label=True,
                )

        # A disabled shared-Teacher attachment is authoritative even when its
        # old URL happens to answer again. Host ports are reused as residents
        # are displaced/restored; dispatching through stale endpoint state can
        # therefore reach a different deployment (or even a different model)
        # while the catalog correctly reports the selected Teacher unavailable.
        # Fail closed before any capability probe or inference. The local NIM
        # lifecycle repairs exact compatible former-consumer attachments when
        # a replacement resident becomes healthy.
        if not endpoint_is_operational(endpoint):
            return (
                "nim_unreachable: The selected Teacher endpoint is disabled or "
                "unhealthy. Choose an available Teacher or restore its NIM "
                "deployment before requesting a proposal."
            )

        # Snapshot data before closing session
        project_dir = project.project_dir
        effective_preset = (
            generation_preset_key_override or project.labeling_generation_preset_key
        )
        thinking_on = (
            (thinking_mode_override == "on")
            if thinking_mode_override is not None
            else project.thinking_default_on
        )
        effective_vb_preset = (
            visual_budget_preset_key_override or project.visual_budget_preset_key
        )
        endpoint_base_url = endpoint.base_url
        snap_auth_mode = endpoint.auth_mode
        model_name = model_config.model_name
        # max_images_per_request resolves via the per-endpoint override
        # (NimEndpoint.max_images_per_request, set on auto-registered
        # local NIMs) and falls back to the per-model value (the gateway-
        # correct hosted cap of e.g. 8). See
        # ``services/image_cap_resolver.py``.
        from vlm_feedback_loop.services.image_cap_resolver import (
            resolve_max_images_per_request,
        )

        mc_input = ModelConfigInput(
            context_window_tokens=model_config.context_window_tokens,
            thinking_toggle_mode=model_config.thinking_toggle_mode or "none",
            thinking_toggle_support=model_config.thinking_toggle_support or "unknown",
            visual_budget_mode=model_config.visual_budget_mode or "none",
            visual_budget_support=model_config.visual_budget_support or "unknown",
            structured_generation_support=model_config.structured_generation_support
            or "unknown",
            max_images_per_request=resolve_max_images_per_request(
                model_config=model_config, nim_endpoint=endpoint
            ),
            default_icl_max_examples=model_config.default_icl_max_examples,
        )
        guidance_schema = guidance.schema or {}
        guidance_fields = guidance_schema.get("fields", [])
        generation_order = guidance_schema.get("generation_order", [])
        derived_json_schema = guidance_schema.get("derived_json_schema", {})
        guidance_description = guidance.description or ""
        guidance_rules = guidance.rules or ""
        guidance_id = guidance.guidance_id
        endpoint_id = endpoint.endpoint_id
        model_config_id = model_config.model_config_id
        storage_ref = example.storage_ref

    # ── 6. Auto-probe capabilities on first use ──────────────────────────
    # Seeded ModelConfigs start at `structured_generation_support="unknown"`
    # (plus thinking_toggle_support, visual_budget_support). If nothing
    # ever probes them, the backend is stuck sending prompt-only JSON
    # even to models that fully support `response_format=json_schema` —
    # which costs the SME a preventable failure class (fenced output,
    # invented field names) on the first proposal. Probe lazily here so
    # the cost is a one-time ~3 s penalty on the first proposal of each
    # (project, model) pair. The probes persist to the DB, so subsequent
    # proposals see "supported"/"unsupported" and skip this block.
    needs_probe = (
        mc_input.structured_generation_support == "unknown"
        or mc_input.thinking_toggle_support == "unknown"
        or mc_input.visual_budget_support == "unknown"
    )
    if needs_probe:
        # Lazy import to avoid tightening the service-dependency graph; the
        # two modules are otherwise peers and reprobe_model_config is the
        # same entry point used by the `:reprobe` endpoint.
        from vlm_feedback_loop.services import model_config_service

        probe_result = await model_config_service.reprobe_model_config(
            project_id,
            model_config_id,
            settings.WORKSPACE_ROOT,
            settings,
        )
        if isinstance(probe_result, ModelConfig):
            mc_input = ModelConfigInput(
                context_window_tokens=probe_result.context_window_tokens,
                thinking_toggle_mode=probe_result.thinking_toggle_mode or "none",
                thinking_toggle_support=probe_result.thinking_toggle_support
                or "unknown",
                visual_budget_mode=probe_result.visual_budget_mode or "none",
                visual_budget_support=probe_result.visual_budget_support or "unknown",
                structured_generation_support=probe_result.structured_generation_support
                or "unknown",
                # Resolve through the same NimEndpoint override as the
                # initial ModelConfigInput build, so the post-probe rebuild
                # honors per-endpoint caps too. ``probe_result`` is the
                # newly-updated ModelConfig row; ``endpoint`` was loaded
                # alongside ``model_config`` upstream and is still in scope.
                max_images_per_request=resolve_max_images_per_request(
                    model_config=probe_result, nim_endpoint=endpoint
                ),
                default_icl_max_examples=probe_result.default_icl_max_examples,
            )

    # ── 7. Generate inference_invocation_id ───────────────────────────────
    inference_invocation_id = generate_uuid4()

    # ── 8. Persist pending OperationRecord (BEFORE NIM call) ────────────
    with Session(engine) as session:
        pending_record = OperationRecord(
            inference_invocation_id=inference_invocation_id,
            project_id=project_id,
            purpose="interactive_proposal",
            example_key=example_key,
            guidance_id=guidance_id,
            model_config_id=model_config_id,
            endpoint_id=endpoint_id,
            model_name=model_name,
            invocation_status="pending",
            label_tier="proposal",
            retry_of_inference_invocation_id=retry_of_inference_invocation_id,
        )
        session.add(pending_record)
        session.commit()

    # ── 9. Resolve auth headers ──────────────────────────────────────────
    # Honor the endpoint's auth_mode via the shared builder — the same one
    # eval/batch use. Only hosted (bearer) endpoints get a Bearer header;
    # self-hosted / local endpoints (auth_mode="none") get none.
    auth_headers = build_endpoint_auth_headers(
        snap_auth_mode,
        get_effective_secret("NVIDIA_API_KEY", settings),
    )

    # ── 10. Query ICL candidates ─────────────────────────────────────────
    with Session(engine) as session:
        icl_candidates = query_icl_candidates(
            session,
            project_id,
            guidance_id,
        )

    # ── 11. (ModelConfigInput, guidance envelope already built above) ────

    # ── 12. Call invoke_teacher() ────────────────────────────────────────
    # One invocation builder for the initial call and the (up to 3)
    # capability-fallback retries below — they differ ONLY in the
    # model_config projection and the thinking flag.
    async def _invoke_with(
        model_config: ModelConfigInput, thinking_flag: bool
    ) -> TeacherInvocationResult:
        return await invoke_teacher(
            project_id=project_id,
            example_key=example_key,
            purpose="interactive_proposal",
            inference_invocation_id=inference_invocation_id,
            guidance_description=guidance_description,
            guidance_rules=guidance_rules,
            guidance_fields=guidance_fields,
            generation_order=generation_order,
            derived_json_schema=derived_json_schema,
            model_name=model_name,
            model_config=model_config,
            endpoint_base_url=endpoint_base_url,
            auth_headers=auth_headers,
            icl_candidates=icl_candidates,
            generation_preset_key=effective_preset,
            thinking_on=thinking_flag,
            visual_budget_preset_key=effective_vb_preset,
            **token_budget_invoke_kwargs(settings),
            icl_max_examples=settings.ICL_MAX_EXAMPLES,
            icl_sim_gap=settings.ICL_SIM_GAP,
            icl_abs_threshold=settings.ICL_ABS_THRESHOLD,
            scope_id=None,  # no seed for interactive proposals
            deadline_s=float(settings.HTTP_DEADLINE_INTERACTIVE_S),
            max_retries=settings.HTTP_MAX_RETRIES,
            query_storage_ref=storage_ref,
            image_transport_max_longest_edge=(
                settings.IMAGE_TRANSPORT_MAX_LONGEST_EDGE
            ),
            settings=settings,
        )

    teacher_result = await _invoke_with(mc_input, thinking_on)

    # ── 13. Handle structured gen runtime rejection ──────────────────────
    # Track the cumulative disabled state across the (up to 3) fallback
    # retries. Each retry preserves prior disablings so a model that
    # rejects structured-gen then rejects thinking-toggle can still
    # complete via prompt-only + Thinking-ON-default + visual-budget.
    structured_gen_fallback = False
    thinking_fallback = False
    visual_budget_fallback = False
    mc_input_current = mc_input
    if _is_structured_gen_rejection(teacher_result):
        logger.info(
            "Structured generation rejected; retrying without response_format",
            extra={
                "component": "schema_validation",
                "project_id": project_id,
                "correlation_id": inference_invocation_id,
            },
        )
        mc_input_current = replace(
            mc_input_current, structured_generation_support="unsupported"
        )
        teacher_result = await _invoke_with(mc_input_current, thinking_on)
        structured_gen_fallback = True

    # ── 14. Handle thinking-toggle runtime rejection ─────────────────────
    # If the model 4xx-rejects ``chat_template_kwargs``, retry once with
    # Thinking-ON (the model's natural reasoning behavior, no override).
    # Permanently demote ``thinking_toggle_support`` to ``"unsupported"`` so
    # subsequent invocations skip the override automatically. Gated by
    # ``ENABLE_THINKING_TOGGLE_FALLBACK`` — when False, the detector still
    # runs and populates the OperationRecord columns, but no retry happens.
    thinking_on_for_retry = thinking_on
    if settings.ENABLE_THINKING_TOGGLE_FALLBACK and _is_thinking_toggle_rejection(
        teacher_result
    ):
        logger.info(
            "Thinking toggle rejected; retrying without chat_template_kwargs",
            extra={
                "component": "capability_probe",
                "project_id": project_id,
                "correlation_id": inference_invocation_id,
            },
        )
        # Persistent demotion: future invocations skip the override.
        _demote_capability_support(engine, model_config_id, "thinking_toggle_support")
        mc_input_current = replace(
            mc_input_current, thinking_toggle_support="unsupported"
        )
        # Override thinking_on=True so the resolver returns no
        # chat_template_kwargs regardless of the SME's preference.
        thinking_on_for_retry = True
        teacher_result = await _invoke_with(mc_input_current, thinking_on_for_retry)
        thinking_fallback = True

    # ── 15. Handle visual-budget runtime rejection ───────────────────────
    # If the model 4xx-rejects ``mm_processor_kwargs``, retry once without
    # them. Demote ``visual_budget_support`` to ``"unsupported"`` for
    # subsequent invocations.
    if settings.ENABLE_VISUAL_BUDGET_FALLBACK and _is_visual_budget_rejection(
        teacher_result
    ):
        logger.info(
            "Visual budget rejected; retrying without mm_processor_kwargs",
            extra={
                "component": "capability_probe",
                "project_id": project_id,
                "correlation_id": inference_invocation_id,
            },
        )
        _demote_capability_support(engine, model_config_id, "visual_budget_support")
        mc_input_current = replace(
            mc_input_current, visual_budget_support="unsupported"
        )
        teacher_result = await _invoke_with(mc_input_current, thinking_on_for_retry)
        visual_budget_fallback = True

    # ── 16. Parse and validate response ──────────────────────────────────
    # Stage 4 timing — schema validation + normalization. Persisted as
    # ``operation_records.t_validation_ms`` to round out the per-stage
    # latency split.
    _validation_t0 = time.monotonic()
    validation_report = validate_proposal(
        teacher_result.content,
        guidance_fields,
        teacher_result.finish_reason,
    )
    t_validation_ms = int((time.monotonic() - _validation_t0) * 1000)

    # ── 17. Determine final invocation_status ────────────────────────────
    final_status = classify_invocation_status(teacher_result, validation_report)

    # ── 18. Write artifact files ─────────────────────────────────────────
    # Interactive proposals write the pretty-printed validation report with
    # per-field results (the Review screen's drill-down reads them).
    artifacts_dir = Path(project_dir) / "artifacts"
    artifact_refs = write_invocation_artifacts(
        artifacts_dir,
        inference_invocation_id,
        teacher_result=teacher_result,
        validation_report=validation_report,
        include_field_results=True,
    )

    # ── 19. Log point 2: schema_validation ───────────────────────────────
    _log_schema_validation(project_id, validation_report, inference_invocation_id)

    # ── 20. Update OperationRecord with all outcome fields ───────────────
    with Session(engine) as session:
        record = (
            session.query(OperationRecord)
            .filter_by(
                inference_invocation_id=inference_invocation_id,
            )
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
                # No run-level structured-generation mode: interactive
                # proposals use the per-invocation fallback retries above,
                # and the fallback flags record which of them fired.
                structured_generation_mode_effective=None,
                structured_generation_fallback_used=(
                    structured_gen_fallback
                    or teacher_result.structured_generation_fallback_used
                ),
                thinking_fallback_used=thinking_fallback,
                visual_budget_fallback_used=visual_budget_fallback,
            )
            session.commit()

    # ── Return result ────────────────────────────────────────────────────
    return ProposalResult(
        inference_invocation_id=inference_invocation_id,
        example_key=example_key,
        proposal_json=validation_report.normalized_json,
        schema_valid_core=validation_report.schema_valid_core,
        validation_errors_core=validation_report.core_errors,
        validation_errors_aux=validation_report.aux_errors,
        invocation_status=final_status,
        latency_ms_end_to_end=teacher_result.latency_ms,
        icl_images_attached_count=teacher_result.icl_images_attached_count,
        icl_example_keys_used=list(teacher_result.icl_example_keys_used),
        used_existing_label=False,
    )
