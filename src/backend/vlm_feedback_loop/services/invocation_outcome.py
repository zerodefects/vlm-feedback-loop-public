# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Teacher-invocation outcome pipeline.

One implementation of the three steps every Teacher invocation finishes
with — artifact writing, final-status classification, and OperationRecord
outcome persistence — previously hand-copied across ``proposal_service``,
``evaluation_service``, and ``batch_label_service``. The per-service
divergences (proposal's per-field validation detail and per-invocation
fallback flags, evaluation's exact-match verdict, the run-level
structured-generation mode) are explicit parameters, so a fix here reaches
all three callers at once without erasing their intentional differences.

A second, transport-result variant of the same pipeline
(``classify_transport_status`` / ``apply_transport_invocation_outcome``)
serves ``rationale_service``, which bypasses ``invoke_teacher`` — it renders
its own prompts, calls ``nim_client.chat_completions`` directly, and
therefore holds a raw ``NimChatCompletionsResult`` rather than a
``TeacherInvocationResult``.

Session management stays with the callers: nothing in this module opens,
queries, or commits a database session.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.services.exact_match_evaluator import SchemaValidationReport
from vlm_feedback_loop.services.nim_client import NimChatCompletionsResult
from vlm_feedback_loop.services.prompt_service import (
    TeacherInvocationResult,
    is_rate_limit_exhaustion,
)

logger = logging.getLogger("vlm_feedback_loop.invocation_outcome")


def write_invocation_artifact(path: Path, content: str) -> str | None:
    """Write an invocation artifact file; swallow OSError with a warn log."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)
    except OSError:
        logger.warning("Failed to write artifact: %s", path, exc_info=True)
        return None


@dataclass(frozen=True)
class InvocationArtifactRefs:
    """Paths of the four per-invocation artifact files (None = write failed
    or, for ``provider_error_ref``, no provider error to record)."""

    raw_ref: str | None
    normalized_ref: str | None
    validation_ref: str | None
    provider_error_ref: str | None


def write_invocation_artifacts(
    artifacts_dir: Path,
    inference_invocation_id: str,
    *,
    teacher_result: TeacherInvocationResult,
    validation_report: SchemaValidationReport,
    include_field_results: bool = False,
) -> InvocationArtifactRefs:
    """Write the four per-invocation artifact files and return their refs.

    ``include_field_results=True`` is the interactive-proposal variant: the
    validation report additionally carries the per-field results (the Review
    screen's drill-down reads them) and is pretty-printed for human
    inspection. Evaluation and batch labeling write the compact five-key
    report — at pool scale the per-field detail would multiply artifact
    volume for data nothing reads.
    """
    raw_ref = write_invocation_artifact(
        artifacts_dir / f"{inference_invocation_id}_raw.txt",
        teacher_result.content or "",
    )
    normalized_ref = write_invocation_artifact(
        artifacts_dir / f"{inference_invocation_id}_normalized.json",
        json.dumps(validation_report.normalized_json, separators=(",", ":"))
        if validation_report.normalized_json is not None
        else "null",
    )

    validation_payload: dict[str, Any] = {
        "schema_valid_core": validation_report.schema_valid_core,
        "core_errors": validation_report.core_errors,
        "aux_errors": validation_report.aux_errors,
        "parse_error": validation_report.parse_error,
        "truncation_attributed_schema_invalid": (
            validation_report.truncation_attributed_schema_invalid
        ),
    }
    if include_field_results:
        validation_payload["field_results"] = [
            {
                "field_name": fr.field_name,
                "field_type": fr.field_type,
                "role": fr.role,
                "valid": fr.valid,
                "error": fr.error,
                "normalization_steps": fr.normalization_steps,
            }
            for fr in validation_report.field_results
        ]
        validation_content = json.dumps(validation_payload, indent=2)
    else:
        validation_content = json.dumps(validation_payload, separators=(",", ":"))
    validation_ref = write_invocation_artifact(
        artifacts_dir / f"{inference_invocation_id}_validation.json",
        validation_content,
    )

    provider_error_ref: str | None = None
    if teacher_result.error:
        provider_error_ref = write_invocation_artifact(
            artifacts_dir / f"{inference_invocation_id}_error.txt",
            teacher_result.error,
        )

    return InvocationArtifactRefs(
        raw_ref=raw_ref,
        normalized_ref=normalized_ref,
        validation_ref=validation_ref,
        provider_error_ref=provider_error_ref,
    )


def classify_invocation_status(
    teacher_result: TeacherInvocationResult,
    validation_report: SchemaValidationReport,
) -> str:
    """Classify the final invocation_status for an invocation.

    A ``timeout``/``endpoint_error`` is reclassified ``rate_limited`` when
    the retry budget was exhausted by 429s — the UI then renders "the hosted
    NIM is rate-limiting, wait and retry" copy instead of the catch-all
    endpoint-failure banner (self-hosting the NIM removes the shared
    per-account cap; the loop stays usable on hosted with operator
    awareness). Otherwise the transport status passes through; a delivered
    response classifies on SchemaCore validity.
    """
    if teacher_result.invocation_status in ("timeout", "endpoint_error"):
        if is_rate_limit_exhaustion(teacher_result):
            return "rate_limited"
        return teacher_result.invocation_status
    if validation_report.schema_valid_core:
        return "success"
    return "schema_invalid"


def apply_invocation_outcome(
    record: OperationRecord,
    *,
    final_status: str,
    teacher_result: TeacherInvocationResult,
    validation_report: SchemaValidationReport,
    t_validation_ms: int,
    artifact_refs: InvocationArtifactRefs,
    structured_generation_mode_effective: str | None,
    structured_generation_fallback_used: bool,
    thinking_fallback_used: bool,
    visual_budget_fallback_used: bool,
    exact_match_pass: bool | None = None,
) -> None:
    """Copy all invocation outcome fields onto a pending OperationRecord.

    Mutates ``record`` in place; the caller owns the session and the commit.
    ``exact_match_pass`` is only meaningful for evaluation (the ground-truth
    verdict); the column is nullable and the record is created pending in
    the same call frame, so the default ``None`` is a no-op for proposal and
    batch labeling.
    """
    usage = teacher_result.usage or {}

    record.invocation_status = final_status
    record.latency_ms_end_to_end = teacher_result.latency_ms

    # Per-stage timings (they sum to within ±5% of latency_ms_end_to_end).
    # t_validation_ms is measured by the calling service — its window is
    # service-specific (evaluation includes ground-truth field matching) —
    # while the other three come from invoke_teacher.
    record.t_image_prep_ms = teacher_result.t_image_prep_ms
    record.t_prompt_render_ms = teacher_result.t_prompt_render_ms
    record.t_nim_call_ms = teacher_result.t_nim_call_ms
    record.t_validation_ms = t_validation_ms

    # Generation Controls
    record.generation_preset_key = teacher_result.generation_preset_key
    record.sampling_params_effective = teacher_result.sampling_params_effective
    record.thinking_mode_effective = teacher_result.thinking_mode_effective
    record.thinking_request_fields_effective = (
        teacher_result.thinking_request_fields_effective
    )
    record.max_tokens_effective = teacher_result.max_tokens_effective
    record.reasoning_headroom_tokens_effective = (
        teacher_result.reasoning_headroom_tokens_effective
    )

    # Visual Budget
    record.visual_budget_preset_key = teacher_result.visual_budget_preset_key
    record.visual_budget_params_effective = (
        teacher_result.visual_budget_params_effective
    )

    # Image transport
    record.image_transport_mode = teacher_result.image_transport_mode
    record.image_format_transmitted = teacher_result.image_format_transmitted

    # Inline ICL image injection
    record.icl_images_attached_count = teacher_result.icl_images_attached_count

    # ICL keys used
    record.icl_example_keys_used = teacher_result.icl_example_keys_used

    # Prompt + seed reproducibility
    record.prompt_hash = teacher_result.prompt_hash
    record.seed_effective = teacher_result.seed_effective

    # Artifact refs
    record.raw_model_response_ref = artifact_refs.raw_ref
    record.normalized_json_ref = artifact_refs.normalized_ref
    record.validation_report_ref = artifact_refs.validation_ref
    record.provider_error_ref = artifact_refs.provider_error_ref

    # Validation report
    record.schema_valid_core = validation_report.schema_valid_core
    record.validation_errors_core = validation_report.core_errors
    record.validation_errors_aux = validation_report.aux_errors

    # Provider token usage + completion/truncation
    record.finish_reason = teacher_result.finish_reason
    record.prompt_tokens = usage.get("prompt_tokens")
    record.completion_tokens = usage.get("completion_tokens")
    record.total_tokens = usage.get("total_tokens")
    record.truncation_attributed_schema_invalid = (
        validation_report.truncation_attributed_schema_invalid
    )

    # Structured generation + Thinking + Visual Budget runtime rejection.
    # ``*_attempted`` reflects whether the dispatched request actually
    # carried the corresponding control (set by invoke_teacher when the
    # resolver returned non-null). ``*_fallback_used`` and the run-level
    # structured-generation mode are caller-supplied: interactive proposals
    # retry per invocation with the rejected control disabled, while
    # evaluation and batch labeling never retry mid-run (mode flips break
    # reproducibility) and fail the whole run instead.
    record.structured_generation_attempted = (
        teacher_result.structured_generation_attempted
    )
    record.structured_generation_fallback_used = structured_generation_fallback_used
    record.structured_generation_mode_effective = structured_generation_mode_effective
    record.thinking_toggle_attempted = teacher_result.thinking_toggle_attempted
    record.thinking_fallback_used = thinking_fallback_used
    record.visual_budget_attempted = teacher_result.visual_budget_attempted
    record.visual_budget_fallback_used = visual_budget_fallback_used

    # Evaluation-specific scoring
    record.exact_match_pass = exact_match_pass


# ── Transport-result variant (invoke_teacher bypass path) ────────────────────
# ``rationale_service`` produces free-form text (a rationale note), not a
# schema-validated label JSON, so it skips ``invoke_teacher`` and calls the
# NIM transport directly.
# The two helpers below are the transport-result counterparts of
# ``classify_invocation_status`` / ``apply_invocation_outcome``: no schema
# validation (there is no ``schema_invalid`` state), no rate-limit
# reclassification (the raw transport error string is preserved verbatim in
# ``provider_error_ref``, truncated to 500 chars).


def classify_transport_status(nim_result: NimChatCompletionsResult) -> str:
    """Classify invocation_status straight from the raw transport result.

    ``success`` / ``timeout`` / ``endpoint_error`` — the three states a
    free-form-text invocation can land in. ``timeout`` keys on the
    ``"timed out"`` message ``nim_client`` puts in ``error`` for connect and
    read timeouts; everything else that failed is ``endpoint_error``.
    """
    if nim_result.success:
        return "success"
    if nim_result.error and "timed out" in nim_result.error.lower():
        return "timeout"
    return "endpoint_error"


def apply_transport_invocation_outcome(
    record: OperationRecord,
    *,
    invocation_status: str,
    nim_result: NimChatCompletionsResult,
    elapsed_ms: int,
    raw_ref: str | None,
    generation_preset_key: str,
    sampling_params_effective: dict[str, float],
    max_tokens_effective: int,
    thinking: dict[str, Any],
    visual_budget: dict[str, Any],
    prompt_hash: str,
    t_image_prep_ms: int | None,
    t_nim_call_ms: int,
    t_prompt_render_ms: int | None = None,
    image_transport_mode: str | None = None,
    image_format_transmitted: str | None = None,
    reasoning_headroom_tokens_effective: int | None = None,
) -> None:
    """Copy transport-invocation outcome fields onto a pending OperationRecord.

    Mutates ``record`` in place; the caller owns the session and the commit.
    ``thinking`` / ``visual_budget`` are the dicts returned by
    ``prompt_service.resolve_thinking_fields`` / ``resolve_visual_budget`` —
    the effective state and ``*_attempted`` flags are derived from them
    exactly as ``invoke_teacher`` records for proposal/eval/batch.

    The trailing keyword defaults cover rationale regeneration's extras:
    it times its prompt render, records image transport, and adds
    reasoning headroom. The record was created pending in the same call
    frame, so assigning the default ``None`` persists the same NULL as
    never touching the column.

    Hardwired by design: no structured generation (free-form text
    output), no thinking / visual-budget runtime fallback (the service
    never retries an invocation with a rejected control disabled), and
    ``t_validation_ms`` stays NULL (no schema validation step).
    """
    usage = nim_result.usage or {}

    record.invocation_status = invocation_status
    record.latency_ms_end_to_end = elapsed_ms
    record.finish_reason = nim_result.finish_reason
    record.prompt_tokens = usage.get("prompt_tokens")
    record.completion_tokens = usage.get("completion_tokens")
    record.total_tokens = usage.get("total_tokens")
    record.raw_model_response_ref = raw_ref

    # Generation Controls
    record.generation_preset_key = generation_preset_key
    record.sampling_params_effective = sampling_params_effective
    record.max_tokens_effective = max_tokens_effective
    record.reasoning_headroom_tokens_effective = reasoning_headroom_tokens_effective
    record.thinking_mode_effective = thinking["thinking_mode_effective"]
    record.thinking_request_fields_effective = thinking["thinking_request_fields"]
    record.thinking_toggle_attempted = thinking["thinking_request_fields"] is not None
    record.thinking_fallback_used = False

    # Visual Budget
    record.visual_budget_preset_key = visual_budget["visual_budget_preset_key"]
    record.visual_budget_params_effective = visual_budget[
        "visual_budget_params_effective"
    ]
    record.visual_budget_attempted = (
        visual_budget["visual_budget_params_effective"] is not None
    )
    record.visual_budget_fallback_used = False

    # Image transport
    record.image_transport_mode = image_transport_mode
    record.image_format_transmitted = image_format_transmitted

    # Per-stage timings — same instrumentation the proposal/eval/batch
    # invocations record via invoke_teacher.
    record.t_image_prep_ms = t_image_prep_ms
    record.t_prompt_render_ms = t_prompt_render_ms
    record.t_nim_call_ms = t_nim_call_ms

    # Structured-generation flags: set explicitly rather than relying on
    # column defaults.
    record.structured_generation_attempted = False
    record.structured_generation_fallback_used = False

    # Prompt reproducibility — seed not applicable on these paths.
    record.prompt_hash = prompt_hash

    if nim_result.error:
        record.provider_error_ref = nim_result.error[:500]
