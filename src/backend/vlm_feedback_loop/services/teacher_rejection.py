# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Teacher runtime-capability rejection detectors.

A Teacher (or hosted function) can reject a generation control with a 4xx even
when the model-config capability flags said it was supported. These heuristics
classify an ``endpoint_error`` as a structured-generation / thinking-toggle /
visual-budget rejection so the caller can retry without that control.

Single source of truth: previously hand-copied into ``proposal_service``,
``evaluation_service``, and ``batch_label_service``. A prior production incident
(the bare-"400" fix below) had to be applied to all three copies at
once — exactly the drift risk this module removes. Inputs are duck-typed
(``Any``) because the three call sites pass different result objects
(``TeacherInvocationResult`` and the eval/batch equivalents) that share the same
attribute names.

Besides the detectors, this module owns the RUN-LEVEL rejection lifecycle
shared by evaluation and batch labeling: per-run registries that a concurrent
or sequential invocation writes into when a detector fires
(:func:`record_runtime_rejections`), the one finalizer that transitions
the rejected run to ``failed`` with a capability-specific ``status_reason``
and emits the ``run_failed`` SSE event (:func:`finalize_runtime_rejection`),
the one cancel finalizer that transitions a canceling run to ``canceled``
(:func:`finalize_canceled`), the outcome-time cancellation classifier
(:func:`mark_operation_ignored_if_canceling`), and the one last-resort
finalizer for an executor's unhandled exception
(:func:`finalize_unhandled_exception`).
Interactive proposals do not use the registries — they retry the single
invocation with the rejected control disabled instead of failing a run.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.run_queries import update_run_if_not_terminal
from vlm_feedback_loop.services.sse import sse_manager

logger = logging.getLogger("vlm_feedback_loop.services.teacher_rejection")


def is_structured_gen_rejection(teacher_result: Any) -> bool:
    """Detect a 4xx ``response_format`` rejection."""
    if not getattr(teacher_result, "structured_generation_attempted", False):
        return False
    if getattr(teacher_result, "invocation_status", None) != "endpoint_error":
        return False
    error_lower = (getattr(teacher_result, "error", None) or "").lower()
    return (
        any(
            sig in error_lower
            # NOTE: a bare "400" OR-signature is deliberately absent. It
            # would make ANY HTTP 400 during an outage look like a schema
            # rejection — the hosted nemotron-omni function once returned
            # 400 "DEGRADED function cannot be invoked" and a batch of
            # evaluation runs was misclassified as
            # structured_generation_rejected, scoring whole pools as EM=0.0
            # "model errors". A genuine response_format rejection names the
            # offending field or the guided-decoding grammar; match on those
            # only. "grammar error"/"unimplemented keys" cover a NIM whose
            # guided-decoding backend can't compile a keyword we emitted
            # (e.g. enum_set's uniqueItems) — surfaced now that the provider
            # body is folded into the error detail (http_client).
            for sig in (
                "response_format",
                "json_schema",
                "json schema",
                "grammar error",
                "unimplemented keys",
            )
        )
        # AND-guard (mirrors the sibling detectors): a genuine
        # response_format rejection is a client-side 4xx — servers answer
        # 400 or 422 depending on the NIM. Without this, a 5xx whose body
        # happens to mention json_schema would cancel the whole run as a
        # capability rejection instead of surfacing the outage.
        and ("400" in error_lower or "422" in error_lower)
    )


def is_thinking_toggle_rejection(teacher_result: Any) -> bool:
    """Detect a 4xx ``chat_template_kwargs`` thinking-toggle rejection."""
    if not getattr(teacher_result, "thinking_toggle_attempted", False):
        return False
    if getattr(teacher_result, "invocation_status", None) != "endpoint_error":
        return False
    error_lower = (getattr(teacher_result, "error", None) or "").lower()
    return (
        any(
            sig in error_lower
            for sig in (
                "chat_template_kwargs",
                "enable_thinking",
                "thinking_kwargs",
                "thinking is not supported",
                "enable_thinking is not supported",
            )
        )
        and "400" in error_lower
    )


def is_visual_budget_rejection(teacher_result: Any) -> bool:
    """Detect a 4xx ``mm_processor_kwargs`` visual-budget rejection."""
    if not getattr(teacher_result, "visual_budget_attempted", False):
        return False
    if getattr(teacher_result, "invocation_status", None) != "endpoint_error":
        return False
    error_lower = (getattr(teacher_result, "error", None) or "").lower()
    return (
        any(
            sig in error_lower
            for sig in (
                "mm_processor_kwargs",
                "max_pixels",
                "min_pixels",
                "image_size",
                "image_pixels",
                "min_image_tokens",
                "max_image_tokens",
            )
        )
        and "400" in error_lower
    )


# ── Run-level rejection registries (evaluation + batch labeling) ────────────

# Per-run rejection registries. When an invocation inside an evaluation or
# batch-label run hits a runtime capability rejection, it records the error
# text here keyed by run_id and signals the run's cancel event so the run's
# remaining work stops. The run-level finalizer then transitions the run to
# ``failed`` with the capability-specific ``status_reason`` — distinct from
# the ``incomplete``/``canceled`` branches, so the UI's failed-run banner can
# offer the matching restart action (e.g. restart with prompt-only) and the
# operator knows which capability flag was demoted. Entries are cleaned up
# after finalization (and defensively in the executors' ``finally`` blocks).
# Structured generation only participates when the run's effective mode is
# ``auto``; the thinking and visual-budget detectors are gated by the
# ``ENABLE_THINKING_TOGGLE_FALLBACK`` / ``ENABLE_VISUAL_BUDGET_FALLBACK``
# settings (default True; flip False for telemetry-only observation).
_structured_gen_rejected: dict[str, str] = {}
_thinking_toggle_rejected: dict[str, str] = {}
_visual_budget_rejected: dict[str, str] = {}

# Ordered rejection kinds: (registry, status_reason, default error text).
# Structured-gen first, then thinking, then visual budget — when multiple
# rejections fire in the same burst (rare), the first wins for the
# SME-facing message. The one finalization path for all three is
# :func:`finalize_runtime_rejection`.
REJECTION_KINDS: tuple[tuple[dict[str, str], str, str], ...] = (
    (
        _structured_gen_rejected,
        "structured_generation_rejected",
        "response_format rejected",
    ),
    (
        _thinking_toggle_rejected,
        "thinking_toggle_rejected",
        "chat_template_kwargs rejected",
    ),
    (
        _visual_budget_rejected,
        "visual_budget_rejected",
        "mm_processor_kwargs rejected",
    ),
)


def record_runtime_rejections(
    run_id: str,
    teacher_result: Any,
    *,
    sgm_effective: str,
    settings: Settings,
    cancel_event: asyncio.Event | None,
) -> None:
    """Record any run-level runtime rejections for one invocation result.

    Runs all three detectors sequentially (not first-match) so a burst that
    trips more than one registry is fully recorded; the finalizer's fixed
    priority order then picks the message. Failing the WHOLE run is the
    right semantics for evaluation/batch labeling because mid-run mode
    flips break reproducibility (silent mode mixing is forbidden) — the
    invocation's own OperationRecord still persists for audit. Each firing
    detector also sets the run's cancel event so sibling concurrent tasks
    (evaluation) break out, or the sequential loop (batch) stops at the top
    of its next iteration.
    """
    if sgm_effective == "auto" and is_structured_gen_rejection(teacher_result):
        _structured_gen_rejected[run_id] = (
            teacher_result.error or "response_format rejected"
        )
        if cancel_event is not None:
            cancel_event.set()

    if settings.ENABLE_THINKING_TOGGLE_FALLBACK and is_thinking_toggle_rejection(
        teacher_result
    ):
        _thinking_toggle_rejected[run_id] = (
            teacher_result.error or "chat_template_kwargs rejected"
        )
        if cancel_event is not None:
            cancel_event.set()

    if settings.ENABLE_VISUAL_BUDGET_FALLBACK and is_visual_budget_rejection(
        teacher_result
    ):
        _visual_budget_rejected[run_id] = (
            teacher_result.error or "mm_processor_kwargs rejected"
        )
        if cancel_event is not None:
            cancel_event.set()


def clear_runtime_rejections(run_id: str) -> None:
    """Drop a run's entries from all three rejection registries."""
    for registry, _status_reason, _default_error in REJECTION_KINDS:
        registry.pop(run_id, None)


@dataclass(frozen=True)
class RunExampleCounts:
    """Per-status example counters a caller wants persisted on the failed run.

    Batch labeling tracks these in loop-local variables (its RunRecord
    counters are only as fresh as the last per-example write), so it passes
    them here; evaluation aggregates counters from OperationRecords at
    finalize time and omits them.
    """

    succeeded: int
    schema_invalid: int
    timeout: int
    endpoint_error: int


async def finalize_runtime_rejection(
    engine: Any,
    project_id: str,
    run_id: str,
    *,
    run_type: str,
    terminal_statuses: frozenset[str],
    example_counts: RunExampleCounts | None = None,
) -> bool:
    """Finalize the run as ``failed`` if a runtime rejection was recorded.

    The one implementation for all three runtime-rejection kinds. Checks
    the registries in the fixed priority order of :data:`REJECTION_KINDS`
    and returns True when it finalized (False when no rejection was
    recorded for ``run_id``).

    Distinct from the services' cancel finalizers: (1) terminal status is
    ``failed`` not ``canceled``; (2) ``status_reason`` is set so the UI's
    failed-run banner renders capability-specific copy. Already-persisted
    OperationRecords stay as-is for audit. No metrics are written — partial
    aggregation from an aborted burst isn't meaningful.

    This helper never touches the services' per-run cancel-event registries —
    the executors' ``finally`` blocks own that cleanup.
    """
    for registry, status_reason, default_error in REJECTION_KINDS:
        if run_id not in registry:
            continue
        error = registry.pop(run_id, default_error)
        clear_runtime_rejections(run_id)
        with Session(engine) as session:
            values: dict[str, Any] = {
                "status": "failed",
                "status_reason": status_reason,
                "completed_at": utc_now(),
            }
            if example_counts is not None:
                values["examples_succeeded"] = example_counts.succeeded
                values["examples_schema_invalid"] = example_counts.schema_invalid
                values["examples_timeout"] = example_counts.timeout
                values["examples_endpoint_error"] = example_counts.endpoint_error
            applied = update_run_if_not_terminal(
                session, run_id, values, terminal_statuses=terminal_statuses
            )
            if applied:
                session.commit()
        # A recorded rejection still tells the executor to stop, but a
        # different terminal owner may already have won the durable race.
        # In that case it owns the client event too.
        if not applied:
            return True
        payload: dict[str, Any] = {
            "run_id": run_id,
            "run_type": run_type,
            "error_summary": error,
            "status_reason": status_reason,
        }
        await sse_manager.emit(project_id, "run_failed", payload)
        return True
    return False


def mark_operation_ignored_if_canceling(
    session: Session,
    operation: OperationRecord,
    run_id: str,
) -> bool:
    """Classify one outcome at the durable cancellation boundary.

    Flushing the terminal outcome acquires SQLite's writer lock before the
    parent status read. The outcome and ``running → canceling`` writes are
    therefore ordered: only an outcome that commits after cancellation is
    marked non-authoritative.
    """
    session.flush()
    run_status = (
        session.query(RunRecord.status).filter(RunRecord.run_id == run_id).scalar()
    )
    if run_status is None:
        raise RuntimeError(f"Operation parent run does not exist: run_id={run_id}")
    if run_status == "canceling":
        operation.ignored_due_to_run_cancellation = True
        return True
    return False


async def finalize_canceled(
    engine: Any,
    project_id: str,
    run_id: str,
    *,
    event_name: str,
) -> bool:
    """Transition ``canceling`` to ``canceled`` and emit its completed SSE.

    Invocation transactions classify their own outcomes at the durable
    cancellation boundary. This finalizer deliberately does not reclassify
    earlier authoritative outcomes. It emits only when the durable transition
    applies, so a failure signal cannot publish a contradictory canceled event.
    Returns whether it performed the transition, and leaves each service's
    in-memory cancel-event registry to that service.
    """
    applied = False
    with Session(engine) as session:
        applied = update_run_if_not_terminal(
            session,
            run_id,
            {"status": "canceled", "completed_at": utc_now()},
            only_status="canceling",
        )
        if applied:
            session.commit()
    if not applied:
        return False
    payload: dict[str, Any] = {
        "run_id": run_id,
        "status": "canceled",
    }
    await sse_manager.emit(project_id, event_name, payload)
    return True


async def finalize_unhandled_exception(
    engine: Any,
    project_id: str,
    run_id: str,
    *,
    run_type: str,
    error_summary: str,
    terminal_statuses: frozenset[str],
) -> None:
    """Last-resort finalizer for an executor's unhandled exception.

    The one implementation shared by evaluation and batch labeling,
    parameterized like the sibling finalizers. Marks the run ``failed``
    with ``status_reason="unhandled_exception"`` unless it already
    terminalized, then emits ``run_failed`` only when that transition wins.
    Never raises: the caller is already inside an ``except`` block, so a
    persistence failure is logged without publishing an uncommitted state.
    Like the siblings, this never touches the services' per-run cancel-event
    registries.
    """
    applied = False
    try:
        with Session(engine) as session:
            applied = update_run_if_not_terminal(
                session,
                run_id,
                {
                    "status": "failed",
                    "status_reason": "unhandled_exception",
                    "completed_at": utc_now(),
                },
                terminal_statuses=terminal_statuses,
            )
            if applied:
                session.commit()
    except Exception:
        logger.exception("Failed to mark run %s as failed", run_id)

    if not applied:
        return
    payload: dict[str, Any] = {
        "run_id": run_id,
        "run_type": run_type,
        "error_summary": error_summary,
    }
    await sse_manager.emit(project_id, "run_failed", payload)
