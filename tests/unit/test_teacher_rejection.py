# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the consolidated Teacher rejection detectors.

Proposal / evaluation / batch_label all consume one shared source
(``services.teacher_rejection``); these tests exercise that shared source
directly. Coverage:

  - canonical hosted-NIM error strings observed in live probes
  - the ``*_attempted`` flag and ``invocation_status`` short-circuit guards
  - heuristic specificity (no false positives on transport errors or
    outages), including the regression guard that a bare "400" must NOT
    be read as a structured-gen rejection
  - the one-source-of-truth invariant: each service reuses the shared
    detectors rather than carrying its own copy
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vlm_feedback_loop.services import teacher_rejection
from vlm_feedback_loop.services.teacher_rejection import (
    is_structured_gen_rejection,
    is_thinking_toggle_rejection,
    is_visual_budget_rejection,
)

# ── Test fixture: minimal TeacherInvocationResult-like surface ──────────────


@dataclass
class _StubResult:
    """Subset of TeacherInvocationResult fields the detectors read.

    Keeping this minimal avoids cross-coupling to the full dataclass —
    detectors only access ``*_attempted``, ``invocation_status``, and
    ``error`` via ``getattr``.
    """

    invocation_status: str = "success"
    error: str | None = None
    structured_generation_attempted: bool = False
    thinking_toggle_attempted: bool = False
    visual_budget_attempted: bool = False


# ── One source of truth across the three consuming services ─────────────────


def test_services_reuse_the_shared_detectors():
    """proposal / evaluation / batch_label consume teacher_rejection's code.

    A prior production incident had to be patched in three hand-copied
    detector implementations at once; this pins that no service carries
    its own copy again. Proposal aliases the detectors directly (it
    retries single invocations); evaluation and batch labeling delegate
    through the shared run-level recorder.
    """
    from vlm_feedback_loop.services import (
        batch_label_service,
        evaluation_service,
        proposal_service,
    )

    assert (
        proposal_service._is_structured_gen_rejection
        is teacher_rejection.is_structured_gen_rejection
    )
    assert (
        proposal_service._is_thinking_toggle_rejection
        is teacher_rejection.is_thinking_toggle_rejection
    )
    assert (
        proposal_service._is_visual_budget_rejection
        is teacher_rejection.is_visual_budget_rejection
    )
    assert (
        evaluation_service.record_runtime_rejections
        is teacher_rejection.record_runtime_rejections
    )
    assert (
        batch_label_service.record_runtime_rejections
        is teacher_rejection.record_runtime_rejections
    )


# ── Structured-generation rejection heuristics ──────────────────────────────


class TestStructuredGenRejection:
    def test_response_format_error_triggers(self):
        assert is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error='HTTP 400: {"error": "response_format is not supported"}',
                structured_generation_attempted=True,
            )
        )

    def test_json_schema_error_triggers(self):
        assert is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error="HTTP 422: json_schema constraint invalid",
                structured_generation_attempted=True,
            )
        )

    def test_grammar_error_400_triggers(self):
        """A NIM guided-decoding grammar rejection (a keyword its backend
        can't compile, e.g. enum_set's uniqueItems) is a response_format
        rejection and must fall back to prompt-only. The provider message is
        folded into the error detail by http_client, so the detector sees it."""
        assert is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error='HTTP 400: Grammar error: Unimplemented keys: ["uniqueItems"]',
                structured_generation_attempted=True,
            )
        )

    def test_5xx_mentioning_json_schema_does_not_trigger(self):
        """A server-side failure whose body mentions json_schema is an
        outage, not a capability rejection — it must surface as
        endpoint_error rather than cancel the run as
        structured_generation_rejected (mirrors the 4xx guard the thinking
        and visual-budget detectors already carry)."""
        assert not is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error="HTTP 500: json_schema processing crashed upstream",
                structured_generation_attempted=True,
            )
        )

    def test_degraded_function_400_does_not_trigger(self):
        # Provider outage, not a schema rejection: nemotron-omni's hosted
        # function returned this while DEGRADED (2026-07-04..07) and the old
        # bare-"400" signature misclassified 16 whole evaluation runs as
        # structured_generation_rejected, scoring their pools as EM=0.0
        # "model errors" that passed every completeness gate.
        assert not is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error=(
                    'HTTP 400: {"status":400,"title":"Bad Request",'
                    '"detail":"Function id \'c4ed50ff\': DEGRADED function '
                    'cannot be invoked"}'
                ),
                structured_generation_attempted=True,
            )
        )

    def test_unrelated_400_does_not_trigger(self):
        assert not is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error="Exhausted 3 retries. Last: HTTP 400",
                structured_generation_attempted=True,
            )
        )

    def test_not_attempted_short_circuits(self):
        assert not is_structured_gen_rejection(
            _StubResult(
                invocation_status="endpoint_error",
                error="response_format rejected",
                structured_generation_attempted=False,
            )
        )

    def test_status_must_be_endpoint_error(self):
        assert not is_structured_gen_rejection(
            _StubResult(
                invocation_status="timeout",
                error="HTTP 400: response_format bad",
                structured_generation_attempted=True,
            )
        )


# ── Thinking-toggle rejection heuristics ────────────────────────────────────


class TestThinkingToggleRejection:
    """Thinking detector: chat_template_kwargs 4xx after a prior supported probe."""

    def test_attempted_false_short_circuits(self):
        """No detection when the request didn't carry chat_template_kwargs."""
        r = _StubResult(
            thinking_toggle_attempted=False,
            invocation_status="endpoint_error",
            error="chat_template_kwargs is not supported (HTTP 400)",
        )
        assert is_thinking_toggle_rejection(r) is False

    def test_status_must_be_endpoint_error(self):
        """A 200 response (success/timeout/schema_invalid) is never a rejection."""
        for status in ("success", "schema_invalid", "timeout", "rate_limited"):
            r = _StubResult(
                thinking_toggle_attempted=True,
                invocation_status=status,
                error="chat_template_kwargs is not supported (HTTP 400)",
            )
            assert is_thinking_toggle_rejection(r) is False, (
                f"status={status} should not trigger"
            )

    def test_canonical_qwen_error_string_triggers(self):
        """Hosted Qwen 3.5 returns this when chat_template_kwargs unsupported."""
        r = _StubResult(
            thinking_toggle_attempted=True,
            invocation_status="endpoint_error",
            error=(
                "HTTP 400: chat_template_kwargs.enable_thinking is not supported "
                "by this model"
            ),
        )
        assert is_thinking_toggle_rejection(r) is True

    def test_canonical_kimi_error_string_triggers(self):
        r = _StubResult(
            thinking_toggle_attempted=True,
            invocation_status="endpoint_error",
            error="HTTP 400: thinking_kwargs is not a recognized field",
        )
        assert is_thinking_toggle_rejection(r) is True

    def test_unrelated_endpoint_error_does_not_trigger(self):
        """Generic transport / 5xx errors must not be misclassified."""
        for err in (
            "Connection reset by peer",
            "HTTP 503: upstream timeout",
            "HTTP 429: too many requests",
            "HTTP 401: unauthorized",
            "HTTP 400: invalid response_format",  # structured_gen, not thinking
        ):
            r = _StubResult(
                thinking_toggle_attempted=True,
                invocation_status="endpoint_error",
                error=err,
            )
            assert is_thinking_toggle_rejection(r) is False, (
                f"err={err!r} false-positive"
            )

    def test_400_required_in_error_text(self):
        """Heuristic guards on ``"400"`` to anchor to client-side rejections."""
        r = _StubResult(
            thinking_toggle_attempted=True,
            invocation_status="endpoint_error",
            error="enable_thinking failed (transport reset)",  # no "400"
        )
        assert is_thinking_toggle_rejection(r) is False

    def test_null_error_does_not_trigger(self):
        r = _StubResult(
            thinking_toggle_attempted=True,
            invocation_status="endpoint_error",
            error=None,
        )
        assert is_thinking_toggle_rejection(r) is False


# ── Visual-budget rejection heuristics ──────────────────────────────────────


class TestVisualBudgetRejection:
    """Visual Budget detector: mm_processor_kwargs 4xx after a prior supported probe."""

    def test_attempted_false_short_circuits(self):
        r = _StubResult(
            visual_budget_attempted=False,
            invocation_status="endpoint_error",
            error="mm_processor_kwargs.max_pixels exceeded (HTTP 400)",
        )
        assert is_visual_budget_rejection(r) is False

    def test_status_must_be_endpoint_error(self):
        for status in ("success", "schema_invalid", "timeout", "rate_limited"):
            r = _StubResult(
                visual_budget_attempted=True,
                invocation_status=status,
                error="mm_processor_kwargs (HTTP 400)",
            )
            assert is_visual_budget_rejection(r) is False, (
                f"status={status} should not trigger"
            )

    def test_canonical_cosmos_error_triggers(self):
        """mm_processor_size mode rejection (Cosmos / Nemotron family)."""
        r = _StubResult(
            visual_budget_attempted=True,
            invocation_status="endpoint_error",
            error=(
                "HTTP 400: mm_processor_kwargs.size.shortest_edge below "
                "image_size.min for this model"
            ),
        )
        assert is_visual_budget_rejection(r) is True

    @pytest.mark.parametrize(
        "signal",
        [
            "mm_processor_kwargs",
            "max_pixels",
            "min_pixels",
            "image_size",
            "image_pixels",
            "min_image_tokens",
            "max_image_tokens",
        ],
    )
    def test_canonical_signals_trigger(self, signal):
        r = _StubResult(
            visual_budget_attempted=True,
            invocation_status="endpoint_error",
            error=f"HTTP 400: {signal} is invalid",
        )
        assert is_visual_budget_rejection(r) is True

    def test_unrelated_endpoint_error_does_not_trigger(self):
        for err in (
            "Connection reset by peer",
            "HTTP 503: upstream timeout",
            "HTTP 401: unauthorized",
            "HTTP 400: chat_template_kwargs unsupported",  # thinking, not vb
            "HTTP 400: invalid response_format",  # structured_gen, not vb
        ):
            r = _StubResult(
                visual_budget_attempted=True,
                invocation_status="endpoint_error",
                error=err,
            )
            assert is_visual_budget_rejection(r) is False, f"err={err!r} false-positive"

    def test_400_required_in_error_text(self):
        r = _StubResult(
            visual_budget_attempted=True,
            invocation_status="endpoint_error",
            error="mm_processor_kwargs reset (transport)",  # no "400"
        )
        assert is_visual_budget_rejection(r) is False
