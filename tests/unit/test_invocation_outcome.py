# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared invocation-outcome pipeline contracts.

Status classification and OperationRecord completeness are pinned
end-to-end in the proposal / evaluation / batch-label / rationale
suites; this file covers the contracts of ``invocation_outcome``
itself that none of those exercise: artifact-write failure tolerance,
the interactive-vs-pool-scale validation artifact divergence, the
machine-readability of failure artifacts, and the transport variant's
record mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from conftest import make_teacher_result
from support import fake_nim_failure, fake_nim_success
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.services.exact_match_evaluator import (
    FieldValidationResult,
    SchemaValidationReport,
)
from vlm_feedback_loop.services.invocation_outcome import (
    apply_transport_invocation_outcome,
    write_invocation_artifacts,
)
from vlm_feedback_loop.services.nim_client import NimChatCompletionsResult

# ── Helpers ──────────────────────────────────────────────────────────────────

# The five keys every validation artifact carries regardless of caller.
REPORT_KEYS = {
    "schema_valid_core",
    "core_errors",
    "aux_errors",
    "parse_error",
    "truncation_attributed_schema_invalid",
}


def make_validation_report(**overrides: Any) -> SchemaValidationReport:
    """Valid single-field report; ``overrides`` flip fields for failures."""
    defaults: dict[str, Any] = {
        "schema_valid_core": True,
        "core_errors": [],
        "aux_errors": [],
        "field_results": [
            FieldValidationResult(
                field_name="severity",
                field_type="enum",
                role="core",
                valid=True,
                normalized_value="high",
                error=None,
                normalization_steps=["lowercased"],
            )
        ],
        "normalized_json": {"severity": "high"},
        "parse_error": None,
        "truncation_attributed_schema_invalid": False,
    }
    defaults.update(overrides)
    return SchemaValidationReport(**defaults)


def apply_transport_outcome(
    record: OperationRecord,
    nim_result: NimChatCompletionsResult,
    *,
    invocation_status: str = "success",
    thinking: dict[str, Any] | None = None,
    visual_budget: dict[str, Any] | None = None,
) -> None:
    """Call ``apply_transport_invocation_outcome`` with rationale-shaped args."""
    apply_transport_invocation_outcome(
        record,
        invocation_status=invocation_status,
        nim_result=nim_result,
        elapsed_ms=1234,
        raw_ref=None,
        generation_preset_key="precise",
        sampling_params_effective={"temperature": 0.0, "top_p": 1.0},
        max_tokens_effective=256,
        thinking=thinking
        or {"thinking_mode_effective": "off", "thinking_request_fields": None},
        visual_budget=visual_budget
        or {"visual_budget_preset_key": None, "visual_budget_params_effective": None},
        prompt_hash="deadbeef" * 8,
        t_image_prep_ms=10,
        t_nim_call_ms=100,
    )


# ── Artifact writing ─────────────────────────────────────────────────────────


class TestArtifactWriteFailureTolerance:
    """An unwritable artifacts directory (disk full, permissions, a file
    squatting on the path) must degrade to None refs — never raise — so the
    invocation still completes and its OperationRecord persists with NULL
    artifact refs instead of the whole proposal/eval item crashing."""

    def test_unwritable_artifacts_dir_yields_none_refs(self, tmp_path: Path):
        blocker = tmp_path / "artifacts"
        blocker.write_text("a file where the artifacts dir should be")

        refs = write_invocation_artifacts(
            blocker / "sub",
            "inv-1",
            teacher_result=make_teacher_result(error="502 Bad Gateway"),
            validation_report=make_validation_report(),
        )

        assert refs.raw_ref is None
        assert refs.normalized_ref is None
        assert refs.validation_ref is None
        # The provider-error write is attempted (error is set) and must
        # fail closed like the others.
        assert refs.provider_error_ref is None


class TestValidationArtifactVariants:
    """The interactive/pool-scale artifact divergence is the parameter the
    shared pipeline exists to preserve: proposals persist per-field detail
    for on-disk inspection of a single invocation, evaluation and batch
    labeling write the compact five-key report so pool-scale runs don't
    multiply artifact volume."""

    def test_interactive_artifact_carries_per_field_results(self, tmp_path: Path):
        refs = write_invocation_artifacts(
            tmp_path / "artifacts",
            "inv-1",
            teacher_result=make_teacher_result(),
            validation_report=make_validation_report(),
            include_field_results=True,
        )

        assert refs.validation_ref is not None
        payload = json.loads(Path(refs.validation_ref).read_text(encoding="utf-8"))
        assert set(payload) == REPORT_KEYS | {"field_results"}
        assert payload["field_results"] == [
            {
                "field_name": "severity",
                "field_type": "enum",
                "role": "core",
                "valid": True,
                "error": None,
                "normalization_steps": ["lowercased"],
            }
        ]
        # No provider error → no error artifact for the audit trail.
        assert refs.provider_error_ref is None

    def test_pool_scale_artifact_is_compact_five_key_report(self, tmp_path: Path):
        refs = write_invocation_artifacts(
            tmp_path / "artifacts",
            "inv-1",
            teacher_result=make_teacher_result(),
            validation_report=make_validation_report(),
        )

        assert refs.validation_ref is not None
        content = Path(refs.validation_ref).read_text(encoding="utf-8")
        payload = json.loads(content)
        assert set(payload) == REPORT_KEYS
        # Compact separators, not pretty-printed — the artifact-volume half
        # of the contract.
        assert "\n" not in content


class TestFailureArtifactsMachineReadable:
    """A failed invocation's artifacts must stay parseable: label_service
    reads ``normalized_json_ref`` back with ``json.loads`` when the SME later
    labels the example, so the no-output case must serialize as JSON null,
    not an empty or missing file."""

    def test_no_output_invocation_writes_parseable_artifacts(self, tmp_path: Path):
        refs = write_invocation_artifacts(
            tmp_path / "artifacts",
            "inv-1",
            teacher_result=make_teacher_result(
                invocation_status="timeout",
                content=None,
                error="Request timed out",
            ),
            validation_report=make_validation_report(
                schema_valid_core=False,
                core_errors=["no response"],
                field_results=[],
                normalized_json=None,
                parse_error="no content",
            ),
        )

        assert refs.raw_ref is not None
        assert Path(refs.raw_ref).read_text(encoding="utf-8") == ""
        assert refs.normalized_ref is not None
        assert json.loads(Path(refs.normalized_ref).read_text(encoding="utf-8")) is None
        # The provider error is preserved verbatim for diagnostics.
        assert refs.provider_error_ref is not None
        assert (
            Path(refs.provider_error_ref).read_text(encoding="utf-8")
            == "Request timed out"
        )


# ── Transport-result variant (rationale path) ────────────────────────────────


class TestTransportOutcomeRecordMapping:
    """The rationale suite asserts only ``invocation_status``/``purpose``;
    the transport variant's record mapping logic is pinned here."""

    def test_provider_token_usage_is_preserved(self):
        """Free-form rationale calls retain exact provider usage just like
        schema-validated proposal, evaluation, and batch-label calls."""
        record = OperationRecord()
        apply_transport_outcome(record, fake_nim_success("a note"))

        assert record.prompt_tokens == 500
        assert record.completion_tokens == 50
        assert record.total_tokens == 550

    def test_provider_error_stored_truncated_to_500_chars(self):
        """A provider failure blob (e.g. a full HTML error page) lands in
        ``provider_error_ref`` capped at 500 chars — long errors must not be
        dumped whole into the DB column, and short ones are kept verbatim."""
        record = OperationRecord()
        long_error = "upstream said: " + "x" * 600
        apply_transport_outcome(
            record,
            fake_nim_failure(long_error),
            invocation_status="endpoint_error",
        )

        assert record.provider_error_ref == long_error[:500]
        assert len(record.provider_error_ref) == 500
        assert record.invocation_status == "endpoint_error"

    def test_attempted_flags_mirror_resolved_controls(self):
        """``thinking_toggle_attempted`` / ``visual_budget_attempted`` record
        whether the dispatched request actually carried the control — an
        operator debugging a Thinking rejection reads these to learn what was
        sent. Structured generation is never attempted on the free-form-text
        path."""
        attempted = OperationRecord()
        apply_transport_outcome(
            attempted,
            fake_nim_success("a note"),
            thinking={
                "thinking_mode_effective": "on",
                "thinking_request_fields": {"chat_template_kwargs": {"thinking": True}},
            },
            visual_budget={
                "visual_budget_preset_key": "balanced",
                "visual_budget_params_effective": {
                    "mm_processor_kwargs": {"size": {"shortest_edge": 672}}
                },
            },
        )
        assert attempted.thinking_toggle_attempted is True
        assert attempted.thinking_mode_effective == "on"
        assert attempted.visual_budget_attempted is True
        assert attempted.structured_generation_attempted is False

        not_attempted = OperationRecord()
        apply_transport_outcome(not_attempted, fake_nim_success("a note"))
        assert not_attempted.thinking_toggle_attempted is False
        assert not_attempted.visual_budget_attempted is False
