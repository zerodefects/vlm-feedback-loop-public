# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operation Record completeness audit for the interactive proposal path.

Operation Records MUST be persisted
"for all outcomes (success/invalid/timeout/endpoint_error), with best
available artifacts and sanitized error payload refs."  Individual tests
elsewhere assert a handful of fields per outcome; this suite does a single
comprehensive audit, exercising each of the four outcome classes and
verifying every required field is populated with a meaningful value
(not a silent default).

The failure paths matter most when an expensive run (e.g. evaluating a
trained Student) hits an inference failure: the OperationRecord is the
only diagnostic signal left, so it must be complete.

Scope: covers ``purpose=interactive_proposal`` via ``create_proposal()``.
The ``purpose=evaluation`` (``_invoke_for_evaluation``) and
``purpose=batch_label`` (``_invoke_for_batch_label``) pipelines delegate
to the same primitives (``invoke_teacher``, ``exact_match_evaluator``);
their OperationRecord coverage lives in
``test_evaluation_service.py::TestProfileBProductionPipeline`` and
``test_batch_label_service.py::TestProfileBProductionPipeline``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from conftest import (
    INVALID_PROPOSAL_JSON,
    VALID_PROPOSAL_JSON,
    make_teacher_result,
    seed_proposal_project,
)
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.services.proposal_service import create_proposal

# ── Required field lists ───────────────────────────────────────────────
#
# Identity + configuration-snapshot fields that MUST be populated on every
# outcome — even on timeout or endpoint_error, the backend knew which project
# /example/guidance/model it was trying to use and what Generation Controls
# applied.  If any of these is None after an invocation, a future diagnostic
# or reproducibility request cannot answer "what was tried here and why?"

REQUIRED_ALWAYS = [
    "inference_invocation_id",
    "project_id",
    "purpose",
    "example_key",
    "guidance_id",
    "model_config_id",
    "endpoint_id",
    "model_name",
    "generation_preset_key",
    "sampling_params_effective",
    "thinking_mode_effective",
    "max_tokens_effective",
    "label_tier",
    # Every invocation persists enough lineage to
    # reconstruct the exact prompt. ``prompt_hash`` is the SHA-256
    # of the rendered messages and is populated even on
    # timeout / endpoint_error (the prompt was rendered before the
    # request went out).
    "prompt_hash",
]


def _assert_required_always(record: OperationRecord) -> None:
    for field in REQUIRED_ALWAYS:
        value = getattr(record, field)
        assert value is not None, (
            f"REQUIRED_ALWAYS field '{field}' is None on "
            f"invocation_id={record.inference_invocation_id} "
            f"status={record.invocation_status}"
        )
    assert isinstance(record.sampling_params_effective, dict)
    assert "temperature" in record.sampling_params_effective
    assert "top_p" in record.sampling_params_effective
    assert record.thinking_mode_effective in ("on", "off")
    assert record.max_tokens_effective > 0
    assert record.purpose in (
        "interactive_proposal",
        "evaluation",
        "batch_label",
        "rationale_regeneration",
    )
    # prompt_hash is SHA-256 hex (64 lowercase hex chars).
    # In tests using ``make_teacher_result`` the canonical placeholder
    # ``"abc123"`` is shorter, so accept any non-empty string here and
    # let the per-test assertions tighten when they use a real hash.
    assert isinstance(record.prompt_hash, str) and record.prompt_hash


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETENESS AUDIT — one test per outcome class for interactive_proposal
# ═══════════════════════════════════════════════════════════════════════════════


class TestInteractiveProposalOperationRecordCompleteness:
    """OperationRecord fields are fully populated across all outcomes."""

    @pytest.mark.asyncio
    async def test_success_populates_all_required_fields(self, tmp_path: Path):
        engine, pid, _, _, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(content=VALID_PROPOSAL_JSON)
            ),
        ):
            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "success"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
        assert rec is not None

        _assert_required_always(rec)
        assert rec.purpose == "interactive_proposal"
        assert rec.label_tier == "proposal"
        assert rec.invocation_status == "success"
        assert rec.generation_preset_key == "precise"

        # image transport populated
        assert rec.image_transport_mode == "base64_inline"
        assert rec.image_format_transmitted == "image/jpeg"

        # visual budget
        assert rec.visual_budget_preset_key == "balanced"
        assert rec.visual_budget_params_effective is not None

        # structured generation
        assert rec.structured_generation_attempted is True
        assert rec.structured_generation_fallback_used is False

        # Success specifics
        assert rec.schema_valid_core is True
        assert rec.validation_errors_core in ([], None)
        assert rec.finish_reason == "stop"
        assert rec.prompt_tokens == 500
        assert rec.completion_tokens == 50
        assert rec.total_tokens == 550
        assert rec.provider_error_ref is None
        assert rec.latency_ms_end_to_end == 150
        assert rec.truncation_attributed_schema_invalid is False

        # Artifact references present for audit trail
        assert rec.raw_model_response_ref is not None
        assert rec.normalized_json_ref is not None
        assert rec.validation_report_ref is not None

        # run linkage
        assert rec.batch_label_run_id is None
        assert rec.evaluation_run_id is None
        assert rec.retry_of_inference_invocation_id is None

    @pytest.mark.asyncio
    async def test_schema_invalid_populates_all_required_fields(self, tmp_path: Path):
        engine, pid, _, _, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(content=INVALID_PROPOSAL_JSON)
            ),
        ):
            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "schema_invalid"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
        assert rec is not None

        _assert_required_always(rec)
        assert rec.purpose == "interactive_proposal"
        assert rec.invocation_status == "schema_invalid"

        # Schema-invalid specifics
        assert rec.schema_valid_core is False
        assert rec.validation_errors_core is not None
        assert len(rec.validation_errors_core) > 0, (
            "Core errors MUST list what went wrong when schema_invalid"
        )

        # The model did respond — finish reason and provider usage persist.
        assert rec.finish_reason is not None
        assert rec.prompt_tokens is not None
        assert rec.completion_tokens is not None
        assert rec.total_tokens is not None

        # Artifacts retained even on schema-invalid for audit.
        assert rec.raw_model_response_ref is not None
        assert rec.validation_report_ref is not None

        # Snapshot preserved
        assert rec.generation_preset_key == "precise"
        assert rec.visual_budget_preset_key == "balanced"

    @pytest.mark.asyncio
    async def test_timeout_populates_all_required_fields(self, tmp_path: Path):
        engine, pid, _, _, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(
                    invocation_status="timeout",
                    content=None,
                    error="Model did not respond within 120s deadline",
                    finish_reason=None,
                    usage=None,
                )
            ),
        ):
            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "timeout"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
        assert rec is not None

        _assert_required_always(rec)
        assert rec.purpose == "interactive_proposal"
        assert rec.invocation_status == "timeout"

        # Timeout specifics — no response → no finish_reason or tokens.
        assert rec.finish_reason is None
        assert rec.prompt_tokens is None
        assert rec.completion_tokens is None
        assert rec.total_tokens is None
        assert rec.schema_valid_core is False  # no usable output

        # Sanitized error MUST be retained so SME sees why this failed.
        assert rec.provider_error_ref is not None

        # Generation Controls / Visual Budget snapshot MUST still be
        # populated — they tell what we tried, not what succeeded.
        assert rec.generation_preset_key == "precise"
        assert rec.max_tokens_effective > 0
        assert rec.visual_budget_preset_key == "balanced"
        assert rec.sampling_params_effective["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_endpoint_error_populates_all_required_fields(self, tmp_path: Path):
        engine, pid, _, _, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(
                    invocation_status="endpoint_error",
                    content=None,
                    error="502 Bad Gateway: endpoint unreachable",
                    finish_reason=None,
                    usage=None,
                    structured_generation_attempted=False,
                )
            ),
        ):
            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "endpoint_error"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
        assert rec is not None

        _assert_required_always(rec)
        assert rec.purpose == "interactive_proposal"
        assert rec.invocation_status == "endpoint_error"

        # Endpoint-error specifics
        assert rec.finish_reason is None
        assert rec.prompt_tokens is None
        assert rec.completion_tokens is None
        assert rec.total_tokens is None
        assert rec.schema_valid_core is False
        assert rec.provider_error_ref is not None

        # The mock reports structured_generation_attempted=False
        # so no fallback was needed.
        assert rec.structured_generation_attempted is False
        assert rec.structured_generation_fallback_used is False


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt + seed reproducibility — round-trip from TeacherInvocationResult
# to the persisted OperationRecord row.
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptHashSeedPersistence:
    """``prompt_hash`` and ``seed_effective`` are computed on every
    invocation and MUST be persisted on the OperationRecord. Guards the
    "computed but dropped at persistence" failure mode."""

    @pytest.mark.asyncio
    async def test_interactive_proposal_persists_prompt_hash_and_null_seed(
        self, tmp_path: Path
    ):
        """Interactive proposals: prompt_hash persists; seed is None
        (interactive purpose does not set a deterministic seed)."""
        engine, pid, _, _, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]
        sentinel_hash = "deadbeef" * 8  # 64 lowercase hex chars — stand-in for SHA-256

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(
                    content=VALID_PROPOSAL_JSON,
                    prompt_hash=sentinel_hash,
                    seed_effective=None,
                )
            ),
        ):
            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "success"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
        assert rec is not None
        assert rec.prompt_hash == sentinel_hash, (
            "prompt_hash MUST round-trip from TeacherInvocationResult "
            "to the persisted OperationRecord row."
        )
        assert rec.seed_effective is None, (
            "Interactive proposals MUST NOT set a deterministic seed "
            "(only evaluation and batch_label do)."
        )


# OperationRecord coverage for the Profile-B inference wrappers
# (``_invoke_for_evaluation`` / ``_invoke_for_batch_label``) lives with
# their end-to-end pipelines against a mocked NIM wire:
#
#   - test_evaluation_service.py::TestProfileBProductionPipeline
#   - test_batch_label_service.py::TestProfileBProductionPipeline
#
# Both suites assert OperationRecord writes, so a refactor that
# accidentally skips those writes fails those tests loudly.
