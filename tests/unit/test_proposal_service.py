# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the proposal service and endpoint.

Covers: the interactive proposal endpoint (request contract, response
shape, retry linkage, use-existing-label short-circuit), structured
generation runtime rejection and fallback, capability auto-probing,
Operation Record completeness, rate-limited reclassification, the
schema-validation log point, no secrets in artifacts, error handling,
and concurrency safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from conftest import (
    INVALID_PROPOSAL_JSON,
    VALID_PROPOSAL_JSON,
    add_endpoint_row,
    add_example_row,
    add_model_config_row,
    add_project_row,
    make_settings,
    make_teacher_result,
    open_project_workspace,
    seed_proposal_project,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord

# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_project_db(tmp_path: Path, project_id: str = "test-proj"):
    """Create a project directory and open its DB."""
    engine, project_dir, _ = open_project_workspace(tmp_path, project_id)
    return engine, str(project_dir)


def _add_project(session: Session, project_id: str, project_dir: str, **overrides):
    add_project_row(
        session, project_id, project_dir, **{"name": "Test Project", **overrides}
    )


def _add_endpoint(session: Session, project_id: str, endpoint_id: str):
    add_endpoint_row(session, project_id, endpoint_id, display_name="Test NIM")


def _add_model_config(
    session: Session, project_id: str, model_config_id: str, endpoint_id: str
):
    add_model_config_row(
        session,
        project_id,
        model_config_id,
        endpoint_id,
        model_name="nvidia/cosmos-reason2-8b",
        eligible_roles=json.dumps(["teacher"]),
        thinking_toggle_mode="qwen_enable_thinking",
        thinking_toggle_support="supported",
        visual_budget_mode="mm_processor_size",
        visual_budget_support="supported",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Interactive proposal endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestProposalRequestContract:
    """Request-shape contract. ProposalRequest was the ONLY
    inbound schema without extra="forbid" — a typo'd override field
    (e.g. ``thinking_override``) silently fell back to project defaults
    instead of 422ing like every sibling endpoint."""

    def test_unknown_field_rejected_at_schema_edge(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/any/proposals",
            json={"example_key": "k1", "thinking_override": "on"},
        )
        assert resp.status_code == 422
        assert "thinking_override" in resp.text

    def test_empty_example_key_rejected(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/any/proposals",
            json={"example_key": ""},
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("example_key" in err.get("loc", []) for err in errors), errors

    def test_zero_batch_run_limit_rejected(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/any/batch_label_runs",
            json={"run_limit": 0},
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("run_limit" in err.get("loc", []) for err in errors), errors


class TestProposalEndpointResponseShape:
    """AC: POST .../proposals returns all required fields."""

    @pytest.mark.asyncio
    async def test_response_has_all_fields(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(return_value=make_teacher_result()),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                settings=settings,
            )

        assert not isinstance(result, str), f"Got error: {result}"
        assert result.inference_invocation_id is not None
        assert result.example_key == ek
        assert result.proposal_json is not None
        assert isinstance(result.schema_valid_core, bool)
        assert isinstance(result.validation_errors_core, list)
        assert isinstance(result.validation_errors_aux, list)
        assert result.invocation_status in (
            "success",
            "schema_invalid",
            "timeout",
            "endpoint_error",
        )
        assert result.used_existing_label is False

    @pytest.mark.asyncio
    async def test_overrides_work_independently(self, tmp_path: Path):
        """Override fields pass through to invoke_teacher independently."""
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        captured_kwargs: list[dict] = []

        async def capturing_teacher(**kwargs):
            captured_kwargs.append(kwargs)
            return make_teacher_result(
                inference_invocation_id=kwargs.get("inference_invocation_id", "x")
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=capturing_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                generation_preset_key_override="explore",
                thinking_mode_override="off",
                visual_budget_preset_key_override="fast",
                settings=settings,
            )

        assert not isinstance(result, str), f"Got error: {result}"
        assert len(captured_kwargs) >= 1
        kw = captured_kwargs[0]
        assert kw["generation_preset_key"] == "explore"
        assert kw["thinking_on"] is False
        assert kw["visual_budget_preset_key"] == "fast"

    @pytest.mark.asyncio
    async def test_proposal_forwards_adaptive_k_params(self, tmp_path: Path):
        """Proposals thread adaptive-K (similarity-gap stopping) from
        ``settings.ICL_SIM_GAP`` / ``ICL_ABS_THRESHOLD`` into ``invoke_teacher``.

        Guards the silent-fallback bug class: if these settings are not
        threaded, interactive proposals quietly use invoke_teacher's fixed-K
        default (icl_sim_gap=None) and the configured ICL behavior never
        reaches the SME's live labeling. Uses non-default values so this
        verifies threading, not the default.
        """
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        settings = settings.model_copy(
            update={"ICL_SIM_GAP": 0.123, "ICL_ABS_THRESHOLD": 0.456}
        )
        ek = keys[0]

        captured_kwargs: list[dict] = []

        async def capturing_teacher(**kwargs):
            captured_kwargs.append(kwargs)
            return make_teacher_result(
                inference_invocation_id=kwargs.get("inference_invocation_id", "x")
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=capturing_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert not isinstance(result, str), f"Got error: {result}"
        assert captured_kwargs[0]["icl_sim_gap"] == 0.123
        assert captured_kwargs[0]["icl_abs_threshold"] == 0.456


class TestProposalRetryLinkage:
    """AC: retry_of_inference_invocation_id links to prior record."""

    @pytest.mark.asyncio
    async def test_retry_of_links_to_prior(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]
        prior_id = generate_uuid4()

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(return_value=make_teacher_result()),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                retry_of_inference_invocation_id=prior_id,
                settings=settings,
            )

        assert not isinstance(result, str)
        # Check the OperationRecord has the retry link
        with Session(engine) as session:
            record = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
            assert record is not None
            assert record.retry_of_inference_invocation_id == prior_id


class TestProposalUseExistingLabel:
    """AC: use_existing_label returns stored Auto-Labeled label."""

    @pytest.mark.asyncio
    async def test_auto_labeled_returned_without_teacher_call(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        # Add an auto-labeled Label
        existing_invocation_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=pid,
                    example_key=ek,
                    label_status="auto_labeled",
                    guidance_id=gid,
                    inference_invocation_id=existing_invocation_id,
                    label_json={"severity": "high", "damaged": True},
                    labeled_at=utc_now(),
                )
            )
            session.commit()

        invoke_mock = AsyncMock()
        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=invoke_mock,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                use_existing_label=True,
                settings=settings,
            )

        assert not isinstance(result, str)
        assert result.used_existing_label is True
        assert result.proposal_json == {"severity": "high", "damaged": True}
        assert result.inference_invocation_id == existing_invocation_id
        invoke_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_label_proceeds_normally(self, tmp_path: Path):
        """use_existing_label=True but no auto_labeled label → Teacher called."""
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(return_value=make_teacher_result()),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                use_existing_label=True,
                settings=settings,
            )

        assert not isinstance(result, str)
        assert result.used_existing_label is False  # fell through to Teacher


class TestProposalValidation:
    """AC: schema_invalid Core errors appear in response."""

    @pytest.mark.asyncio
    async def test_schema_invalid_core_errors_in_response(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(content=INVALID_PROPOSAL_JSON)
            ),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                settings=settings,
            )

        assert not isinstance(result, str)
        assert result.invocation_status == "schema_invalid"
        assert result.schema_valid_core is False
        assert len(result.validation_errors_core) > 0
        assert any("severity" in e for e in result.validation_errors_core)


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Structured gen runtime rejection
# ══════════════════════════════════════════════════════════════════════════════


class TestStructuredGenFallback:
    """AC: Runtime json_schema rejection retries without response_format."""

    @pytest.mark.asyncio
    async def test_runtime_rejection_retries_prompt_only(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        call_count = 0

        async def counting_teacher(**kwargs):
            nonlocal call_count
            call_count += 1
            invocation_id = kwargs.get("inference_invocation_id", "x")
            if call_count == 1:
                # First call: structured gen rejection
                return make_teacher_result(
                    inference_invocation_id=invocation_id,
                    invocation_status="endpoint_error",
                    content=None,
                    error="400 Bad Request: response_format json_schema not supported",
                    structured_generation_attempted=True,
                )
            # Second call: success without structured gen
            return make_teacher_result(
                inference_invocation_id=invocation_id,
                content=VALID_PROPOSAL_JSON,
                structured_generation_attempted=False,
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=counting_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                settings=settings,
            )

        assert not isinstance(result, str)
        assert call_count == 2  # original + fallback retry
        assert result.invocation_status == "success"

    @pytest.mark.asyncio
    async def test_fallback_sets_flag_on_record(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        call_count = 0

        async def counting_teacher(**kwargs):
            nonlocal call_count
            call_count += 1
            iid = kwargs.get("inference_invocation_id", "x")
            if call_count == 1:
                return make_teacher_result(
                    inference_invocation_id=iid,
                    invocation_status="endpoint_error",
                    content=None,
                    error="400: response_format rejected",
                    structured_generation_attempted=True,
                )
            return make_teacher_result(
                inference_invocation_id=iid,
                content=VALID_PROPOSAL_JSON,
                structured_generation_attempted=False,
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=counting_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                settings=settings,
            )

        assert not isinstance(result, str)
        with Session(engine) as session:
            record = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
            assert record is not None
            assert record.structured_generation_fallback_used is True

    @pytest.mark.asyncio
    async def test_non_format_error_no_retry(self, tmp_path: Path):
        """Endpoint error NOT attributable to response_format → no retry."""
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        call_count = 0

        async def counting_teacher(**kwargs):
            nonlocal call_count
            call_count += 1
            iid = kwargs.get("inference_invocation_id", "x")
            return make_teacher_result(
                inference_invocation_id=iid,
                invocation_status="endpoint_error",
                content=None,
                error="502 Bad Gateway: upstream server error",
                structured_generation_attempted=True,
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=counting_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(
                pid,
                example_key=ek,
                settings=settings,
            )

        assert not isinstance(result, str)
        assert call_count == 1  # no retry
        assert result.invocation_status == "endpoint_error"


# ══════════════════════════════════════════════════════════════════════════════
# Section B2: Auto-probe capabilities on first use
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoProbeOnFirstUse:
    """AC: Capability probes fire lazily when a ModelConfig still has any of
    structured_generation_support / thinking_toggle_support /
    visual_budget_support = "unknown". Operator-added ModelConfigs start
    "unknown" (catalog models are preseeded); without a lazy probe
    the only path is the on-demand :reprobe endpoint, which leaves
    prompt-only JSON as the default until an SME manually probes.
    Probe-on-first-use closes that gap.
    """

    @pytest.mark.asyncio
    async def test_reprobe_fires_when_support_is_unknown(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        # Flip the fixture ModelConfig back to "unknown" (fixture default
        # is "supported" for tests that don't care about the probe path).
        with Session(engine) as session:
            mc = session.get(ModelConfig, mcid)
            mc.structured_generation_support = "unknown"
            mc.thinking_toggle_support = "unknown"
            mc.visual_budget_support = "unknown"
            session.commit()
        ek = keys[0]

        async def _fake_reprobe(project_id, model_config_id, workspace_root, settings_):
            # Persist "supported" so the caller can re-read the ModelConfig.
            with Session(engine) as session:
                mc = session.get(ModelConfig, model_config_id)
                mc.structured_generation_support = "supported"
                mc.thinking_toggle_support = "supported"
                mc.visual_budget_support = "supported"
                session.commit()
                session.refresh(mc)
                session.expunge(mc)
                return mc

        with (
            patch(
                "vlm_feedback_loop.services.model_config_service.reprobe_model_config",
                new=AsyncMock(side_effect=_fake_reprobe),
            ) as mock_reprobe,
            patch(
                "vlm_feedback_loop.services.proposal_service.invoke_teacher",
                new=AsyncMock(return_value=make_teacher_result()),
            ),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert not isinstance(result, str), f"Got error: {result}"
        assert mock_reprobe.await_count == 1
        # DB now reflects probed capabilities.
        with Session(engine) as session:
            mc = session.get(ModelConfig, mcid)
            assert mc.structured_generation_support == "supported"

    @pytest.mark.asyncio
    async def test_reprobe_skipped_when_all_capabilities_known(self, tmp_path: Path):
        # Fixture default: all three = "supported" — no probe needed.
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with (
            patch(
                "vlm_feedback_loop.services.model_config_service.reprobe_model_config",
                new=AsyncMock(),
            ) as mock_reprobe,
            patch(
                "vlm_feedback_loop.services.proposal_service.invoke_teacher",
                new=AsyncMock(return_value=make_teacher_result()),
            ),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert not isinstance(result, str), f"Got error: {result}"
        mock_reprobe.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Operation Record completeness
# ══════════════════════════════════════════════════════════════════════════════


class TestOperationRecordCompleteness:
    """Record lifecycle around the NIM call: a pending row exists before
    invoke_teacher returns, and artifact refs point at real files on disk.
    The per-field completeness audit for all outcomes lives in
    test_operation_record_completeness.py."""

    @pytest.mark.asyncio
    async def test_pending_before_nim_call(self, tmp_path: Path):
        """OperationRecord exists with status='pending' BEFORE invoke_teacher returns."""
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]
        found_pending = False

        async def check_pending_teacher(**kwargs):
            nonlocal found_pending
            iid = kwargs.get("inference_invocation_id", "x")
            # Mid-flight: check DB for pending record
            with Session(engine) as session:
                rec = (
                    session.query(OperationRecord)
                    .filter_by(
                        inference_invocation_id=iid,
                    )
                    .first()
                )
                found_pending = rec is not None and rec.invocation_status == "pending"
            return make_teacher_result(inference_invocation_id=iid)

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=check_pending_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            await create_proposal(pid, example_key=ek, settings=settings)

        assert found_pending, "OperationRecord was not pending before NIM call"

    @pytest.mark.asyncio
    async def test_artifacts_written_and_referenced(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(return_value=make_teacher_result()),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
            assert rec is not None
            assert rec.raw_model_response_ref is not None
            assert rec.normalized_json_ref is not None
            assert rec.validation_report_ref is not None
            # Verify files exist on disk
            assert Path(rec.raw_model_response_ref).exists()
            assert Path(rec.normalized_json_ref).exists()
            assert Path(rec.validation_report_ref).exists()


class TestRateLimitedReclassification:
    """429-retry-exhaustion endpoint errors are reclassified as
    rate_limited so the UI can render wait-and-retry copy; other endpoint
    errors stay generic."""

    @pytest.mark.asyncio
    async def test_rate_limited_status_on_429_exhaustion(self, tmp_path: Path):
        """Hosted-NIM 429 exhaustion surfaces as ``rate_limited``, not ``endpoint_error``.

        Sustained labeling against hosted endpoints can exhaust the 429
        retry budget; a bare ``endpoint_error`` gives the SME no actionable
        hint. ``rate_limited`` lets the UI render "wait and retry" copy
        distinct from generic failure.
        """
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(
                    invocation_status="endpoint_error",
                    content=None,
                    error="Exhausted 3 retries. Last: HTTP 429",
                    structured_generation_attempted=False,
                )
            ),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "rate_limited"
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
            assert rec is not None
            assert rec.invocation_status == "rate_limited"

    @pytest.mark.asyncio
    async def test_non_429_endpoint_error_stays_generic(self, tmp_path: Path):
        """Sanity — non-429 endpoint errors must NOT be reclassified."""
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(
                return_value=make_teacher_result(
                    invocation_status="endpoint_error",
                    content=None,
                    error="502 Bad Gateway",
                    structured_generation_attempted=False,
                )
            ),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert result.invocation_status == "endpoint_error"


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Log point 2 — schema_validation
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaValidationLogPoint:
    """AC: Log point 2 emitted at INFO with required fields."""

    @pytest.mark.asyncio
    async def test_info_log_with_required_fields(self, tmp_path: Path, caplog):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with caplog.at_level(
            logging.DEBUG, logger="vlm_feedback_loop.proposal_service"
        ):
            with patch(
                "vlm_feedback_loop.services.proposal_service.invoke_teacher",
                new=AsyncMock(return_value=make_teacher_result()),
            ):
                from vlm_feedback_loop.services.proposal_service import create_proposal

                await create_proposal(pid, example_key=ek, settings=settings)

        records = [r for r in caplog.records if "Schema validation" in r.message]
        assert len(records) >= 1
        rec = records[0]
        assert rec.levelno == logging.INFO
        details = getattr(rec, "details", None)
        assert details is not None
        assert "schema_valid_core" in details
        assert "core_error_count" in details
        assert "normalization_steps" in details


# ══════════════════════════════════════════════════════════════════════════════
# Section F: No secrets in artifacts
# ══════════════════════════════════════════════════════════════════════════════


class TestNoSecretsInOutput:
    """AC: OperationRecords and artifacts contain no API keys."""

    @pytest.mark.asyncio
    async def test_artifacts_no_api_keys(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        ek = keys[0]

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=AsyncMock(return_value=make_teacher_result()),
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            result = await create_proposal(pid, example_key=ek, settings=settings)

        assert not isinstance(result, str)
        # Check artifact files for API key patterns
        with Session(engine) as session:
            rec = (
                session.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result.inference_invocation_id,
                )
                .first()
            )
            assert rec is not None

            for ref_field in [
                rec.raw_model_response_ref,
                rec.normalized_json_ref,
                rec.validation_report_ref,
            ]:
                if ref_field and Path(ref_field).exists():
                    content = Path(ref_field).read_text()
                    assert "nvapi-" not in content
                    assert "Bearer " not in content


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Error handling
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """AC: Missing project/guidance/example returns appropriate errors."""

    @pytest.mark.asyncio
    async def test_project_not_found(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        from vlm_feedback_loop.services.proposal_service import create_proposal

        result = await create_proposal(
            "nonexistent-project",
            example_key="img_000",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_no_active_guidance(self, tmp_path: Path):
        engine, project_dir = _setup_project_db(tmp_path)
        endpoint_id = generate_uuid4()
        mc_id = generate_uuid4()
        with Session(engine) as session:
            _add_project(
                session,
                "test-proj",
                project_dir,
                active_guidance_id=None,
                teacher_model_config_id=mc_id,
            )
            _add_endpoint(session, "test-proj", endpoint_id)
            _add_model_config(session, "test-proj", mc_id, endpoint_id)
            add_example_row(session, "test-proj", "img_000")
            session.commit()

        from vlm_feedback_loop.services.project_service import set_project_engine

        set_project_engine("test-proj", engine)

        settings = make_settings(tmp_path / "workspace")
        from vlm_feedback_loop.services.proposal_service import create_proposal

        result = await create_proposal(
            "test-proj",
            example_key="img_000",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "guidance" in result.lower()

    @pytest.mark.asyncio
    async def test_example_not_found(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(tmp_path)
        from vlm_feedback_loop.services.proposal_service import create_proposal

        result = await create_proposal(
            pid,
            example_key="nonexistent_key",
            settings=settings,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# Section H: Concurrency safety
# ══════════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """AC: 4 concurrent proposals produce distinct, correct records."""

    @pytest.mark.asyncio
    async def test_four_concurrent_proposals_no_contamination(self, tmp_path: Path):
        engine, pid, gid, mcid, keys, settings = seed_proposal_project(
            tmp_path,
            num_examples=4,
        )

        # Deterministic stub keyed on example_key
        async def deterministic_teacher(**kwargs):
            ek = kwargs["example_key"]
            iid = kwargs["inference_invocation_id"]
            content = json.dumps(
                {
                    "rationale_note": f"rationale for {ek}",
                    "severity": "high",
                    "damaged": True,
                }
            )
            return make_teacher_result(
                inference_invocation_id=iid,
                content=content,
            )

        with patch(
            "vlm_feedback_loop.services.proposal_service.invoke_teacher",
            new=deterministic_teacher,
        ):
            from vlm_feedback_loop.services.proposal_service import create_proposal

            # Fire 4 concurrent proposals
            tasks = [
                create_proposal(pid, example_key=ek, settings=settings) for ek in keys
            ]
            results = await asyncio.gather(*tasks)

        # All should succeed
        for r in results:
            assert not isinstance(r, str), f"Got error: {r}"
            assert r.invocation_status == "success"

        # All have distinct inference_invocation_ids
        iids = [r.inference_invocation_id for r in results]
        assert len(set(iids)) == 4, "Expected 4 distinct invocation IDs"

        # Each result has correct example_key
        result_keys = {r.example_key for r in results}
        assert result_keys == set(keys)

        # 4 distinct OperationRecords in DB
        with Session(engine) as session:
            records = (
                session.query(OperationRecord)
                .filter_by(
                    project_id=pid,
                )
                .all()
            )
            record_iids = {r.inference_invocation_id for r in records}
            assert len(record_iids) == 4

            # Each record has correct example_key mapping
            for rec in records:
                matching_result = [
                    r
                    for r in results
                    if r.inference_invocation_id == rec.inference_invocation_id
                ]
                assert len(matching_result) == 1
                assert rec.example_key == matching_result[0].example_key
