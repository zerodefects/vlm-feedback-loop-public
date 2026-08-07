# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rationale regeneration service and its prompt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    make_settings,
    open_project_workspace,
)
from support import (
    fake_nim_failure,
    fake_nim_success,
    fake_nim_timeout,
    fake_prepare_result,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.prompt_service import (
    render_rationale_regeneration_prompt,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _setup_project(tmp_path: Path, *, rationale_enabled: bool = True):
    pid = "test-proj"
    gid = generate_uuid4()
    mcid = generate_uuid4()
    eid = generate_uuid4()

    engine, project_dir, _ = open_project_workspace(tmp_path, pid, register_engine=True)

    with Session(engine) as session:
        add_project_row(
            session,
            pid,
            str(project_dir),
            active_guidance_id=gid,
            teacher_model_config_id=mcid,
        )
        add_endpoint_row(session, pid, eid)
        add_model_config_row(
            session, pid, mcid, eid, model_name="nvidia/cosmos-reason2-8b"
        )
        add_example_row(session, pid, "img_000")
        fields = [
            {
                "field_name": "severity",
                "type": "enum",
                "role": "core",
                "allowed_values": ["low", "medium", "high"],
                "display_order": 0,
            },
            {
                "field_name": "damaged",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            },
        ]
        generation_order = ["severity", "damaged"]
        if rationale_enabled:
            fields.insert(
                0,
                {
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": -1,
                },
            )
            generation_order.insert(0, "rationale_note")
        add_guidance_row(
            session,
            pid,
            gid,
            {
                "fields": fields,
                "generation_order": generation_order,
                "derived_json_schema": {},
                "schema_hash": "test_hash",
            },
            description="Classify damage severity.",
            rules="Focus on visible defects.",
        )
        session.commit()

    settings = make_settings(tmp_path / "workspace", NVIDIA_API_KEY="nvapi-test")
    return engine, pid, mcid, settings


def _fake_nim_success(content="Clear dent visible on front panel."):
    return fake_nim_success(content)


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Rationale regeneration
# ══════════════════════════════════════════════════════════════════════════════


class TestRationaleRegeneration:
    """AC: Rationale regeneration endpoint."""

    @pytest.mark.asyncio
    async def test_success_returns_rationale_text(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)

        prepare_mock = AsyncMock(return_value=fake_prepare_result(1))
        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert not isinstance(result, str), f"Error: {result}"
        assert result["invocation_status"] == "success"
        assert "dent" in result["rationale_note"].lower()
        prepare_mock.assert_called_once()
        args, kwargs = prepare_mock.call_args
        assert len(args) == 1
        assert isinstance(args[0], list) and len(args[0]) == 1
        assert kwargs["settings"] is settings

    @pytest.mark.asyncio
    async def test_image_preparation_failure_never_dispatches_text_only(self, tmp_path):
        engine, pid, _mcid, settings = _setup_project(tmp_path)
        failed_prep = fake_prepare_result(1)
        failed_prep.success = False
        failed_prep.images[0].error = "Path is outside IMAGE_ROOT"
        chat_mock = AsyncMock(return_value=_fake_nim_success())

        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=chat_mock,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=failed_prep),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert not isinstance(result, str)
        assert result["invocation_status"] == "endpoint_error"
        assert result["rationale_note"] == ""
        chat_mock.assert_not_awaited()
        with Session(engine) as session:
            record = session.get(
                OperationRecord,
                result["inference_invocation_id"],
            )
            assert record is not None
            assert "Image preparation failed" in (record.provider_error_ref or "")

    @pytest.mark.asyncio
    async def test_disabled_guidance_rejects_regeneration_without_dispatch(
        self, tmp_path
    ):
        engine, pid, mcid, settings = _setup_project(tmp_path, rationale_enabled=False)
        chat_mock = AsyncMock(return_value=_fake_nim_success())
        prepare_mock = AsyncMock(return_value=fake_prepare_result(1))
        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=chat_mock,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert result == "conflict: rationale_note is disabled in the active Guidance"
        chat_mock.assert_not_awaited()
        prepare_mock.assert_not_awaited()
        with Session(engine) as session:
            assert (
                session.query(OperationRecord)
                .filter_by(purpose="rationale_regeneration")
                .first()
                is None
            )

    @pytest.mark.asyncio
    async def test_creates_operation_record_with_purpose(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)

        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert not isinstance(result, str)
        with Session(engine) as s:
            rec = (
                s.query(OperationRecord)
                .filter_by(
                    inference_invocation_id=result["inference_invocation_id"],
                )
                .first()
            )
            assert rec is not None
            assert rec.purpose == "rationale_regeneration"
            assert rec.invocation_status == "success"

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)

        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=AsyncMock(return_value=fake_nim_timeout()),
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert not isinstance(result, str)
        assert result["invocation_status"] == "timeout"

    @pytest.mark.asyncio
    async def test_endpoint_error_returns_failure(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)

        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=AsyncMock(
                    return_value=fake_nim_failure("502 Bad Gateway", status_code=502)
                ),
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            result = await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert not isinstance(result, str)
        assert result["invocation_status"] == "endpoint_error"

    @pytest.mark.asyncio
    async def test_example_not_found(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)

        from vlm_feedback_loop.services.rationale_service import regenerate_rationale

        result = await regenerate_rationale(
            pid,
            "nonexistent",
            None,
            settings.WORKSPACE_ROOT,
            settings,
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_uses_project_teacher_when_null(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)
        captured_model: list[str] = []

        async def capture_chat(*args, **kwargs):
            captured_model.append(args[2])  # model argument
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=capture_chat,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            await regenerate_rationale(
                pid,
                "img_000",
                None,  # None = use project default
                settings.WORKSPACE_ROOT,
                settings,
            )

        assert captured_model[0] == "nvidia/cosmos-reason2-8b"


# ══════════════════════════════════════════════════════════════════════════════
# Section A2: reasoning-headroom-aware budget
# ══════════════════════════════════════════════════════════════════════════════


class TestRationaleBudgetReasoningHeadroom:
    """rationale_service must add ``MODEL_REASONING_HEADROOM_TOKENS``
    to ``max_tokens`` when ``thinking_mode_effective == "on"``.

    A fixed ``max_tokens=256`` truncates Qwen/Kimi reasoners' visible
    content because their ``<think>`` block consumes the entire ceiling.
    The budget therefore scales the base (``max(BASE_OUTPUT_TOKENS_FLOOR,
    2 * RATIONALE_NOTE_ESTIMATE_TOKENS)``) and adds reasoning headroom
    when Thinking is effective-on.
    """

    @pytest.mark.asyncio
    async def test_thinking_on_budget_includes_headroom(self, tmp_path):
        engine, pid, mcid, settings = _setup_project(tmp_path)
        # Fixture project has thinking_default_on=True. The seeded
        # ModelConfig is created with thinking_toggle_mode="none" — flip
        # it to "always_on_reasoning" for this test so the resolver
        # returns thinking_mode_effective="on" (the mode that allocates
        # reasoning headroom in the token budget).
        with Session(engine) as session:
            mc = session.query(ModelConfig).filter_by(model_config_id=mcid).first()
            mc.thinking_toggle_mode = "always_on_reasoning"
            session.commit()
        nim_mock = AsyncMock(return_value=_fake_nim_success())
        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=nim_mock,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        nim_mock.assert_called_once()
        kwargs = nim_mock.call_args.kwargs
        # Token budget formula:
        #   base = max(256, 2*160) = 320
        #   max_tokens = base + headroom = 320 + 16384 = 16704
        # capped at ctx (256000) * 0.25 = 64000.
        assert kwargs["max_tokens"] >= 16384, (
            f"expected reasoning headroom (>=16384), got {kwargs['max_tokens']}"
        )
        assert kwargs["max_tokens"] == 320 + 16384

    @pytest.mark.asyncio
    async def test_thinking_off_budget_omits_headroom(self, tmp_path):
        """When thinking_default_on=False AND model supports the toggle,
        no reasoning headroom is added — the rationale is short text,
        the smaller ceiling is appropriate.
        """
        engine, pid, mcid, settings = _setup_project(tmp_path)
        # Override to a model that supports the thinking toggle, then
        # turn the toggle off project-wide.
        with Session(engine) as session:
            project = session.query(Project).filter_by(project_id=pid).first()
            project.thinking_default_on = False
            mc = session.query(ModelConfig).filter_by(model_config_id=mcid).first()
            mc.thinking_toggle_mode = "qwen_enable_thinking"
            mc.thinking_toggle_support = "supported"
            session.commit()

        nim_mock = AsyncMock(return_value=_fake_nim_success())
        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=nim_mock,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        kwargs = nim_mock.call_args.kwargs
        assert kwargs["max_tokens"] == 320, (
            f"thinking-off should use base only (320), got {kwargs['max_tokens']}"
        )
        # And the toggle override is on the wire when thinking is OFF.
        assert kwargs.get("chat_template_kwargs") == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_unsupported_toggle_is_omitted_and_keeps_headroom(self, tmp_path):
        """Rationale regeneration honors the persisted capability result."""
        engine, pid, mcid, settings = _setup_project(tmp_path)
        with Session(engine) as session:
            project = session.query(Project).filter_by(project_id=pid).first()
            project.thinking_default_on = False
            mc = session.query(ModelConfig).filter_by(model_config_id=mcid).first()
            mc.thinking_toggle_mode = "qwen_enable_thinking"
            mc.thinking_toggle_support = "unsupported"
            session.commit()

        nim_mock = AsyncMock(return_value=_fake_nim_success())
        with (
            patch(
                "vlm_feedback_loop.services.rationale_service.nim_client.chat_completions",
                new=nim_mock,
            ),
            patch(
                "vlm_feedback_loop.services.rationale_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            from vlm_feedback_loop.services.rationale_service import (
                regenerate_rationale,
            )

            await regenerate_rationale(
                pid,
                "img_000",
                None,
                settings.WORKSPACE_ROOT,
                settings,
            )

        kwargs = nim_mock.call_args.kwargs
        assert "chat_template_kwargs" not in kwargs
        assert kwargs["max_tokens"] == 320 + 16384


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Rationale prompt
# ══════════════════════════════════════════════════════════════════════════════


class TestRationalePrompt:
    """The rationale regeneration prompt has the expected template structure."""

    _DEFAULT_KWARGS = {
        "guidance_description": "Classify damage severity.",
        "guidance_rules": "Focus on visible defects.",
        "guidance_fields": [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "aux",
                "display_order": -1,
            },
            {
                "field_name": "severity",
                "type": "enum",
                "role": "core",
                "allowed_values": ["low", "medium", "high"],
                "display_order": 0,
            },
            {
                "field_name": "damaged",
                "type": "boolean",
                "role": "core",
                "display_order": 1,
            },
        ],
    }

    def test_prompt_structure_matches_d3(self):
        messages, _ = render_rationale_regeneration_prompt(
            **self._DEFAULT_KWARGS,
        )
        assert len(messages) == 2
        system, user = messages
        assert system["role"] == "system"
        assert "vision-labeling records" in system["content"].lower()
        assert "inspect the image independently" in system["content"].lower()
        assert "subject the task asks about" in system["content"].lower()
        assert "only when that distinction matters to the task" in (
            system["content"].lower()
        )
        assert "do not infer observations" in system["content"].lower()
        assert user["role"] == "user"
        # Task context must be injected
        assert "Classify damage severity." in user["content"]
        assert "Focus on visible defects." in user["content"]
        # Schema summary lists the enum's allowed values explicitly.
        assert "severity: enum — one of" in user["content"]
        # The reviewed answer must not be exposed to the writer. It would
        # anchor the model into fabricating a plausible post-hoc explanation.
        assert '"severity":"high"' not in user["content"]
        assert "Reviewed field values" not in user["content"]

    def test_contains_no_icl(self):
        messages, _ = render_rationale_regeneration_prompt(**self._DEFAULT_KWARGS)
        user_text = messages[1]["content"]
        assert "E01" not in user_text
        assert "ICL" not in user_text

    def test_requests_plain_text(self):
        messages, _ = render_rationale_regeneration_prompt(**self._DEFAULT_KWARGS)
        user_text = messages[1]["content"]
        assert "Return only the rationale text, no JSON" in user_text

    def test_prompt_hash_deterministic(self):
        _, h1 = render_rationale_regeneration_prompt(**self._DEFAULT_KWARGS)
        _, h2 = render_rationale_regeneration_prompt(**self._DEFAULT_KWARGS)
        assert h1 == h2

    def test_prompt_contains_word_constraints(self):
        messages, _ = render_rationale_regeneration_prompt(**self._DEFAULT_KWARGS)
        user_text = messages[1]["content"]
        assert "80 words" in user_text
        # Case-insensitive — the prompt embeds "do not speculate" mid-sentence.
        assert "do not speculate" in user_text.lower()

    # ── Task-awareness extensions (spec-compatible enhancement) ──────────────

    def test_includes_guidance_description_when_provided(self):
        messages, _ = render_rationale_regeneration_prompt(
            guidance_description="Classify the hand gesture.",
        )
        user_text = messages[1]["content"]
        assert "Task:" in user_text
        assert "Classify the hand gesture." in user_text

    def test_includes_guidance_rules_when_provided(self):
        messages, _ = render_rationale_regeneration_prompt(
            guidance_rules="Use only the visible hand; ignore background.",
        )
        user_text = messages[1]["content"]
        assert "Rules:" in user_text
        assert "Use only the visible hand; ignore background." in user_text

    def test_omits_task_block_when_description_empty(self):
        messages, _ = render_rationale_regeneration_prompt(
            guidance_description="",
            guidance_rules="",
        )
        user_text = messages[1]["content"]
        assert "Task:" not in user_text
        assert "Rules:" not in user_text

    def test_schema_summary_lists_enum_alternatives(self):
        fields = [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "aux",
                "display_order": -1,
            },
            {
                "field_name": "answer",
                "type": "enum",
                "role": "core",
                "allowed_values": ["rock", "paper", "scissors"],
                "display_order": 0,
            },
        ]
        messages, _ = render_rationale_regeneration_prompt(
            guidance_fields=fields,
        )
        user_text = messages[1]["content"]
        assert "Label schema (for context):" in user_text
        assert "answer: enum — one of" in user_text
        # All three allowed values, JSON-quoted, in order.
        assert '"rock"' in user_text
        assert '"paper"' in user_text
        assert '"scissors"' in user_text
        # Aux fields (rationale_note) excluded from the schema summary.
        assert "rationale_note:" not in user_text

    def test_schema_summary_formats_integer_range(self):
        fields = [
            {
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 4,
                "display_order": 0,
            },
        ]
        messages, _ = render_rationale_regeneration_prompt(
            guidance_fields=fields,
        )
        user_text = messages[1]["content"]
        assert "severity: integer — 0..4" in user_text

    def test_schema_summary_formats_enum_set(self):
        fields = [
            {
                "field_name": "damage_types",
                "type": "enum_set",
                "role": "core",
                "allowed_values": ["crush", "dent", "scratch"],
                "display_order": 0,
            },
        ]
        messages, _ = render_rationale_regeneration_prompt(
            guidance_fields=fields,
        )
        user_text = messages[1]["content"]
        assert "damage_types: enum_set — any of" in user_text

    def test_schema_summary_omitted_when_no_core_fields(self):
        # Only Aux fields → no schema summary block at all.
        fields = [
            {
                "field_name": "rationale_note",
                "type": "string",
                "role": "aux",
                "display_order": -1,
            },
        ]
        messages, _ = render_rationale_regeneration_prompt(
            guidance_fields=fields,
        )
        user_text = messages[1]["content"]
        assert "Label schema" not in user_text

    def test_prompt_prioritizes_visible_evidence_over_value_compliance(self):
        """No proposed or reviewed answer can anchor the visual observation."""
        messages, _ = render_rationale_regeneration_prompt()
        system_text = messages[0]["content"].lower()
        user_text = messages[1]["content"].lower()
        assert "inspect the image independently" in system_text
        assert "do not infer observations from a proposed or reviewed label" in (
            system_text
        )
        assert '"answer":"rock"' not in user_text
        assert "invent supporting details" in user_text
        assert "authoritative" not in system_text
        assert "never contradict" not in system_text

    def test_prompt_makes_carrier_distinction_task_conditional(self):
        """Carrier/content grounding must not assume every task is about an item."""
        messages, _ = render_rationale_regeneration_prompt()
        system_text = messages[0]["content"].lower()
        assert "subject the task asks about" in system_text
        assert "physical carrier" in system_text
        assert "content shown on it" in system_text
        assert "only when that distinction matters to the task" in system_text
        assert "physical foreground item" not in system_text

    def test_prompt_allows_domain_appropriate_evidence(self):
        """Counts, identities, comparisons, and locations can be valid evidence."""
        messages, _ = render_rationale_regeneration_prompt()
        user_text = messages[1]["content"].lower()
        assert (
            "quantities, identities, comparisons, and locations may be described"
            in (user_text)
        )
        assert "natural vocabulary appropriate to this domain" in user_text
        assert "do not name, count" not in user_text
        assert "counting discrete items is unreliable" not in user_text

    def test_prompt_does_not_prescribe_rps_words_or_visual_palette(self):
        messages, _ = render_rationale_regeneration_prompt()
        user_text = messages[1]["content"].lower()
        for leaked_word in (
            "fingers",
            "curled",
            "clenched",
            "outer contour",
            "edge sharpness",
            "region compactness",
        ):
            assert leaked_word not in user_text

    def test_prompt_rejects_generic_or_value_echo_rationales(self):
        messages, _ = render_rationale_regeneration_prompt()
        user_text = messages[1]["content"].lower()
        assert "do not merely name an outcome" in user_text
        assert "generic visual-feature checklist" in user_text
        assert "from your own inspection of the image" in user_text
