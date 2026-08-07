# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for prompt rendering, Teacher invocation, and Generation Controls.

Covers Generation Controls, Visual Budget Controls, per-log-point
operational logging, and the prompt template structure.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_settings
from support import fake_nim_success, fake_prepare_result
from vlm_feedback_loop._defaults import DEFAULTS
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_8B,
    MISTRAL_MEDIUM_3_5,
    NEMOTRON_3_NANO_OMNI_REASONING,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

FIXTURE_FIELDS = [
    {
        "field_id": "f0",
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": -1,
    },
    {
        "field_id": "f1",
        "field_name": "damage_visible",
        "type": "boolean",
        "role": "aux",
        "display_order": 0,
    },
    {
        "field_id": "f2",
        "field_name": "severity",
        "type": "enum",
        "role": "core",
        "allowed_values": ["low", "medium", "high"],
        "display_order": 1,
    },
]

FIXTURE_GENERATION_ORDER = ["rationale_note", "damage_visible", "severity"]

FIXTURE_DESCRIPTION = "Classify damage severity from inspection images."
FIXTURE_RULES = "Focus on visible surface defects. Ignore background objects."

FIXTURE_DERIVED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale_note": {"type": "string"},
        "damage_visible": {"type": "boolean"},
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["severity"],
    "additionalProperties": False,
    "x-generation-order": FIXTURE_GENERATION_ORDER,
}

LABELING_PRESETS = DEFAULTS["LABELING_PRESETS"]
VISUAL_BUDGET_PRESETS = DEFAULTS["VISUAL_BUDGET_PRESETS"]


def _make_model_config(**overrides):
    from vlm_feedback_loop.services.prompt_service import ModelConfigInput

    defaults = {
        "context_window_tokens": 256000,
        "thinking_toggle_mode": "qwen_enable_thinking",
        "thinking_toggle_support": "supported",
        "visual_budget_mode": "mm_processor_size",
        "visual_budget_support": "supported",
        "structured_generation_support": "supported",
    }
    defaults.update(overrides)
    return ModelConfigInput(**defaults)


def _fake_nim_success(content: str = '{"severity":"high"}') -> Any:
    return fake_nim_success(content)


def _fake_nim_truncated() -> Any:
    from vlm_feedback_loop.services.nim_client import NimChatCompletionsResult

    return NimChatCompletionsResult(
        success=True,
        content='{"severity":"hi',
        finish_reason="length",
        usage={"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
        status_code=200,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Section A: Generation Presets
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveGenerationParams:
    """AC: Preset sampling parameter resolution."""

    def test_precise_preset(self):
        from vlm_feedback_loop.services.prompt_service import resolve_generation_params

        result = resolve_generation_params("precise", LABELING_PRESETS)
        assert result["temperature"] == 0.0
        assert result["top_p"] == 1.0

    def test_explore_preset(self):
        from vlm_feedback_loop.services.prompt_service import resolve_generation_params

        result = resolve_generation_params("explore", LABELING_PRESETS)
        assert result["temperature"] == 0.3
        assert result["top_p"] == 0.9

    def test_unknown_preset_raises(self):
        from vlm_feedback_loop.services.prompt_service import resolve_generation_params

        with pytest.raises(ValueError, match="Unknown preset"):
            resolve_generation_params("nonexistent", LABELING_PRESETS)


# ═══════════════════════════════════════════════════════════════════════════
# Section B: Thinking Toggle
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveThinkingFields:
    """AC: Thinking toggle resolution for model-specific fields."""

    def test_thinking_on_no_override(self):
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(True, "qwen_enable_thinking")
        assert result["thinking_mode_effective"] == "on"
        assert result["thinking_request_fields"] is None

    def test_thinking_off_qwen(self):
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(False, "qwen_enable_thinking")
        assert result["thinking_mode_effective"] == "off"
        assert result["thinking_request_fields"] == {"enable_thinking": False}

    def test_thinking_off_kimi(self):
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(False, "kimi_thinking")
        assert result["thinking_mode_effective"] == "off"
        assert result["thinking_request_fields"] == {"thinking": False}

    @pytest.mark.parametrize("support", ["unknown", "unsupported"])
    def test_unavailable_toggle_uses_natural_reasoning_default(self, support):
        """A mode name alone does not authorize unsupported wire kwargs."""
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(
            False,
            "qwen_enable_thinking",
            support,
        )
        assert result == {
            "thinking_mode_effective": "on",
            "thinking_request_fields": None,
            "thinking_hidden": True,
        }

    def test_thinking_hidden_mode_none(self):
        """``mode="none"`` — no toggle, no reasoning (Mistral, Pixtral,
        Nemotron Nano 12B VL). Resolver returns ``mode_effective="off"``
        so the token budget does not allocate reasoning headroom.
        """
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(False, "none")
        assert result["thinking_mode_effective"] == "off"
        assert result["thinking_request_fields"] is None
        assert result["thinking_hidden"] is True

    def test_thinking_hidden_mode_none_user_toggle_ignored(self):
        """``mode="none"`` ignores the user's ``thinking_on`` input —
        the model can't be told to reason, so resolver always returns
        ``mode_effective="off"`` regardless of user preference.
        """
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        on_result = resolve_thinking_fields(True, "none")
        off_result = resolve_thinking_fields(False, "none")
        assert on_result == off_result
        assert on_result["thinking_mode_effective"] == "off"

    def test_thinking_always_on_reasoning(self):
        """``mode="always_on_reasoning"`` — no toggle, model ALWAYS
        reasons (Nemotron-3 Nano Omni Reasoning). Resolver returns
        ``mode_effective="on"`` so the token budget allocates
        ``MODEL_REASONING_HEADROOM_TOKENS``. Toggle hidden in UI.
        """
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        result = resolve_thinking_fields(False, "always_on_reasoning")
        assert result["thinking_mode_effective"] == "on"
        assert result["thinking_request_fields"] is None
        assert result["thinking_hidden"] is True

    def test_thinking_always_on_reasoning_user_toggle_ignored(self):
        """``mode="always_on_reasoning"`` ignores the user's
        ``thinking_on`` input — the model reasons regardless.
        """
        from vlm_feedback_loop.services.prompt_service import resolve_thinking_fields

        on_result = resolve_thinking_fields(True, "always_on_reasoning")
        off_result = resolve_thinking_fields(False, "always_on_reasoning")
        assert on_result == off_result
        assert on_result["thinking_mode_effective"] == "on"


# ═══════════════════════════════════════════════════════════════════════════
# Section C: Visual Budget
# ═══════════════════════════════════════════════════════════════════════════


class TestResolveVisualBudget:
    """AC: Visual budget preset resolution to mm_processor_kwargs."""

    def test_mm_processor_size_balanced(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "balanced",
            "mm_processor_size",
            "supported",
            VISUAL_BUDGET_PRESETS,
        )
        params = result["visual_budget_params_effective"]
        assert params is not None
        assert "mm_processor_kwargs" in params
        kw = params["mm_processor_kwargs"]
        assert kw["size"]["shortest_edge"] == 1568
        assert kw["size"]["longest_edge"] == 131072

    def test_mm_processor_pixels_fast(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "fast",
            "mm_processor_pixels",
            "supported",
            VISUAL_BUDGET_PRESETS,
        )
        kw = result["visual_budget_params_effective"]["mm_processor_kwargs"]
        assert kw["images_kwargs"]["min_pixels"] == 1568
        assert kw["images_kwargs"]["max_pixels"] == 65536

    def test_mm_processor_tiles_high_detail(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "high_detail",
            "mm_processor_tiles",
            "supported",
            VISUAL_BUDGET_PRESETS,
        )
        kw = result["visual_budget_params_effective"]["mm_processor_kwargs"]
        assert kw["max_num_tiles"] == 32

    def test_mode_none_returns_none(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "balanced",
            "none",
            "supported",
            VISUAL_BUDGET_PRESETS,
        )
        assert result["visual_budget_preset_key"] is None
        assert result["visual_budget_params_effective"] is None

    def test_unsupported_returns_none(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "balanced",
            "mm_processor_size",
            "unsupported",
            VISUAL_BUDGET_PRESETS,
        )
        assert result["visual_budget_preset_key"] is None
        assert result["visual_budget_params_effective"] is None

    def test_preset_key_in_result(self):
        from vlm_feedback_loop.services.prompt_service import resolve_visual_budget

        result = resolve_visual_budget(
            "fast",
            "mm_processor_size",
            "supported",
            VISUAL_BUDGET_PRESETS,
        )
        assert result["visual_budget_preset_key"] == "fast"


# ═══════════════════════════════════════════════════════════════════════════
# Section D: Seed Derivation
# ═══════════════════════════════════════════════════════════════════════════


class TestDeriveSeed:
    """AC: Deterministic seed for evaluation/batch."""

    def test_deterministic_value(self):
        from vlm_feedback_loop.services.prompt_service import derive_seed

        s1 = derive_seed("run-123", "img-001")
        s2 = derive_seed("run-123", "img-001")
        assert s1 == s2

    def test_different_inputs_different_seed(self):
        from vlm_feedback_loop.services.prompt_service import derive_seed

        s1 = derive_seed("run-123", "img-001")
        s2 = derive_seed("run-123", "img-002")
        assert s1 != s2

    def test_non_negative_int32(self):
        from vlm_feedback_loop.services.prompt_service import derive_seed

        s = derive_seed("scope", "key")
        assert isinstance(s, int)
        assert s >= 0
        assert s <= 2**31 - 1


# ═══════════════════════════════════════════════════════════════════════════
# Section E: Request Construction
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildChatRequestKwargs:
    """AC: Combined 6-step request kwargs."""

    def test_max_tokens_always_present(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields=None,
            seed=None,
            visual_budget_params=None,
            max_tokens=1024,
            response_format=None,
        )
        assert kwargs["max_tokens"] == 1024

    def test_json_schema_when_supported(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        rf = {
            "type": "json_schema",
            "json_schema": {"name": "test", "strict": True, "schema": {}},
        }
        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields=None,
            seed=None,
            visual_budget_params=None,
            max_tokens=1024,
            response_format=rf,
        )
        assert kwargs["response_format"] == rf

    def test_no_seed_for_interactive(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields=None,
            seed=None,
            visual_budget_params=None,
            max_tokens=1024,
            response_format=None,
        )
        assert "seed" not in kwargs

    def test_seed_included_when_provided(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields=None,
            seed=42,
            visual_budget_params=None,
            max_tokens=1024,
            response_format=None,
        )
        assert kwargs["seed"] == 42

    def test_chat_template_kwargs_when_thinking_off(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields={"enable_thinking": False},
            seed=None,
            visual_budget_params=None,
            max_tokens=1024,
            response_format=None,
        )
        assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}

    def test_visual_budget_included(self):
        from vlm_feedback_loop.services.prompt_service import build_chat_request_kwargs

        vb = {"mm_processor_kwargs": {"size": {"shortest_edge": 672}}}
        kwargs = build_chat_request_kwargs(
            sampling_params={"temperature": 0.0, "top_p": 1.0},
            thinking_request_fields=None,
            seed=None,
            visual_budget_params=vb,
            max_tokens=1024,
            response_format=None,
        )
        assert kwargs["mm_processor_kwargs"] == {"size": {"shortest_edge": 672}}


# ═══════════════════════════════════════════════════════════════════════════
# Section F: Prompt Template
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderPrompt:
    """The rendered prompt has the expected template structure."""

    def test_system_message_structure(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        assert messages[0]["role"] == "system"
        assert "vision labeling assistant" in messages[0]["content"]
        assert "Output valid JSON only" in messages[0]["content"]

    def test_icl_markers_no_example_key(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        icl_examples = [
            {"rationale_note": "crack visible", "severity": "high"},
            {"rationale_note": "dent on panel", "severity": "medium"},
        ]
        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            icl_examples,
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"]
        assert "E01:" in user_text
        assert "E02:" in user_text
        # No example_key should appear
        assert "example_key" not in user_text.lower()

    def test_rationale_is_requested_last(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"]
        assert "Add rationale_note last." in user_text
        assert "one or two short, image-specific sentences" in user_text
        assert "First, populate rationale_note" not in user_text

    def test_rationale_instructions_are_omitted_when_disabled(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        schema_without_rationale = {
            "type": "object",
            "properties": {"severity": {"type": "string", "enum": ["low", "high"]}},
            "required": ["severity"],
            "additionalProperties": False,
            "x-generation-order": ["severity"],
        }
        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            schema_without_rationale,
            True,
        )
        user_text = messages[1]["content"]
        assert "rationale_note" not in user_text
        assert "image-specific sentences" not in user_text

    def test_prompt_only_render_includes_compact_schema(self):
        """Prompt-only mode carries field names, types, and enum values."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            False,
        )
        user_text = messages[1]["content"]
        assert "Label schema:" in user_text
        assert '- severity (required): "low" | "medium" | "high"' in user_text
        assert "- damage_visible (optional): boolean" in user_text
        assert "schema described above" not in user_text

    def test_default_omits_verbose_guidance(self):
        """Production uses the single compact prompt contract."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            False,
        )
        user_text = messages[1]["content"]
        assert FIXTURE_DESCRIPTION not in user_text
        assert FIXTURE_RULES not in user_text
        assert "Task Description:" not in user_text
        assert "Rules:" not in user_text

    def test_return_json_only_closing(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"]
        assert user_text.strip().endswith("Return JSON only.")

    def test_prompt_hash_deterministic(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        _, h1 = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        _, h2 = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_empty_icl_omits_section(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"]
        # The image-mode heading is "ICL Examples (each block is the
        # IMAGE …)" — no colon — so asserting on "ICL Examples:" would
        # pass even if the heading leaked into a cold-start prompt. The
        # bare "ICL Examples" prefix is the stricter check.
        assert "ICL Examples" not in user_text
        assert "ICL Example 1" not in user_text

    # ── Section F.1: Inline ICL image injection ──────────────

    def test_attach_icl_images_emits_slot_sentinels(self):
        """When ``attach_icl_images=True`` the rendered text wraps each
        ICL example with explicit "ICL Example N image (E0N):" header,
        the slot sentinel, and "ICL Example N label (E0N): {json}"
        post-image anchor (image before label, binding each label to
        its image position). ``invoke_teacher`` step 7 splits on those
        sentinels to interleave prepared image content parts."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        icl = [
            {"rationale_note": "rock fist", "category": "rock"},
            {"rationale_note": "paper", "category": "paper"},
            {"rationale_note": "scissors V", "category": "scissors"},
        ]
        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            icl,
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
            attach_icl_images=True,
        )
        user_text = messages[1]["content"]

        # All three ICL slots present
        assert "<<<__VLM_ICL_IMG_SLOT_1__>>>" in user_text
        assert "<<<__VLM_ICL_IMG_SLOT_2__>>>" in user_text
        assert "<<<__VLM_ICL_IMG_SLOT_3__>>>" in user_text
        # Both pre-image and post-image position anchors per example
        assert "ICL Example 1 image (E01):" in user_text
        assert "ICL Example 1 label (E01):" in user_text
        assert "ICL Example 2 image (E02):" in user_text
        assert "ICL Example 2 label (E02):" in user_text
        assert "ICL Example 3 image (E03):" in user_text
        assert "ICL Example 3 label (E03):" in user_text
        # Per-example ordering: image header → sentinel → label line
        for n in (1, 2, 3):
            assert user_text.index(f"ICL Example {n} image (E0{n}):") < user_text.index(
                f"<<<__VLM_ICL_IMG_SLOT_{n}__>>>"
            )
            assert user_text.index(f"<<<__VLM_ICL_IMG_SLOT_{n}__>>>") < user_text.index(
                f"ICL Example {n} label (E0{n}):"
            )
            # Cross-modal binding: structured "_slot" key is the FIRST
            # field of each ICL JSON label dict so the model sees the
            # position commitment in three channels (header text, label
            # text, structured JSON). The leading-underscore non-domain
            # key avoids Nemotron's tile-coord vocabulary collision
            # that a "position" key triggers.
            assert f'"_slot":"E0{n}"' in user_text
        # Cross-example ordering: example N's label precedes example N+1's image
        assert user_text.index("ICL Example 1 label (E01):") < user_text.index(
            "ICL Example 2 image (E02):"
        )
        assert user_text.index("ICL Example 2 label (E02):") < user_text.index(
            "ICL Example 3 image (E03):"
        )
        # Query slot appears after the "Now label" instruction line
        assert "<<<__VLM_QUERY_IMG_SLOT__>>>" in user_text
        assert user_text.index("Now label the QUERY image below.") < user_text.index(
            "<<<__VLM_QUERY_IMG_SLOT__>>>"
        )
        # Demonstrations teach the task but never become nearest-neighbor
        # answers for the query. Every label is scoped to its paired image.
        assert "Each JSON label applies only to its paired example image." in user_text
        assert "Use the examples to infer field meanings and decision boundaries." in (
            user_text
        )
        assert "determine every value independently from the QUERY image" in user_text
        assert "closest visual match" not in user_text
        assert "prefer the closest" not in user_text

    def test_attach_icl_images_false_omits_icl_sentinels(self):
        """``attach_icl_images=False`` (the legacy text-only fallback for
        single-image-only Vision models whose max_images_per_request=1
        cannot fit ICL images) emits NO ICL slot sentinels — only the
        QUERY slot sentinel, which is always emitted."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        icl = [
            {"rationale_note": "rock fist", "category": "rock"},
            {"rationale_note": "paper", "category": "paper"},
        ]
        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            icl,
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
            attach_icl_images=False,
        )
        user_text = messages[1]["content"]

        # ICL slots NOT emitted
        assert "<<<__VLM_ICL_IMG_SLOT_" not in user_text
        # ICL JSON labels still rendered (text-only ICL)
        assert "E01:" in user_text and "E02:" in user_text
        # The "_slot" cross-modal field is image-mode-only — the
        # text-only fallback already names the position via the "E0N:"
        # text marker, so the JSON-level injection is suppressed to
        # keep the legacy path's prompt-hash byte-identical.
        assert '"_slot":"E01"' not in user_text
        assert '"_slot":"E02"' not in user_text
        # Query slot always emitted (query image is always attached)
        assert "<<<__VLM_QUERY_IMG_SLOT__>>>" in user_text
        assert "Each JSON label applies only to its paired example image." in user_text
        assert "closest visual match" not in user_text


# ═══════════════════════════════════════════════════════════════════════════
# Section G: Invoke Teacher (async, mocked) — 5 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestInvokeTeacher:
    """AC: Full Teacher invocation pipeline with mocked NIM."""

    @pytest.mark.asyncio
    async def test_success_basic_invocation(self, tmp_path):
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        settings = make_settings(tmp_path / "workspace")
        prepare_mock = AsyncMock(return_value=fake_prepare_result(1))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=COSMOS_REASON2_8B,
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/img1.jpg",
                settings=settings,
            )

        assert result.invocation_status == "success"
        assert result.content is not None
        assert result.generation_preset_key == "precise"
        assert result.thinking_mode_effective == "on"
        assert result.structured_generation_attempted is True
        assert result.prompt_hash is not None
        assert len(result.prompt_hash) == 64
        prepare_mock.assert_called_once()
        args, kwargs = prepare_mock.call_args
        assert args[0] == ["/fake/img1.jpg"]
        assert kwargs["settings"] is settings

    @pytest.mark.asyncio
    async def test_unsupported_thinking_toggle_is_not_sent(self):
        """Persisted capability support gates the actual request kwargs."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_kwargs: dict[str, Any] = {}

        async def capture_completions(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(thinking_toggle_support="unsupported"),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=False,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/img1.jpg",
            )

        assert "chat_template_kwargs" not in captured_kwargs
        assert result.thinking_mode_effective == "on"
        assert result.thinking_toggle_attempted is False
        assert result.reasoning_headroom_tokens_effective == 16384

    @pytest.mark.asyncio
    async def test_default_guided_schema_places_rationale_last(self):
        """The response grammar matches the rationale-last production prompt."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_kwargs: dict[str, Any] = {}

        async def capture_completions(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=False,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/img1.jpg",
            )

        schema = captured_kwargs["response_format"]["json_schema"]["schema"]
        assert list(schema["properties"]) == [
            "damage_visible",
            "severity",
            "rationale_note",
        ]
        assert schema["required"] == ["severity"]

    @pytest.mark.asyncio
    async def test_truncation_retry(self):
        """``finish_reason="length"`` triggers exactly one retry with a
        larger max_tokens, and the retry's response becomes the result."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_max_tokens: list[int] = []

        async def counting_completions(*args, **kwargs):
            captured_max_tokens.append(kwargs["max_tokens"])
            if len(captured_max_tokens) == 1:
                return _fake_nim_truncated()
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=counting_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/img1.jpg",
            )

        assert len(captured_max_tokens) == 2  # original + retry
        assert captured_max_tokens[1] > captured_max_tokens[0]
        assert result.invocation_status == "success"

    @pytest.mark.asyncio
    async def test_retry_preserves_preset_thinking_visual(self):
        """Truncation retry changes only max_tokens."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_kwargs: list[dict] = []

        async def capturing_completions(*args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                return _fake_nim_truncated()
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capturing_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="explore",
                thinking_on=False,
                visual_budget_preset_key="fast",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/img1.jpg",
            )

        assert len(captured_kwargs) == 2
        # Preset/thinking/visual unchanged between calls
        assert captured_kwargs[0]["temperature"] == captured_kwargs[1]["temperature"]
        assert captured_kwargs[0]["top_p"] == captured_kwargs[1]["top_p"]
        assert captured_kwargs[0].get("chat_template_kwargs") == captured_kwargs[1].get(
            "chat_template_kwargs"
        )
        # max_tokens increased
        assert captured_kwargs[1]["max_tokens"] > captured_kwargs[0]["max_tokens"]

    @pytest.mark.asyncio
    async def test_no_image_still_works(self):
        """query_storage_ref=None → no image prep, text-only prompt."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ) as mock_prep,
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref=None,  # No image
            )

        assert result.invocation_status == "success"
        assert result.image_transport_mode is None
        mock_prep.assert_not_called()

    @pytest.mark.asyncio
    async def test_eval_batch_uses_seed(self):
        """evaluation/batch_label purpose includes deterministic seed."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_kwargs: list[dict] = []

        async def capturing(*args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return _fake_nim_success()

        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capturing,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(1)),
            ),
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="img1",
                purpose="evaluation",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="test-model",
                model_config=_make_model_config(),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                scope_id="eval-run-42",
                query_storage_ref="/fake/img1.jpg",
            )

        assert "seed" in captured_kwargs[0]
        assert result.seed_effective is not None
        assert isinstance(result.seed_effective, int)


# ═══════════════════════════════════════════════════════════════════════════
# Section G.1: Inline ICL image injection
# ═══════════════════════════════════════════════════════════════════════════


def _icl_example(key: str, label: dict[str, Any], storage_ref: str | None = None):
    """Build an ``ICLExample`` for invoke_teacher tests."""
    from vlm_feedback_loop.services.icl_service import ICLExample

    return ICLExample(
        example_key=key,
        label_json=label,
        labeled_at="2026-04-24T20:00:00Z",
        phash="0" * 16,
        storage_ref=(storage_ref if storage_ref is not None else f"/fake/{key}.jpg"),
    )


def _failed_prepare_result(n: int, fail_idx: int = 0):
    """Build a partial-failure BatchPrepareResult — one entry has
    ``error`` set and the batch ``success=False``."""
    from vlm_feedback_loop.services.image_transport import (
        BatchPrepareResult,
        PreparedImage,
    )

    return BatchPrepareResult(
        images=[
            PreparedImage(
                content_part=(
                    None
                    if i == fail_idx
                    else {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,IMG{i}"},
                    }
                ),
                transport_mode="base64_inline",
                format_transmitted="image/jpeg",
                error="missing_file" if i == fail_idx else None,
            )
            for i in range(n)
        ],
        success=False,
    )


class TestInvokeTeacherInlineIclImageInjection:
    """ICL example images are dispatched alongside the query image
    so the model receives ``[icl_text, icl_image]`` content-part pairs.

    Text-only ICL (JSON labels without their images) gives the model no
    visual grounding. These tests pin the content-list shape, the
    per-model max_images fallback, cold start, sentinel-collision
    defense, and partial-prep abort.
    """

    @pytest.mark.asyncio
    async def test_inline_interleaves_icl_images(self):
        """Happy path: 3 ICL examples + 1 query, max_images=10. Content
        list interleaves text and image content parts so each ICL image
        sits between its "ICL Example N image (E0N):" header and its
        "ICL Example N label (E0N): {json}" anchor (image before label,
        binding each label to its image position). The query image
        follows "Now label the QUERY image below."
        """
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_messages: list[Any] = []

        async def capture_completions(*args, **kwargs):
            # The signature is (base_url, auth_headers, model_name, messages, deadline_s, ...)
            captured_messages.append(args[3])
            return _fake_nim_success()

        prepare_mock = AsyncMock(return_value=fake_prepare_result(4))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            icl_candidates = [
                _icl_example(
                    "icl_a",
                    {"rationale_note": "rock", "severity": "low"},
                ),
                _icl_example(
                    "icl_b",
                    {"rationale_note": "paper", "severity": "medium"},
                ),
                _icl_example(
                    "icl_c",
                    {"rationale_note": "scissors", "severity": "high"},
                ),
            ]
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=COSMOS_REASON2_8B,
                model_config=_make_model_config(max_images_per_request=10),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                icl_candidates=icl_candidates,
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        # Three ICL images + 1 query = 4 storage_refs in one batched prep call
        prepare_mock.assert_called_once()
        prep_args = prepare_mock.call_args.args
        assert prep_args[0] == [
            "/fake/icl_a.jpg",
            "/fake/icl_c.jpg",
            "/fake/icl_b.jpg",
            "/fake/query.jpg",
        ], "ICL refs must use bookend order, followed by the query"
        assert result.icl_example_keys_used == ["icl_a", "icl_c", "icl_b"]

        # Result accounting
        assert result.invocation_status == "success"
        assert result.icl_images_attached_count == 3
        rendered_text = "\n".join(
            part["text"]
            for part in captured_messages[0][-1]["content"]
            if part.get("type") == "text"
        )
        label_lines = [
            line
            for line in rendered_text.splitlines()
            if "ICL Example" in line and "label (E" in line
        ]
        assert len(label_lines) == 3
        assert all("rationale_note" not in line for line in label_lines)

        # User-message content list has interleaved text/image parts.
        # Locate image parts in dispatch order and confirm 4 of them.
        msgs = captured_messages[0]
        user_content = msgs[-1]["content"]
        assert isinstance(user_content, list)
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 4, (
            f"Expected 4 image parts (3 ICL + 1 query); got {len(image_parts)}"
        )
        # Order: IMG0..IMG2 are ICL, IMG3 is query
        urls = [p["image_url"]["url"] for p in image_parts]
        assert urls == [
            "data:image/jpeg;base64,IMG0",
            "data:image/jpeg;base64,IMG1",
            "data:image/jpeg;base64,IMG2",
            "data:image/jpeg;base64,IMG3",
        ]

        # No sentinel survived into any text part
        for part in user_content:
            if part.get("type") == "text":
                assert "<<<__VLM_" not in part["text"]

        # Each ICL image sits between its image-header and label-anchor
        # text parts (text → image → text → image …). The image-header
        # for example N appears in the text part IMMEDIATELY preceding
        # ICL image N; the label anchor for example N appears in the
        # text part IMMEDIATELY following.
        text_parts_idx = [
            i for i, p in enumerate(user_content) if p.get("type") == "text"
        ]
        image_parts_idx = [
            i for i, p in enumerate(user_content) if p.get("type") == "image_url"
        ]
        first_img_idx = image_parts_idx[0]
        preceding_text = " ".join(
            user_content[i]["text"] for i in text_parts_idx if i < first_img_idx
        )
        # Image-before-label: header is in the preceding text, label is NOT
        assert "ICL Example 1 image (E01):" in preceding_text
        assert "ICL Example 1 label (E01):" not in preceding_text

    @pytest.mark.asyncio
    async def test_drops_all_icl_when_max_images_one(self):
        """Single-image Vision model: max_images_per_request=1.

        Cannot fit ICL images alongside the query, and inline
        injection forbids text-only ICL. Image-budget pruning drops
        every ICL example (cap = max - 1 = 0), so the 4th proposal
        runs cold-start-style: only the query image, no ICL JSON
        markers in the prompt.
        """
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_messages: list[Any] = []

        async def capture_completions(*args, **kwargs):
            captured_messages.append(args[3])
            return _fake_nim_success()

        prepare_mock = AsyncMock(return_value=fake_prepare_result(1))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            icl_candidates = [
                _icl_example("icl_a", {"category": "rock"}),
                _icl_example("icl_b", {"category": "paper"}),
                _icl_example("icl_c", {"category": "scissors"}),
            ]
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name="custom/single-image-vlm",
                model_config=_make_model_config(max_images_per_request=1),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=icl_candidates,
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        # Only the query image was prepared (no ICL refs in batch)
        prep_args = prepare_mock.call_args.args
        assert prep_args[0] == ["/fake/query.jpg"]

        # All 3 ICL candidates dropped by image-budget pruning.
        assert result.icl_images_attached_count == 0
        assert result.icl_example_keys_used == []

        # Content list has exactly 1 image part (query)
        user_content = captured_messages[0][-1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 1

        # No ICL JSON markers in the rendered prompt — every retained
        # example must be image-grounded, so when the image budget is
        # exhausted the ICL section is empty (cold-start-equivalent).
        all_text = " ".join(p["text"] for p in user_content if p.get("type") == "text")
        assert "E01:" not in all_text
        assert "E02:" not in all_text
        assert "E03:" not in all_text
        assert "ICL Example 1" not in all_text
        # No sentinel leaks
        assert "<<<__VLM_" not in all_text

    @pytest.mark.asyncio
    async def test_image_budget_caps_retained_examples(self):
        """max_images_per_request=5, 8 ICL candidates: image-budget
        pruning trims retained to 4 (cap = 5 - 1 query). Every retained
        ICL has its image attached; the dispatched content list shows
        4 ICL images + 1 query."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_messages: list[Any] = []

        async def capture_completions(*args, **kwargs):
            captured_messages.append(args[3])
            return _fake_nim_success()

        # 4 ICL + 1 query = 5 prepared images.
        prepare_mock = AsyncMock(return_value=fake_prepare_result(5))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            # 8 candidates, all distinct phashes so similarity scoring
            # has clear ranking; non-pinned so all are eligible to drop.
            icl_candidates = [
                _icl_example(f"icl_{c}", {"rationale_note": c, "category": c})
                for c in (
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "eta",
                    "theta",
                )
            ]
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=MISTRAL_MEDIUM_3_5,
                model_config=_make_model_config(max_images_per_request=5),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                icl_candidates=icl_candidates,
                generation_preset_key="precise",
                thinking_on=False,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        # Image-budget pruning enforces cap = max(5) - 1(query) = 4:
        # 8 candidates → exactly 4 retained, every one image-grounded.
        assert result.icl_images_attached_count == 4
        assert len(result.icl_example_keys_used) == 4
        # Invariant: count(images attached) == count(retained).
        assert result.icl_images_attached_count == len(result.icl_example_keys_used)
        # The 4 dropped candidates must not appear in the retained set.
        assert set(result.icl_example_keys_used) <= {
            f"icl_{c}"
            for c in (
                "alpha",
                "beta",
                "gamma",
                "delta",
                "epsilon",
                "zeta",
                "eta",
                "theta",
            )
        }

        # Wire shape: 4 ICL images + 1 query = 5 content_parts of type
        # image_url, in order (ICL refs dispatched first).
        user_content = captured_messages[0][-1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 5

    @pytest.mark.asyncio
    async def test_invariant_holds_under_image_budget_pruning(self):
        """Happy path: when image-budget pruning fires, the retained set
        and the dispatched image-content-parts agree exactly. Calls
        invoke_teacher with parameters that force the image-budget
        pruning code path and verifies the assertion path is silent."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        prepare_mock = AsyncMock(return_value=fake_prepare_result(5))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            icl_candidates = [
                _icl_example(f"icl_{i}", {"category": f"v{i}"}) for i in range(7)
            ]
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=NEMOTRON_3_NANO_OMNI_REASONING,
                model_config=_make_model_config(max_images_per_request=5),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                icl_candidates=icl_candidates,
                generation_preset_key="precise",
                thinking_on=False,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        # Invariant assertion in invoke_teacher passes silently.
        assert result.invocation_status == "success"
        assert result.icl_images_attached_count == len(result.icl_example_keys_used)
        assert result.icl_images_attached_count == 4

    @pytest.mark.asyncio
    async def test_cold_start_no_icl_examples(self):
        """Cold start: 0 ICL retained → only query image prepared.
        Same content-list shape as text-only fallback for the no-ICL
        case. icl_images_attached_count==0."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        captured_messages: list[Any] = []

        async def capture_completions(*args, **kwargs):
            captured_messages.append(args[3])
            return _fake_nim_success()

        prepare_mock = AsyncMock(return_value=fake_prepare_result(1))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=capture_completions,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=COSMOS_REASON2_8B,
                model_config=_make_model_config(max_images_per_request=8),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        prep_args = prepare_mock.call_args.args
        assert prep_args[0] == ["/fake/query.jpg"]
        assert result.icl_images_attached_count == 0

        user_content = captured_messages[0][-1]["content"]
        image_parts = [p for p in user_content if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        # No sentinels and no E0N markers (no ICL section)
        all_text = " ".join(p["text"] for p in user_content if p.get("type") == "text")
        assert "<<<__VLM_" not in all_text
        # Cold-start prompt has no ICL section heading at all
        # (image-mode heading omits the colon, text-only heading is also
        # gated on ICL_RENDERED_EXAMPLES being non-empty).
        assert "ICL Examples" not in all_text
        assert "ICL Example 1" not in all_text

    @pytest.mark.asyncio
    async def test_partial_image_prep_failure_aborts_dispatch(self):
        """All images for one invocation (ICL + query) MUST be
        prepared before the model request is dispatched. If any image fails
        to prepare, ``nim_client.chat_completions`` MUST NOT be called
        and the invocation finalizes as endpoint_error."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        completions_mock = AsyncMock(return_value=_fake_nim_success())
        prepare_mock = AsyncMock(return_value=_failed_prepare_result(2, fail_idx=0))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=completions_mock,
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            icl_candidates = [
                _icl_example("icl_a", {"category": "rock"}),
            ]
            result = await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=COSMOS_REASON2_8B,
                model_config=_make_model_config(max_images_per_request=8),
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                icl_candidates=icl_candidates,
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
            )

        # The model was NEVER called — abort precondition holds
        completions_mock.assert_not_called()
        # Result reflects the abort
        assert result.invocation_status == "endpoint_error"
        assert result.icl_images_attached_count == 0
        assert "image_transport_failed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_sentinel_collision_in_user_content_raises(self):
        """Defensive: if user-provided Guidance/rationale text happens to
        contain a literal slot sentinel string, the renderer MUST raise
        rather than silently mis-split the content list."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        prepare_mock = AsyncMock(return_value=fake_prepare_result(2))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            # A rendered Core value carries a literal sentinel marker.
            poisoned = "<<<__VLM_ICL_IMG_SLOT_2__>>>"
            icl_candidates = [
                _icl_example("icl_a", {"severity": poisoned}),
            ]
            with pytest.raises(ValueError, match="[Ss]entinel collision"):
                await invoke_teacher(
                    project_id="p1",
                    example_key="query",
                    purpose="interactive_proposal",
                    inference_invocation_id="inv-1",
                    guidance_description=FIXTURE_DESCRIPTION,
                    guidance_rules=FIXTURE_RULES,
                    guidance_fields=FIXTURE_FIELDS,
                    generation_order=FIXTURE_GENERATION_ORDER,
                    derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                    model_name=COSMOS_REASON2_8B,
                    model_config=_make_model_config(max_images_per_request=8),
                    endpoint_base_url="https://test.nvidia.com/v1",
                    auth_headers={},
                    icl_candidates=icl_candidates,
                    generation_preset_key="precise",
                    thinking_on=True,
                    visual_budget_preset_key="balanced",
                    labeling_presets=LABELING_PRESETS,
                    visual_budget_presets=VISUAL_BUDGET_PRESETS,
                    query_storage_ref="/fake/query.jpg",
                )


# ═══════════════════════════════════════════════════════════════════════════
# Section G.2: Per-model ICL depth default resolution
# ═══════════════════════════════════════════════════════════════════════════


class TestPerModelIclDepthDefault:
    """The per-model ICL depth default (§6.2) is resolved inside
    ``invoke_teacher`` — the single funnel shared by proposals, evaluation,
    and batch labeling — as: explicit ``icl_max_examples`` override wins
    outright (in either direction); else ``ModelConfigInput.
    default_icl_max_examples``; else uncapped. July 2026 depth studies:
    useful depth is model-specific (Nemotron Nano VL 2, Omni 4, CR3 8,
    CR2-2B 8, CR2-8B 16, MiniMax 8, ceiling-class 2)."""

    async def _invoke(self, *, n_candidates: int, n_attached: int, **kwargs):
        """Run invoke_teacher with ``n_candidates`` ICL edits available and
        a mocked NIM; ``n_attached`` sizes the prepared-image batch
        (attached ICL + 1 query). Returns the invocation result."""
        from vlm_feedback_loop.services.prompt_service import invoke_teacher

        prepare_mock = AsyncMock(return_value=fake_prepare_result(n_attached + 1))
        with (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=_fake_nim_success()),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=prepare_mock,
            ),
        ):
            return await invoke_teacher(
                project_id="p1",
                example_key="query",
                purpose="interactive_proposal",
                inference_invocation_id="inv-1",
                guidance_description=FIXTURE_DESCRIPTION,
                guidance_rules=FIXTURE_RULES,
                guidance_fields=FIXTURE_FIELDS,
                generation_order=FIXTURE_GENERATION_ORDER,
                derived_json_schema=FIXTURE_DERIVED_JSON_SCHEMA,
                model_name=COSMOS_REASON2_8B,
                endpoint_base_url="https://test.nvidia.com/v1",
                auth_headers={},
                icl_candidates=[
                    _icl_example(f"icl_{i}", {"category": "rock"})
                    for i in range(n_candidates)
                ],
                generation_preset_key="precise",
                thinking_on=True,
                visual_budget_preset_key="balanced",
                labeling_presets=LABELING_PRESETS,
                visual_budget_presets=VISUAL_BUDGET_PRESETS,
                query_storage_ref="/fake/query.jpg",
                **kwargs,
            )

    @pytest.mark.asyncio
    async def test_model_default_caps_selection_when_no_override(self):
        """No explicit icl_max_examples + model default 2 → exactly 2 of
        the 5 eligible Edits are attached (a nemotron-family Teacher at
        product defaults labels with shallow ICL)."""
        result = await self._invoke(
            n_candidates=5,
            n_attached=2,
            model_config=_make_model_config(
                max_images_per_request=10, default_icl_max_examples=2
            ),
        )
        assert result.invocation_status == "success"
        assert len(result.icl_example_keys_used) == 2
        assert result.icl_images_attached_count == 2

    @pytest.mark.asyncio
    async def test_explicit_override_wins_upward_over_model_default(self):
        """An explicit icl_max_examples=4 exceeds the model default of 2 —
        diagnostic depth sweeps must be able to out-vote the default, so
        4 examples attach."""
        result = await self._invoke(
            n_candidates=5,
            n_attached=4,
            icl_max_examples=4,
            model_config=_make_model_config(
                max_images_per_request=10, default_icl_max_examples=2
            ),
        )
        assert result.invocation_status == "success"
        assert len(result.icl_example_keys_used) == 4

    @pytest.mark.asyncio
    async def test_null_model_default_leaves_selection_uncapped(self):
        """default_icl_max_examples=None (unmeasured / operator-registered
        model) + no override → all eligible Edits attach, exactly the
        pre-column behavior."""
        result = await self._invoke(
            n_candidates=5,
            n_attached=5,
            model_config=_make_model_config(
                max_images_per_request=10, default_icl_max_examples=None
            ),
        )
        assert result.invocation_status == "success"
        assert len(result.icl_example_keys_used) == 5

    @pytest.mark.asyncio
    async def test_image_budget_still_prunes_within_model_default(self):
        """The model default does not bypass the image budget: default 8
        but max_images_per_request=3 (2 ICL + 1 query) → only 2 attach."""
        result = await self._invoke(
            n_candidates=5,
            n_attached=2,
            model_config=_make_model_config(
                max_images_per_request=3, default_icl_max_examples=8
            ),
        )
        assert result.invocation_status == "success"
        assert result.icl_images_attached_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# Section H: Log Point 1 (model invocation) — 2 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLogModelInvocation:
    """AC: Log point 1 emits INFO with required fields."""

    def test_info_log_with_required_fields(self, caplog):
        from vlm_feedback_loop.services.prompt_service import log_model_invocation

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.prompt_service"):
            log_model_invocation(
                "p1",
                COSMOS_REASON2_8B,
                "https://test.nvidia.com/v1",
                {"temperature": 0.0, "top_p": 1.0},
                None,
                "stop",
                150,
                {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550},
            )

        records = [r for r in caplog.records if "Model invocation" in r.message]
        assert len(records) >= 1
        rec = records[0]
        assert rec.levelno == logging.INFO
        details = getattr(rec, "details", None)
        assert details is not None
        assert details["model_name"] == COSMOS_REASON2_8B
        assert details["sampling_params"] == {"temperature": 0.0, "top_p": 1.0}
        assert details["finish_reason"] == "stop"
        assert details["latency_ms"] == 150
        assert details["prompt_tokens"] == 500

    def test_retry_logging(self, caplog):
        from vlm_feedback_loop.services.prompt_service import log_model_invocation

        with caplog.at_level(logging.DEBUG, logger="vlm_feedback_loop.prompt_service"):
            log_model_invocation(
                "p1",
                "test-model",
                "https://test.nvidia.com/v1",
                {"temperature": 0.0, "top_p": 1.0},
                None,
                "length",
                200,
                None,
                is_retry=True,
                retry_trigger="finish_reason=length",
                retry_change="max_tokens 512 → 768",
            )

        records = [r for r in caplog.records if "Model invocation" in r.message]
        details = getattr(records[0], "details", None)
        assert details["is_retry"] is True
        assert details["retry_trigger"] == "finish_reason=length"


# ═══════════════════════════════════════════════════════════════════════════
# Section I: Concurrency — 1 test
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrency:
    """AC: Pure functions are safe for concurrent execution."""

    def test_concurrent_render_prompt_independent(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        results = []
        for i in range(4):
            desc = f"Task {i}"
            msgs, h = render_prompt(
                desc,
                FIXTURE_RULES,
                [],
                FIXTURE_DERIVED_JSON_SCHEMA,
                True,
            )
            results.append((msgs, h))

        # Guidance prose is intentionally absent from the compact production
        # prompt, so description-only changes do not alter the rendered bytes.
        hashes = {h for _, h in results}
        assert len(hashes) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Section J: Edge Cases — 2 tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge case handling."""

    def test_serving_prompt_render_is_pinned_byte_for_byte(self):
        """Golden pin of the D.1 serving prompt (zero-shot, no schema attach).

        Every Teacher inference, evaluation run, batch label, and training
        export flows through this render; ICL studies compare prompt hashes
        across runs. Any byte change here is a serving-distribution change
        and must be a deliberate, reviewed decision — update the pin only
        alongside a changelog entry explaining the prompt change.
        """
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            "Classify the hand gesture.",
            "Judge by silhouette only.",
            [],
            None,
            False,
        )
        assert messages[0]["content"] == (
            "You are a vision labeling assistant. Output valid JSON only."
        )
        assert messages[1]["content"] == (
            "Now label the QUERY image below.\n"
            "<<<__VLM_QUERY_IMG_SLOT__>>>\n\n"
            "Return one valid JSON object.\n"
            "Return JSON only."
        )

    def test_empty_rules(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            "",
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"]
        # Rules section should be omitted when empty
        assert "Rules:" not in user_text

    def test_empty_description_omits_task_description_header(self):
        """An empty Task Description renders no "Task Description:" block —
        with no Rules either, the user turn opens with the Output Contract
        (the same omit-when-empty guard the Rules block uses)."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt("", "", [], None, False)
        user_text = messages[1]["content"]
        assert "Task Description:" not in user_text
        assert user_text.strip().startswith("Now label the QUERY image below.")
        assert "schema described above" not in user_text

    def test_whitespace_description_omits_task_description_header(self):
        """A whitespace-only Task Description is treated as empty — no
        "Task Description:" block in the rendered user turn."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt("   ", "", [], None, False)
        user_text = messages[1]["content"]
        assert "Task Description:" not in user_text
        assert user_text.strip().startswith("Now label the QUERY image below.")
        assert "schema described above" not in user_text

    def test_rules_are_omitted_from_compact_production_prompt(self):
        """Verbose Rules are not reintroduced when Description is empty."""
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt("", FIXTURE_RULES, [], None, False)
        user_text = messages[1]["content"]
        assert "Task Description:" not in user_text
        assert "Rules:" not in user_text
        assert FIXTURE_RULES not in user_text
        assert user_text.strip().startswith("Now label the QUERY image below.")

    def test_structured_gen_not_attempted(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            False,
        )
        user_text = messages[1]["content"]
        assert "Label schema:" in user_text
        assert "Return one JSON object matching the label schema above." in user_text


class TestProposalRationaleGrounding:
    """The proposal rationale is image-specific, honest, and domain-agnostic."""

    def test_proposal_uses_task_conditional_grounding(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"].lower()
        assert "describe the subject the task asks about" in user_text
        assert "only when that distinction matters to the task" in user_text
        assert "physical foreground item" not in user_text

    def test_proposal_has_no_domain_specific_or_fixed_palette_instructions(self):
        from vlm_feedback_loop.services.prompt_service import render_prompt

        messages, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        user_text = messages[1]["content"].lower()
        for leaked_word in (
            "fingers",
            "curled",
            "clenched",
            "outer contour",
            "region compactness",
        ):
            assert leaked_word not in user_text

    def test_proposal_and_regen_share_honesty_contract(self):
        from vlm_feedback_loop.services.prompt_service import (
            render_prompt,
            render_rationale_regeneration_prompt,
        )

        prop_msgs, _ = render_prompt(
            FIXTURE_DESCRIPTION,
            FIXTURE_RULES,
            [],
            FIXTURE_DERIVED_JSON_SCHEMA,
            True,
        )
        regen_msgs, _ = render_rationale_regeneration_prompt(
            guidance_description=FIXTURE_DESCRIPTION,
        )
        prop_text = prop_msgs[1]["content"].lower()
        regen_text = " ".join(str(msg["content"]).lower() for msg in regen_msgs)
        for term in (
            "image-specific",
            "do not",
            "invent",
            "subject the task asks about",
            "physical carrier",
            "content shown on it",
        ):
            assert term in prop_text and term in regen_text
        assert "uncertainty" in prop_text
        assert "uncertainty" in regen_text
