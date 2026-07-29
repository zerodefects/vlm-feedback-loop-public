# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token budget derivation service.

Pure stateless functions for per-field token estimation and prompt budget
computation.  No database or HTTP dependencies — config values are passed
as explicit parameters with defaults from ``_defaults.py``.
:func:`token_budget_invoke_kwargs` is the one mapping from ``Settings`` to
those parameters for ``invoke_teacher`` callers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from vlm_feedback_loop._defaults import DEFAULTS
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.services.schema_core import RESERVED_FIELD_NAME

# ── Encoder singleton ──────────────────────────────────────────────────────

try:
    import tiktoken

    _encoder = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _encoder = None


# ── Token counting ─────────────────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """Count tokens using ``cl100k_base``; fall back to ``len(text) // 4``.

    The fallback absorbs tokenizer mismatch; the safety margin
    provides additional headroom.
    """
    if _encoder is not None:
        return len(_encoder.encode(text))
    return max(1, len(text) // 4)  # pragma: no cover


# ── Per-field estimation ───────────────────────────────────────────────────

# Single source of truth is
# ``_defaults.py:DEFAULTS``; values are derived rather than copied.
_RATIONALE_NOTE_ESTIMATE = DEFAULTS["RATIONALE_NOTE_ESTIMATE_TOKENS"]
_DEFAULT_UNBOUNDED_STRING_BUDGET = DEFAULTS["DEFAULT_UNBOUNDED_STRING_BUDGET"]


def estimate_field_tokens(
    field: dict[str, Any],
    *,
    rationale_note_estimate: int = _RATIONALE_NOTE_ESTIMATE,
    default_unbounded_string_budget: int = _DEFAULT_UNBOUNDED_STRING_BUDGET,
) -> int:
    """Worst-case token estimate for a single SchemaCore field.

    Counting model (per field type):

    - Keyed types (enum, enum_set, bounded string) measure the serialized
      key prefix ``"name": `` explicitly and add a per-type value term —
      enum's ``+ 6`` and enum_set's array-syntax term cover value quotes,
      separators, and headroom on top of that.
    - Flat types (boolean, integer, unbounded string, rationale_note) use
      a single spec-pinned constant that already embeds key + value +
      separator overhead; no key term is added.

    Estimates are deliberately loose upper bounds — the ×2 multiplier and
    ``BASE_OUTPUT_TOKENS_FLOOR`` downstream absorb any remaining slack, so
    do not "tighten" individual branches against the spec formulas.

    Args:
        field: A field dict from ``Guidance.schema["fields"]``.
        rationale_note_estimate: Override for RATIONALE_NOTE_ESTIMATE_TOKENS.
        default_unbounded_string_budget: Override for DEFAULT_UNBOUNDED_STRING_BUDGET.
    """
    field_name = field.get("field_name", "")
    field_type = field.get("type", "string")

    # Reserved rationale_note gets a flat estimate
    if field_name == RESERVED_FIELD_NAME:
        return rationale_note_estimate

    key_overhead = count_tokens(f'"{field_name}": ')

    if field_type == "boolean":
        return 6

    if field_type == "integer":
        return 8

    if field_type == "enum":
        allowed = field.get("allowed_values", [])
        max_val_tokens = max((count_tokens(v) for v in allowed), default=1)
        return key_overhead + max_val_tokens + 6

    if field_type == "enum_set":
        allowed = field.get("allowed_values", [])
        sum_val_tokens = sum(count_tokens(v) for v in allowed)
        # Array syntax: brackets, commas, quotes
        array_overhead = 4 + max(0, len(allowed) - 1) * 2
        return key_overhead + sum_val_tokens + array_overhead

    if field_type == "string":
        max_length = field.get("max_length")
        if max_length is not None:
            return key_overhead + math.ceil(max_length / 4)
        return default_unbounded_string_budget

    # Fallback for unknown types
    return default_unbounded_string_budget


# ── Field-mode filtering ──────────────────────────────────────────────────


def _include_field(
    field: dict[str, Any],
    field_mode: Literal["all", "aux_and_core", "core_only"],
) -> bool:
    """Determine whether a field is included under the given field mode."""
    role = field.get("role", "core")
    name = field.get("field_name", "")

    if field_mode == "all":
        return True
    if field_mode == "aux_and_core":
        # Include aux + core, but exclude rationale_note
        return name != RESERVED_FIELD_NAME
    # core_only
    return role == "core"


# ── Schema output estimate ────────────────────────────────────────────────


def estimate_schema_output_tokens(
    fields: list[dict[str, Any]],
    field_mode: Literal["all", "aux_and_core", "core_only"],
    *,
    json_structural_overhead: int = 48,
    rationale_note_estimate: int = _RATIONALE_NOTE_ESTIMATE,
    default_unbounded_string_budget: int = _DEFAULT_UNBOUNDED_STRING_BUDGET,
) -> int:
    """Sum per-field estimates for fields matching *field_mode*.

    Returns ``json_structural_overhead + sum(per_field_estimates)``.
    """
    total = json_structural_overhead
    for f in fields:
        if _include_field(f, field_mode):
            total += estimate_field_tokens(
                f,
                rationale_note_estimate=rationale_note_estimate,
                default_unbounded_string_budget=default_unbounded_string_budget,
            )
    return total


# ── Budget derivation ─────────────────────────────────────────────────────


@dataclass
class TokenBudget:
    """Complete token budget for a single VLM invocation."""

    schema_output_estimate: int
    base_output_tokens: int
    reasoning_headroom: int
    max_output_tokens: int
    effective_max_input_tokens: int
    context_window_tokens: int


def derive_token_budget(
    fields: list[dict[str, Any]],
    output_field_mode: Literal["all", "aux_and_core", "core_only"],
    context_window_tokens: int,
    thinking_on: bool,
    *,
    base_output_tokens_floor: int = 256,
    json_structural_overhead: int = 48,
    max_output_fraction: float = 0.25,
    rationale_note_estimate: int = _RATIONALE_NOTE_ESTIMATE,
    default_unbounded_string_budget: int = _DEFAULT_UNBOUNDED_STRING_BUDGET,
    model_reasoning_headroom_tokens: int = 16384,
    runtime_prompt_output_max_tokens_override: int | None = None,
    token_safety_margin: float = 0.85,
) -> TokenBudget:
    """Derive the full token budget for a single VLM invocation.

    All config values are explicit parameters with built-in defaults,
    keeping this function pure and testable.
    """
    # 1. Schema output estimate
    schema_output_estimate = estimate_schema_output_tokens(
        fields,
        output_field_mode,
        json_structural_overhead=json_structural_overhead,
        rationale_note_estimate=rationale_note_estimate,
        default_unbounded_string_budget=default_unbounded_string_budget,
    )

    # 2. Base output tokens
    base_output_tokens = max(base_output_tokens_floor, 2 * schema_output_estimate)

    # 3. Reasoning headroom
    reasoning_headroom = model_reasoning_headroom_tokens if thinking_on else 0

    # 4. Max output tokens
    if runtime_prompt_output_max_tokens_override is not None:
        max_output_tokens = runtime_prompt_output_max_tokens_override
    else:
        max_output_tokens = min(
            base_output_tokens + reasoning_headroom,
            math.floor(context_window_tokens * max_output_fraction),
        )

    # 5. Effective max input tokens
    effective_max_input_tokens = math.floor(
        (context_window_tokens - max_output_tokens) * token_safety_margin
    )

    return TokenBudget(
        schema_output_estimate=schema_output_estimate,
        base_output_tokens=base_output_tokens,
        reasoning_headroom=reasoning_headroom,
        max_output_tokens=max_output_tokens,
        effective_max_input_tokens=effective_max_input_tokens,
        context_window_tokens=context_window_tokens,
    )


# ── Settings → invoke_teacher token-budget kwargs ─────────────────────────


class TokenBudgetInvokeKwargs(TypedDict):
    """Token-budget/preset kwargs of ``prompt_service.invoke_teacher``.

    A TypedDict (rather than ``dict[str, Any]``) so type checking validates
    the ``**`` splat against ``invoke_teacher``'s explicit-primitive
    signature at every call site.
    """

    labeling_presets: dict[str, dict[str, float]]
    visual_budget_presets: dict[str, Any]
    base_output_tokens_floor: int
    json_structural_overhead: int
    max_output_fraction: float
    rationale_note_estimate: int
    default_unbounded_string_budget: int
    model_reasoning_headroom_tokens: int
    runtime_prompt_output_max_tokens_override: int | None
    token_safety_margin: float


def token_budget_invoke_kwargs(settings: Settings) -> TokenBudgetInvokeKwargs:
    """Map ``Settings`` to ``invoke_teacher``'s token-budget/preset kwargs.

    The one Settings-to-parameters mapping shared by the three
    ``invoke_teacher`` callers (proposal, evaluation, batch labeling), so a
    new token-budget knob is wired once instead of three times.
    """
    return {
        "labeling_presets": settings.LABELING_PRESETS,
        "visual_budget_presets": settings.VISUAL_BUDGET_PRESETS,
        "base_output_tokens_floor": settings.BASE_OUTPUT_TOKENS_FLOOR,
        "json_structural_overhead": settings.JSON_STRUCTURAL_OVERHEAD_TOKENS,
        "max_output_fraction": settings.MAX_OUTPUT_FRACTION,
        "rationale_note_estimate": settings.RATIONALE_NOTE_ESTIMATE_TOKENS,
        "default_unbounded_string_budget": settings.DEFAULT_UNBOUNDED_STRING_BUDGET,
        "model_reasoning_headroom_tokens": settings.MODEL_REASONING_HEADROOM_TOKENS,
        "runtime_prompt_output_max_tokens_override": (
            settings.RUNTIME_PROMPT_OUTPUT_MAX_TOKENS_OVERRIDE
        ),
        "token_safety_margin": settings.RUNTIME_PROMPT_TOKEN_SAFETY_MARGIN,
    }


# ── ICL per-example token estimation ──────────────────────────────────────


def estimate_icl_example_tokens(
    label_json: dict[str, Any],
    fields: list[dict[str, Any]],
    generation_order: list[str],
    field_mode: Literal["all", "aux_and_core", "core_only"],
) -> int:
    """Estimate tokens for a single rendered ICL example.

    Renders the example to a JSON string and counts tokens, adding
    structural overhead for markers and formatting (~10 tokens).
    """
    rendered = render_icl_fields(label_json, fields, generation_order, field_mode)
    json_str = json.dumps(rendered, separators=(",", ":"))
    return count_tokens(json_str) + 10  # E01 marker + newlines


# ── ICL field rendering ───────────────────────────────────────────────────


def render_icl_fields(
    label_json: dict[str, Any],
    fields: list[dict[str, Any]],
    generation_order: list[str],
    field_mode: Literal["all", "aux_and_core", "core_only"],
) -> dict[str, Any]:
    """Render an ICL example's label filtered by *field_mode* in generation order.

    Args:
        label_json: The full label dict (all fields).
        fields: Schema field definitions from ``Guidance.schema["fields"]``.
        generation_order: Canonical field ordering from the guidance envelope.
        field_mode: Which fields to include.

    Returns:
        An ordered dict with keys in ``generation_order``, filtered by mode.
    """
    field_map = {f["field_name"]: f for f in fields}
    result: dict[str, Any] = {}
    for name in generation_order:
        f = field_map.get(name)
        if f is None:
            continue
        if not _include_field(f, field_mode):
            continue
        if name in label_json:
            result[name] = label_json[name]
    return result
