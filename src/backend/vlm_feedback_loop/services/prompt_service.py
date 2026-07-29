# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt rendering, Teacher invocation, and Generation Controls.

A family of pure, stateless functions (banner sections "Function 0"–"6":
the rationale-regeneration prompt, generation-param /
thinking-toggle / visual-budget resolution, seed derivation, prompt
rendering, request-kwargs construction) plus one async orchestrator
(``invoke_teacher``, "Function 7"), with a template loader and prompt-variant
registries up top.  The pure functions are independently testable; the
orchestrator coordinates NIM calls, image preparation, and result assembly.

Covers prompt rendering, structured generation, Generation Controls,
the 6-step request construction, Visual Budget Controls, the seed
policy, and the Teacher prompt template.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.clip_embedding_service import embedding_cache
from vlm_feedback_loop.services.icl_service import (
    ICLExample,
    bookend_icl_examples,
    log_icl_selection,
    prune_icl_by_budget,
    prune_icl_by_image_budget,
    select_icl_examples,
)
from vlm_feedback_loop.services.image_transport import prepare_images
from vlm_feedback_loop.services.schema_core import (
    place_rationale_last,
    strip_guided_decoding_unsupported_keys,
)
from vlm_feedback_loop.services.token_budget_service import (
    count_tokens,
    derive_token_budget,
    render_icl_fields,
)

logger = logging.getLogger("vlm_feedback_loop.prompt_service")


# ── Prompt template loader ─────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _compact_json_filter(value: Any) -> str:
    # Jinja2's built-in ``tojson`` sorts keys and HTML-escapes <>&'" so the
    # output is safe to embed in <script> tags. For prompt rendering both of
    # those defaults are wrong: sorting destroys the generation-order
    # signal carried by dict insertion order, and HTML escaping mutates the
    # prompt bytes (' becomes an HTML entity), breaking the byte-identical
    # compact-JSON contract that prompt hashing depends on.
    return json.dumps(value, separators=(",", ":"))


@lru_cache(maxsize=1)
def _prompt_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(_PROMPTS_DIR)),
        keep_trailing_newline=False,
        autoescape=False,
        undefined=StrictUndefined,
    )
    env.filters["compact_json"] = _compact_json_filter
    return env


def _render_prompt_template(name: str, **variables: Any) -> tuple[str, str]:
    """Render a prompt template file into (system_text, user_text).

    Templates live in ``src/backend/vlm_feedback_loop/prompts/*.txt`` and
    separate the system and user turns with literal ``===SYSTEM===`` and
    ``===USER===`` markers. Variables are Jinja2-substituted. An empty
    SYSTEM block (useful for capability-probe prompts) renders as an empty
    system_text; callers decide whether to emit a system message.
    """
    rendered = _prompt_env().get_template(name).render(**variables)
    if "===SYSTEM===" not in rendered or "===USER===" not in rendered:
        raise ValueError(
            f"Prompt template {name!r} is missing ===SYSTEM=== or ===USER=== marker."
        )
    _, _, after_system = rendered.partition("===SYSTEM===")
    system_text, _, user_text = after_system.partition("===USER===")
    return system_text.strip(), user_text.strip()


def load_probe_prompt(name: str) -> str:
    """Load the user-turn text of a single-turn probe template.

    Probe templates (``probe_*.txt``) typically have an empty SYSTEM block.
    Callers attach model-specific request kwargs (``response_format``,
    ``chat_template_kwargs``, ``mm_processor_kwargs``, or an image
    content-part) around this text themselves.
    """
    _, user_text = _render_prompt_template(name)
    return user_text


_TEACHER_PROPOSAL_TEMPLATE = "teacher_interactive_proposal.txt"


# ── Data structures ────────────────────────────────────────────────────────


@dataclass
class ModelConfigInput:
    """Lightweight projection of ModelConfig fields for invocation.

    Keeps ``invoke_teacher`` free of ORM / database session dependencies.
    """

    context_window_tokens: int
    thinking_toggle_mode: str = "none"
    thinking_toggle_support: str = "unknown"
    visual_budget_mode: str = "none"
    visual_budget_support: str = "unknown"
    structured_generation_support: str = "unknown"
    # Per-model cap on image content parts in a single chat/completions
    # request. Drives the image-budget pruning step in
    # ``invoke_teacher`` (inline ICL image injection). Default 5 is a
    # conservative starting point for an SME-registered model whose cap
    # has not been live-probed yet; seeded models carry live-probed
    # caps (Qwen 8, Mistral 8, Nemotron 10, Kimi 32). When the
    # SME registers a single-image-only model (cap=1), the
    # image-budget pruner drops all ICL examples and the prompt
    # renders via the legacy text-only fallback.
    max_images_per_request: int = 5
    # Per-model default ICL depth cap (ModelConfig.default_icl_max_examples).
    # Applied by ``invoke_teacher`` only when the caller passes no explicit
    # ``icl_max_examples`` override — an override (per-run API field or a
    # non-null ``ICL_MAX_EXAMPLES`` setting) always wins, in either
    # direction (diagnostic depth sweeps may exceed the model default).
    # None = no model default; selection is bounded only by adaptive-K,
    # token budget, and the image budget, exactly as before.
    default_icl_max_examples: int | None = None


@dataclass
class TeacherInvocationResult:
    """Everything the caller needs to populate an OperationRecord."""

    inference_invocation_id: str
    content: str | None
    finish_reason: str | None
    invocation_status: str  # success | timeout | endpoint_error
    latency_ms: int | None
    usage: dict[str, int] | None
    icl_example_keys_used: list[str]
    prompt_hash: str
    structured_generation_attempted: bool
    structured_generation_fallback_used: bool
    # Thinking + Visual Budget runtime rejection plumbing.
    # ``*_attempted`` is True whenever the dispatched request actually
    # included the corresponding kwargs; caller services combine them
    # with the rejection-detector heuristics (and track any fallback
    # retry in their own locals, as they do for
    # ``structured_generation_fallback_used``).
    thinking_toggle_attempted: bool = False
    visual_budget_attempted: bool = False
    # Generation Controls state
    generation_preset_key: str | None = None
    sampling_params_effective: dict[str, float] | None = None
    thinking_mode_effective: str | None = None
    thinking_request_fields_effective: dict[str, Any] | None = None
    max_tokens_effective: int | None = None
    reasoning_headroom_tokens_effective: int | None = None
    visual_budget_preset_key: str | None = None
    visual_budget_params_effective: dict[str, Any] | None = None
    seed_effective: int | None = None
    # Image transport
    image_transport_mode: str | None = None
    image_format_transmitted: str | None = None
    # Inline ICL image injection. Number of ICL example
    # images actually attached to the dispatched request. 0 means
    # text-only ICL — either cold start, max-images budget exhausted,
    # or all ICL image transport prep failed (in which case the run
    # continues with text-only ICL rather than aborting; aborting is
    # reserved for the QUERY image, since labeling without it is
    # impossible).
    icl_images_attached_count: int = 0
    # Error
    error: str | None = None
    # Per-stage timings (OperationRecord cols
    # ``t_image_prep_ms`` / ``t_prompt_render_ms`` / ``t_nim_call_ms`` /
    # ``t_validation_ms``). Nullable when the stage didn't run
    # (e.g. ``t_image_prep_ms`` is None for invocations with no images).
    # ``t_validation_ms`` is filled in by the caller (proposal /
    # evaluation / batch_label service) since validation runs there,
    # not inside ``invoke_teacher``.
    t_image_prep_ms: int | None = None
    t_prompt_render_ms: int | None = None
    t_nim_call_ms: int | None = None


def is_rate_limit_exhaustion(result: TeacherInvocationResult) -> bool:
    """Detect a hosted-NIM rate-limit failure that exhausted retries.

    The HTTP client surfaces 429s with the message
    ``"Exhausted N retries. Last: HTTP 429"``. Distinguishing this from a
    generic ``endpoint_error`` lets the UI show a "wait + retry" hint
    rather than the catch-all failure banner.

    Shared across proposal, evaluation, and batch-label services so all
    three classify hosted-NIM rate-limit exhaustion identically.
    """
    if result.invocation_status != "endpoint_error":
        return False
    error_lower = (result.error or "").lower()
    return "http 429" in error_lower


# ── Function 0: Rationale regeneration prompt ────────────────────────────


def _render_schema_summary_for_rationale(
    guidance_fields: list[dict[str, Any]],
) -> str:
    """Render a compact, human-readable schema summary for the rationale prompt.

    Lists every Core field with its type and, for categorical fields, the
    allowed values.  Aux fields (including ``rationale_note``) are omitted —
    the rationale explains Core correctness, not Aux scaffolding.

    Example output::

        Label schema (for context):
          answer: enum — one of ["rock", "paper", "scissors"]
          severity: integer — 0..4
          damage_types: enum_set — any of ["crush", "dent", "scratch"]
          is_blurry: boolean
    """
    core_fields = [f for f in guidance_fields if f.get("role") == "core"]
    # Sort by display_order for stability; matches the generation order.
    core_fields.sort(key=lambda f: f.get("display_order", 0))

    if not core_fields:
        return ""

    lines = ["Label schema (for context):"]
    for field_def in core_fields:
        name = field_def.get("field_name", "?")
        ftype = field_def.get("type", "?")
        values = field_def.get("allowed_values")
        minimum = field_def.get("minimum")
        maximum = field_def.get("maximum")

        if ftype == "enum" and values:
            vals = ", ".join(json.dumps(v) for v in values)
            lines.append(f"  {name}: enum — one of [{vals}]")
        elif ftype == "enum_set" and values:
            vals = ", ".join(json.dumps(v) for v in values)
            lines.append(f"  {name}: enum_set — any of [{vals}]")
        elif ftype == "integer" and (minimum is not None or maximum is not None):
            lo = minimum if minimum is not None else "…"
            hi = maximum if maximum is not None else "…"
            lines.append(f"  {name}: integer — {lo}..{hi}")
        else:
            lines.append(f"  {name}: {ftype}")

    return "\n".join(lines)


def _render_output_schema_contract(derived_json_schema: dict[str, Any] | None) -> str:
    """Render the derived schema as a compact, prompt-visible field contract.

    The full JSON Schema remains authoritative for validation and guided
    decoding. This summary makes prompt-only generation self-contained without
    repeating verbose Guidance prose or dumping implementation-only schema
    extensions into the prompt.
    """
    if not derived_json_schema:
        return ""

    raw_properties: object = derived_json_schema.get("properties")
    if not isinstance(raw_properties, dict) or not raw_properties:
        return ""
    properties = cast("dict[str, object]", raw_properties)

    raw_required: object = derived_json_schema.get("required")
    required: set[str] = (
        {
            value
            for value in cast("list[object]", raw_required)
            if isinstance(value, str)
        }
        if isinstance(raw_required, list)
        else set()
    )
    lines = ["Label schema:"]
    for name, raw_property in properties.items():
        prop = (
            cast("dict[str, object]", raw_property)
            if isinstance(raw_property, dict)
            else {}
        )
        qualifier = "required" if name in required else "optional"
        raw_type = prop.get("type", "value")
        prop_type = raw_type if isinstance(raw_type, str) else "value"

        enum_values = prop.get("enum")
        if isinstance(enum_values, list):
            values = " | ".join(
                json.dumps(value) for value in cast("list[object]", enum_values)
            )
            shape = values or "string"
        elif prop_type == "array" and isinstance(prop.get("items"), dict):
            item_schema = cast("dict[str, object]", prop["items"])
            item_enum_values = item_schema.get("enum")
            if isinstance(item_enum_values, list):
                values = " | ".join(
                    json.dumps(value)
                    for value in cast("list[object]", item_enum_values)
                )
                shape = f"array of ({values})"
            else:
                shape = f"array of {item_schema.get('type', 'values')}"
        elif prop_type in {"integer", "number"} and (
            "minimum" in prop or "maximum" in prop
        ):
            lower = prop.get("minimum", "…")
            upper = prop.get("maximum", "…")
            shape = f"{prop_type} {lower}..{upper}"
        elif prop_type == "string" and ("minLength" in prop or "maxLength" in prop):
            lower = prop.get("minLength", 0)
            upper = prop.get("maxLength", "…")
            shape = f"string length {lower}..{upper}"
        else:
            shape = str(prop_type)

        lines.append(f"- {name} ({qualifier}): {shape}")

    if derived_json_schema.get("additionalProperties") is False:
        lines.append("Use only these fields.")
    return "\n".join(lines)


def render_rationale_regeneration_prompt(
    guidance_description: str = "",
    guidance_rules: str = "",
    guidance_fields: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Render the rationale regeneration prompt.

    Called when the SME edits a label and requests an AI-generated rationale.
    Builds the messages inline (system + user with task / schema summary /
    instructions); image splicing is the caller's responsibility (see
    ``services/rationale_service.py``).

    The writer is not shown either the corrected label or the original
    proposal. Doing so creates a post-hoc rationalization task: the model can
    invent pixels that would make a supplied value sound right instead of
    independently observing the image.
    The active task, rules, and Core schema provide all context needed to write
    a relevant rationale without disclosing a proposed answer.

    The production contract is deliberately domain-agnostic:

    1. **Independent source.** No previous rationale, corrected answer, or
       prior proposal value is rendered.
    2. **Task-aware observation.** Guidance and the Core schema focus the
       response without prescribing a universal visual-vocabulary checklist.
    3. **Task-conditional grounding.** The response describes the subject the
       task asks about and distinguishes a carrier from depicted content only
       when that distinction matters to the task.
    4. **Honesty over compliance.** When pixels are ambiguous, the response
       says so instead of inventing support.

    Counts, identities, positions, and comparisons are neither universally
    required nor universally forbidden. The active task and what is actually
    visible determine whether they are useful evidence.

    Returns ``(messages, prompt_hash)`` matching the signature of
    :func:`render_prompt` for consistency.
    """
    system_msg: dict[str, Any] = {
        "role": "system",
        "content": (
            "You write rationale_note text for vision-labeling records. "
            "Inspect the image independently and describe concrete, "
            "image-specific evidence relevant to the active task. Describe the "
            "subject the task asks about. Distinguish a physical carrier from "
            "content shown on it only when that distinction matters to the task. "
            "Do not infer observations from a proposed or reviewed label. If the "
            "subject or relevant visual evidence is ambiguous, state the uncertainty. "
            "Return only the new rationale text."
        ),
    }

    user_parts: list[str] = []

    if guidance_description.strip():
        user_parts.append(f"Task:\n{guidance_description.strip()}")

    if guidance_rules.strip():
        user_parts.append(f"Rules:\n{guidance_rules.strip()}")

    schema_summary = _render_schema_summary_for_rationale(guidance_fields or [])
    if schema_summary:
        user_parts.append(schema_summary)

    user_parts.append(
        "Write one new rationale_note from your own inspection of the image. "
        "Focus on visible properties of the subject the task asks about. Use "
        "natural vocabulary appropriate to this domain. Quantities, identities, "
        "comparisons, and locations may be described when they are visibly "
        "supported and useful for this task.\n"
        "Do not copy or paraphrase a previous rationale. Do not merely name an "
        "outcome, recite a generic visual-feature checklist, or invent "
        "supporting details. Use only what is visible; do not speculate about "
        "causes or events outside the image. State uncertainty when the "
        "subject or relevant evidence is ambiguous.\n"
        "Target 30–60 words. Never exceed 80 words.\n"
        "Prefer 2–4 short sentences or one compact paragraph.\n"
        "\n"
        "Return only the rationale text, no JSON."
    )

    user_msg: dict[str, Any] = {"role": "user", "content": "\n\n".join(user_parts)}
    messages = [system_msg, user_msg]

    hash_input = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    prompt_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    return messages, prompt_hash


# ── Function 1: Generation param resolution ───────────────────────────────


def resolve_generation_params(
    preset_key: str,
    presets_dict: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Resolve Output Stability preset to sampling parameters.

    Returns ``{"temperature": ..., "top_p": ...}``.
    """
    if preset_key not in presets_dict:
        raise ValueError(
            f"Unknown preset key {preset_key!r}; available: {sorted(presets_dict)}"
        )
    return dict(presets_dict[preset_key])


# ── Function 2: Thinking toggle resolution ────────────────────────────────


def resolve_thinking_fields(
    thinking_on: bool,
    thinking_toggle_mode: str,
    thinking_toggle_support: str = "supported",
) -> dict[str, Any]:
    """Resolve Thinking toggle to request fields (step 2 of request construction).

    Returns a dict with:
      - ``thinking_mode_effective``: ``"on"`` or ``"off"``
      - ``thinking_request_fields``: ``chat_template_kwargs`` dict or None
      - ``thinking_hidden``: True when model has no runtime toggle
        (``mode="none"`` or ``mode="always_on_reasoning"``)

    Mode semantics:
      - ``none`` — no runtime toggle, model does NOT reason (e.g.,
        Mistral, Pixtral, Nemotron Nano 12B VL). ``mode_effective="off"``;
        no reasoning headroom needed.
      - ``always_on_reasoning`` — no runtime toggle, model ALWAYS
        reasons (e.g., Nemotron-3 Nano Omni Reasoning). User's
        ``--thinking off`` has no wire effect; the model reasons
        regardless. ``mode_effective="on"`` so the budget allocates
        ``MODEL_REASONING_HEADROOM_TOKENS``.
      - ``qwen_enable_thinking`` / ``kimi_thinking`` — runtime toggle
        via ``chat_template_kwargs`` only when capability support is
        ``supported``. If support is unknown or unsupported, no override is
        sent and the model's natural reasoning default is treated as on.
    """
    if thinking_toggle_mode == "none":
        return {
            "thinking_mode_effective": "off",
            "thinking_request_fields": None,
            "thinking_hidden": True,
        }

    if thinking_toggle_mode == "always_on_reasoning":
        return {
            "thinking_mode_effective": "on",
            "thinking_request_fields": None,
            "thinking_hidden": True,
        }

    if thinking_toggle_support != "supported":
        return {
            "thinking_mode_effective": "on",
            "thinking_request_fields": None,
            "thinking_hidden": True,
        }

    if thinking_on:
        return {
            "thinking_mode_effective": "on",
            "thinking_request_fields": None,
            "thinking_hidden": False,
        }

    # Thinking OFF — model-specific fields
    if thinking_toggle_mode == "qwen_enable_thinking":
        fields = {"enable_thinking": False}
    elif thinking_toggle_mode == "kimi_thinking":
        fields = {"thinking": False}
    else:
        fields = None

    return {
        "thinking_mode_effective": "off",
        "thinking_request_fields": fields,
        "thinking_hidden": False,
    }


# ── Function 3: Visual budget resolution ──────────────────────────────────


def resolve_visual_budget(
    preset_key: str,
    visual_budget_mode: str,
    visual_budget_support: str,
    presets_dict: dict[str, Any],
) -> dict[str, Any]:
    """Resolve Visual Budget preset to mm_processor_kwargs (step 4 of request construction).

    Returns:
      - ``visual_budget_preset_key``: the resolved key or None
      - ``visual_budget_params_effective``: the mm_processor_kwargs dict or None
    """
    if visual_budget_mode == "none" or visual_budget_support != "supported":
        return {
            "visual_budget_preset_key": None,
            "visual_budget_params_effective": None,
        }

    preset_data = presets_dict.get(preset_key, {})
    mode_params = preset_data.get(visual_budget_mode)

    if mode_params is None:
        return {
            "visual_budget_preset_key": preset_key,
            "visual_budget_params_effective": None,
        }

    return {
        "visual_budget_preset_key": preset_key,
        "visual_budget_params_effective": {"mm_processor_kwargs": mode_params},
    }


# ── Function 4: Seed derivation ───────────────────────────────────────────


def derive_seed(scope_id: str, example_key: str) -> int:
    """Derive a deterministic seed for evaluation/batch.

    ``seed_effective = abs(int32(first_4_bytes(sha256("<scope_id>:<example_key>"))))``
    """
    digest = hashlib.sha256(f"{scope_id}:{example_key}".encode()).digest()
    (value,) = struct.unpack(">i", digest[:4])
    return abs(value)


# ── Function 5: Prompt rendering ──────────────────────────────────────────


def render_prompt(
    guidance_description: str,
    guidance_rules: str,
    icl_rendered_examples: list[dict[str, Any]],
    derived_json_schema: dict[str, Any] | None,
    structured_generation_attempted: bool,
    *,
    attach_icl_images: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """Render the Teacher proposal prompt.

    Template lives at ``prompts/teacher_interactive_proposal.txt``.
    Returns ``(messages, prompt_hash)`` where *messages* is a list of
    message dicts (SYSTEM + USER) and *prompt_hash* is the SHA-256 of the
    text content. Image content parts are inserted by the caller.

    When ``attach_icl_images=True``, the rendered USER text contains
    ``<<<__VLM_ICL_IMG_SLOT_{idx}__>>>`` placeholders after each ICL JSON
    line and a ``<<<__VLM_QUERY_IMG_SLOT__>>>`` placeholder after the
    "Now label the following image." sentence. The caller (typically
    ``invoke_teacher`` step 7) splits the text on these sentinels and
    interleaves prepared image content parts to satisfy the
    inline ICL image injection contract. When ``attach_icl_images=False``
    only the QUERY slot sentinel is emitted (the query image is always
    attached); ICL examples are sent as text-JSON labels only — the
    legacy text-only behavior used by the per-model fallback path.
    """
    # Position-binding cross-modal channel:
    # when ATTACH_ICL_IMAGES is True, prepend a ``"_slot": "E0N"``
    # field to each rendered ICL example dict so the model sees the
    # position commitment as STRUCTURED data inside the JSON label,
    # redundant with the textual "ICL Example N image (E0N):" /
    # "ICL Example N label (E0N):" anchors in the template. Three
    # independent binding channels (header text, label text, structured
    # JSON key) for multi-image teachers like Qwen / Kimi / Mistral
    # that fail implicit positional binding.
    #
    # The key is ``"_slot"`` rather than ``"position"``: Nemotron Nano
    # VL's training data ties the literal token "position" to
    # within-image tile coordinates (the Nemotron tile-grid layout).
    # Using ``"position"`` as a slot identifier silently activates
    # tile-coordinate attention — in probing it collapsed Nemotron from
    # 36% → 6% E02 color recall. The leading-underscore non-domain key
    # avoids that collision.
    # Source: NVIDIA-Nemotron-Nano-12B-v2-VL-BF16 model card +
    # Nano V2 VL technical report (arXiv 2511.03929v2).
    #
    # When ATTACH_ICL_IMAGES is False (single-image-cap fallback for
    # SME-registered models with max_images_per_request=1)
    # the legacy ``E0N: {json}`` text marker already names the
    # position; no JSON-level injection is needed.
    if attach_icl_images:
        icl_for_template: list[dict[str, Any]] = [
            {"_slot": f"E{idx:02d}", **ex}
            for idx, ex in enumerate(icl_rendered_examples, start=1)
        ]
    else:
        icl_for_template = list(icl_rendered_examples)

    schema_properties = (
        derived_json_schema.get("properties")
        if isinstance(derived_json_schema, dict)
        else None
    )
    system_text, user_text = _render_prompt_template(
        _TEACHER_PROPOSAL_TEMPLATE,
        GUIDANCE_DESCRIPTION=guidance_description,
        GUIDANCE_RULES=guidance_rules or "",
        STRUCTURED_GENERATION_ATTEMPTED=bool(
            structured_generation_attempted and derived_json_schema
        ),
        ICL_RENDERED_EXAMPLES=icl_for_template,
        ATTACH_ICL_IMAGES=bool(attach_icl_images),
        RATIONALE_NOTE_ENABLED=bool(
            isinstance(schema_properties, dict)
            and "rationale_note" in schema_properties
        ),
    )
    schema_contract = _render_output_schema_contract(derived_json_schema)
    if schema_contract:
        user_text = f"{schema_contract}\n\n{user_text}"
    else:
        # Keep direct prompt-only renders honest when a caller supplies no schema.
        for dangling_contract in (
            "Return valid JSON matching the label schema described above.",
            "Return one JSON object matching the label schema above.",
        ):
            user_text = user_text.replace(
                dangling_contract,
                "Return one valid JSON object.",
            )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]

    hash_input = json.dumps(messages, sort_keys=True, separators=(",", ":"))
    prompt_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    return messages, prompt_hash


def render_training_conversation_prompt(
    guidance_description: str,
    guidance_rules: str,
    derived_json_schema: dict[str, Any] | None = None,
) -> str:
    """Render the §6 serving prompt as the §9.3 export human turn.

    Students are served (and serving-evaluated, §9.5) with the full D.1
    proposal prompt, so training data must carry that same prompt or the
    fine-tune is systematically off-distribution at deployment (train/serve
    prompt skew: measured 3.4× EM loss on an otherwise healthy student).

    Renders the production serving prompt exactly as a deployed Student
    receives it — zero ICL examples (Student serving evals run
    ``icl_mode="disabled"``), no schema attachment — as a single text
    block, with two deliberate residual deltas from the live request:

    - The SYSTEM message text is folded into the top of the human turn.
      The Cosmos-RL LLaVA annotations contract (§9.3.2) is a strict
      two-turn ``human``/``gpt`` conversation with no system slot; folding
      keeps the instruction tokens in-distribution even though they arrive
      as a separate system message at serve time.
    - The compact label-schema contract remains in the text while the
      serve-time ``response_format`` attachment itself is absent.

    The query-image sentinel becomes the literal ``<image>`` token the
    training framework expands; it appears exactly once, mid-prompt at the
    same position the query image occupies at serve time.
    """
    messages, _ = render_prompt(
        guidance_description,
        guidance_rules,
        icl_rendered_examples=[],
        derived_json_schema=derived_json_schema,
        structured_generation_attempted=False,
        attach_icl_images=False,
    )
    system_text = str(messages[0]["content"]).strip()
    user_text = str(messages[1]["content"]).strip()
    combined = f"{system_text}\n\n{user_text}" if system_text else user_text
    rendered = combined.replace("<<<__VLM_QUERY_IMG_SLOT__>>>", "<image>")
    if rendered.count("<image>") != 1 or "<<<__VLM_" in rendered:
        raise ValueError(
            "training conversation prompt must contain exactly one <image> "
            "token and no unresolved slot sentinels"
        )
    return rendered


# ── Function 6: Request kwargs builder ────────────────────────────────────


def build_chat_request_kwargs(
    *,
    sampling_params: dict[str, float],
    thinking_request_fields: dict[str, Any] | None,
    seed: int | None,
    visual_budget_params: dict[str, Any] | None,
    max_tokens: int,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    """Combine all 6-step resolution results into NIM request kwargs.

    The returned dict is passed as ``**kwargs`` to
    :func:`nim_client.chat_completions`.
    """
    kwargs: dict[str, Any] = {}

    # Step 1: sampling
    kwargs["temperature"] = sampling_params["temperature"]
    kwargs["top_p"] = sampling_params["top_p"]

    # Step 2: thinking toggle
    if thinking_request_fields is not None:
        kwargs["chat_template_kwargs"] = thinking_request_fields

    # Step 3: seed
    if seed is not None:
        kwargs["seed"] = seed

    # Step 4: visual budget
    if visual_budget_params is not None:
        kwargs.update(visual_budget_params)

    # Step 5: output budget + structured generation
    kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    return kwargs


# ── Logging helper (log point 1) ──────────────────────────────────────────


def log_model_invocation(
    project_id: str,
    model_name: str,
    endpoint_url: str,
    sampling_params: dict[str, float],
    visual_budget_params: dict[str, Any] | None,
    finish_reason: str | None,
    latency_ms: int | None,
    usage: dict[str, int] | None,
    *,
    is_retry: bool = False,
    retry_trigger: str | None = None,
    retry_change: str | None = None,
) -> None:
    """Log point 1: model invocation.

    Always at INFO level.
    """
    details: dict[str, Any] = {
        "model_name": model_name,
        "endpoint": endpoint_url,
        "sampling_params": sampling_params,
        "visual_budget_params": visual_budget_params,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
    }
    if usage:
        details["prompt_tokens"] = usage.get("prompt_tokens")
        details["completion_tokens"] = usage.get("completion_tokens")
        details["total_tokens"] = usage.get("total_tokens")
    if is_retry:
        details["is_retry"] = True
        details["retry_trigger"] = retry_trigger
        details["retry_change"] = retry_change

    logger.info(
        "Model invocation: %s → %s (%dms%s)",
        model_name,
        finish_reason or "no_response",
        latency_ms or 0,
        "; retry" if is_retry else "",
        extra={
            "component": "model_invocation",
            "project_id": project_id,
            "details": details,
        },
    )


# ── Function 7: Async Teacher invocation orchestrator ─────────────────────


async def invoke_teacher(
    *,
    # Identity
    project_id: str,
    example_key: str,
    purpose: Literal[
        "interactive_proposal",
        "evaluation",
        "batch_label",
        "rationale_regeneration",
    ],
    inference_invocation_id: str,
    # Guidance
    guidance_description: str,
    guidance_rules: str,
    guidance_fields: list[dict[str, Any]],
    generation_order: list[str],
    derived_json_schema: dict[str, Any],
    # Model
    model_name: str,
    model_config: ModelConfigInput,
    # Endpoint
    endpoint_base_url: str,
    auth_headers: dict[str, str],
    # ICL (pre-queried by caller)
    icl_candidates: list[ICLExample],
    # Settings from config
    generation_preset_key: str,
    thinking_on: bool,
    visual_budget_preset_key: str,
    labeling_presets: dict[str, dict[str, float]],
    visual_budget_presets: dict[str, Any],
    # Token budget config
    base_output_tokens_floor: int = 256,
    json_structural_overhead: int = 48,
    max_output_fraction: float = 0.25,
    rationale_note_estimate: int = 160,
    default_unbounded_string_budget: int = 200,
    model_reasoning_headroom_tokens: int = 16384,
    runtime_prompt_output_max_tokens_override: int | None = None,
    token_safety_margin: float = 0.85,
    # ICL config
    icl_max_examples: int | None = None,
    # Adaptive per-query ICL depth (similarity-gap stopping): after
    # relevance ranking, keep the prefix of neighbors close enough to the
    # query (relative gap to the best AND/OR absolute floor), always >=1,
    # then cap at ``icl_max_examples``. None/None = fixed-K behavior.
    icl_sim_gap: float | None = None,
    icl_abs_threshold: float | None = None,
    # Seed / scope
    scope_id: str | None = None,
    # Deadline
    deadline_s: float = 120.0,
    max_retries: int = 3,
    # Image preparation
    query_storage_ref: str | None = None,
) -> TeacherInvocationResult:
    """Orchestrate a full Teacher invocation (the 6-step request flow).

    This is the only function with I/O (NIM call + image prep).  All
    inputs are explicit; the function is stateless across calls.
    """
    start_time = time.monotonic()
    # Per-stage checkpoints. Each stage records its own elapsed time
    # so we can localise the per-Teacher latency profile (persisted on
    # OperationRecord).
    t_prompt_render_ms: int | None = None
    t_image_prep_ms: int | None = None
    t_nim_call_ms: int | None = None
    stage_t0 = start_time

    schema_for_output, _ = place_rationale_last(derived_json_schema, generation_order)
    structured_gen_attempted = model_config.structured_generation_support == "supported"
    response_format: dict[str, Any] | None = None
    if structured_gen_attempted and schema_for_output:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "label_output",
                # Guided decoding gets a grammar-compilable subset of the
                # derived schema.
                "strict": True,
                "schema": strip_guided_decoding_unsupported_keys(schema_for_output),
            },
        }

    # ── 1. Derive token budget ────────────────────────────────────────
    # Resolve the thinking toggle FIRST so the budget can use the
    # corrected ``thinking_mode_effective``. This matters for models
    # whose ``thinking_toggle_mode == "none"`` (e.g., Cosmos Reason 2,
    # Mistral, Nemotron-3 Omni Reasoning): the user's --thinking off
    # override has no wire effect for them — the model thinks
    # regardless. Without this resolution, the budget would trust the
    # raw ``thinking_on`` flag, allocate ``reasoning_headroom=0``, and
    # the model would burn its entire ``max_tokens`` budget on internal
    # reasoning before producing any visible content
    # (``finish_reason=length``, empty response body, schema_invalid).
    thinking = resolve_thinking_fields(
        thinking_on,
        model_config.thinking_toggle_mode,
        model_config.thinking_toggle_support,
    )
    effective_thinking_on = thinking["thinking_mode_effective"] == "on"

    output_field_mode: Literal["all", "aux_and_core", "core_only"] = "all"
    budget = derive_token_budget(
        guidance_fields,
        output_field_mode,
        model_config.context_window_tokens,
        effective_thinking_on,
        base_output_tokens_floor=base_output_tokens_floor,
        json_structural_overhead=json_structural_overhead,
        max_output_fraction=max_output_fraction,
        rationale_note_estimate=rationale_note_estimate,
        default_unbounded_string_budget=default_unbounded_string_budget,
        model_reasoning_headroom_tokens=model_reasoning_headroom_tokens,
        runtime_prompt_output_max_tokens_override=runtime_prompt_output_max_tokens_override,
        token_safety_margin=token_safety_margin,
    )

    # ── 2. Select and prune ICL ───────────────────────────────────────
    # The selector ranks Edits by CLIP similarity to THIS query image; look
    # the query embedding up from the per-project cache (None if embeddings
    # aren't ready → selector falls back to newest-first). Cheap dict lookup.
    query_clip_embedding = embedding_cache.get(project_id, example_key)
    # Effective ICL depth cap (Spec §6.2): an explicit ``icl_max_examples``
    # override (per-run API field or a non-null ``ICL_MAX_EXAMPLES`` setting,
    # both arriving via the kwarg) wins outright; otherwise the model's own
    # depth default applies. Resolved HERE — the single funnel all three
    # pipelines (proposal / evaluation / batch) pass through — so no caller
    # can drift out of the law. Adaptive-K still trims
    # per query within this cap, and the image budget still prunes after
    # selection; the cap bounds the adaptive mechanism, it does not replace
    # it.
    effective_icl_max_examples = (
        icl_max_examples
        if icl_max_examples is not None
        else model_config.default_icl_max_examples
    )
    selection = select_icl_examples(
        icl_candidates,
        icl_max_examples=effective_icl_max_examples,
        query_clip_embedding=query_clip_embedding,
        icl_sim_gap=icl_sim_gap,
        icl_abs_threshold=icl_abs_threshold,
    )

    # Count the actual fixed prompt text, including the rendered field schema,
    # instead of guessing from raw Guidance plus a magic 200-token constant.
    # The per-example estimator below adds each rendered label + marker cost.
    base_messages, _ = render_prompt(
        guidance_description,
        guidance_rules,
        [],
        schema_for_output,
        structured_gen_attempted,
        attach_icl_images=False,
    )
    prompt_overhead_tokens = count_tokens(
        json.dumps(base_messages, separators=(",", ":"))
    )
    max_icl_tokens = max(0, budget.effective_max_input_tokens - prompt_overhead_tokens)

    retained, token_dropped = prune_icl_by_budget(
        selection.examples,
        max_icl_tokens,
        guidance_fields,
        generation_order,
        "core_only",
    )

    # Image-budget pruning (inline ICL injection): cap retained at
    # ``max_images_per_request - 1`` (subtracting 1 for the query image
    # when present) so every retained ICL example will be image-grounded.
    # Budget exhaustion prunes examples rather than degrading them to
    # text-only labels — every retained example stays image-grounded;
    # the image-budget invariant is enforced at the dispatch boundary.
    image_budget_cap = model_config.max_images_per_request - (
        1 if query_storage_ref else 0
    )
    image_dropped: list[str] = []
    if len(retained) > image_budget_cap:
        retained, image_dropped = prune_icl_by_image_budget(retained, image_budget_cap)
        logger.info(
            "icl_image_budget_pruned retained=%d cap=%d image_dropped=%d "
            "max_per_request=%d",
            len(retained),
            image_budget_cap,
            len(image_dropped),
            model_config.max_images_per_request,
            extra={"component": "prompt_service", "project_id": project_id},
        )

    # Budget pruning operates on relevance order so the weakest tail is always
    # removed first. Only then apply the production bookend presentation:
    # top-1 first, top-2 last, remaining examples in the middle.
    retained = bookend_icl_examples(retained)

    # Combined audit list — token-budget and image-cap drops together feed
    # the ICL-selection log (log point 4).
    dropped = list(token_dropped) + image_dropped
    selection.pruned_count = len(dropped)
    selection.pruned_keys = dropped
    selection.examples = retained
    selection.selected_keys = [ex.example_key for ex in retained]
    selection.total_count = len(retained)

    log_icl_selection(project_id, selection)

    # ── 3. Render ICL fields ──────────────────────────────────────────
    icl_rendered: list[dict[str, Any]] = []
    for ex in retained:
        rendered = render_icl_fields(
            ex.label_json,
            guidance_fields,
            generation_order,
            "core_only",
        )
        icl_rendered.append(rendered)

    # ── 4. Resolve generation controls ────────────────────────────────
    sampling_params = resolve_generation_params(
        generation_preset_key,
        labeling_presets,
    )
    # ``thinking`` was already resolved in step 1 (so the budget could
    # honor models that can't be told to disable reasoning). Reuse it
    # here for sampling-params merging + persistence below.
    vb = resolve_visual_budget(
        visual_budget_preset_key,
        model_config.visual_budget_mode,
        model_config.visual_budget_support,
        visual_budget_presets,
    )
    seed_effective = (
        derive_seed(scope_id, example_key)
        if scope_id and purpose in ("evaluation", "batch_label")
        else None
    )

    # ── 5. Decide ICL image inline mode + render prompt ──────────────
    # Inline ICL image injection: each retained ICL example's
    # image is dispatched alongside its JSON marker so the model can
    # ground the labeled-example pattern visually. Image-budget pruning
    # in section 2 already capped retained at the per-model limit, so
    # the only path to text-only ICL here is the defensive
    # missing-storage_ref branch (rare — Example.storage_ref is NOT
    # NULL by schema and the eligibility query projects it). Invariant:
    # ``len(retained) > 0`` ⇒ every example has its image attached.
    icl_storage_refs: list[str] = [ex.storage_ref for ex in retained if ex.storage_ref]
    all_icl_have_refs = bool(retained) and len(icl_storage_refs) == len(retained)

    if retained and not all_icl_have_refs:
        # Defensive: future bug or alternate eligibility path could pass
        # an ICLExample without ``storage_ref``. Log and degrade to
        # text-only ICL so the loop still runs.
        logger.warning(
            "icl_image_inline=False reason=missing_storage_ref retained=%d with_ref=%d",
            len(retained),
            len(icl_storage_refs),
            extra={"component": "prompt_service", "project_id": project_id},
        )

    attach_icl_images = bool(retained) and all_icl_have_refs

    messages, prompt_hash = render_prompt(
        guidance_description,
        guidance_rules,
        icl_rendered,
        schema_for_output,
        structured_gen_attempted,
        attach_icl_images=attach_icl_images,
    )

    # End of prompt-render stage (sections 1-6: token budget, ICL,
    # generation params, structured-gen, prompt template). Image prep
    # is a separate stage so we can split image-encode cost from
    # template-assembly cost in the per-stage timing columns.
    t_prompt_render_ms = int((time.monotonic() - stage_t0) * 1000)
    stage_t0 = time.monotonic()

    # ── 7. Batch-prepare images and interleave into user message ─────
    # All images for one model invocation go in one call —
    # single prepare_images() with ICL refs first, query last. Each
    # image is read, normalised, and encoded as inline base64. Partial
    # failure (read/normalise error on any image) returns success=False
    # and we abort the invocation cleanly with endpoint_error.
    image_transport_mode: str | None = None
    image_format_transmitted: str | None = None
    icl_images_attached_count = 0

    refs_to_prep: list[str] = []
    if attach_icl_images:
        refs_to_prep.extend(icl_storage_refs)
    if query_storage_ref:
        refs_to_prep.append(query_storage_ref)

    if refs_to_prep:
        prep = await prepare_images(refs_to_prep)

        if not prep.success:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            first_error = next(
                (img.error for img in prep.images if img.error),
                "unknown",
            )
            # Capture the transport mode the failure occurred under so the
            # caller can persist ``image_transport_mode`` even on failure.
            failed_transport_mode = next(
                (img.transport_mode for img in prep.images if img.error),
                None,
            )
            logger.warning(
                "Image transport failed before model dispatch: %s (refs=%d)",
                first_error,
                len(refs_to_prep),
                extra={
                    "component": "prompt_service",
                    "project_id": project_id,
                    "correlation_id": inference_invocation_id,
                },
            )
            # Image prep failed mid-way; attribute the elapsed time to
            # the image-prep stage. Prompt-render had already completed,
            # NIM call never ran.
            t_image_prep_ms = int((time.monotonic() - stage_t0) * 1000)
            return TeacherInvocationResult(
                inference_invocation_id=inference_invocation_id,
                content=None,
                finish_reason=None,
                invocation_status="endpoint_error",
                latency_ms=elapsed_ms,
                usage=None,
                icl_example_keys_used=[ex.example_key for ex in retained],
                prompt_hash=prompt_hash,
                structured_generation_attempted=structured_gen_attempted,
                structured_generation_fallback_used=False,
                generation_preset_key=generation_preset_key,
                sampling_params_effective=sampling_params,
                thinking_mode_effective=thinking["thinking_mode_effective"],
                thinking_request_fields_effective=thinking["thinking_request_fields"],
                max_tokens_effective=budget.max_output_tokens,
                reasoning_headroom_tokens_effective=(
                    budget.reasoning_headroom if effective_thinking_on else None
                ),
                visual_budget_preset_key=vb["visual_budget_preset_key"],
                visual_budget_params_effective=vb["visual_budget_params_effective"],
                seed_effective=seed_effective,
                icl_images_attached_count=0,
                error=f"image_transport_failed: {first_error}",
                t_image_prep_ms=t_image_prep_ms,
                t_prompt_render_ms=t_prompt_render_ms,
                t_nim_call_ms=None,
                image_transport_mode=failed_transport_mode,
            )

        # Successful prep — interleave content parts into user message.
        # When attach_icl_images=True, the first len(icl_storage_refs)
        # entries are ICL images (parallel to retained); the last entry
        # (when query_storage_ref is set) is the query image.
        if attach_icl_images:
            prep_icl_imgs = prep.images[: len(icl_storage_refs)]
            prep_query_img = prep.images[-1] if query_storage_ref else None
        else:
            prep_icl_imgs = []
            prep_query_img = prep.images[0] if query_storage_ref else None

        user_text = messages[-1]["content"]
        parts: list[dict[str, Any]] = []
        remaining = user_text

        for idx, prepared_icl in enumerate(prep_icl_imgs, start=1):
            sentinel = f"<<<__VLM_ICL_IMG_SLOT_{idx}__>>>"
            head, sep, remaining = remaining.partition(sentinel)
            if not sep:
                raise ValueError(
                    f"Rendered prompt missing ICL image slot sentinel "
                    f"{sentinel!r}; template/render mismatch."
                )
            if head.strip():
                parts.append({"type": "text", "text": head.rstrip()})
            parts.append(prepared_icl.content_part)
            icl_images_attached_count += 1

        if prep_query_img is not None:
            query_sentinel = "<<<__VLM_QUERY_IMG_SLOT__>>>"
            pre_query, sep, post_query = remaining.partition(query_sentinel)
            if not sep:
                raise ValueError("Rendered prompt missing query image slot sentinel.")
            if pre_query.strip():
                parts.append({"type": "text", "text": pre_query.rstrip()})
            parts.append(prep_query_img.content_part)
            if post_query.strip():
                parts.append({"type": "text", "text": post_query.lstrip()})
            image_transport_mode = prep_query_img.transport_mode
            image_format_transmitted = prep_query_img.format_transmitted
        else:
            # No query image: strip the unconditionally-emitted QUERY
            # sentinel from the rendered text so it does not leak into
            # the model request.
            remaining = remaining.replace("<<<__VLM_QUERY_IMG_SLOT__>>>", "")
            if remaining.strip():
                parts.append({"type": "text", "text": remaining.strip()})

        # Defensive: catch any sentinel that survived into a text part
        # because user-provided content (Guidance description, rules,
        # or an ICL rationale string) contained a literal sentinel.
        for part in parts:
            if part.get("type") == "text" and "<<<__VLM_" in part.get("text", ""):
                raise ValueError(
                    "Sentinel collision detected: user-provided content "
                    "(Guidance description, rules, or ICL rationale) "
                    "contains a literal '<<<__VLM_*_SLOT_*__>>>' marker."
                )

        messages[-1]["content"] = parts

    # Invariant (inline ICL image injection): every retained ICL
    # example MUST have its image attached. Image-budget pruning at
    # section 2 enforces ``len(retained) <= max_images_per_request - 1``;
    # the defensive missing-storage_ref branch flips
    # ``attach_icl_images`` to False (and ``icl_images_attached_count``
    # stays 0) when the precondition fails.
    if attach_icl_images:
        assert icl_images_attached_count == len(retained), (
            f"ICL inline-injection invariant violated: "
            f"{icl_images_attached_count} images attached vs "
            f"{len(retained)} retained examples (spec §6.3)."
        )

    # End of image-prep stage (only if any images were prepared; for
    # text-only invocations this stage is None).
    if refs_to_prep:
        t_image_prep_ms = int((time.monotonic() - stage_t0) * 1000)
    stage_t0 = time.monotonic()

    # ── 8. Build request kwargs ───────────────────────────────────────
    kwargs = build_chat_request_kwargs(
        sampling_params=sampling_params,
        thinking_request_fields=thinking["thinking_request_fields"],
        seed=seed_effective,
        visual_budget_params=vb["visual_budget_params_effective"],
        max_tokens=budget.max_output_tokens,
        response_format=response_format,
    )

    # ── 9. Call NIM ───────────────────────────────────────────────────
    nim_call_t0 = time.monotonic()
    nim_result = await nim_client.chat_completions(
        endpoint_base_url,
        auth_headers,
        model_name,
        messages,
        deadline_s,
        max_retries=max_retries,
        **kwargs,
    )

    elapsed_ms = int((time.monotonic() - start_time) * 1000)
    t_nim_call_ms = int((time.monotonic() - nim_call_t0) * 1000)

    # ── 10. Handle truncation retry (step 6 of request flow) ─────────
    # Applies to all label-generating invocations. Evaluation +
    # batch_label routinely encounter ``finish_reason="length"`` under
    # Thinking-ON; without retry, those examples count as
    # ``truncation_attributed_schema_invalid`` and bias evaluation accuracy
    # downward (which feeds the Scale-Up Readiness Gate).
    # ``rationale_regeneration`` does not reach this code path because it
    # bypasses invoke_teacher.
    if nim_result.finish_reason == "length" and purpose in (
        "interactive_proposal",
        "retry",
        "evaluation",
        "batch_label",
        "rationale_regeneration",
    ):
        # Re-derive from the formula floor (base + reasoning_headroom) and
        # apply the 1.5× inflation, capped at ctx × MAX_OUTPUT_FRACTION. The
        # explicit recomputation keeps intent visible and makes the retry
        # robust to any future change in how ``budget.max_output_tokens`` is
        # initially clamped.
        formula_floor = budget.base_output_tokens + budget.reasoning_headroom
        new_max = min(
            math.floor(formula_floor * 1.5),
            math.floor(model_config.context_window_tokens * max_output_fraction),
        )
        if new_max > budget.max_output_tokens:
            log_model_invocation(
                project_id,
                model_name,
                endpoint_base_url,
                sampling_params,
                vb["visual_budget_params_effective"],
                nim_result.finish_reason,
                elapsed_ms,
                nim_result.usage,
                is_retry=True,
                retry_trigger="finish_reason=length",
                retry_change=f"max_tokens {budget.max_output_tokens} → {new_max}",
            )
            kwargs["max_tokens"] = new_max
            retry_t0 = time.monotonic()
            nim_result = await nim_client.chat_completions(
                endpoint_base_url,
                auth_headers,
                model_name,
                messages,
                deadline_s,
                max_retries=max_retries,
                **kwargs,
            )
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            # Truncation retry: cumulative NIM time across both calls.
            t_nim_call_ms += int((time.monotonic() - retry_t0) * 1000)

    # ── 11. Classify result ───────────────────────────────────────────
    if nim_result.success:
        invocation_status = "success"
    elif nim_result.error and "timed out" in nim_result.error.lower():
        invocation_status = "timeout"
    else:
        invocation_status = "endpoint_error"

    # ── 12. Log point 1 ──────────────────────────────────────────────
    log_model_invocation(
        project_id,
        model_name,
        endpoint_base_url,
        sampling_params,
        vb["visual_budget_params_effective"],
        nim_result.finish_reason,
        elapsed_ms,
        nim_result.usage,
    )

    # ── 13. Return result ─────────────────────────────────────────────
    return TeacherInvocationResult(
        inference_invocation_id=inference_invocation_id,
        content=nim_result.content,
        finish_reason=nim_result.finish_reason,
        invocation_status=invocation_status,
        latency_ms=elapsed_ms,
        usage=nim_result.usage,
        icl_example_keys_used=[ex.example_key for ex in retained],
        prompt_hash=prompt_hash,
        structured_generation_attempted=structured_gen_attempted,
        structured_generation_fallback_used=False,
        # ``*_attempted`` flags reflect whether the dispatched request
        # actually carried the corresponding kwargs. Caller services use
        # these together with the rejection-detector heuristics to decide
        # whether a 4xx ``endpoint_error`` is attributable to thinking /
        # visual-budget kwargs (vs an unrelated transport failure).
        thinking_toggle_attempted=thinking["thinking_request_fields"] is not None,
        visual_budget_attempted=vb["visual_budget_params_effective"] is not None,
        generation_preset_key=generation_preset_key,
        sampling_params_effective=sampling_params,
        thinking_mode_effective=thinking["thinking_mode_effective"],
        thinking_request_fields_effective=thinking["thinking_request_fields"],
        max_tokens_effective=kwargs.get("max_tokens", budget.max_output_tokens),
        reasoning_headroom_tokens_effective=(
            budget.reasoning_headroom if effective_thinking_on else None
        ),
        visual_budget_preset_key=vb["visual_budget_preset_key"],
        visual_budget_params_effective=vb["visual_budget_params_effective"],
        seed_effective=seed_effective,
        image_transport_mode=image_transport_mode,
        image_format_transmitted=image_format_transmitted,
        icl_images_attached_count=icl_images_attached_count,
        error=nim_result.error if not nim_result.success else None,
        t_image_prep_ms=t_image_prep_ms,
        t_prompt_render_ms=t_prompt_render_ms,
        t_nim_call_ms=t_nim_call_ms,
    )
