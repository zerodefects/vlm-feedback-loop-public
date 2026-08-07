# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical uncapped Student serving-request construction.

Quality evaluation deliberately keeps its derived output limit and retry
policy.  Production verification and serving benchmarks use this module so
their prompt, image placement, Inference Contract, and generation controls
cannot drift while both omit an output-token cap.
"""

from __future__ import annotations

from typing import Any, cast

from vlm_feedback_loop.services.prompt_service import (
    filter_output_schema_for_field_mode,
    render_prompt,
)
from vlm_feedback_loop.services.schema_core import (
    place_rationale_last,
    strip_guided_decoding_unsupported_keys,
)


def build_uncapped_student_request(
    *,
    served_model: str,
    guidance_description: str,
    guidance_rules: str,
    guidance_fields: list[dict[str, Any]],
    generation_order: list[str],
    derived_json_schema: dict[str, Any],
    inference_contract: dict[str, Any],
    structured_generation_attempted: bool,
    sampling_params: dict[str, Any] | None,
    thinking_request_fields: dict[str, Any] | None,
    visual_budget_params: dict[str, Any] | None,
    image_content_part: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build one production Student request without an output-token cap."""
    raw_mode = inference_contract.get("output_field_mode", "all")
    output_field_mode = (
        raw_mode if raw_mode in {"all", "aux_and_core", "core_only"} else "all"
    )
    projected_schema, projected_order = filter_output_schema_for_field_mode(
        derived_json_schema,
        generation_order,
        guidance_fields,
        cast("Any", output_field_mode),
    )
    schema_for_output, _ = place_rationale_last(projected_schema, projected_order)

    messages, prompt_hash = render_prompt(
        guidance_description,
        guidance_rules,
        icl_rendered_examples=[],
        derived_json_schema=schema_for_output,
        structured_generation_attempted=structured_generation_attempted,
        attach_icl_images=False,
    )

    user_text = str(messages[-1]["content"])
    pre_image, sentinel, post_image = user_text.partition(
        "<<<__VLM_QUERY_IMG_SLOT__>>>"
    )
    if not sentinel:
        raise ValueError("Rendered prompt missing query image slot sentinel")
    user_parts: list[dict[str, Any]] = []
    if pre_image.strip():
        user_parts.append({"type": "text", "text": pre_image.rstrip()})
    user_parts.append(image_content_part)
    if post_image.strip():
        user_parts.append({"type": "text", "text": post_image.lstrip()})
    messages[-1]["content"] = user_parts

    request: dict[str, Any] = {"model": served_model, "messages": messages}
    if sampling_params:
        request.update(sampling_params)
    if thinking_request_fields:
        request["chat_template_kwargs"] = thinking_request_fields
    if visual_budget_params:
        request.update(visual_budget_params)
    if structured_generation_attempted and schema_for_output:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "label_output",
                "strict": True,
                "schema": strip_guided_decoding_unsupported_keys(schema_for_output),
            },
        }

    # Deliberate serving contract: neither OpenAI spelling is allowed here.
    request.pop("max_tokens", None)
    request.pop("max_completion_tokens", None)
    return request, prompt_hash


__all__ = ["build_uncapped_student_request"]
