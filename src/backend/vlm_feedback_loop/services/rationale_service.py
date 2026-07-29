# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rationale regeneration service.

Calls the Teacher with the image and active task context to produce a new,
independently observed rationale. Reviewed label values are deliberately not
shown to the writer. Creates an OperationRecord with
``purpose="rationale_regeneration"``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services import nim_client
from vlm_feedback_loop.services.image_transport import prepare_images
from vlm_feedback_loop.services.invocation_outcome import (
    apply_transport_invocation_outcome,
    classify_transport_status,
    write_invocation_artifact,
)
from vlm_feedback_loop.services.nim_client import build_endpoint_auth_headers
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.prompt_service import (
    build_chat_request_kwargs,
    render_rationale_regeneration_prompt,
    resolve_generation_params,
    resolve_thinking_fields,
    resolve_visual_budget,
)
from vlm_feedback_loop.services.runtime_secrets import get_effective_secret
from vlm_feedback_loop.services.schema_core import rationale_note_enabled

logger = logging.getLogger("vlm_feedback_loop.rationale_service")

# Minimum cleaned-rationale length below which we treat the regeneration as
# failed rather than store a degenerate note (scaffolding-only leaks from
# omni/Qwen reasoners collapse to a few characters after sanitization).
_MIN_RATIONALE_CHARS = 10


# ── Teacher output sanitizer (omni/Qwen scaffolding leaks) ──────────────────
# Cosmos-Reason / Qwen-omni style reasoners sometimes emit tool-call or
# thinking scaffolding, or a leading JSON/coordinate fragment, around the
# free-form rationale. These patterns strip that scaffolding before
# ``_clean_rationale_output`` does its preamble/quote trim. Conservative: a
# no-op on already-clean prose.
_PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s|\s+is|\s+are)?\b[^:\n]*|rationale|observation|evidence|answer|note)"
    r"\s*:[ \t]+",
    re.IGNORECASE,
)
_TOOLCALL_RE = re.compile(r"<TOOLCALL>.*?(?:</TOOLCALL>|$)", re.IGNORECASE | re.DOTALL)
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
# An unclosed thinking block — strip from the opening tag to the end.
_THINK_OPEN_RE = re.compile(r"<think(?:ing)?>.*$", re.IGNORECASE | re.DOTALL)
# A dangling closing thinking tag with no opener (model emitted reasoning
# then "</thinking>" before the visible answer): drop everything up to and
# including it.
_THINK_CLOSE_LEAD_RE = re.compile(r"^.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
# A leading JSON / coordinate fragment the omni grounding head leaks, e.g.
# '{"point_2d": [1,2]}' or '[{"bbox": ...}]'. Strip a single leading balanced
# {...} object or [...] array (greedy to the matching final bracket) so the
# prose after it survives.
_LEADING_JSON_OBJ_RE = re.compile(r"^\s*\{.*\}\s*", re.DOTALL)
_LEADING_JSON_ARR_RE = re.compile(r"^\s*\[.*\]\s*", re.DOTALL)


def _clean_rationale_output(raw: str) -> str:
    """Remove simple chat wrapping while leaving clean prose unchanged."""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        inner = text[3:]
        if "\n" in inner:
            first, rest = inner.split("\n", 1)
            if first.strip().isalpha():
                inner = rest
        inner = inner.strip().rstrip("`").strip()
        if inner:
            text = inner
    text = _PREAMBLE_RE.sub("", text, count=1).strip()
    for quote in ('"', "'", "`"):
        if len(text) >= 2 and text[0] == quote and text[-1] == quote:
            text = text[1:-1].strip()
            break
    return text


def _sanitize_teacher_rationale(raw: str) -> str:
    """Strip omni/Qwen scaffolding leaks, then reuse ``_clean_rationale_output``.

    Removes ``<TOOLCALL>…``, ``<thinking>…</thinking>`` (open or closed) and a
    leading JSON/coordinate fragment (``{"point_2d…}`` / ``[{"bbox…}]``), then
    performs conservative preamble/code-fence/quote trimming. Returns ``""``
    when nothing usable remains.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    # 1. Drop tool-call scaffolding (closed or dangling-open).
    text = _TOOLCALL_RE.sub("", text).strip()
    # 2. Drop thinking blocks: fully-closed first, then a leading dangling
    #    "</thinking>" (reasoning-before-answer), then an unclosed "<thinking>".
    text = _THINK_RE.sub("", text).strip()
    if "</think" in text.lower():
        text = _THINK_CLOSE_LEAD_RE.sub("", text, count=1).strip()
    text = _THINK_OPEN_RE.sub("", text).strip()
    # 3. Drop a single leading JSON object/array coordinate fragment.
    if text.startswith("{"):
        text = _LEADING_JSON_OBJ_RE.sub("", text, count=1).strip()
    elif text.startswith("["):
        text = _LEADING_JSON_ARR_RE.sub("", text, count=1).strip()
    # 4. Preamble / code-fence / quote trim.
    return _clean_rationale_output(text)


async def regenerate_rationale(
    project_id: str,
    example_key: str,
    teacher_model_config_id: str | None,
    workspace_root: str,
    settings: Settings,
) -> dict[str, Any] | str:
    """Regenerate an independently observed rationale with the Teacher."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        return f"Project not found: {project_id}"

    # ── Load project, example, model config, endpoint ────────────────────
    with Session(engine) as session:
        project = (
            session.query(Project)
            .filter_by(
                project_id=project_id,
            )
            .first()
        )
        if project is None:
            return f"Project not found: {project_id}"

        example = (
            session.query(Example)
            .filter_by(
                project_id=project_id,
                example_key=example_key,
            )
            .first()
        )
        if example is None:
            return f"Example not found: {example_key}"

        effective_mc_id = teacher_model_config_id or project.teacher_model_config_id
        if not effective_mc_id:
            return "No teacher model configured"

        model_config = (
            session.query(ModelConfig)
            .filter_by(
                project_id=project_id,
                model_config_id=effective_mc_id,
            )
            .first()
        )
        if model_config is None:
            return f"Model config not found: {effective_mc_id}"

        endpoint = (
            session.query(NimEndpoint)
            .filter_by(
                project_id=project_id,
                endpoint_id=model_config.endpoint_id,
            )
            .first()
        )
        if endpoint is None:
            return f"NIM endpoint not found for model config: {effective_mc_id}"

        # Guidance context for task-aware rationale. The endpoint is available
        # only when the active Guidance explicitly opts into rationale_note.
        guidance_id = project.active_guidance_id
        guidance_description = ""
        guidance_rules = ""
        guidance_fields: list[dict[str, Any]] = []
        if guidance_id:
            guidance = (
                session.query(Guidance)
                .filter_by(
                    project_id=project_id,
                    guidance_id=guidance_id,
                )
                .first()
            )
            if guidance is not None:
                guidance_description = guidance.description or ""
                guidance_rules = guidance.rules or ""
                guidance_fields = (guidance.schema or {}).get("fields", []) or []
        if not rationale_note_enabled(guidance_fields):
            return "conflict: rationale_note is disabled in the active Guidance"

        # Snapshot before closing session
        storage_ref = example.storage_ref
        model_name = model_config.model_name
        endpoint_base_url = endpoint.base_url
        endpoint_auth_mode = endpoint.auth_mode
        endpoint_id = endpoint.endpoint_id
        model_config_id = model_config.model_config_id
        preset_key = project.labeling_generation_preset_key or "precise"
        project_dir = project.project_dir
        # Snapshot the capability flags so the
        # bypass path can call ``resolve_thinking_fields`` /
        # ``resolve_visual_budget`` outside the session. Mirror what
        # ``proposal_service`` reads off ``ModelConfigInput``.
        thinking_toggle_mode = model_config.thinking_toggle_mode or "none"
        thinking_toggle_support = model_config.thinking_toggle_support or "unknown"
        visual_budget_mode = model_config.visual_budget_mode or "none"
        visual_budget_support = model_config.visual_budget_support or "unknown"
        thinking_default_on = bool(project.thinking_default_on)
        vb_preset_key = project.visual_budget_preset_key or "high_detail"
        context_window_tokens = model_config.context_window_tokens

    # ── Generate invocation ID and persist pending OperationRecord ────────
    inference_invocation_id = generate_uuid4()

    with Session(engine) as session:
        pending = OperationRecord(
            inference_invocation_id=inference_invocation_id,
            project_id=project_id,
            purpose="rationale_regeneration",
            example_key=example_key,
            guidance_id=guidance_id,
            model_config_id=model_config_id,
            endpoint_id=endpoint_id,
            model_name=model_name,
            invocation_status="pending",
        )
        session.add(pending)
        session.commit()

    # ── Build the regeneration prompt ────────────────────────────────────
    # Instrument per-stage timings here: this service bypasses
    # ``invoke_teacher`` and would otherwise leave t_*_ms NULL on its
    # OperationRecord.
    start_time = time.monotonic()
    stage_t0 = start_time
    messages, prompt_hash = render_rationale_regeneration_prompt(
        guidance_description=guidance_description,
        guidance_rules=guidance_rules,
        guidance_fields=guidance_fields,
    )
    t_prompt_render_ms = int((time.monotonic() - stage_t0) * 1000)

    # ── Prepare image ────────────────────────────────────────────────────
    deadline_s = float(settings.HTTP_DEADLINE_INTERACTIVE_S)

    image_transport_mode: str | None = None
    image_format_transmitted: str | None = None
    t_image_prep_ms: int | None = None

    if storage_ref:
        stage_t0 = time.monotonic()
        prep = await prepare_images([storage_ref])
        t_image_prep_ms = int((time.monotonic() - stage_t0) * 1000)
        if prep.success and prep.images:
            img = prep.images[0]
            image_transport_mode = img.transport_mode
            image_format_transmitted = img.format_transmitted

            user_msg = messages[-1]
            user_text = user_msg["content"]
            user_msg["content"] = [
                img.content_part,
                {"type": "text", "text": user_text},
            ]
        elif prep.images:
            # Capture the transport mode for OperationRecord persistence.
            # The model call still proceeds without the image — rationale
            # regeneration text-only is degraded but non-fatal.
            failed = next((p for p in prep.images if p.error), None)
            if failed:
                image_transport_mode = failed.transport_mode

    # ── Build request kwargs ─────────────────────────────────────────────
    # Run the full request-construction sequence even on the bypass path:
    # sampling preset →
    # thinking-toggle fields → visual-budget kwargs → reasoning-headroom-
    # aware max_tokens. Rationale never uses structured generation (the
    # output is free-form text, not a JSON object) and never uses a per-
    # invocation seed (Step 3 omits ``seed`` for ``rationale_regeneration``).
    sampling_params = resolve_generation_params(
        preset_key,
        settings.LABELING_PRESETS,
    )
    thinking = resolve_thinking_fields(
        thinking_default_on,
        thinking_toggle_mode,
        thinking_toggle_support,
    )
    vb = resolve_visual_budget(
        vb_preset_key,
        visual_budget_mode,
        visual_budget_support,
        settings.VISUAL_BUDGET_PRESETS,
    )

    # Reasoning-headroom-aware budget for rationale regeneration.
    # The Teacher generates a 30-80 word rationale (~160 tokens worst
    # case per RATIONALE_NOTE_ESTIMATE_TOKENS), but a Qwen/Kimi reasoner
    # with Thinking ON spends most of its output on a ``<think>`` block
    # before emitting visible content. Without headroom the 256-token
    # ceiling truncates the rationale entirely. Mirror the prompt-budget
    # formula:
    # ``base_output = max(BASE_OUTPUT_TOKENS_FLOOR, 2 * rationale_estimate);
    # if Thinking ON add MODEL_REASONING_HEADROOM_TOKENS; cap at
    # ctx*MAX_OUTPUT_FRACTION``.
    base_output_tokens = max(
        settings.BASE_OUTPUT_TOKENS_FLOOR,
        2 * settings.RATIONALE_NOTE_ESTIMATE_TOKENS,
    )
    reasoning_headroom = (
        settings.MODEL_REASONING_HEADROOM_TOKENS
        if thinking["thinking_mode_effective"] == "on"
        else 0
    )
    max_tokens_effective = min(
        base_output_tokens + reasoning_headroom,
        math.floor(context_window_tokens * settings.MAX_OUTPUT_FRACTION),
    )

    extra_kwargs = build_chat_request_kwargs(
        sampling_params=sampling_params,
        thinking_request_fields=thinking["thinking_request_fields"],
        seed=None,  # no seed for rationale_regeneration
        visual_budget_params=vb["visual_budget_params_effective"],
        max_tokens=max_tokens_effective,
        response_format=None,  # rationale is free-form text
    )

    # Honor the endpoint's auth_mode via the shared builder. Resolve the key
    # through get_effective_secret so a session-only UI-pasted key reaches
    # the Teacher call, matching the interactive proposal path.
    auth_headers = build_endpoint_auth_headers(
        endpoint_auth_mode,
        get_effective_secret("NVIDIA_API_KEY", settings),
    )

    # ── Call NIM ─────────────────────────────────────────────────────────
    stage_t0 = time.monotonic()
    nim_result = await nim_client.chat_completions(
        endpoint_base_url,
        auth_headers,
        model_name,
        messages,
        deadline_s,
        max_retries=settings.HTTP_MAX_RETRIES,
        **extra_kwargs,
    )
    t_nim_call_ms = int((time.monotonic() - stage_t0) * 1000)

    elapsed_ms = int((time.monotonic() - start_time) * 1000)

    # ── Classify result ──────────────────────────────────────────────────
    invocation_status = classify_transport_status(nim_result)

    # Sanitize the Teacher/hosted-author rationale BEFORE storing it. The
    # Omni/Qwen reasoners leak ``<TOOLCALL>…``, ``<thinking>…`` and
    # leading JSON/coordinate fragments around the prose; strip them here.
    rationale_note = ""
    if nim_result.content:
        rationale_note = _sanitize_teacher_rationale(nim_result.content)

    # If cleaning leaves an empty/degenerate note (scaffolding-only output),
    # treat the regeneration as failed so the harness retries / falls back,
    # rather than persisting a useless note that would poison downstream ICL.
    # Only downgrade an otherwise-successful call — a real timeout /
    # endpoint_error keeps its own classification.
    if invocation_status == "success" and len(rationale_note) < _MIN_RATIONALE_CHARS:
        logger.warning(
            "Teacher rationale for %s/%s collapsed to %d chars after "
            "sanitization (scaffolding-only output); marking failed",
            project_id,
            example_key,
            len(rationale_note),
        )
        invocation_status = "endpoint_error"
        rationale_note = ""

    # ── Write artifact ───────────────────────────────────────────────────
    raw_ref = write_invocation_artifact(
        Path(project_dir)
        / "artifacts"
        / f"{inference_invocation_id}_rationale_raw.txt",
        nim_result.content or "",
    )

    # ── Update OperationRecord ───────────────────────────────────────────
    with Session(engine) as session:
        record = (
            session.query(OperationRecord)
            .filter_by(
                inference_invocation_id=inference_invocation_id,
            )
            .first()
        )
        if record is not None:
            apply_transport_invocation_outcome(
                record,
                invocation_status=invocation_status,
                nim_result=nim_result,
                elapsed_ms=elapsed_ms,
                raw_ref=raw_ref,
                generation_preset_key=preset_key,
                sampling_params_effective=sampling_params,
                max_tokens_effective=max_tokens_effective,
                thinking=thinking,
                visual_budget=vb,
                prompt_hash=prompt_hash,
                t_image_prep_ms=t_image_prep_ms,
                t_nim_call_ms=t_nim_call_ms,
                t_prompt_render_ms=t_prompt_render_ms,
                image_transport_mode=image_transport_mode,
                image_format_transmitted=image_format_transmitted,
                # Persisted only when Thinking actually added headroom.
                reasoning_headroom_tokens_effective=(
                    reasoning_headroom if reasoning_headroom > 0 else None
                ),
            )
            session.commit()

    return {
        "inference_invocation_id": inference_invocation_id,
        "rationale_note": rationale_note,
        "invocation_status": invocation_status,
    }
