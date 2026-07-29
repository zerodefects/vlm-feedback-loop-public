# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OperationRecord — single per-invocation record across all purposes."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Index, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, uuid_pk


class OperationRecord(ProjectBase):
    __tablename__ = "operation_records"
    # Indexes on the hot filter columns — this is the largest table (one row
    # per invocation).
    __table_args__ = (
        Index("ix_operation_records_evaluation_run_id", "evaluation_run_id"),
        Index("ix_operation_records_batch_label_run_id", "batch_label_run_id"),
        Index("ix_operation_records_example_key", "example_key"),
    )

    inference_invocation_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    purpose: Mapped[str] = mapped_column(
        String, nullable=False
    )  # interactive_proposal | evaluation | batch_label | rationale_regeneration
    example_key: Mapped[str | None] = mapped_column(String, nullable=True)
    guidance_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_config_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    endpoint_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    icl_example_keys_used: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    invocation_status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )

    # Provider/latency metadata
    latency_ms_end_to_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Per-stage timings ────────────────────────────────────────────
    # Stage breakdown of ``latency_ms_end_to_end``. Each stage is a
    # disjoint slice of the per-invocation pipeline:
    #   * ``t_image_prep_ms``     — read + normalise + base64-encode all
    #                                images (query + ICL).
    #   * ``t_prompt_render_ms``  — token-budget derivation, ICL
    #                                selection + diversity-based pruning,
    #                                Jinja2 prompt assembly, JSON-Schema
    #                                derivation.
    #   * ``t_nim_call_ms``       — outbound HTTP POST to the NIM
    #                                ``/v1/chat/completions`` endpoint
    #                                including any retry/backoff inside
    #                                the resilient client.
    #   * ``t_validation_ms``     — schema validation + normalization +
    #                                Exact Match scoring.
    # The four stages SHOULD sum to within ±5% of
    # ``latency_ms_end_to_end``; the unaccounted slack is dispatch
    # bookkeeping. Used to localise the per-Teacher latency profile
    # (e.g. hosted Qwen measured ~38s/call vs. ~3s for a bare control
    # request; the per-stage split tells us which side of the wire to
    # optimise).
    t_image_prep_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    t_prompt_render_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    t_nim_call_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    t_validation_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Generation Controls ───────────────────────────────────
    generation_preset_key: Mapped[str | None] = mapped_column(String, nullable=True)
    sampling_params_effective: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    thinking_mode_effective: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # on | off
    thinking_request_fields_effective: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    max_tokens_effective: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_headroom_tokens_effective: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )

    # ── Prompt + seed reproducibility ─────────────────────────
    # SHA-256 hex of the rendered prompt (system + user messages serialized).
    # Every model invocation persists enough lineage to reconstruct the
    # exact prompt context: model_config_id, guidance_id, decoding
    # params, icl_example_keys_used[], and a prompt_hash or rendered
    # prompt reference. Populated for every invocation that
    # reaches a terminal status (success / schema_invalid / timeout /
    # endpoint_error).
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Deterministic per-example seed for evaluation and batch_label
    # invocations. Null for interactive_proposal, retry,
    # rationale_regeneration. Captured here in addition to being passed
    # to NIM so the run can be reproduced from persisted state alone.
    seed_effective: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Visual Budget ─────────────────────────────────────────
    visual_budget_preset_key: Mapped[str | None] = mapped_column(String, nullable=True)
    visual_budget_params_effective: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )

    # ── Image transport ───────────────────────────────────────
    image_transport_mode: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # base64_inline
    image_format_transmitted: Mapped[str | None] = mapped_column(String, nullable=True)

    # ── Inline ICL image injection ───────────────────────────
    # Number of ICL example images actually attached to the dispatched
    # request (inline ICL image injection). 0 means the model
    # received text-only ICL — either because no Edits exist (cold start),
    # because the active model's `max_images_per_request` budget couldn't
    # fit `1 + len(retained_icl)` images, or because all ICL images failed
    # transport prep. Audit signal for diagnosing weak ICL behavior.
    icl_images_attached_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # ── Label tier ───────────────────────────────────────────────────
    label_tier: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # proposal | auto_labeled

    # ── Artifact references ──────────────────────────────────────────
    raw_model_response_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    normalized_json_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    validation_report_ref: Mapped[str | None] = mapped_column(String, nullable=True)

    # ── Validation report ────────────────────────────────────────────
    schema_valid_core: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    validation_errors_core: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True
    )
    validation_errors_aux: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    structured_generation_fallback_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    structured_generation_mode_effective: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # auto | prompt_only
    structured_generation_attempted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # Thinking + Visual Budget runtime rejection plumbing.
    # ``*_attempted`` is True whenever the request included the
    # corresponding kwargs; ``*_fallback_used`` is True only on the
    # interactive retry path that dropped them after a runtime rejection.
    # Mirrors the structured-generation pattern.
    thinking_toggle_attempted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    thinking_fallback_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    visual_budget_attempted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    visual_budget_fallback_used: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ── Provider token usage, completion, and truncation ─────────────
    finish_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    truncation_attributed_schema_invalid: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ── Error ────────────────────────────────────────────────────────
    provider_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)

    # ── Linkage ──────────────────────────────────────────────────────
    retry_of_inference_invocation_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    evaluation_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    batch_label_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    ignored_due_to_run_cancellation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ── Purpose-dependent outcome ────────────────────────────────────
    exact_match_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
