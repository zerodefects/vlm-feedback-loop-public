# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RunRecord — single-table for evaluation and batch-label runs."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk

# Statuses meaning a run has (or should have) a live background task. The
# one-active-run invariant (batch_label_service), the model-config re-probe
# block, the schema-evolution cancel sweep, and startup recovery (main.py)
# all key on this set. "paused" is non-terminal but has no task and is
# deliberately excluded; per-service TERMINAL_STATUSES stay local because
# the terminal sets differ (evaluation adds "incomplete").
ACTIVE_RUN_STATUSES: frozenset[str] = frozenset({"queued", "running", "canceling"})


class RunRecord(ProjectBase):
    __tablename__ = "run_records"

    run_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    run_type: Mapped[str] = mapped_column(
        String, nullable=False
    )  # evaluation_run | batch_label_run
    status: Mapped[str] = mapped_column(String, nullable=False)
    status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    cancel_requested_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    recovered_from_restart: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[str] = created_at_col()
    started_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)

    # ── Evaluation run fields ──────────────────────────────
    pool_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    guidance_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    model_config_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    icl_mode: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # enabled | disabled
    evaluation_source: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # tao | nim
    generation_preset_key: Mapped[str | None] = mapped_column(String, nullable=True)
    thinking_mode_effective: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # on | off
    visual_budget_preset_key: Mapped[str | None] = mapped_column(String, nullable=True)
    structured_generation_mode_effective: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # auto | prompt_only
    inference_contract: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    icl_eligible_count_at_start: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    icl_eligible_count_at_completion: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    tao_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    tao_native_metrics: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    rescored_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Returning vs New
    previous_pool_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    returning_example_keys: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True
    )
    new_example_keys: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    previous_overall_exact_match: Mapped[float | None] = mapped_column(nullable=True)
    coverage_gaps: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )

    # ── Per-run evaluation provenance ────────
    student_model_config_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    nim_model_profile_requested: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    nim_model_profile_selected: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    nim_profile_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    quantization_method: Mapped[str | None] = mapped_column(String, nullable=True)
    gpu_type: Mapped[str | None] = mapped_column(String, nullable=True)
    gpu_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dataset_manifest_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # ── Batch labeling fields ──────────────────────────────
    paused_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    examples_succeeded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    examples_schema_invalid: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    examples_timeout: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    examples_endpoint_error: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    examples_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Metrics (shared across run types)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
