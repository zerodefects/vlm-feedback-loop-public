# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import (
    ProjectBase,
    created_at_col,
    updated_at_col,
    uuid_pk,
)


class Project(ProjectBase):
    __tablename__ = "projects"

    project_id: Mapped[str] = uuid_pk()
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    project_dir: Mapped[str] = mapped_column(String, nullable=False)

    # ── Selection pointers ───────────────────────────────────────────
    teacher_model_config_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    active_guidance_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    active_student_model_config_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    # ── Generation Controls defaults ──────────────────────────
    labeling_generation_preset_key: Mapped[str] = mapped_column(
        String, nullable=False, default="precise"
    )
    thinking_default_on: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )

    # ── Visual Budget ─────────────────────────────────────────
    visual_budget_preset_key: Mapped[str] = mapped_column(
        String, nullable=False, default="high_detail"
    )

    # ── Structured generation ─────────────────────────────────
    structured_generation_mode_default: Mapped[str] = mapped_column(
        String, nullable=False, default="auto"
    )

    # ── Rationale ─────────────────────────────────────────────
    rationale_anti_anchoring: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )

    # ── Evaluation ────────────────────────────────────────────
    auto_evaluate_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    icl_recommendation_dismissed_at_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # ── Export ────────────────────────────────────────────────────────
    export_field_mode: Mapped[str] = mapped_column(
        String, nullable=False, default="all"
    )

    # ── Embedding ────────────────────────────────────────────────────
    embedding_provider: Mapped[str] = mapped_column(
        String, nullable=False, default="none"
    )
    embedding_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_endpoint_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # ── pHash ────────────────────────────────────────────────────────
    phash_algorithm: Mapped[str] = mapped_column(
        String, nullable=False, default="dct_phash_64"
    )

    # ── Feature flags ────────────────────────────────────────────────
    feature_flags: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # ── Schema refinement reminders ───────────────────────────
    schema_refinement_reminders_dismissed: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    schema_change_context_example_key: Mapped[str | None] = mapped_column(
        String, nullable=True
    )

    # ── Test Pool ─────────────────────────────────────────────
    test_pool_fraction: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.40
    )

    # ── Scale-Up Readiness Gate ───────────────────────────────
    scaleup_exact_match_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.80
    )
    scaleup_per_field_match_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.80
    )
    scaleup_min_per_value_f1_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.80
    )
    scaleup_accept_rate_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.80
    )
    scaleup_accept_rate_window: Mapped[int] = mapped_column(
        Integer, nullable=False, default=50
    )
    scaleup_min_test_pool_size: Mapped[int] = mapped_column(
        Integer, nullable=False, default=60
    )

    # ── Review selector ──────────────────────────────────────
    review_selector_scheduler_state: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )

    # ── Archive (soft) ───────────────────────────────────────────────
    # When non-null, the project is archived: hidden from the default list,
    # mutating endpoints return 409, background workers skip it. Source of
    # truth; the {project_dir}/.archived sentinel file is a lazy index used
    # to short-circuit list/recovery scans without opening every DB.
    archived_at: Mapped[str | None] = mapped_column(String(24), nullable=True)

    # ── FTU acknowledgment ────────────────────
    # Stamped on first transition through onboarding (NIMConnectionPage
    # auto-skip or manual [Continue], whichever fires first). When null,
    # ProjectIndexRedirect routes to /setup; when non-null the SME has
    # acknowledged the configuration once and the gate stays open.
    setup_completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)

    # ── Timestamps ───────────────────────────────────────────────────
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()
