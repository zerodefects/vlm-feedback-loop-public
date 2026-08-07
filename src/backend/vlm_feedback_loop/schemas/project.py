# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request/response schemas for Project endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Request schemas ──────────────────────────────────────────────────────────


class ProjectCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1)
    description: str | None = None


class ProjectUpdate(BaseModel):
    """Partial update — only non-None fields are applied.

    ``extra="forbid"`` rejects unknown fields.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = None

    # Selection pointers
    teacher_model_config_id: str | None = None
    active_guidance_id: str | None = None
    active_student_model_config_id: str | None = None

    # Generation controls
    labeling_generation_preset_key: str | None = None
    thinking_default_on: bool | None = None
    visual_budget_preset_key: str | None = None
    structured_generation_mode_default: str | None = None

    # Rationale
    rationale_anti_anchoring: bool | None = None

    # Evaluation
    auto_evaluate_enabled: bool | None = None

    # Schema refinement reminder dismissals are NOT patchable here:
    # the SME's X-dismiss goes through ``POST .../guidance:dismiss_reminder``,
    # which owns the counter increment (guidance_service).

    # Export. Constrained to the SchemaCore field-mode enum — the
    # service applies these via blind setattr, so bounds belong on the schema.
    export_field_mode: Literal["all", "aux_and_core", "core_only"] | None = None

    # Embedding
    embedding_provider: str | None = None

    # Feature flags
    feature_flags: dict[str, Any] | None = None

    # Test Pool — a fraction; a value outside [0, 1] would corrupt pool routing.
    test_pool_fraction: float | None = Field(default=None, ge=0.0, le=1.0)

    # Scale-Up Gate. Thresholds are rates in [0, 1]; window/size are counts ≥ 0.
    scaleup_exact_match_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    scaleup_per_field_match_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    scaleup_min_per_value_f1_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    scaleup_accept_rate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    scaleup_accept_rate_window: int | None = Field(default=None, ge=0)
    scaleup_min_test_pool_size: int | None = Field(default=None, ge=0)


# ── Response schemas ─────────────────────────────────────────────────────────


class ProjectCounts(BaseModel):
    verified: int = 0
    unlabeled: int = 0
    auto_labeled: int = 0
    omitted: int = 0
    pending_relabel: int = 0
    # Verified examples that carry a ``prior_verified_label_ref`` — i.e.,
    # priors the SME has already re-labeled after a semantic Core change.
    # Powers the "Prior labels: N of M re-labeled" progress strip
    # (M = prior_relabeled + pending_relabel).
    prior_relabeled: int = 0


class ProjectResponse(BaseModel):
    """Full Project record returned from create/get/update.

    ``counts`` mirrors the field exposed on ``ProjectListItem`` so that a
    single ``GET /v1/projects/{id}`` fetch surfaces the data totals the
    Student Training screen needs. Defaults to zeros when
    the router does not populate it (e.g., immediately after create).
    """

    model_config = ConfigDict(from_attributes=True)

    project_id: str
    name: str
    description: str | None
    project_dir: str
    counts: ProjectCounts = Field(default_factory=ProjectCounts)

    # Selection pointers
    teacher_model_config_id: str | None
    active_guidance_id: str | None
    active_student_model_config_id: str | None

    # Generation Controls
    labeling_generation_preset_key: str
    thinking_default_on: bool
    visual_budget_preset_key: str
    structured_generation_mode_default: str

    # Rationale
    rationale_anti_anchoring: bool

    # Evaluation
    auto_evaluate_enabled: bool
    icl_recommendation_dismissed_at_count: int

    # Export
    export_field_mode: str

    # Embedding
    embedding_provider: str
    embedding_model_id: str | None
    embedding_dim: int | None
    embedding_endpoint_id: str | None

    # pHash
    phash_algorithm: str

    # Feature flags
    feature_flags: Any

    # Schema refinement reminders
    schema_refinement_reminders_dismissed: int
    schema_change_context_example_key: str | None

    # Test Pool
    test_pool_fraction: float

    # Scale-Up Gate
    scaleup_exact_match_threshold: float
    scaleup_per_field_match_threshold: float
    scaleup_min_per_value_f1_threshold: float
    scaleup_accept_rate_threshold: float
    scaleup_accept_rate_window: int
    scaleup_min_test_pool_size: int

    # Review selector
    review_selector_scheduler_state: Any

    # Archive (soft) — non-null ISO 8601 string when the project is archived.
    archived_at: str | None = None

    # FTU acknowledgment — non-null ISO 8601 string once the SME has
    # walked through the NIM connection and embedding setup screens
    # (auto-skip or manual). ProjectIndexRedirect gates the setup route
    # on this field.
    setup_completed_at: str | None = None

    # Timestamps
    created_at: str
    updated_at: str


class ProjectListItem(BaseModel):
    project_id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    counts: ProjectCounts
    # Archive (soft) — non-null ISO 8601 string when archived.
    archived_at: str | None = None
    # FTU acknowledgment — non-null ISO 8601 string
    # once the SME has acknowledged onboarding for this project.
    setup_completed_at: str | None = None


class ProjectListResponse(BaseModel):
    items: list[ProjectListItem]
    next_cursor: str | None
    # Workspace-global: True when at least one project is archived,
    # computed from the ``.archived`` marker files alone. Lets the project
    # list screen decide whether to render the "Show archived" affordance
    # without a second archived-inclusive fetch (which opens every project
    # DB in the workspace).
    has_archived: bool = False


class MarkSetupCompletedRequest(BaseModel):
    """Request body for ``POST /v1/projects/{id}:mark_setup_completed``.

    Captures the recommendation context the SME walked through at the
    moment of acknowledgment. Stored verbatim on the ``setup_completed``
    AuditEvent for forensic audit — what mode did the system recommend,
    did it auto-skip, what embedding provider was effective at the time.
    Project-state fields (``embedding_provider`` etc.) on the row itself
    remain the source of truth; this payload captures the *recommendation*
    surface that may have differed from the persisted state during
    transient deploys.
    """

    model_config = ConfigDict(extra="forbid")

    auto_skip: bool
    teacher_mode: str
    embedding_mode: str
    embedding_provider: str
    # Names of local NIM models the FTUE queued for background
    # deployment at gate confirm (e.g. ``["nvidia/cosmos-reason2-8b",
    # "nvidia/llama-nemotron-embed-vl-1b-v2"]``). Empty when the SME
    # walked the hosted-only path. Persisted verbatim on the
    # ``setup_completed`` AuditEvent — the only forensic record of which
    # local deploys the FTUE kicked off at gate-confirm time.
    local_deploy_queued: list[str] = []


class MarkSetupCompletedResponse(BaseModel):
    """Response for ``POST /v1/projects/{id}:mark_setup_completed``.

    ``transitioned`` is True only on the first call (when
    ``setup_completed_at`` was previously null). Subsequent calls are
    idempotent no-ops and return ``transitioned=false`` with the existing
    project payload unchanged.
    """

    transitioned: bool
    project: ProjectResponse
