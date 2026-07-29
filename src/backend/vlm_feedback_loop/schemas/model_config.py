# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for ModelConfig CRUD and capability probes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Shared Literal types for enum-like fields
ThinkingToggleMode = Literal[
    "none", "always_on_reasoning", "qwen_enable_thinking", "kimi_thinking"
]
VisualBudgetMode = Literal[
    "none", "mm_processor_size", "mm_processor_pixels", "mm_processor_tiles"
]
EligibleRole = Literal["teacher", "student_base"]
UnavailableReason = Literal[
    "no_nvidia_api_key",
    "hosted_not_compatible",
    "endpoint_unhealthy",
    "local_not_running",
    "endpoint_missing",
    "unknown_endpoint_mode",
]


class ModelAvailability(BaseModel):
    """Whether a model can actually be invoked from the current environment.

    Computed at API response time by combining the bound NimEndpoint's
    mode + last_probe_status, the cached environment assessment
    (credential / GPU presence), and ``ModelConfig.hosted_compatible``.
    Stable machine-readable reason codes let UI consumers choose copy
    or hide the entry entirely; the labeling Teacher dropdown hides
    every entry where ``available`` is false.
    """

    available: bool
    reason: UnavailableReason | None = None


class ModelConfigCreate(BaseModel):
    """Request body for creating a ModelConfig entry."""

    model_config = ConfigDict(extra="forbid")

    endpoint_id: str
    model_name: str
    context_window_tokens: int
    eligible_roles: list[EligibleRole]
    supports_image_input: bool
    thinking_toggle_mode: ThinkingToggleMode = "none"
    visual_budget_mode: VisualBudgetMode = "none"
    max_images_per_request: int = Field(default=5, ge=1)
    # Per-model default ICL depth cap (§6.2). None = no default; an explicit
    # icl_max_examples override always wins over this value.
    default_icl_max_examples: int | None = Field(default=None, ge=1)
    model_quantization: str | None = None
    nim_model_profile: str | None = None
    nim_profile_metadata: dict[str, Any] | None = None
    local_deploy_metadata: dict[str, Any] | None = None


class ModelConfigUpdate(BaseModel):
    """Partial update for a ModelConfig entry.

    ``model_config_id`` and ``model_name`` are immutable — excluded from this
    schema so ``extra="forbid"`` rejects them.
    """

    model_config = ConfigDict(extra="forbid")

    endpoint_id: str | None = None
    context_window_tokens: int | None = None
    eligible_roles: list[EligibleRole] | None = None
    supports_image_input: bool | None = None
    thinking_toggle_mode: ThinkingToggleMode | None = None
    visual_budget_mode: VisualBudgetMode | None = None
    # Operator tuning knob for the per-model ICL depth default (§6.2).
    # PATCHing it re-tunes the model's depth cap for this project; note
    # partial-update semantics cannot express "clear back to NULL" —
    # unsetting a seeded default is a deliberate non-goal (override depth
    # per run via the evaluation API instead).
    default_icl_max_examples: int | None = Field(default=None, ge=1)
    model_quantization: str | None = None
    nim_model_profile: str | None = None
    nim_profile_metadata: dict[str, Any] | None = None
    local_deploy_metadata: dict[str, Any] | None = None


class ModelConfigResponse(BaseModel):
    """Full ModelConfig record for API responses."""

    model_config = ConfigDict(from_attributes=True)

    model_config_id: str
    project_id: str
    endpoint_id: str
    model_name: str
    context_window_tokens: int
    eligible_roles: list[str]
    supports_image_input: bool
    structured_generation_support: str
    thinking_toggle_mode: str
    thinking_toggle_support: str
    visual_budget_mode: str
    visual_budget_support: str
    max_images_per_request: int
    image_cap_support: str
    default_icl_max_examples: int | None
    model_quantization: str | None
    nim_model_profile: str | None
    nim_profile_metadata: dict[str, Any] | None
    local_deploy_metadata: dict[str, Any] | None
    # TAO base-experiment provisioning state is read-only via this
    # endpoint. Setting these fields requires
    # `vlm-feedback-loop tao-pull-base-experiments` (self-service)
    # or `vlm-feedback-loop tao-bootstrap` (admin-managed);
    # PATCH /model_configs MUST continue to reject these as extra fields.
    # Surfacing them on GET unblocks operator visibility into provisioning
    # state without requiring direct sqlite3 access to project.db.
    tao_base_experiment_id: str | None = None
    tao_base_experiment_pull_status: str | None = None
    hosted_compatible: bool = True
    # Computed per request from environment + bound endpoint state.
    # Default kept on the model so unit tests that build a response
    # without going through the router (e.g. service-layer tests using
    # ``model_validate(mc)`` directly) get a sensible value without
    # having to construct it; the router overrides with the real
    # capability check.
    availability: ModelAvailability = Field(
        default_factory=lambda: ModelAvailability(available=True, reason=None)
    )
    created_at: str


class ModelConfigListResponse(BaseModel):
    """Paginated list of ModelConfig records."""

    items: list[ModelConfigResponse]
    next_cursor: str | None = None
