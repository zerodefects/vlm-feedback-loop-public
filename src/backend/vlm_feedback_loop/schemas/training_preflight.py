# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the training preflight endpoint.

The training preflight confirms TAO reachability, safe job-timeout support,
workspace readiness, per-student-base experiment readiness, gated-model
credential availability when first-use provisioning is required,
``student_base`` role and training-mode compatibility on each selected base
model, and that both the training and held-out evaluation datasets meet their
project-configured requirements.
Deliberately distinct from the NIM deployment preflight which does Docker /
GPU / NGC checks.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingPreflightRequest(BaseModel):
    """Request body for ``POST /v1/projects/{project_id}/training_preflight``."""

    model_config = ConfigDict(extra="forbid")

    student_base_model_config_ids: list[str] = Field(..., min_length=1)
    include_auto_labeled: bool = True
    enable_lora: bool = True
    quantization_schemes: list[Literal["FP8_DYNAMIC", "W8A8", "W8A16", "W4A16"]] = (
        Field(default_factory=lambda: ["FP8_DYNAMIC"])
    )


class TrainingPreflightCheck(BaseModel):
    """A single preflight check result."""

    model_config = ConfigDict(extra="forbid")

    check_name: Literal[
        "tao_reachable",
        "tao_job_timeout_supported",
        "tao_workspace_reachable",
        "tao_base_experiment_ready",
        "hf_token_configured",
        "lora_merge_runtime",
        "student_base_role",
        "training_mode_compatible",
        "quantization_compatible",
        "verified_train_examples",
        "min_test_pool_size",
    ]
    passed: bool
    message: str
    model_config_id: str | None = None
    # True when the check passes because Start Training can provision the
    # selected base automatically. Lets the UI explain the first-run work
    # without treating it as an infrastructure failure.
    provisioning_required: bool = False
    # Shared follow-up instruction, kept separate from ``message`` so the
    # UI can render it once even when several per-model checks fail with
    # the same fix.
    remediation: str | None = None
    # Optional backend-owned capability restriction for a configuration that
    # may still pass in a reduced mode (for example, a baseline-only Student).
    restriction: Literal["baseline_only"] | None = None


class TrainingDataSummary(BaseModel):
    """Backend-authoritative counts for the exact training export selection."""

    model_config = ConfigDict(extra="forbid")

    verified_training_count: int = Field(ge=0)
    test_pool_count: int = Field(ge=0)
    required_test_pool_count: int = Field(ge=1)
    auto_labeled_eligible_count: int = Field(ge=0)
    auto_labeled_included_count: int = Field(ge=0)
    excluded_test_pool_count: int = Field(ge=0)
    excluded_auto_labeled_count: int = Field(ge=0)
    usable_training_count: int = Field(ge=0)


class TrainingPreflightResponse(BaseModel):
    """Response body — overall status plus per-check detail."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["passed", "failed"]
    checks: list[TrainingPreflightCheck]
    data_summary: TrainingDataSummary
    # Server-resolved training-preset hyperparameter patches, keyed
    # model_config_id → preset key → patch. The Advanced expander on the
    # Training screen renders these verbatim — the backend resolver is
    # the single source of truth (the UI's former mirror drifted).
    resolved_presets: dict[str, dict[str, Any]] = Field(default_factory=dict)
