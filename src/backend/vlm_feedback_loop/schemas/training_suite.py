# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for training suite endpoints.

"Start Training" durably owns its frozen exports before transfer, then
atomically creates every TAOJob chain after setup succeeds.
``TrainingSuiteCreateRequest`` is the composite counterpart to single-job TAO
CRUD.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Request ─────────────────────────────────────────────────────────────────


class TrainingSuiteCreateRequest(BaseModel):
    """POST /v1/projects/{project_id}/training_suites request body."""

    model_config = ConfigDict(extra="forbid")

    student_base_model_config_ids: list[str] = Field(
        ..., min_length=1, description="One or more ModelConfigs with student_base role"
    )
    training_preset: Literal["quick", "standard", "high_quality", "max_quality"]
    include_auto_labeled: bool = True
    export_field_mode: Literal["all", "aux_and_core", "core_only"] = "all"
    # LoRA-first is the Spec §2 default training mode; ``false`` opts a
    # suite into full-weight fine-tuning (increased memory + no adapter
    # packaging path). The value is retained on the suite and per-TAOJob,
    # then emitted on the cosmos-rl wire as ``specs.policy.lora`` (§9.7.3.2).
    enable_lora: bool = True
    quantization_schemes: list[Literal["FP8_DYNAMIC", "W8A8", "W8A16", "W4A16"]] = (
        Field(default_factory=lambda: ["FP8_DYNAMIC"])
    )
    idempotency_key: str = Field(..., min_length=1, max_length=128)


class TrainingPresetResolveRequest(BaseModel):
    """Selected Student bases whose deterministic preset patches are requested."""

    model_config = ConfigDict(extra="forbid")

    student_base_model_config_ids: list[str] = Field(..., min_length=1)


class TrainingPresetResolveResponse(BaseModel):
    """Backend-resolved preset patches for the read-only Advanced disclosure."""

    model_config = ConfigDict(extra="forbid")

    resolved_presets: dict[str, dict[str, Any]]


# ── Response ────────────────────────────────────────────────────────────────


class TrainingSuiteJobResponse(BaseModel):
    """A single TAOJob as it appears in the chain view."""

    tao_job_id: str
    action: str
    chain_sequence: int
    status: str
    tao_external_job_id: str | None = None
    chain_halted_reason: str | None = None
    outputs_fetch_status: Literal["pending", "in_progress", "completed", "failed"]
    outputs_fetch_error_ref: str | None = None


class TrainingSuiteChainResponse(BaseModel):
    """A single per-model chain."""

    chain_id: str
    student_base_model_config_id: str
    base_model_name: str
    jobs: list[TrainingSuiteJobResponse]


class TrainingSuiteResponse(BaseModel):
    """Full training-suite record with chains rendered for the Training Job Monitor."""

    training_suite_id: str
    project_id: str
    idempotency_key: str
    guidance_id: str
    training_preset: str
    export_field_mode: str
    include_auto_labeled: bool
    enable_lora: bool
    quantization_schemes: list[str]

    training_dataset_export_id: str | None
    evaluation_dataset_export_id: str | None
    training_example_count: int | None = None
    evaluation_example_count: int | None = None
    evaluation_dataset_checksum_sha256: str | None = None

    selected_student_base_model_config_ids: list[str]
    chain_ids_ordered: list[str]
    chains: list[TrainingSuiteChainResponse]
    student_model_ids: list[str] = Field(default_factory=list)

    provisioning_run_id: str | None = None
    provisioning_model_names: list[str] = Field(default_factory=list)
    setup_error_ref: str | None = None
    setup_retryable: bool = False

    status: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


class TrainingSuiteListResponse(BaseModel):
    """Paginated list of training suites."""

    items: list[TrainingSuiteResponse]
    next_cursor: str | None = None


class TrainingSuiteCancelFailure(BaseModel):
    """One TAO-side cancellation that could not be confirmed."""

    tao_job_id: str
    error: str


class TrainingSuiteCancelResponse(BaseModel):
    """Best-effort result for canceling every remaining job in one suite."""

    training_suite: TrainingSuiteResponse
    jobs_canceled: int
    jobs_already_terminal: int
    setup_tasks_canceled: int
    remote_cancel_failures: list[TrainingSuiteCancelFailure]
