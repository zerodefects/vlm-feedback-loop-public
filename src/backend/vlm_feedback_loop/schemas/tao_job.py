# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for TAO job endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ── Nested configuration models ─────────────────────────────────────────────


class LoRAConfig(BaseModel):
    """LoRA fine-tuning configuration persisted on every TAOJob."""

    model_config = ConfigDict(extra="forbid")

    enable_lora: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    modules_to_save: list[str] | None = None


class JobConfig(BaseModel):
    """``job_config`` payload persisted alongside every TAOJob.

    ``extra="allow"`` lets callers carry deployment-specific extensions
    (e.g. ``resolved_training_fields`` sub-maps, parallelism extras, opaque
    dataset_refs) without forcing schema churn every time TAO grows a field.
    """

    model_config = ConfigDict(extra="allow")

    training_backend: Literal["cosmos_rl_tao_vlm"] = "cosmos_rl_tao_vlm"
    training_preset: str
    training_policy_type: Literal["sft"] = "sft"
    lora_config: LoRAConfig
    hyperparameters: dict[str, Any] = {}
    parallelism_config: dict[str, Any] | None = None
    num_nodes: int = 1
    num_gpus_per_node: int | None = None
    redis_config: dict[str, Any] | None = None
    tao_release_version: str
    cosmos_rl_container_tag: str
    dataset_refs: dict[str, Any] = {}
    intended_outputs: dict[str, Any] = {}
    resolved_training_fields: dict[str, Any] = {}


class TAOCreateJobRequest(BaseModel):
    """Exact payload submitted to TAO (persisted verbatim).

    ``extra="allow"`` is deliberate: the payload must preserve TAO-native
    field names (``custom.dataset.media_path`` for train, ``media_dir`` for
    evaluate/quantize, plus job-level fields like ``name``, ``workspace``,
    ``train_datasets``, ``eval_dataset``, ``base_experiment_ids``,
    ``timeout_minutes``, etc.) so that our payload matches TAO documentation
    exactly rather than being normalized.
    """

    model_config = ConfigDict(extra="allow")

    kind: str
    action: Literal["train", "evaluate", "inference", "quantize"]
    specs: dict[str, Any] = {}


# ── Request / response envelopes ─────────────────────────────────────────────


class TAOJobCreateRequest(BaseModel):
    """Request body for ``POST .../tao_jobs``."""

    model_config = ConfigDict(extra="forbid")

    student_base_model_config_id: str
    dataset_export_ids: list[str]
    job_config: JobConfig
    tao_create_job_request: TAOCreateJobRequest


class TAOJobResponse(BaseModel):
    """Full TAOJob detail returned by create/get/list."""

    tao_job_id: str
    project_id: str
    status: str
    tao_status_raw: str | None = None

    # Action and identity
    action: str
    training_backend: str
    training_policy_type: str | None = None
    student_base_model_config_id: str
    dataset_export_ids: list[str]

    # Persisted payloads
    job_config: dict[str, Any]
    tao_create_job_request: dict[str, Any]
    tao_external_job_id: str | None = None

    # Progress and outputs
    progress: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    outputs_fetch_status: Literal["pending", "in_progress", "completed", "failed"]
    outputs_fetch_error_ref: str | None = None

    # Chain + linkage
    parent_tao_job_id: str | None = None
    chain_id: str | None = None
    chain_sequence: int | None = None
    chain_halted_reason: str | None = None

    # Lifecycle
    preflight_result: dict[str, Any] | None = None
    error_ref: str | None = None
    poll_error_ref: str | None = None

    # Timestamps (UTC ISO 8601 with Z suffix)
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    last_polled_at: str | None = None


class TAOJobListResponse(BaseModel):
    """Paginated list of TAOJob records."""

    items: list[TAOJobResponse]
    next_cursor: str | None = None
