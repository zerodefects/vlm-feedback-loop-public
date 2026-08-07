# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic response schemas for StudentModel endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

CheckpointPackagingStatus = Literal["pending", "validated", "failed"]
QualityStatus = Literal["pending", "validated", "partial", "failed"]
ServingStatus = Literal["pending", "validated", "failed", "not_attempted"]


class StudentModelResponse(BaseModel):
    """Full StudentModel record."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    student_model_id: str
    project_id: str
    training_suite_id: str | None
    student_base_model_config_id: str
    # Joined from ModelConfig.model_name so consumers (Compare & Benchmark
    # UI, full_stack_validation classifier, deployment-handoff renderers)
    # don't need a second round-trip to get the base model identity. May
    # be ``None`` only if the referenced ModelConfig was deleted out from
    # under the StudentModel row, which is an invariant violation; it is
    # otherwise always populated.
    base_model_name: str | None = None
    tao_job_id: str
    guidance_id: str
    dataset_export_ids: list[str]
    training_preset: str
    lora_config: dict[str, Any]
    created_at: str

    # Checkpoint status
    checkpoint_packaging_status: CheckpointPackagingStatus
    nim_checkpoint_ref: str | None

    # Two-part readiness
    quality_status: QualityStatus
    quality_evaluation_run_id: str | None
    serving_status: ServingStatus
    serving_evaluation_run_id: str | None
    # ``serving_status`` is durable historical state. This response-only
    # assessment prevents pre-AIPerf latency sweeps from satisfying the
    # current production handoff gate after a workspace upgrade.
    serving_benchmark_current: bool = False
    serving_benchmark_blocker: str | None = None

    # NIM deployment state
    nim_preflight_status: str | None
    nim_preflight_details: dict[str, Any] | None
    nim_preflight_at: str | None
    nim_deployment_mode: str | None
    nim_container_id: str | None
    nim_endpoint_url: str | None

    # Deployment metadata
    nim_vlm_release_version: str | None
    nim_model_profile_requested: str | None
    nim_model_profile_selected: str | None
    nim_profile_metadata: dict[str, Any] | None
    gpu_type: str | None
    gpu_count: int | None

    # Quantization provenance
    quantization_method: str | None
    quantize_tao_job_id: str | None


class StudentModelListResponse(BaseModel):
    """Cursor-paginated list response."""

    model_config = ConfigDict(extra="forbid")

    items: list[StudentModelResponse]
    next_cursor: str | None = None
