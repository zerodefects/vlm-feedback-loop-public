# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""StudentModel record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class StudentModel(ProjectBase):
    __tablename__ = "student_models"

    student_model_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    student_base_model_config_id: Mapped[str] = mapped_column(
        String(36), nullable=False
    )
    tao_job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    guidance_id: Mapped[str] = mapped_column(String(36), nullable=False)
    dataset_export_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    training_preset: Mapped[str] = mapped_column(String, nullable=False)
    lora_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[str] = created_at_col()

    # Checkpoint status
    checkpoint_packaging_status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )
    nim_checkpoint_ref: Mapped[str | None] = mapped_column(String, nullable=True)

    # Two-part readiness
    quality_status: Mapped[str] = mapped_column(
        String, nullable=False, default="pending"
    )
    quality_evaluation_run_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    serving_status: Mapped[str] = mapped_column(
        String, nullable=False, default="not_attempted"
    )
    serving_evaluation_run_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    # NIM deployment state
    nim_preflight_status: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_preflight_details: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    nim_preflight_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    nim_deployment_mode: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # local | external
    nim_container_id: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_endpoint_url: Mapped[str | None] = mapped_column(String, nullable=True)

    # Deployment metadata
    nim_vlm_release_version: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_model_profile_requested: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    nim_model_profile_selected: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    nim_profile_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
    gpu_type: Mapped[str | None] = mapped_column(String, nullable=True)
    gpu_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Quantization provenance
    quantization_method: Mapped[str | None] = mapped_column(String, nullable=True)
    quantize_tao_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Inference Contract snapshot
    # Derived from the training DatasetExport's ``export_field_mode`` at
    # registration. When null, the ``deployment_handoff`` generator falls back
    # to ``services.student_nim_lifecycle._student_inference_contract``.
    training_inference_contract: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
