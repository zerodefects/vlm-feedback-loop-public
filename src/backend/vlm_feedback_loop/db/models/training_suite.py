# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrainingSuite record.

A TrainingSuite aggregates a single "Start Training" action: the training
and Test-Pool evaluation DatasetExports produced at suite creation, plus
one TAOJob chain per selected student_base model (train → evaluate →
quantize/evaluate per scheme). ``chain_ids_ordered`` preserves the
intended sequential-model execution order.

``idempotency_key`` is UNIQUE per project so a retry-safe POST returns the
existing suite response instead of creating duplicates (see
``training_suite_service.create_training_suite``).
"""

from __future__ import annotations

from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class TrainingSuite(ProjectBase):
    __tablename__ = "training_suites"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_training_suites_project_idempotency",
        ),
    )

    training_suite_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String, nullable=False)

    guidance_id: Mapped[str] = mapped_column(String(36), nullable=False)
    training_preset: Mapped[str] = mapped_column(
        String, nullable=False
    )  # quick | standard | high_quality | max_quality
    export_field_mode: Mapped[str] = mapped_column(
        String, nullable=False
    )  # all | aux_and_core | core_only
    include_auto_labeled: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Null only while first-use base provisioning / suite preparation is active.
    training_dataset_export_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    evaluation_dataset_export_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    # JSON list[str] — order preserved.
    selected_student_base_model_config_ids: Mapped[list[str]] = mapped_column(
        JSON, nullable=False
    )
    # JSON list[str] — quantization schemes selected at suite creation.
    quantization_schemes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    # JSON list[str] — chain_ids in the order chains should run sequentially.
    chain_ids_ordered: Mapped[list[str]] = mapped_column(JSON, nullable=False)

    # A non-null run id means this suite required automatic first-use
    # provisioning. The model names are duplicated here so the project-scoped
    # Training Jobs response does not depend on joining deployment.db.
    provisioning_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    provisioning_model_names: Mapped[list[str] | None] = mapped_column(
        JSON, nullable=True
    )
    setup_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[str] = mapped_column(
        String, nullable=False, default="initialized"
    )  # provisioning | preparing | initialized | running | completed | failed | canceled

    created_at: Mapped[str] = created_at_col()
    started_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
