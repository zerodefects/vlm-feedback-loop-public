# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocalNimDeployment record."""

from __future__ import annotations

from sqlalchemy import Boolean, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class LocalNimDeployment(ProjectBase):
    __tablename__ = "local_nim_deployments"

    local_nim_deployment_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_config_id: Mapped[str] = mapped_column(String(36), nullable=False)
    role: Mapped[str] = mapped_column(
        String, nullable=False
    )  # teacher | embedding | student
    nim_container_image: Mapped[str] = mapped_column(String, nullable=False)
    container_name: Mapped[str] = mapped_column(String, nullable=False)
    container_id: Mapped[str | None] = mapped_column(String, nullable=True)
    host_port: Mapped[int] = mapped_column(Integer, nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String, nullable=False)
    gpu_assignment: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default="starting"
    )  # starting | running | stopped | failed
    status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    # Post-onboarding NIM Configuration can request that a Teacher become the
    # project's active Teacher only after the new NIM has passed health,
    # served-model, and inference verification. Persisting the intent keeps it
    # durable across backend restarts during a cold deploy.
    activate_on_success: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    deployed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    stopped_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    created_at: Mapped[str] = created_at_col()

    # Student-specific extensions (populated only when role == "student").
    student_model_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    checkpoint_mount_path: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_served_model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    nim_model_name_path: Mapped[str | None] = mapped_column(String, nullable=True)
    precision_method: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # bf16 | fp8 | w4a16 (mirror of StudentModel.quantization_method)

    # One-NIM-per-GPU displacement audit and restoration source.
    # Populated when this deployment was stopped by another
    # deployment's acquire-GPU call (the *replace* semantics). The audit link
    # also lets a failed replacement find and best-effort restore its prior
    # resident without relying on in-memory state.
    displaced_by_deployment_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    displaced_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
