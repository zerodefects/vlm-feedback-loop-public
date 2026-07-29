# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TAOJob record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class TAOJob(ProjectBase):
    __tablename__ = "tao_jobs"

    tao_job_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    student_base_model_config_id: Mapped[str] = mapped_column(
        String(36), nullable=False
    )
    dataset_export_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    action: Mapped[str] = mapped_column(
        String, nullable=False
    )  # train | evaluate | inference | quantize
    status: Mapped[str] = mapped_column(String, nullable=False, default="not_started")
    tao_status_raw: Mapped[str | None] = mapped_column(String, nullable=True)
    training_backend: Mapped[str] = mapped_column(
        String, nullable=False, default="cosmos_rl_tao_vlm"
    )
    training_policy_type: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # sft (when action=train)
    job_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    tao_create_job_request: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    tao_external_job_id: Mapped[str | None] = mapped_column(String, nullable=True)

    # Progress and outputs
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    outputs: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Linkage and lifecycle
    parent_tao_job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    preflight_result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[str] = created_at_col()
    started_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    last_polled_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    error_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    poll_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)

    # Job chaining
    chain_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    chain_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chain_halted_reason: Mapped[str | None] = mapped_column(String, nullable=True)

    # Outputs-fetch lifecycle.
    # ``pending`` (default) → ``in_progress`` → ``completed`` | ``failed``.
    # Polling tick scans ``succeeded + outputs_fetch_status IN (pending,
    # in_progress)`` and re-fires ``_handle_succeeded`` so an interrupted
    # multi-GB artifact download doesn't silently halt the chain.
    outputs_fetch_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", server_default="pending"
    )
    outputs_fetch_error_ref: Mapped[str | None] = mapped_column(String, nullable=True)
