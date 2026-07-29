# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guidance record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String, UniqueConstraint
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class Guidance(ProjectBase):
    __tablename__ = "guidances"
    # One Guidance per (project, version_number). This blocks a concurrent
    # create race from assigning duplicate version numbers.
    __table_args__ = (
        UniqueConstraint(
            "project_id", "version_number", name="ux_guidances_project_version"
        ),
    )

    guidance_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    rules: Mapped[str] = mapped_column(String, nullable=False, default="")
    created_at: Mapped[str] = created_at_col()
    semantic_core_change_from_guidance_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )
    schema_change_summary: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )
