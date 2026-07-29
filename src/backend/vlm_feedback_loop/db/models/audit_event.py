# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AuditEvent record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class AuditEvent(ProjectBase):
    __tablename__ = "audit_events"

    audit_event_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    example_key: Mapped[str | None] = mapped_column(String, nullable=True)
    event_data: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    created_at: Mapped[str] = created_at_col()
