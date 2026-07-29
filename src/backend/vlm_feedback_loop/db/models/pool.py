# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pool record — frozen Test Pool version snapshots."""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class Pool(ProjectBase):
    __tablename__ = "pools"

    pool_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    pool_type: Mapped[str] = mapped_column(String, nullable=False, default="test_pool")
    pool_version: Mapped[int] = mapped_column(Integer, nullable=False)
    member_example_keys: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    member_count: Mapped[int] = mapped_column(Integer, nullable=False)
    guidance_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[str] = created_at_col()
