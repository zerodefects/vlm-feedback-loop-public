# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClipEmbedding record.

Composite PK (project_id, example_key) with FK to Example + ON DELETE CASCADE.
"""

from __future__ import annotations

from sqlalchemy import ForeignKeyConstraint, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, updated_at_col


class ClipEmbedding(ProjectBase):
    __tablename__ = "clip_embeddings"
    # Runtime project connections enforce this exact ownership pair. Migration
    # 054 adds the corresponding unique parent key before enforcement begins.
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "example_key"],
            ["examples.project_id", "examples.example_key"],
            ondelete="CASCADE",
        ),
    )

    project_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    example_key: Mapped[str] = mapped_column(String, primary_key=True)
    embedding_provider: Mapped[str] = mapped_column(
        String, nullable=False
    )  # hosted_nvclip | self_hosted_nvclip
    clip_embedding_model_id: Mapped[str] = mapped_column(String, nullable=False)
    clip_embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector_blob_f32: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[str] = created_at_col()
    updated_at: Mapped[str] = updated_at_col()
