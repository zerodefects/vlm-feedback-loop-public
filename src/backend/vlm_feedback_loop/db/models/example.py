# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, Index, Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase


class Example(ProjectBase):
    __tablename__ = "examples"
    # The review selector and pool/schema-evolution passes filter by state on
    # this one-row-per-image table.
    __table_args__ = (
        Index("ix_examples_state", "state"),
        Index(
            "ux_examples_project_key",
            "project_id",
            "example_key",
            unique=True,
        ),
    )

    # example_key is content-derived, NOT UUID4
    example_key: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    storage_ref: Mapped[str] = mapped_column(String, nullable=False)
    ingested_at: Mapped[str] = mapped_column(String(24), nullable=False)
    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    state: Mapped[str] = mapped_column(
        String, nullable=False, default="Unlabeled"
    )  # Unlabeled | Auto-Labeled | Verified | Omitted

    # pHash
    phash: Mapped[str | None] = mapped_column(String, nullable=True)

    # Omission provenance
    omitted_source: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # sme_skip
    omitted_at: Mapped[str | None] = mapped_column(String(24), nullable=True)

    # Embedding fields — presence flags only, not the vector
    clip_embedding_present: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    clip_embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clip_embedding_model_id: Mapped[str | None] = mapped_column(String, nullable=True)
    embedding_provider: Mapped[str | None] = mapped_column(String, nullable=True)

    # Prior-label reference fields
    prior_verified_label_ref: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # JSON snapshot
    prior_verified_outcome: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # Accept | Edit
