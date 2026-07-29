# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Index, String, text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, uuid_pk


class Label(ProjectBase):
    __tablename__ = "labels"

    # At most one auto_labeled Label per example. The partial predicate leaves
    # ``verified`` labels unconstrained. Consumers
    # fetch the auto-label with ``.scalar_one_or_none()``, so a duplicate would
    # crash the review queue; this makes the invariant structural.
    __table_args__ = (
        Index(
            "ux_labels_auto_labeled_example",
            "project_id",
            "example_key",
            unique=True,
            sqlite_where=text("label_status = 'auto_labeled'"),
        ),
        # Hot lookup path (example_key + status).
        Index("ix_labels_example_key_status", "example_key", "label_status"),
    )

    label_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    example_key: Mapped[str] = mapped_column(String, nullable=False)
    label_status: Mapped[str] = mapped_column(
        String, nullable=False
    )  # verified | auto_labeled
    guidance_id: Mapped[str] = mapped_column(String(36), nullable=False)
    inference_invocation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    label_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    labeled_at: Mapped[str] = mapped_column(String(24), nullable=False)

    # Verification fields (populated when label_status=verified; null for auto_labeled)
    verified_outcome: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # Accept | Edit
    verified_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    edited_core_fields: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    edited_aux_fields: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    rationale_source: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # teacher_proposal | sme_edited | teacher_regenerated_approved
    rationale_regeneration_invocation_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    # Batch Labeling lineage
    batch_label_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    # Pool assignment
    pool_assignment: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # test_pool | null
