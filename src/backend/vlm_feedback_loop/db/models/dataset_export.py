# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DatasetExport record."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Integer, String
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from vlm_feedback_loop.db.base import ProjectBase, created_at_col, uuid_pk


class DatasetExport(ProjectBase):
    __tablename__ = "dataset_exports"

    dataset_export_id: Mapped[str] = uuid_pk()
    project_id: Mapped[str] = mapped_column(String(36), nullable=False)
    dataset_intent: Mapped[str] = mapped_column(
        String, nullable=False
    )  # training | evaluation | testing
    export_field_mode: Mapped[str] = mapped_column(
        String, nullable=False
    )  # all | aux_and_core | core_only
    guidance_id: Mapped[str] = mapped_column(String(36), nullable=False)
    label_tier_filter: Mapped[str] = mapped_column(
        String, nullable=False
    )  # verified_only | auto_labeled_only | combined
    selection_definition_snapshot: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False
    )
    # Status lifecycle: the standalone POST endpoint creates the row
    # `running` and a background task builds the archive; the training-suite
    # path commits a completed row, links it to its durable preparing suite,
    # and only then starts workspace upload.
    # `artifact_refs`/`manifest_ref` stay NULL until the build completes.
    status: Mapped[str] = mapped_column(
        String, nullable=False, server_default="completed"
    )  # running | completed | failed
    status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    progress: Mapped[dict[str, Any] | None] = mapped_column(
        JSON, nullable=True
    )  # {images_written, images_total}
    started_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(24), nullable=True)
    artifact_refs: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    manifest_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    example_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Workspace S3 upload tracking. Null until the per-training
    # dataset upload service streams the .tar.gz archive to the workspace
    # bucket. `dataset_upload_ref` holds the S3 key; `dataset_upload_uri`
    # holds the full s3:// or http://<endpoint>/<bucket>/<key> reference
    # that TAO job specs consume.
    dataset_upload_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_upload_uri: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[str] = created_at_col()
