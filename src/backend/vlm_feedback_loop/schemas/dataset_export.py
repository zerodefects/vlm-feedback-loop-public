# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for dataset export endpoints and schema-invalid manifest."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Dataset Export Create ──────────────────────────────────────────────────


class DatasetExportSelectionFilters(BaseModel):
    """Optional selection filters for dataset export."""

    model_config = ConfigDict(extra="forbid")

    guidance_id: str | None = None


class DatasetExportCreateRequest(BaseModel):
    """Request body for ``POST .../dataset_exports``."""

    model_config = ConfigDict(extra="forbid")

    dataset_intent: Literal["training", "evaluation", "testing"]
    label_tier_filter: Literal["verified_only", "auto_labeled_only", "combined"] = (
        "verified_only"
    )
    export_field_mode: Literal["all", "aux_and_core", "core_only"] = "all"
    batch_label_run_id: str | None = Field(default=None, min_length=1)
    selection_filters: DatasetExportSelectionFilters | None = None


# ── Dataset Export Record ──────────────────────────────────────────────────


class DatasetExportResponse(BaseModel):
    """The full dataset export record — create, get, and list items alike.

    The archive builds in the background: the ``POST`` response carries
    ``status="running"`` with null ``artifact_refs``/``manifest_ref``;
    both populate when the record reaches ``completed``.
    """

    dataset_export_id: str
    project_id: str
    dataset_intent: str
    export_field_mode: str
    guidance_id: str
    label_tier_filter: str
    selection_definition_snapshot: dict[str, Any]
    example_count: int
    status: str  # running | completed | failed
    status_reason: str | None = None
    progress: dict[str, int] | None = None  # {images_written, images_total}
    started_at: str | None = None
    completed_at: str | None = None
    artifact_refs: (
        dict[str, str] | None
    )  # {archive_path, annotations_path, checksum_sha256} once completed
    manifest_ref: str | None
    created_at: str


# ── List ───────────────────────────────────────────────────────────────────


class DatasetExportListResponse(BaseModel):
    """Paginated list of dataset exports."""

    items: list[DatasetExportResponse]
    next_cursor: str | None = None


# ── Schema-Invalid Manifest ─────────────────────────────────────


class SchemaInvalidExample(BaseModel):
    """A single schema-invalid example in the manifest."""

    example_key: str
    validation_errors_core: list[str]
    inference_invocation_id: str


class SchemaInvalidManifestResponse(BaseModel):
    """Response from ``GET .../batch_label_runs/{run_id}/schema_invalid_manifest``."""

    batch_label_run_id: str
    schema_invalid_examples: list[SchemaInvalidExample]
    total_count: int
