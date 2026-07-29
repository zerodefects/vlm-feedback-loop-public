# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for image ingestion, example query, and image serving."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Ingestion ───────────────────────────────────────────────────────────────


class ExampleIngestItem(BaseModel):
    """Single item in an ingestion batch."""

    model_config = ConfigDict(extra="forbid")

    example_key: str
    storage_ref: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None  # merged into source_metadata if present
    state: str = "Unlabeled"


class ExampleIngestRequest(BaseModel):
    """Request body for ``POST .../examples:ingest``."""

    model_config = ConfigDict(extra="forbid")

    examples: list[ExampleIngestItem]


class ExampleResponse(BaseModel):
    """Serialised Example record."""

    model_config = ConfigDict(from_attributes=True)

    example_key: str
    project_id: str
    storage_ref: str
    ingested_at: str
    source_metadata: dict[str, Any]
    state: str
    phash: str | None = None
    clip_embedding_present: bool = False
    clip_embedding_dim: int | None = None
    clip_embedding_model_id: str | None = None
    embedding_provider: str | None = None
    omitted_source: str | None = None
    omitted_at: str | None = None
    prior_verified_label_ref: str | None = None
    prior_verified_outcome: str | None = None


class ExampleIngestResultItem(BaseModel):
    """Per-item result from ingestion."""

    example_key: str
    status: Literal["created", "exists", "error"]
    error: str | None = None
    error_code: str | None = None
    warnings: list[str] = Field(default_factory=list)
    example: ExampleResponse | None = None


class ExampleIngestResponse(BaseModel):
    """Response body for ``POST .../examples:ingest``."""

    results: list[ExampleIngestResultItem]


# ── Query ───────────────────────────────────────────────────────────────────


class VerifiedLabelResponse(BaseModel):
    """Serialised Label record (subset for query responses)."""

    model_config = ConfigDict(from_attributes=True)

    label_id: str
    example_key: str
    project_id: str
    label_status: str
    guidance_id: str
    inference_invocation_id: str
    label_json: dict[str, Any]
    labeled_at: str
    verified_outcome: str | None = None
    verified_at: str | None = None
    edited_core_fields: list[str] | None = None
    edited_aux_fields: list[str] | None = None
    rationale_source: str | None = None
    batch_label_run_id: str | None = None
    pool_assignment: str | None = None


class ExampleQueryItem(BaseModel):
    """Example record with optional nested verified_label."""

    example: ExampleResponse
    verified_label: VerifiedLabelResponse | None = None


class ExampleQueryResponse(BaseModel):
    """Response body for ``GET .../examples``."""

    items: list[ExampleQueryItem]
    next_cursor: str | None = None


class EmbeddingStatusResponse(BaseModel):
    """Response body for ``GET .../examples:embedding_status``.

    The authoritative REST counterpart to the ``embedding_progress`` /
    ``embedding_completed`` SSE events — lets a caller (the UI on reconnect, or
    an embeddings-first seeding barrier) poll whether CLIP embeddings have
    finished computing before relying on CLIP-diverse review selection,
    Test-Pool assignment, or ICL diversity policies.
    """

    total_examples: int
    embedded: int
    pending: int
    worker_active: bool
    auto_compute: bool
    provider: str | None = None
    model_id: str | None = None
    dim: int | None = None
    complete: bool


# ── Path Remapping ──────────────────────────────────────────────────────────


class RemapPathsRequest(BaseModel):
    """Request body for ``POST .../examples:remap_paths``."""

    model_config = ConfigDict(extra="forbid")

    old_prefix: str
    new_prefix: str
    dry_run: bool = True


class RemapSampleItem(BaseModel):
    """Single sample remapping in a dry-run response."""

    example_key: str
    old_storage_ref: str
    new_storage_ref: str


class RemapValidation(BaseModel):
    """Dry-run validation summary — spot-check of remapped paths."""

    sample_checked: int
    sample_resolved: int
    sample_missing: int
    missing_examples: list[str] = Field(default_factory=list)


class RemapResponse(BaseModel):
    """Response body for ``POST .../examples:remap_paths``."""

    dry_run: bool
    # Dry-run fields
    matched_count: int | None = None
    sample_remappings: list[RemapSampleItem] | None = None
    unmatched_count: int | None = None
    validation: RemapValidation | None = None
    # Commit fields
    remapped_count: int | None = None
    audit_event_id: str | None = None
