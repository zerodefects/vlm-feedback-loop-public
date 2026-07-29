# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the guidance:edit endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vlm_feedback_loop.schemas.guidance import GuidanceResponse, SchemaFieldEditInput


class GuidanceEditRequest(BaseModel):
    """Request body for ``POST /v1/projects/{project_id}/guidance:edit``."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    schema_fields: list[SchemaFieldEditInput] = Field(..., alias="schema")
    rules: str = ""
    dry_run: bool = False
    schema_change_context_example_key: str | None = None


class FieldChangeResponse(BaseModel):
    """A single detected change between schema versions."""

    field_id: str
    change_type: str
    classification: str  # "in_place" or "semantic"
    detail: dict[str, Any] = {}


class EditPreviewResponse(BaseModel):
    """Response from ``dry_run=true``: classification + affected counts."""

    edit_type: str  # "in_place", "semantic", or "no_change"
    changes: list[FieldChangeResponse]
    verified_count: int
    auto_labeled_count: int
    change_summary: dict[str, Any]


class EditExecuteResponse(BaseModel):
    """Response from ``dry_run=false``: new guidance + mutation results."""

    guidance: GuidanceResponse
    edit_type: str  # "in_place", "semantic", or "no_change"
    verified_reverted_count: int
    auto_labeled_reverted_count: int
    changes: list[FieldChangeResponse]
