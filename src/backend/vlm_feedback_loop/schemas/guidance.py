# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for Guidance CRUD and draft validation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vlm_feedback_loop.db.models.guidance import Guidance

# ── Shared types ────────────────────────────────────────────────────────────

FieldType = Literal["enum", "enum_set", "boolean", "integer", "string"]
FieldRole = Literal["core", "aux"]


# ── Request schemas ─────────────────────────────────────────────────────────


class SchemaFieldInput(BaseModel):
    """A field definition for new Guidance; the backend assigns its identity."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    type: FieldType
    role: FieldRole = "core"
    allowed_values: list[str] | None = None
    minimum: int | None = None
    maximum: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    display_order: int = 0


class SchemaFieldEditInput(SchemaFieldInput):
    """A field in an edit request; existing fields carry backend-issued identity."""

    field_id: str | None = None


class GuidanceCreate(BaseModel):
    """Request body for creating a new Guidance version."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    schema_fields: list[SchemaFieldInput] = Field(..., alias="schema")
    rules: str = ""


class DraftValidationRequest(BaseModel):
    """Request body for draft validation."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    schema_fields: list[SchemaFieldInput] = Field(..., alias="schema")
    rules: str = ""


# ── Response schemas ────────────────────────────────────────────────────────


class SchemaFieldResponse(BaseModel):
    """A single SchemaCore field in a response — includes system-generated field_id."""

    field_id: str
    field_name: str
    type: str
    role: str
    allowed_values: list[str] | None = None
    minimum: int | None = None
    maximum: int | None = None
    min_length: int | None = None
    max_length: int | None = None
    display_order: int = 0


class SchemaIssueResponse(BaseModel):
    """A validation issue from draft validation or create."""

    severity: str
    code: str
    message: str
    field_path: str | None = None


class GuidanceResponse(BaseModel):
    """Full Guidance record for API responses.

    Uses ``from_guidance`` classmethod to unpack the JSON envelope stored
    in the ``schema`` column into typed response fields.
    """

    guidance_id: str
    project_id: str
    version_number: int
    description: str
    schema_fields: list[SchemaFieldResponse]
    rules: str
    derived_json_schema: dict[str, Any]
    generation_order: list[str]
    schema_hash: str
    created_at: str
    semantic_core_change_from_guidance_id: str | None = None
    schema_change_summary: dict[str, Any] | None = None

    @classmethod
    def from_guidance(cls, g: Guidance) -> GuidanceResponse:
        """Construct a response from a Guidance ORM object.

        The ``schema`` column stores a JSON envelope::

            {
                "fields": [...],
                "derived_json_schema": {...},
                "generation_order": [...],
                "schema_hash": "..."
            }
        """
        envelope = g.schema
        fields_raw = envelope.get("fields", [])
        schema_fields = [SchemaFieldResponse(**f) for f in fields_raw]

        return cls(
            guidance_id=g.guidance_id,
            project_id=g.project_id,
            version_number=g.version_number,
            description=g.description,
            schema_fields=schema_fields,
            rules=g.rules,
            derived_json_schema=envelope.get("derived_json_schema", {}),
            generation_order=envelope.get("generation_order", []),
            schema_hash=envelope.get("schema_hash", ""),
            created_at=g.created_at,
            semantic_core_change_from_guidance_id=g.semantic_core_change_from_guidance_id,
            schema_change_summary=g.schema_change_summary,
        )


class GuidanceListResponse(BaseModel):
    """Paginated list of Guidance records."""

    items: list[GuidanceResponse]
    next_cursor: str | None = None


class DraftValidationResponse(BaseModel):
    """Response from the draft validation endpoint."""

    issues: list[SchemaIssueResponse]
    derived_json_schema: dict[str, Any] | None = None
    schema_hash: str | None = None
    save_allowed: bool


# ── ICL count ──────────────────────────────────────────────────


class IclCountResponse(BaseModel):
    """Response from ``GET .../guidance:icl_count``.

    Count of ICL-eligible Edits — non-pool Verified Edits under the
    active Guidance — displayed as "ICL: N edits" on the Batch pre-run
    configuration snapshot.
    """

    eligible_count: int


# ── Schema Refinement Reminders ───────────────────────────────────


class ReminderStatusResponse(BaseModel):
    """Response from ``GET .../guidance:reminder_status``."""

    active_reminder: int | None  # 1 or 2, or None if no reminder is active
    verified_count: int
    threshold_1: int
    threshold_2: int
    dismissed_count: int


class ReminderDismissResponse(BaseModel):
    """Response from ``POST .../guidance:dismiss_reminder``."""

    dismissed_count: int
