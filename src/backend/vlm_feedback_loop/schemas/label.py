# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for label save, skip, restore, and rationale endpoints."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# ── Label Save ───────────────────────────────────────────────────────────────


class LabelSaveRequest(BaseModel):
    """Request body for ``POST .../labels``."""

    model_config = ConfigDict(extra="forbid")

    example_key: str
    inference_invocation_id: str
    label_json: dict[str, Any]
    rationale_source: (
        Literal["teacher_proposal", "sme_edited", "teacher_regenerated_approved"] | None
    ) = None
    rationale_regeneration_invocation_id: str | None = None


class LabelSaveResponse(BaseModel):
    """Response from ``POST .../labels``."""

    example_key: str
    label_status: str  # "verified"
    verified_outcome: str  # "Accept" | "Edit"
    verified_at: str
    edited_core_fields: list[str]
    edited_aux_fields: list[str]
    pool_assignment: str | None


class SkipResponse(BaseModel):
    """Response from ``POST .../examples/{key}:skip``."""

    example_key: str
    state: str  # "Omitted"
    omitted_at: str


# ── Restore Omitted ──────────────────────────────────────────────────────────


class RestoreOmittedResponse(BaseModel):
    """Response from ``POST .../examples:restore_omitted``."""

    restored_count: int


# ── Rationale Regeneration ───────────────────────────────────────────────────


class RationaleRegenerationRequest(BaseModel):
    """Request body for ``POST .../examples/{key}:regenerate_rationale``."""

    model_config = ConfigDict(extra="forbid")

    teacher_model_config_id: str | None = None


class RationaleRegenerationResponse(BaseModel):
    """Response from ``POST .../examples/{key}:regenerate_rationale``."""

    inference_invocation_id: str
    rationale_note: str
    invocation_status: str  # success | timeout | endpoint_error
