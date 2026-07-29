# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for batch labeling run endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CommonErrorEntry(BaseModel):
    """Aggregated error signature for a batch labeling run."""

    code: str  # e.g. "schema_invalid:primary_damage", "timeout", "endpoint_error"
    count: int
    sample: str | None = None  # First representative error message


# ── Batch Label Run Create ─────────────────────────────────────────────────


class BatchLabelRunCreateRequest(BaseModel):
    """Request body for ``POST .../batch_label_runs``."""

    model_config = ConfigDict(extra="forbid")

    include_auto_labeled: bool = False
    # Bounded at the schema edge: 0/negative limits previously reached the
    # service and produced degenerate runs instead of a 422.
    run_limit: int | None = Field(default=None, ge=1)
    structured_generation_mode: Literal["auto", "prompt_only"] | None = None
    ingested_after: str | None = None  # ISO 8601 timestamp filter
    ingested_before: str | None = None  # ISO 8601 timestamp filter
    # Dispatch-width override. Default (None) resolves provider-aware:
    # BATCH_LABEL_CONCURRENCY_HOSTED for hosted endpoints,
    # BATCH_LABEL_CONCURRENCY_SELF_HOSTED for self-hosted/local NIMs.
    concurrency: int | None = Field(default=None, ge=1, le=64)
    # Mirror of the evaluation API's icl_mode. "disabled" runs the
    # batch at the Teacher's zero-shot form — the supported path for
    # ICL-negative teachers (§8.3), replacing the process-global
    # ICL_MAX_EXAMPLES=0 workaround. Default "enabled" (shipped behavior).
    icl_mode: Literal["enabled", "disabled"] = "enabled"


class BatchLabelRunCreateResponse(BaseModel):
    """Response from ``POST .../batch_label_runs``."""

    run_id: str
    run_type: str
    status: str
    guidance_id: str | None
    model_config_id: str | None
    generation_preset_key: str | None
    thinking_mode_effective: str | None
    visual_budget_preset_key: str | None
    structured_generation_mode_effective: str | None
    icl_mode: str | None = None
    examples_total: int
    created_at: str


# ── Batch Label Run Detail ─────────────────────────────────────────────────


class BatchLabelRunResponse(BaseModel):
    """Full batch label run detail for ``GET .../batch_label_runs/{run_id}``."""

    run_id: str
    run_type: str
    status: str
    status_reason: str | None = None
    paused_reason: str | None = None

    # Config snapshot
    guidance_id: str | None = None
    guidance_version_number: int | None = None
    model_config_id: str | None = None
    model_name: str | None = None
    generation_preset_key: str | None = None
    thinking_mode_effective: str | None = None
    visual_budget_preset_key: str | None = None
    structured_generation_mode_effective: str | None = None
    icl_mode: str | None = None

    # Progress
    progress: dict[str, int] | None = None  # {processed, total}

    # Per-outcome counters
    examples_succeeded: int = 0
    examples_schema_invalid: int = 0
    examples_timeout: int = 0
    examples_endpoint_error: int = 0
    examples_total: int = 0

    # Top-N aggregated error signatures for the run monitor's "Common errors:" block
    common_errors: list[CommonErrorEntry] = []

    # Timestamps
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    cancel_requested_at: str | None = None

    recovered_from_restart: bool = False


# ── List ───────────────────────────────────────────────────────────────────


class BatchLabelRunListResponse(BaseModel):
    """Paginated list of batch label runs."""

    items: list[BatchLabelRunResponse]
    next_cursor: str | None = None


# ── Resume / Cancel ────────────────────────────────────────────────────────


class BatchLabelRunResumeResponse(BaseModel):
    """Response from ``POST .../batch_label_runs/{run_id}:resume``."""

    run_id: str
    status: str


class BatchLabelRunCancelResponse(BaseModel):
    """Response from ``POST .../batch_label_runs/{run_id}:cancel``."""

    run_id: str
    status: str
    cancel_requested_at: str | None = None
