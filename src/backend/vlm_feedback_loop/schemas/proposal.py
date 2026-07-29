# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the interactive proposal endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ProposalRequest(BaseModel):
    """Request body for ``POST .../proposals``.

    ``extra="forbid"`` matches every sibling inbound schema — a typo'd
    override field (e.g. ``thinking_override``) must 422, not silently
    fall back to project defaults. The ``*_override`` field names are the
    established wire contract (the UI's Retry panel sends them).
    """

    model_config = ConfigDict(extra="forbid")

    example_key: str = Field(..., min_length=1)
    teacher_model_config_id_override: str | None = None
    guidance_id_override: str | None = None
    generation_preset_key_override: str | None = None
    thinking_mode_override: Literal["on", "off"] | None = None
    visual_budget_preset_key_override: str | None = None
    retry_of_inference_invocation_id: str | None = None
    use_existing_label: bool = False


class ProposalResponse(BaseModel):
    """Response from ``POST .../proposals``."""

    inference_invocation_id: str
    example_key: str
    proposal_json: dict[str, Any] | None
    schema_valid_core: bool
    validation_errors_core: list[str]
    validation_errors_aux: list[str]
    # success | schema_invalid | timeout | endpoint_error | rate_limited
    # ``rate_limited`` distinguishes hosted-NIM 429s from generic
    # endpoint errors so the UI can show "wait + retry" copy instead
    # of the generic failure banner. The backend-level status string
    # is a free-form ``str`` rather than a ``Literal`` to keep room
    # for future failure categories without a schema migration.
    invocation_status: str
    latency_ms_end_to_end: int | None
    icl_images_attached_count: int = 0
    icl_example_keys_used: list[str] = []
    used_existing_label: bool
