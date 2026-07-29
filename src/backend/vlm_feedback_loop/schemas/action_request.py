# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic request/response schemas for Action Request endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ActionRequestContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    student_model_id: str | None = None
    example_keys: list[str] | None = None
    error_ref: str | None = None
    tao_job_id: str | None = None


class ActionRequestGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_type: str
    context: ActionRequestContext | None = None


class ActionRequestGenerateResponse(BaseModel):
    request_type: str
    generated_at: str
    project_name: str
    technical_requirements: dict[str, Any]
    current_environment: dict[str, Any]
    rendered_text: str


class ActionRequestLogCopyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_type: str
    rendered_text: str


class ActionRequestLogCopyResponse(BaseModel):
    audit_event_id: str
