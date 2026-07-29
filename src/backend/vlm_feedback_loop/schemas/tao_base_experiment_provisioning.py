# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schemas for on-demand TAO Student-base provisioning."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TAOBaseExperimentProvisioningRequest(BaseModel):
    """Selected Student bases to ensure before a training suite starts."""

    model_config = ConfigDict(extra="forbid")

    student_base_model_config_ids: list[str] = Field(..., min_length=1)


class TAOBaseExperimentProvisioningFailure(BaseModel):
    """One target-specific provisioning failure."""

    model_config = ConfigDict(extra="forbid")

    target: str
    error: str


class TAOBaseExperimentProvisioningResponse(BaseModel):
    """Durable status for one background provisioning attempt."""

    model_config = ConfigDict(extra="forbid")

    provisioning_run_id: str
    project_id: str
    requested_model_config_ids: list[str]
    requested_model_names: list[str]
    status: Literal["queued", "running", "succeeded", "failed"]
    registered: list[str]
    already_registered: list[str]
    failures: list[TAOBaseExperimentProvisioningFailure]
    error_ref: str | None
    started_at: str | None
    completed_at: str | None
    created_at: str
    updated_at: str
