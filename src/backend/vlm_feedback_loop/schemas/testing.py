# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for test-only endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class TestEventEmitRequest(BaseModel):
    """Emit a synthetic SSE event for testing."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    data: dict[str, Any]


class TestEventEmitResponse(BaseModel):
    ok: bool = True
