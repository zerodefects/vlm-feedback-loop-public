# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test-only endpoints for emitting synthetic SSE events.

Used by SSE acceptance tests to drive events without a real background
task producer.  The endpoint calls ``sse_manager.emit()`` directly
so that connected EventSource clients receive the event through the
normal SSE infrastructure.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from vlm_feedback_loop.routers.projects import require_project
from vlm_feedback_loop.schemas.testing import (
    TestEventEmitRequest,
    TestEventEmitResponse,
)
from vlm_feedback_loop.services.sse import sse_manager

router = APIRouter(prefix="/testing", tags=["testing"])


@router.post(
    "/projects/{project_id}/events:emit",
    dependencies=[Depends(require_project)],
)
async def emit_test_event(
    project_id: str,
    body: TestEventEmitRequest,
) -> TestEventEmitResponse:
    """Emit a synthetic SSE event for a project (test harness)."""
    await sse_manager.emit(
        project_id=project_id,
        event_type=body.event_type,
        data=body.data,
    )
    return TestEventEmitResponse()
