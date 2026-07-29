# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the Student NIM deployment endpoint.

POST ``/v1/projects/{project_id}/student_models/{student_model_id}:deploy_nim``
returns 202 with these shapes; the lifecycle runs in the background and
emits SSE events (``nim_benchmark_progress`` / ``nim_benchmark_completed``
/ ``run_failed``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class DeployNimRequest(BaseModel):
    """Request body for ``:deploy_nim``.

    ``nim_endpoint_url`` is the mode discriminator: ``None`` means local
    Docker orchestration; a non-empty URL skips local entirely and
    registers the supplied endpoint as a permanent Student endpoint.
    """

    model_config = ConfigDict(extra="forbid")

    nim_endpoint_url: str | None = None
    nim_container_image: str | None = None
    nim_release_version: str | None = None
    gpu_assignment: str | None = None
    auth_mode: Literal["none", "bearer"] = "none"


class DeployNimResponse(BaseModel):
    """202 response shape.

    Fields:

      - ``student_model_id``: echoed for client UX.
      - ``nim_deployment_mode``: ``"local"`` or ``"external"``.
      - ``serving_status``: snapshot at dispatch time. Always ``"pending"``
        on a fresh attempt; lifecycle flips to ``"validated"`` or
        ``"failed"`` later, surfaced via SSE + the StudentModel detail
        endpoint.
      - ``task_id``: background_manager task id; clients can correlate
        SSE events using ``student_model_id`` (the task id is also
        rendered for support / debugging).
      - ``created_at``: ISO 8601 UTC timestamp.

    Preflight runs asynchronously in the lifecycle; its outcome is
    surfaced on the StudentModel record (``nim_preflight_status``), not
    in this dispatch snapshot.
    """

    student_model_id: str
    nim_deployment_mode: Literal["local", "external"]
    serving_status: Literal["pending"]
    task_id: str
    created_at: str
