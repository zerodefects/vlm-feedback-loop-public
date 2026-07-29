# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the operator-run live smoke scripts."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


@dataclass
class StageResult:
    """One named smoke stage and its result."""

    name: str
    ok: bool
    detail: str = ""
    duration_s: float = 0.0


async def wait_backend(
    client: httpx.AsyncClient,
    deadline_s: float = 30.0,
    *,
    base_url: str = DEFAULT_BACKEND_URL,
) -> None:
    """Poll the health endpoint until the backend is ready."""
    end = time.monotonic() + deadline_s
    last_error = ""
    while time.monotonic() < end:
        try:
            response = await client.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            last_error = type(exc).__name__
        await asyncio.sleep(0.5)
    raise RuntimeError(f"Backend not ready at {base_url}: {last_error}")


async def resolve_teacher_model_config_id(
    client: httpx.AsyncClient,
    project_id: str,
    model_name: str,
    *,
    base_url: str = DEFAULT_BACKEND_URL,
) -> str | None:
    """Return the seeded model-config ID for a Teacher model name."""
    response = await client.get(f"{base_url}/v1/projects/{project_id}/model_configs")
    if response.status_code != 200:
        return None
    for item in response.json().get("items", []):
        if item.get("model_name") == model_name:
            return item.get("model_config_id") or item.get("id")
    return None
