# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NimEndpoint probe semantics (services/nim_endpoint_service.py).

The router suite (test_nim_router.py) covers CRUD plumbing with an
always-healthy probe. These tests pin the probe failure taxonomy, the
credential resolution for probes, and the re-probe-on-update rules that
the endpoint health display depends on. HTTP is mocked at the
``resilient_request`` boundary so the real ``nim_client.list_models``
mapping runs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_settings, open_project_workspace
from vlm_feedback_loop.services import runtime_secrets
from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.nim_endpoint_service import (
    create_nim_endpoint,
    get_nim_endpoint,
    update_nim_endpoint,
)

PID = "nim-ep-proj"

HEALTHY_HTTP = HttpResult(
    status_code=200,
    body={"data": [{"id": "model-a"}]},
    error_class=None,
    attempts=1,
)
TIMEOUT_HTTP = HttpResult(
    status_code=None,
    body=None,
    error_class="timeout",
    error_detail="Request timed out",
    attempts=1,
)


def _http_error(status: int) -> HttpResult:
    return HttpResult(
        status_code=status,
        body=None,
        error_class="endpoint_error",
        error_detail=f"HTTP {status}",
        attempts=1,
    )


def _patch_http(result: HttpResult):
    return patch(
        "vlm_feedback_loop.services.nim_client.resilient_request",
        new_callable=AsyncMock,
        return_value=result,
    )


def _bearer_endpoint_data(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "display_name": "Hosted",
        "endpoint_mode": "hosted",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "auth_mode": "bearer",
    }
    data.update(overrides)
    return data


@pytest.fixture(autouse=True)
def _clean_runtime_secrets(monkeypatch):
    """Keep UI-applied secret overrides from bleeding into credential tests."""
    monkeypatch.setattr(runtime_secrets, "_runtime_overrides", {})


@pytest.fixture
def workspace(tmp_path):
    _, _, workspace = open_project_workspace(tmp_path, PID, register_engine=True)
    return workspace


class TestProbeFailureTaxonomy:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("http_result", "expected_status"),
        [
            (_http_error(401), "auth_failed"),
            (_http_error(403), "auth_failed"),
            (_http_error(500), "unhealthy"),
            (TIMEOUT_HTTP, "unreachable"),
        ],
        ids=["401", "403", "500", "no-http-response"],
    )
    async def test_create_persists_endpoint_with_classified_probe_failure(
        self, workspace, http_result, expected_status
    ):
        """A failed auto-probe never blocks endpoint creation: the record
        persists with the failure classified (401/403 → auth_failed, other
        HTTP error → unhealthy, no HTTP response at all → unreachable) so
        the health display can say WHY the endpoint is down."""
        settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test-123")

        with _patch_http(http_result):
            created = await create_nim_endpoint(
                PID, _bearer_endpoint_data(), str(workspace), settings
            )

        assert created is not None
        assert created.last_probe_status == expected_status
        assert created.last_probe_error_ref  # human-readable reason retained
        assert created.last_probe_at is not None

        stored = get_nim_endpoint(PID, created.endpoint_id, str(workspace))
        assert stored is not None
        assert stored.last_probe_status == expected_status


class TestProbeCredentials:
    @pytest.mark.asyncio
    async def test_bearer_probe_carries_deployment_key(self, workspace):
        """Bearer endpoints probe with the deployment-level NVIDIA_API_KEY
        (the endpoint record never stores a key value); a probe that
        silently ran unauthenticated would misreport healthy endpoints as
        auth_failed."""
        settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test-123")

        with _patch_http(HEALTHY_HTTP) as mock_req:
            created = await create_nim_endpoint(
                PID, _bearer_endpoint_data(), str(workspace), settings
            )

        assert created is not None
        assert created.last_probe_status == "healthy"
        headers = mock_req.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer nvapi-test-123"

    @pytest.mark.asyncio
    async def test_bearer_without_key_records_unknown_without_probing(self, workspace):
        """Creating a bearer endpoint before any NVIDIA_API_KEY is
        configured (the fresh-install case) must not crash the create or
        hit the network: the record lands with probe status "unknown" and
        an error ref pointing at the missing credential."""
        settings = make_settings(workspace)  # no NVIDIA_API_KEY

        with _patch_http(HEALTHY_HTTP) as mock_req:
            created = await create_nim_endpoint(
                PID, _bearer_endpoint_data(), str(workspace), settings
            )

        assert created is not None
        assert created.last_probe_status == "unknown"
        assert "credential" in (created.last_probe_error_ref or "")
        mock_req.assert_not_called()


class TestUpdateReprobeRules:
    async def _create_healthy(self, workspace, settings):
        with _patch_http(HEALTHY_HTTP):
            created = await create_nim_endpoint(
                PID, _bearer_endpoint_data(), str(workspace), settings
            )
        assert created is not None
        assert created.last_probe_status == "healthy"
        return created

    @pytest.mark.asyncio
    async def test_rename_and_echoed_base_url_do_not_reprobe(self, workspace):
        """Updating non-probe fields — including echoing back the unchanged
        base_url — must not re-probe: a rename can't churn health state or
        add probe latency to every save."""
        settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test-123")
        created = await self._create_healthy(workspace, settings)

        with _patch_http(TIMEOUT_HTTP) as mock_req:
            updated = await update_nim_endpoint(
                PID,
                created.endpoint_id,
                {"display_name": "Renamed", "base_url": created.base_url},
                str(workspace),
                settings,
            )

        mock_req.assert_not_called()
        assert updated is not None
        assert updated.display_name == "Renamed"
        assert updated.last_probe_status == "healthy"
        assert updated.last_probe_at == created.last_probe_at

    @pytest.mark.asyncio
    async def test_base_url_change_reprobes_and_overwrites_status(self, workspace):
        """Changing base_url re-probes immediately and the stored health
        state reflects the NEW address — a stale "healthy" from the old URL
        must not survive the move."""
        settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test-123")
        created = await self._create_healthy(workspace, settings)

        with _patch_http(TIMEOUT_HTTP) as mock_req:
            updated = await update_nim_endpoint(
                PID,
                created.endpoint_id,
                {"base_url": "http://moved:8000/v1"},
                str(workspace),
                settings,
            )

        assert mock_req.call_count == 1
        assert "http://moved:8000/v1" in mock_req.call_args.args[1]
        assert updated is not None
        assert updated.base_url == "http://moved:8000/v1"
        assert updated.last_probe_status == "unreachable"

        stored = get_nim_endpoint(PID, created.endpoint_id, str(workspace))
        assert stored is not None
        assert stored.last_probe_status == "unreachable"
