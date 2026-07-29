# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TAO FTMS client (services/tao_client.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from conftest import make_settings, make_tao_settings
from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.tao_client import login_tao, probe_tao_connection

# ── Fixture ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tao_settings(tmp_path):
    """Settings with TAO credentials configured."""
    return make_tao_settings(
        tmp_path / "workspace",
        TAO_API_BASE_URL="http://tao-host:8090/api/v2",
        TAO_API_KEY="jwt-token-abc123",
    )


@pytest.fixture()
def empty_tao_settings(tmp_path):
    """Settings with no TAO credentials."""
    return make_settings(tmp_path / "workspace")


# ── probe_tao_connection ───────────────────────────────────────────────────


class TestProbeTaoConnection:
    @pytest.mark.asyncio
    async def test_success_with_valid_credentials(self, tao_settings, monkeypatch):
        jobs_result = HttpResult(
            status_code=200,
            body={"jobs": []},
            error_class=None,
            attempts=1,
        )
        openapi_result = HttpResult(
            status_code=200,
            body={
                "components": {
                    "schemas": {
                        "ExperimentJobReq": {
                            "properties": {"timeout_minutes": {"type": "integer"}}
                        }
                    }
                }
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(side_effect=[jobs_result, openapi_result])
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        result = await probe_tao_connection(tao_settings)
        assert result["success"] is True
        assert result["error"] is None
        assert result["status_code"] == 200
        assert result["job_timeout_supported"] is True
        assert result["job_timeout_error"] is None

        # Verify both the authenticated reachability probe and non-mutating
        # timeout compatibility check.
        assert "/orgs/example-org/jobs" in mock_fn.call_args_list[0].args[1]
        assert mock_fn.call_args_list[1].args[1].endswith("/openapi.json")

    @pytest.mark.asyncio
    async def test_success_reports_unpatched_timeout_schema(
        self, tao_settings, monkeypatch
    ):
        jobs_result = HttpResult(
            status_code=200,
            body={"jobs": []},
            error_class=None,
            attempts=1,
        )
        openapi_result = HttpResult(
            status_code=200,
            body={
                "components": {
                    "schemas": {"ExperimentJobReq": {"properties": {"kind": {}}}}
                }
            },
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request",
            AsyncMock(side_effect=[jobs_result, openapi_result]),
        )

        result = await probe_tao_connection(tao_settings)

        assert result["success"] is True
        assert result["job_timeout_supported"] is False
        assert "timeout_minutes" in result["job_timeout_error"]

    @pytest.mark.asyncio
    async def test_auth_failure(self, tao_settings, monkeypatch):
        mock_result = HttpResult(
            status_code=401,
            body={"error": "unauthorized"},
            error_class="endpoint_error",
            error_detail="HTTP 401",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        result = await probe_tao_connection(tao_settings)
        assert result["success"] is False
        assert "Could not connect to TAO" in result["error"]
        assert result["status_code"] == 401

    @pytest.mark.asyncio
    async def test_unreachable(self, tao_settings, monkeypatch):
        mock_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        result = await probe_tao_connection(tao_settings)
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_base_url_returns_error(self, empty_tao_settings):
        result = await probe_tao_connection(empty_tao_settings)
        assert result["success"] is False
        assert "TAO_API_BASE_URL" in result["error"]
        assert result["status_code"] is None

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_error(self, tmp_path):
        settings = make_settings(
            tmp_path,
            TAO_API_BASE_URL="http://host:8090/api/v2",
            TAO_ORG_NAME="org",
        )
        result = await probe_tao_connection(settings)
        assert result["success"] is False
        assert "TAO_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_api_key_not_in_result(self, tao_settings, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"jobs": []},
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )

        result = await probe_tao_connection(tao_settings)
        result_str = str(result)
        assert "jwt-token-abc123" not in result_str
        assert "nvapi-" not in result_str

    @pytest.mark.asyncio
    async def test_uses_bearer_auth(self, tao_settings, monkeypatch):
        mock_fn = AsyncMock(
            return_value=HttpResult(
                status_code=200,
                body={"jobs": []},
                error_class=None,
                attempts=1,
            )
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        await probe_tao_connection(tao_settings)
        headers = mock_fn.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer jwt-token-abc123"


# ── login_tao ───────────────────────────────────────────────────────────────


class TestLoginTao:
    @pytest.mark.asyncio
    async def test_successful_login_returns_token(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"token": "eyJhbGci.jwt.payload", "user_id": "abc-123"},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        result = await login_tao(
            "http://tao-host:8090/api/v2", "ngc-personal-key", "example-org", 30.0
        )
        assert result["success"] is True
        assert result["token"] == "eyJhbGci.jwt.payload"
        assert result["error"] is None

        # Verify login endpoint called with correct body
        url = mock_fn.call_args.args[1]
        assert url.endswith("/login")
        body = mock_fn.call_args.kwargs["json_body"]
        assert body["ngc_key"] == "ngc-personal-key"
        assert body["ngc_org_name"] == "example-org"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            status_code=401,
            body={"error_desc": "Unauthorized", "error_code": 1},
            error_class="endpoint_error",
            error_detail="HTTP 401",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.resilient_request", mock_fn
        )

        result = await login_tao(
            "http://tao-host:8090/api/v2", "bad-key", "example-org", 30.0
        )
        assert result["success"] is False
        assert result["token"] is None
        assert "failed" in result["error"].lower()
