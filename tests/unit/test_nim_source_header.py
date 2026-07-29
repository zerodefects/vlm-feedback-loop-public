# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the NIM source header ships on every outbound call to NVIDIA infrastructure.

Spec reference: CLAUDE.md "Blueprint Patterns → Backend Patterns → HTTP client":
*"All outbound NIM requests MUST include a `{"source": "vlm-feedback-loop"}`
default header for API usage tracking, following the NVIDIA Blueprint
convention (e.g., RAG Blueprint uses `{"source": "rag-blueprint"}`)."*

Covers all five paths that send traffic to NVIDIA infrastructure:
  1. nim_client.list_models          — GET  /v1/models
  2. nim_client.chat_completions     — POST /v1/chat/completions
  3. nim_client.create_embeddings    — POST /v1/embeddings
  4. tao_client.probe_tao_connection — GET  TAO FTMS /orgs/{org}/jobs
  5. tao_client.login_tao            — POST TAO FTMS /login

Each test captures the outgoing request headers via a mocked
``resilient_request`` and asserts ``headers["source"] == "vlm-feedback-loop"``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_settings
from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.nim_client import (
    chat_completions,
    create_embeddings,
    list_models,
)

EXPECTED_SOURCE = "vlm-feedback-loop"


def _capture_request_headers(
    mock_resilient_request: AsyncMock,
) -> dict[str, str]:
    """Extract the ``headers`` kwarg from the most recent call."""
    assert mock_resilient_request.await_count == 1, (
        f"expected 1 resilient_request call, got {mock_resilient_request.await_count}"
    )
    _, kwargs = mock_resilient_request.await_args
    headers = kwargs.get("headers") or {}
    return dict(headers)


# ── NIM client layer ────────────────────────────────────────────────────────


class TestNimClientSourceHeader:
    """Every nim_client operation must ship the source header."""

    @pytest.mark.asyncio
    async def test_list_models_sends_source_header(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "m1"}]},
                error_class=None,
            )
            await list_models(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                deadline_s=10.0,
            )

        headers = _capture_request_headers(mock_req)
        assert headers.get("source") == EXPECTED_SOURCE
        # Auth still intact — source header doesn't clobber existing headers
        assert headers.get("Authorization") == "Bearer test"

    @pytest.mark.asyncio
    async def test_chat_completions_sends_source_header(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                error_class=None,
            )
            await chat_completions(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                model="cosmos-reason-2-8b",
                messages=[{"role": "user", "content": "hi"}],
                deadline_s=10.0,
            )

        headers = _capture_request_headers(mock_req)
        assert headers.get("source") == EXPECTED_SOURCE
        assert headers.get("Authorization") == "Bearer test"

    @pytest.mark.asyncio
    async def test_create_embeddings_sends_source_header(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "data": [{"embedding": [0.1, 0.2], "index": 0}],
                    "model": "nvidia/llama-nemotron-embed-vl-1b-v2",
                },
                error_class=None,
            )
            await create_embeddings(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                model="nvidia/llama-nemotron-embed-vl-1b-v2",
                input_items=["probe"],
                deadline_s=10.0,
            )

        headers = _capture_request_headers(mock_req)
        assert headers.get("source") == EXPECTED_SOURCE
        assert headers.get("Authorization") == "Bearer test"

    @pytest.mark.asyncio
    async def test_nim_client_with_empty_auth_still_has_source(self) -> None:
        """No auth (local NIM, auth_mode=none) still gets the source header."""
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": []},
                error_class=None,
            )
            await list_models(
                base_url="http://localhost:8000/v1",
                auth_headers={},
                deadline_s=5.0,
            )

        headers = _capture_request_headers(mock_req)
        assert headers.get("source") == EXPECTED_SOURCE


# ── TAO client ──────────────────────────────────────────────────────────────


class TestTaoClientSourceHeader:
    """TAO FTMS probe and login both ship the source header."""

    @pytest.mark.asyncio
    async def test_probe_tao_connection_sends_source_header(self) -> None:
        from vlm_feedback_loop.services.tao_client import probe_tao_connection

        settings = make_settings(
            "/tmp/ws",
            TAO_API_BASE_URL="https://tao-test/api/v2",
            TAO_API_KEY="test-jwt",
            TAO_ORG_NAME="test-org",
        )

        with patch(
            "vlm_feedback_loop.services.tao_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"jobs": []},
                error_class=None,
            )
            await probe_tao_connection(settings)

        assert mock_req.await_count == 2
        for call in mock_req.await_args_list:
            headers = call.kwargs["headers"]
            assert headers.get("source") == EXPECTED_SOURCE
            # Both the jobs probe and OpenAPI compatibility check carry the
            # resolved Bearer JWT.
            assert headers.get("Authorization") == "Bearer test-jwt"

    @pytest.mark.asyncio
    async def test_login_tao_sends_source_header(self) -> None:
        from vlm_feedback_loop.services.tao_client import login_tao

        with patch(
            "vlm_feedback_loop.services.tao_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"token": "jwt-xyz"},
                error_class=None,
            )
            await login_tao(
                tao_api_base_url="https://tao-test/api/v2",
                ngc_api_key="ngc-key",
                org_name="test-org",
            )

        headers = _capture_request_headers(mock_req)
        assert headers.get("source") == EXPECTED_SOURCE
