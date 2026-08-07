# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the NIM API client layer (services/nim_client.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.nim_client import (
    build_auth_headers,
    build_endpoint_auth_headers,
    chat_completions,
    create_embeddings,
    list_models,
)

# ── build_auth_headers ──────────────────────────────────────────────────────


class TestBuildAuthHeaders:
    def test_bearer_mode(self):
        headers = build_auth_headers("bearer", credential="nvapi-test-key")
        assert headers == {"Authorization": "Bearer nvapi-test-key"}

    def test_none_mode(self):
        # Self-hosted / local endpoints run on a trusted network — no auth.
        headers = build_auth_headers("none")
        assert headers == {}

    def test_bearer_missing_credential_raises(self):
        with pytest.raises(ValueError, match="bearer auth_mode requires a credential"):
            build_auth_headers("bearer")

    def test_bearer_empty_credential_raises(self):
        with pytest.raises(ValueError, match="bearer auth_mode requires a credential"):
            build_auth_headers("bearer", credential="")

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown auth_mode"):
            build_auth_headers("oauth2")


class TestBuildEndpointAuthHeaders:
    """The service-side auth resolver used by every outbound NIM path."""

    def test_bearer(self):
        assert build_endpoint_auth_headers("bearer", "k") == {
            "Authorization": "Bearer k"
        }

    def test_none_returns_empty_even_with_credential(self):
        # Self-hosted / local endpoints never get a Bearer header, so a
        # configured NVIDIA_API_KEY is never leaked to a no-auth endpoint.
        assert build_endpoint_auth_headers("none", "k") == {}

    def test_missing_bearer_credential_blocks_dispatch(self):
        # None is distinct from a legitimate no-auth {} endpoint and is
        # consumed by chat_completions without opening a network connection.
        assert build_endpoint_auth_headers("bearer", None) is None
        assert build_endpoint_auth_headers("bearer", "") is None

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown auth_mode"):
            build_endpoint_auth_headers("oauth2", "token")


# ── list_models ─────────────────────────────────────────────────────────────


class TestListModels:
    @pytest.mark.asyncio
    async def test_success_parses_model_ids(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "data": [
                    {"id": "model-a", "object": "model"},
                    {"id": "model-b", "object": "model"},
                ]
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await list_models("http://localhost:8000/v1", {}, deadline_s=10.0)
        assert result.success is True
        assert result.models == ["model-a", "model-b"]
        assert result.error is None

    @pytest.mark.asyncio
    async def test_auth_failure_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            status_code=401,
            body={"error": "unauthorized"},
            error_class="endpoint_error",
            error_detail="HTTP 401",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await list_models("http://localhost:8000/v1", {}, deadline_s=10.0)
        assert result.success is False
        assert "Authentication failed" in result.error
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await list_models("http://unreachable:8000/v1", {}, deadline_s=5.0)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unexpected_body_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body="not json",
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await list_models("http://localhost:8000/v1", {}, deadline_s=10.0)
        assert result.success is False
        assert "Unexpected response" in result.error

    @pytest.mark.asyncio
    async def test_endpoint_error_returns_detail(self, monkeypatch):
        mock_result = HttpResult(
            status_code=503,
            body=None,
            error_class="endpoint_error",
            error_detail="Exhausted 3 retries. Last: HTTP 503",
            attempts=3,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await list_models("http://localhost:8000/v1", {}, deadline_s=10.0)
        assert result.success is False
        assert "Endpoint error" in result.error

    @pytest.mark.asyncio
    async def test_custom_models_path(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"data": [{"id": "m1"}]},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        await list_models(
            "http://host:8000/v1",
            {},
            deadline_s=10.0,
            models_path="/v2/models",
        )
        call_args = mock_fn.call_args
        assert "/v2/models" in call_args.args[1]


# ── chat_completions ────────────────────────────────────────────────────────


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_missing_bearer_credential_never_dispatches(self, monkeypatch):
        """A hosted endpoint without a key fails locally, before HTTP."""
        mock_fn = AsyncMock()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await chat_completions(
            "https://integrate.api.nvidia.com/v1",
            None,
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            deadline_s=120.0,
        )

        assert result.success is False
        assert "credential is not configured" in (result.error or "")
        mock_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_extracts_fields(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok": true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await chat_completions(
            "http://localhost:8000/v1",
            {"Authorization": "Bearer test"},
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            deadline_s=120.0,
        )
        assert result.success is True
        assert result.content == '{"ok": true}'
        assert result.finish_reason == "stop"
        assert result.usage == {"prompt_tokens": 10, "completion_tokens": 5}

    @pytest.mark.asyncio
    async def test_kwargs_pass_through(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        await chat_completions(
            "http://localhost:8000/v1",
            {},
            model="m",
            messages=[],
            deadline_s=10.0,
            temperature=0.3,
            max_tokens=100,
            response_format={"type": "json_schema"},
            chat_template_kwargs={"enable_thinking": False},
        )
        sent_body = mock_fn.call_args.kwargs["json_body"]
        assert sent_body["temperature"] == 0.3
        assert sent_body["max_tokens"] == 100
        assert sent_body["response_format"] == {"type": "json_schema"}
        assert sent_body["chat_template_kwargs"] == {"enable_thinking": False}

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await chat_completions("http://host:8000/v1", {}, "model", [], 10.0)
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_choices_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"choices": []},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await chat_completions("http://host:8000/v1", {}, "model", [], 10.0)
        assert result.success is False
        assert "No choices" in result.error


# ── create_embeddings ───────────────────────────────────────────────────────


class TestCreateEmbeddings:
    @pytest.mark.asyncio
    async def test_success_reorders_by_index(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={
                "data": [
                    {"index": 1, "embedding": [0.2, 0.3]},
                    {"index": 0, "embedding": [0.1, 0.4]},
                ],
                "model": "nvclip",
                "usage": {"total_tokens": 2},
            },
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await create_embeddings(
            "http://host:8000/v1",
            {},
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            input_items=["img1", "img2"],
            deadline_s=10.0,
        )
        assert result.success is True
        # Should be reordered: index 0 first, index 1 second
        assert result.embeddings == [[0.1, 0.4], [0.2, 0.3]]
        assert result.model == "nvclip"

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            error_class="timeout",
            error_detail="Request timed out",
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await create_embeddings(
            "http://host:8000/v1", {}, "model", ["test"], 10.0
        )
        assert result.success is False
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_unexpected_format_returns_error(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"result": "not embeddings"},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        result = await create_embeddings(
            "http://host:8000/v1", {}, "model", ["test"], 10.0
        )
        assert result.success is False
        assert "Unexpected response" in result.error

    @pytest.mark.asyncio
    async def test_malformed_data_item_returns_error_instead_of_raising(
        self, monkeypatch
    ):
        """A reachable but incompatible NIM remains an actionable probe failure."""
        mock_result = HttpResult(
            status_code=200,
            body={"data": [{"index": 0, "not_embedding": [0.1]}]},
            error_class=None,
            attempts=1,
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            AsyncMock(return_value=mock_result),
        )

        result = await create_embeddings(
            "http://host:8000/v1", {}, "model", ["test"], 10.0
        )

        assert result.success is False
        assert result.error == "Unexpected response format from /embeddings"

    @pytest.mark.asyncio
    async def test_request_body_format(self, monkeypatch):
        mock_result = HttpResult(
            status_code=200,
            body={"data": [{"index": 0, "embedding": [0.1]}], "model": "m"},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        await create_embeddings(
            "http://host:8000/v1",
            {},
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            input_items=["data:image/png;base64,abc123", "text input"],
            deadline_s=10.0,
        )
        sent_body = mock_fn.call_args.kwargs["json_body"]
        assert sent_body["model"] == "nvidia/llama-nemotron-embed-vl-1b-v2"
        assert sent_body["input"] == ["data:image/png;base64,abc123", "text input"]
        # input_type omitted by default (NV-CLIP wire-contract preserved)
        assert "input_type" not in sent_body

    @pytest.mark.asyncio
    async def test_input_type_included_when_set(self, monkeypatch):
        """NeMo Retriever asymmetric models require input_type — verify it
        appears verbatim in the request body when passed."""
        mock_result = HttpResult(
            status_code=200,
            body={"data": [{"index": 0, "embedding": [0.1]}], "model": "m"},
            error_class=None,
            attempts=1,
        )
        mock_fn = AsyncMock(return_value=mock_result)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.nim_client.resilient_request", mock_fn
        )

        await create_embeddings(
            "http://host:8000/v1",
            {},
            model="nvidia/llama-nemotron-embed-vl-1b-v2",
            input_items=["text"],
            deadline_s=10.0,
            input_type="passage",
        )
        sent_body = mock_fn.call_args.kwargs["json_body"]
        assert sent_body["input_type"] == "passage"
