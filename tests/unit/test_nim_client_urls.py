# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``nim_client`` URL construction.

The seeded ``NimEndpoint.base_url`` (``https://integrate.api.nvidia.com/v1``)
already carries ``/v1``, so a client that hardcodes ``/v1/chat/completions``
or ``/v1/embeddings`` double-prefixes every hosted invocation to
``.../v1/v1/...`` and 404s. Tests that mock ``resilient_request`` without
inspecting the outgoing URL cannot catch that failure.

These tests pin the URL contract so the bug cannot re-emerge silently.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from vlm_feedback_loop.services.http_client import HttpResult
from vlm_feedback_loop.services.nim_client import (
    chat_completions,
    create_embeddings,
    list_models,
)


def _captured_url(mock_req: AsyncMock) -> str:
    """Pull the URL positional argument from the last resilient_request call."""
    assert mock_req.await_count == 1
    call = mock_req.await_args_list[0]
    # resilient_request(method, url, ...) — url is positional arg 1.
    return call.args[1]


class TestChatCompletionsUrl:
    @pytest.mark.asyncio
    async def test_hosted_nim_appends_chat_completions_without_duplicating_v1(
        self,
    ) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                error_class=None,
            )
            await chat_completions(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                model="nvidia/cosmos-reason2-8b",
                messages=[{"role": "user", "content": "hi"}],
                deadline_s=10.0,
            )

        url = _captured_url(mock_req)
        assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert "/v1/v1" not in url

    @pytest.mark.asyncio
    async def test_local_nim_also_composes_correctly(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                error_class=None,
            )
            await chat_completions(
                base_url="http://localhost:8000/v1",
                auth_headers={},
                model="nvidia/cosmos-reason2-2b",
                messages=[{"role": "user", "content": "hi"}],
                deadline_s=10.0,
            )

        assert _captured_url(mock_req) == "http://localhost:8000/v1/chat/completions"

    @pytest.mark.asyncio
    async def test_trailing_slash_on_base_url_is_normalized(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                error_class=None,
            )
            await chat_completions(
                base_url="https://integrate.api.nvidia.com/v1/",
                auth_headers={},
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                deadline_s=10.0,
            )

        assert (
            _captured_url(mock_req)
            == "https://integrate.api.nvidia.com/v1/chat/completions"
        )


class TestCreateEmbeddingsUrl:
    @pytest.mark.asyncio
    async def test_hosted_nvclip_appends_embeddings_without_duplicating_v1(
        self,
    ) -> None:
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

        url = _captured_url(mock_req)
        assert url == "https://integrate.api.nvidia.com/v1/embeddings"
        assert "/v1/v1" not in url


class TestListModelsUrl:
    @pytest.mark.asyncio
    async def test_default_models_path_is_slash_models(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "some-model"}]},
                error_class=None,
            )
            await list_models(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={"Authorization": "Bearer test"},
                deadline_s=10.0,
            )

        url = _captured_url(mock_req)
        assert url == "https://integrate.api.nvidia.com/v1/models"
        assert "/v1/v1" not in url

    @pytest.mark.asyncio
    async def test_caller_provided_models_path_is_honored(self) -> None:
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
                base_url="https://custom.example/ai",
                auth_headers={},
                deadline_s=10.0,
                models_path="/catalog/v2/models",
            )

        assert _captured_url(mock_req) == "https://custom.example/ai/catalog/v2/models"


class TestRequestBodyModelField:
    """The `model` field of the request body is sent verbatim (no rewriting)."""

    @pytest.mark.asyncio
    async def test_chat_completions_passes_model_name_through(self) -> None:
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
                },
                error_class=None,
            )
            await chat_completions(
                base_url="https://integrate.api.nvidia.com/v1",
                auth_headers={},
                model="mistralai/mistral-large-3-675b-instruct-2512",
                messages=[{"role": "user", "content": "hi"}],
                deadline_s=10.0,
            )

        json_body: dict[str, Any] = mock_req.await_args_list[0].kwargs["json_body"]
        assert json_body["model"] == "mistralai/mistral-large-3-675b-instruct-2512"
