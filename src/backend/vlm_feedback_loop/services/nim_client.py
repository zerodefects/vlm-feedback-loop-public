# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIM API client layer — typed operations on top of resilient_request().

Three operations covering the NIM OpenAI-compatible surface:
  1. list_models     — GET {base_url}{models_path} (connection test)
  2. chat_completions — POST {base_url}/chat/completions (inference dispatch)
  3. create_embeddings — POST {base_url}/embeddings (embedding computation)

``base_url`` is assumed to already carry the version prefix (e.g. ``/v1``)
matching the NVIDIA Blueprint seeded-endpoint convention: hosted NIM seeds
``https://integrate.api.nvidia.com/v1`` as the base, with path suffixes like
``/chat/completions`` — not ``/v1/chat/completions`` — appended below.

All operations delegate to ``resilient_request()`` for deadline enforcement,
bounded retries, and error classification. Auth injection is handled by
``build_auth_headers()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from vlm_feedback_loop.services.http_client import HttpResult, resilient_request

# Default header identifying this Blueprint as the request source.
# Follows the NVIDIA Blueprint convention (e.g., RAG Blueprint uses
# ``{"source": "rag-blueprint"}``).  Attached to every outbound NIM request.
NIM_DEFAULT_HEADERS: dict[str, str] = {"source": "vlm-feedback-loop"}


# ── Auth ────────────────────────────────────────────────────────────────────


def build_auth_headers(
    auth_mode: str,
    credential: str | None = None,
) -> dict[str, str]:
    """Build HTTP auth headers for a NIM endpoint.

    Args:
        auth_mode: ``"bearer"`` | ``"none"``.
        credential: Secret value (API key / token).

    Returns:
        Dict of header name → value.  Empty when ``auth_mode="none"``
        (self-hosted / local NIMs run on a trusted network, no auth).

    Raises:
        ValueError: when ``bearer`` is missing its credential, or the mode
            is unknown.
    """
    if auth_mode == "none":
        return {}

    if auth_mode == "bearer":
        if not credential:
            raise ValueError("bearer auth_mode requires a credential")
        return {"Authorization": f"Bearer {credential}"}

    raise ValueError(f"Unknown auth_mode: {auth_mode!r}")


def build_endpoint_auth_headers(
    auth_mode: str | None,
    credential: str | None,
) -> dict[str, str]:
    """Auth headers for an outbound NIM call, tolerant of a missing credential.

    The single builder for every service-side NIM dispatch (interactive
    proposal, evaluation, batch labeling, rationale regeneration). Unlike
    :func:`build_auth_headers` it returns ``{}`` when no credential is
    configured instead of raising.

    ``auth_mode == "none"`` (self-hosted / local system-managed NIMs)
    returns ``{}`` even when a credential is incidentally present, so a
    configured ``NVIDIA_API_KEY`` is never leaked as a Bearer header to a
    no-auth endpoint. Only ``bearer`` (hosted NVIDIA endpoints) attaches an
    ``Authorization`` header.
    """
    if auth_mode == "none":
        return {}
    if not credential:
        return {}
    return {"Authorization": f"Bearer {credential}"}


# ── list_models ─────────────────────────────────────────────────────────────


@dataclass
class NimListModelsResult:
    """Result of a GET /v1/models call."""

    success: bool
    models: list[str] | None = None
    error: str | None = None
    status_code: int | None = None


async def list_models(
    base_url: str,
    auth_headers: dict[str, str],
    deadline_s: float,
    models_path: str = "/models",
    max_retries: int = 3,
) -> NimListModelsResult:
    """Query the NIM model list endpoint.

    Parses the OpenAI-compatible ``{"data": [{"id": "..."}]}`` response.
    ``models_path`` defaults to ``/models`` because ``base_url`` already
    carries the version prefix (``/v1``) per the seeded-endpoint convention.
    """
    url = f"{base_url.rstrip('/')}{models_path}"
    headers = {**NIM_DEFAULT_HEADERS, **auth_headers}
    result: HttpResult = await resilient_request(
        "GET",
        url,
        deadline_s=deadline_s,
        max_retries=max_retries,
        headers=headers,
    )

    if result.error_class == "timeout":
        return NimListModelsResult(
            success=False,
            error="Connection timed out",
            status_code=result.status_code,
        )

    if result.error_class == "endpoint_error":
        status = result.status_code
        if status in (401, 403):
            return NimListModelsResult(
                success=False,
                error="Authentication failed — check your API key",
                status_code=status,
            )
        return NimListModelsResult(
            success=False,
            error=f"Endpoint error: {result.error_detail}",
            status_code=status,
        )

    # Success — parse model list
    body: Any = result.body
    if isinstance(body, dict) and "data" in body:
        body_dict = cast("dict[str, Any]", body)
        data_list = cast("list[Any]", body_dict.get("data") or [])
        model_ids = [
            cast("dict[str, Any]", entry).get("id", "")
            for entry in data_list
            if isinstance(entry, dict)
        ]
        return NimListModelsResult(
            success=True,
            models=model_ids,
            status_code=result.status_code,
        )

    # Unexpected response shape
    return NimListModelsResult(
        success=False,
        error="Unexpected response format from /models",
        status_code=result.status_code,
    )


# ── chat_completions ────────────────────────────────────────────────────────


@dataclass
class NimChatCompletionsResult:
    """Result of a POST {base_url}/chat/completions call."""

    success: bool
    content: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None
    status_code: int | None = None


async def chat_completions(
    base_url: str,
    auth_headers: dict[str, str],
    model: str,
    messages: list[dict[str, Any]],
    deadline_s: float,
    max_retries: int = 3,
    **kwargs: Any,
) -> NimChatCompletionsResult:
    """Send a chat completions request to a NIM endpoint.

    ``**kwargs`` passes through extra fields (temperature, max_tokens,
    response_format, chat_template_kwargs, mm_processor_kwargs, seed, etc.)
    directly into the request body — needed by capability probes and
    Teacher invocation.
    """
    # ``base_url`` already includes the API root (e.g. ``.../v1``) by contract,
    # so we append only the path segment — not ``/v1/chat/completions``.
    url = f"{base_url.rstrip('/')}/chat/completions"

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    body.update(kwargs)

    headers = {**NIM_DEFAULT_HEADERS, **auth_headers}
    result: HttpResult = await resilient_request(
        "POST",
        url,
        deadline_s=deadline_s,
        max_retries=max_retries,
        headers=headers,
        json_body=body,
    )

    if result.error_class == "timeout":
        return NimChatCompletionsResult(
            success=False,
            error="Request timed out",
            status_code=result.status_code,
        )

    if result.error_class == "endpoint_error":
        return NimChatCompletionsResult(
            success=False,
            error=f"Endpoint error: {result.error_detail}",
            status_code=result.status_code,
        )

    # Success — parse response
    resp: Any = result.body
    if not isinstance(resp, dict):
        return NimChatCompletionsResult(
            success=False,
            error="Unexpected response format",
            status_code=result.status_code,
        )
    resp_dict = cast("dict[str, Any]", resp)

    choices_raw: Any = resp_dict.get("choices") or []
    if not choices_raw:
        return NimChatCompletionsResult(
            success=False,
            error="No choices in response",
            status_code=result.status_code,
        )
    choices = cast("list[dict[str, Any]]", choices_raw)

    choice: dict[str, Any] = choices[0]
    message_raw: Any = choice.get("message") or {}
    message: dict[str, Any] = (
        cast("dict[str, Any]", message_raw) if isinstance(message_raw, dict) else {}
    )
    content: Any = message.get("content")
    finish_reason: Any = choice.get("finish_reason")
    usage: Any = resp_dict.get("usage")

    return NimChatCompletionsResult(
        success=True,
        content=content,
        finish_reason=finish_reason,
        usage=usage,
        status_code=result.status_code,
    )


# ── create_embeddings ───────────────────────────────────────────────────────


@dataclass
class NimEmbeddingsResult:
    """Result of a POST {base_url}/embeddings call."""

    success: bool
    embeddings: list[list[float]] | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    error: str | None = None
    status_code: int | None = None


async def create_embeddings(
    base_url: str,
    auth_headers: dict[str, str],
    model: str,
    input_items: list[str],
    deadline_s: float,
    max_retries: int = 3,
    input_type: str | None = None,
) -> NimEmbeddingsResult:
    """Send an embeddings request to a NIM endpoint.

    ``input_items`` are strings — either text or base64 data URLs
    (``data:image/<format>;base64,<data>``).  The endpoint returns
    embeddings indexed by input order.

    ``input_type`` is an optional asymmetric-retrieval signal required by
    NeMo Retriever embedding models (typical values: ``"query"`` /
    ``"passage"``).  Symmetric CLIP-style models (NV-CLIP) do not accept
    this field — pass ``None`` (the default) to omit it.
    """
    # ``base_url`` already includes the API root (e.g. ``.../v1``) by contract.
    url = f"{base_url.rstrip('/')}/embeddings"

    body: dict[str, Any] = {
        "model": model,
        "input": input_items,
    }
    if input_type is not None:
        body["input_type"] = input_type

    headers = {**NIM_DEFAULT_HEADERS, **auth_headers}
    result: HttpResult = await resilient_request(
        "POST",
        url,
        deadline_s=deadline_s,
        max_retries=max_retries,
        headers=headers,
        json_body=body,
    )

    if result.error_class == "timeout":
        return NimEmbeddingsResult(
            success=False,
            error="Request timed out",
            status_code=result.status_code,
        )

    if result.error_class == "endpoint_error":
        return NimEmbeddingsResult(
            success=False,
            error=f"Endpoint error: {result.error_detail}",
            status_code=result.status_code,
        )

    # Success — parse and reorder by index
    resp: Any = result.body
    if not isinstance(resp, dict) or "data" not in resp:
        return NimEmbeddingsResult(
            success=False,
            error="Unexpected response format from /embeddings",
            status_code=result.status_code,
        )

    data_items = cast("list[dict[str, Any]]", resp["data"])
    # Sort by index to guarantee input-order alignment
    sorted_items = sorted(data_items, key=lambda x: x.get("index", 0))
    embeddings: list[list[float]] = [
        cast("list[float]", item["embedding"]) for item in sorted_items
    ]

    resp_dict = cast("dict[str, Any]", resp)
    return NimEmbeddingsResult(
        success=True,
        embeddings=embeddings,
        model=resp_dict.get("model"),
        usage=resp_dict.get("usage"),
        status_code=result.status_code,
    )
