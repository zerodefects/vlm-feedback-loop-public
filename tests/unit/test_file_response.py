# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Descriptor lifecycle and range behavior for authorized file responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from starlette.responses import FileResponse

from vlm_feedback_loop.routers.file_response import FileDescriptorResponse
from vlm_feedback_loop.services.authorized_file import open_regular_file_beneath


def _scope(
    *, range_header: bytes | None = None, pathsend: bool = False
) -> dict[str, Any]:
    headers = [] if range_header is None else [(b"range", range_header)]
    extensions = {"http.response.pathsend": {}} if pathsend else {}
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/download",
        "raw_path": b"/download",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
        "extensions": extensions,
    }


async def _receive() -> dict[str, Any]:
    return {"type": "http.disconnect"}


def _headers(messages: list[dict[str, Any]]) -> dict[bytes, bytes]:
    start = next(
        message for message in messages if message["type"] == "http.response.start"
    )
    return dict(start["headers"])


def _collector(messages: list[dict[str, Any]]):
    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    return send


@pytest.mark.asyncio
async def test_response_streams_authorized_bytes_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"0123456789")
    opened = open_regular_file_beneath(path, tmp_path)
    response = FileDescriptorResponse(
        opened,
        media_type="application/octet-stream",
        filename=path.name,
    )
    messages: list[dict[str, Any]] = []

    await response(_scope(pathsend=True), _receive, _collector(messages))

    body = b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )
    assert body == b"0123456789"
    assert all(message["type"] != "http.response.pathsend" for message in messages)
    assert opened._closed is True


@pytest.mark.asyncio
async def test_response_supports_exact_byte_range_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"0123456789")
    opened = open_regular_file_beneath(path, tmp_path)
    response = FileDescriptorResponse(
        opened,
        media_type="application/octet-stream",
        filename=path.name,
    )
    messages: list[dict[str, Any]] = []

    await response(_scope(range_header=b"bytes=2-5"), _receive, _collector(messages))

    start = messages[0]
    assert start["status"] == 206
    assert _headers(messages)[b"content-range"] == b"bytes 2-5/10"
    assert (
        b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        == b"2345"
    )
    assert opened._closed is True


@pytest.mark.asyncio
async def test_unsatisfiable_range_returns_416_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"0123456789")
    opened = open_regular_file_beneath(path, tmp_path)
    response = FileDescriptorResponse(
        opened,
        media_type="application/octet-stream",
        filename=path.name,
    )
    messages: list[dict[str, Any]] = []

    await response(
        _scope(range_header=b"bytes=20-30"),
        _receive,
        _collector(messages),
    )

    assert messages[0]["status"] == 416
    assert _headers(messages)[b"content-range"] == b"bytes */10"
    assert opened._closed is True


@pytest.mark.asyncio
async def test_malformed_range_returns_400_and_closes_descriptor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"0123456789")
    opened = open_regular_file_beneath(path, tmp_path)
    response = FileDescriptorResponse(
        opened,
        media_type="application/octet-stream",
        filename=path.name,
    )
    messages: list[dict[str, Any]] = []

    await response(
        _scope(range_header=b"items=0-3"),
        _receive,
        _collector(messages),
    )

    assert messages[0]["status"] == 400
    assert opened._closed is True


@pytest.mark.asyncio
async def test_send_failure_still_closes_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"0123456789")
    opened = open_regular_file_beneath(path, tmp_path)
    response = FileDescriptorResponse(
        opened,
        media_type="application/octet-stream",
        filename=path.name,
    )

    async def failing_send(_message: dict[str, Any]) -> None:
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await response(_scope(), _receive, failing_send)
    assert opened._closed is True


def test_constructor_failure_closes_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"bytes")
    opened = open_regular_file_beneath(path, tmp_path)

    with (
        patch.object(FileResponse, "__init__", side_effect=RuntimeError("bad init")),
        pytest.raises(RuntimeError, match="bad init"),
    ):
        FileDescriptorResponse(
            opened,
            media_type="application/octet-stream",
            filename=path.name,
        )
    assert opened._closed is True
