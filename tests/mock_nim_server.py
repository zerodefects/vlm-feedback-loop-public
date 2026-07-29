# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process mock NIM server for closing-smoke mock validation.

Implements exactly the surface the mock-mode tests exercise: ``POST
/v1/chat/completions`` returning a schema-valid label JSON (the handoff
re-execution round-trip in ``test_full_stack_validation_mock.py``) and
``POST /v1/embeddings`` returning deterministic vectors (the local
embedding NIM wire tests in ``test_clip_embedding.py``).

The server runs in a background thread on an OS-assigned localhost port.
Tests use it as a context manager::

    with MockNIMServer() as mock:
        monkeypatch.setenv("LOCAL_NIM_MOCK_ENDPOINT_URL", mock.base_url)
        ...

``set_label_payload`` overrides what /v1/chat/completions returns (a dict
is serialized to JSON; a plain string exercises the text-response path).

Implementation uses stdlib ``http.server.ThreadingHTTPServer`` to avoid
adding a new dependency (aiohttp is not in the project's dep tree).
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

# ── Mock state ──────────────────────────────────────────────────────────────


class _MockState:
    """Per-server mutable state, mutated via the MockNIMServer handle."""

    def __init__(self) -> None:
        self.label_payload: dict[str, Any] | str = {
            "rationale_note": "stub mock label",
            "gesture": "rock",
        }
        self.embedding_dim: int = 2048
        self.embeddings_requests: list[dict[str, Any]] = []


# ── Handler ─────────────────────────────────────────────────────────────────


def _make_handler(state: _MockState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        # Keep the test output clean — we don't need access logs.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D401
            return

        def do_POST(self) -> None:  # noqa: N802 — stdlib API
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"null")
            except json.JSONDecodeError:
                payload = None
            if self.path.startswith("/v1/chat/completions"):
                # Echo back a deterministic schema-valid label JSON — the
                # closing-smoke's handoff re-execution round-trip consumes this.
                content = json.dumps(state.label_payload)
                self._json(
                    200,
                    {
                        "id": f"mock-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": (payload or {}).get("model") or "stub-cosmos-reason2",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": content,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                )
                return
            if self.path.startswith("/v1/embeddings"):
                # Deterministic vectors sized by ``embedding_dim``. Each
                # request is recorded so tests can assert wire shape —
                # auth headers, model, batch size.
                state.embeddings_requests.append(
                    {
                        "path": self.path,
                        "headers": dict(self.headers.items()),
                        "body": payload,
                    }
                )
                inputs = (payload or {}).get("input") or []
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "index": i,
                                "object": "embedding",
                                "embedding": [float(i + 1) / 1000]
                                * state.embedding_dim,
                            }
                            for i in range(len(inputs))
                        ],
                        "model": (payload or {}).get("model") or "stub-embed",
                        "usage": {
                            "prompt_tokens": len(inputs),
                            "total_tokens": len(inputs),
                        },
                    },
                )
                return
            self._json(404, {"error": "not found", "path": self.path})

        def _json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return _Handler


# ── Public handle ───────────────────────────────────────────────────────────


class MockNIMServer:
    """Test handle for an in-process mock NIM server."""

    def __init__(self) -> None:
        self._state = _MockState()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int = 0

    # ── Lifecycle ──

    def __enter__(self) -> MockNIMServer:
        # Bind the real server directly to port 0 and read back the
        # OS-assigned port. (A probe-then-rebind dance would be a TOCTOU
        # race: another process can grab the probed port between the
        # probe socket closing and the server binding.)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self._state))
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="MockNIMServer"
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, tb: Any) -> bool:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        return False

    # ── Public surface ──

    @property
    def base_url(self) -> str:
        # The lifecycle's _build_endpoint_url returns ``…/v1`` so the mock
        # URL must include the same suffix for the test seam to work.
        return f"http://127.0.0.1:{self._port}/v1"

    def set_label_payload(self, payload: dict[str, Any] | str) -> None:
        """Override the /v1/chat/completions message content.

        A dict is the JSON-label happy path; a plain string exercises the
        text-response acceptance path (Bug N regression shape).
        """
        self._state.label_payload = (
            dict(payload) if isinstance(payload, dict) else payload
        )

    def set_embedding_dim(self, dim: int) -> None:
        """Override the /v1/embeddings vector dimension (default 2048)."""
        self._state.embedding_dim = dim

    @property
    def embeddings_requests(self) -> list[dict[str, Any]]:
        """Captured /v1/embeddings requests: path, headers, parsed body."""
        return list(self._state.embeddings_requests)
