# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance tests for the SSE client and recovery contract.

The contract's six behaviors:
  1. REST-first render — UI renders from REST before SSE connects.
     The backend half (project data served over plain REST through the
     same proxy, no SSE involved) is pinned by test_project_list.py;
     nothing SSE-specific to cover here.
  2. SSE events update UI without redundant REST calls.
  3. Disconnect with active work → ~5 s REST polling.
  4. Reconnect → immediate REST refresh.
  5. Terminal events (*_completed, run_failed) → immediate REST refresh —
     the refresh/invalidation half is vitest
     (src/ui/src/stores/__tests__/sse-store.test.ts); delivery of terminal
     event types is pinned here alongside progress events (they share one
     opaque-passthrough backend path).
  6. Browser tab closure does not affect backend background tasks.

Tests use the synthetic test-emit endpoint
(POST /v1/testing/projects/{id}/events:emit) to inject events through the
backend's SSE infrastructure, independent of any real background task.

Server fixtures (backend_server, frontend_server) are provided by
tests/integration/conftest.py.
"""

from __future__ import annotations

import json
import threading
import time

import httpx

FRONTEND_URL = "http://127.0.0.1:5173"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_project(base_url: str, name: str) -> str:
    """Create a project and return its ID."""
    resp = httpx.post(
        f"{base_url}/v1/projects",
        json={"name": name},
        timeout=10,
    )
    assert resp.status_code == 201, f"Create failed: {resp.text}"
    return resp.json()["project_id"]


def _emit_test_event(
    base_url: str,
    project_id: str,
    event_type: str,
    data: dict,
) -> None:
    """Emit a synthetic SSE event via the test harness."""
    resp = httpx.post(
        f"{base_url}/v1/testing/projects/{project_id}/events:emit",
        json={"event_type": event_type, "data": data},
        timeout=10,
    )
    assert resp.status_code == 200, f"Emit failed: {resp.text}"
    assert resp.json()["ok"] is True


def _read_sse_events(
    url: str,
    timeout_seconds: float = 3.0,
) -> list[dict]:
    """Open an SSE stream and collect named events until timeout.

    Returns a list of dicts with ``event_type`` and ``data`` keys.
    """
    events: list[dict] = []

    try:
        with httpx.stream(
            "GET",
            url,
            timeout=httpx.Timeout(connect=5, read=timeout_seconds, write=5, pool=5),
            headers={"Accept": "text/event-stream"},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")

            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                # Parse complete SSE frames from the buffer
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    event_type = None
                    data_str = None
                    for line in frame.strip().split("\n"):
                        if line.startswith("event: "):
                            event_type = line[len("event: ") :]
                        elif line.startswith("data: "):
                            data_str = line[len("data: ") :]
                    if event_type and data_str:
                        events.append(
                            {
                                "event_type": event_type,
                                "data": json.loads(data_str),
                            }
                        )
    except httpx.ReadTimeout:
        pass  # Expected — SSE streams have no natural end

    return events


# ---------------------------------------------------------------------------
# SSE events received without redundant REST
# ---------------------------------------------------------------------------


class TestSSEEventsReceived:
    """While SSE is healthy, events update UI without redundant REST.

    The backend treats event_type as an opaque passthrough string, so one
    stream carrying a progress and a terminal event pins delivery for
    every event family; what the frontend does with terminal events
    (immediate REST refresh) is vitest-covered (sse-store.test.ts).
    """

    def test_events_arrive_in_order_with_timestamp(self, frontend_server):  # noqa: ARG002
        """Emitted events arrive in order, types intact, timestamp injected."""
        project_id = _create_project(FRONTEND_URL, "SSE Delivery Test")

        # Emit in a background thread after a short delay so the stream
        # has time to open.
        def emit_delayed():
            time.sleep(0.5)
            _emit_test_event(
                FRONTEND_URL,
                project_id,
                "evaluation_progress",
                {"run_id": "r1", "processed": 5, "total": 10},
            )
            _emit_test_event(
                FRONTEND_URL,
                project_id,
                "run_failed",
                {
                    "run_id": "r1",
                    "run_type": "evaluation_run",
                    "error_summary": "timeout",
                },
            )

        thread = threading.Thread(target=emit_delayed)
        thread.start()

        events = _read_sse_events(
            f"{FRONTEND_URL}/v1/projects/{project_id}/events",
            timeout_seconds=3.0,
        )
        thread.join(timeout=5)

        assert len(events) >= 2, events
        first, second = events[0], events[1]
        assert first["event_type"] == "evaluation_progress"
        assert first["data"]["processed"] == 5
        assert "timestamp" in first["data"]  # auto-injected by the emitter
        assert second["event_type"] == "run_failed"
        assert second["data"]["error_summary"] == "timeout"


# ---------------------------------------------------------------------------
# Backend half of disconnect/reconnect/tab-close: the SSE endpoint is
# reconnectable
# ---------------------------------------------------------------------------


def _open_sse_once(url: str) -> None:
    """Open the SSE endpoint, assert it streams, then close the connection."""
    try:
        with httpx.stream(
            "GET",
            url,
            timeout=httpx.Timeout(connect=5, read=1, write=5, pool=5),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
    except httpx.ReadTimeout:
        pass  # Expected — SSE streams have no natural end


class TestSSEReconnectable:
    """Client disconnects (polling fallback, reconnect, tab close) must
    not wedge the stream endpoint or the server.

    The corresponding frontend behaviors (~5 s REST polling on
    disconnect, immediate REST refresh on reconnect) live in the Zustand
    SSE store and are covered by vitest
    (src/ui/src/stores/__tests__/sse-store.test.ts).
    """

    def test_sse_endpoint_is_reconnectable(self, frontend_server):  # noqa: ARG002
        """Open → client closes (disconnect / tab close) → reopen succeeds."""
        project_id = _create_project(FRONTEND_URL, "SSE Reconnect Test")
        url = f"{FRONTEND_URL}/v1/projects/{project_id}/events"

        _open_sse_once(url)  # first connection, closed by the client

        # Backend still healthy after the disconnect …
        health = httpx.get(f"{FRONTEND_URL}/health", timeout=5)
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        _open_sse_once(url)  # … and a new connection succeeds immediately


# ---------------------------------------------------------------------------
# Tab close does not affect backend
# ---------------------------------------------------------------------------


class TestTabCloseSafe:
    """Browser tab closure does not affect backend background tasks.

    The open→close→health→reopen sequence itself is pinned by
    TestSSEReconnectable above; this class covers the emit-with-no-
    listeners half.
    """

    def test_event_emitted_after_disconnect_does_not_crash_backend(
        self,
        frontend_server,  # noqa: ARG002
    ):
        """Emit event while no clients connected — backend doesn't crash."""
        project_id = _create_project(FRONTEND_URL, "Emit After Disconnect")

        # No SSE clients connected — emit should succeed silently
        _emit_test_event(
            FRONTEND_URL,
            project_id,
            "evaluation_progress",
            {"run_id": "r1", "processed": 1, "total": 5},
        )

        # Backend still healthy
        health = httpx.get(f"{FRONTEND_URL}/health", timeout=5)
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
