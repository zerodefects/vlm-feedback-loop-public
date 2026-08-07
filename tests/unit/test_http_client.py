# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the HTTP retry client."""

from __future__ import annotations

from datetime import UTC

import httpx
import pytest

from vlm_feedback_loop.services import http_client
from vlm_feedback_loop.services.http_client import resilient_request


# Helper: no-op backoff for fast tests
async def _noop_backoff(attempt_index: int, max_wait: float | None = None) -> None:
    pass


def _mock_transport(handler):
    """Create an httpx MockTransport from a sync handler."""
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _reset_throttle(monkeypatch):
    """Isolate the module-level adaptive per-host throttle between tests."""
    monkeypatch.setattr(http_client, "_HOST_MIN_INTERVAL", {})
    monkeypatch.setattr(http_client, "_HOST_LAST_CALL", {})
    monkeypatch.setattr(http_client, "_HOST_LOCKS", {})


class TestTimeout:
    """Timeout at the configured deadline."""

    @pytest.mark.asyncio
    async def test_timeout_returns_error_class(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.TimeoutException("simulated timeout")

        result = await resilient_request(
            "GET",
            "http://test/slow",
            deadline_s=1.0,
            _transport=_mock_transport(handler),
        )
        assert result.error_class == "timeout"
        assert result.status_code is None
        assert result.attempts == 1


class TestMaxRetries:
    """Automatic retries are bounded at MAX_RETRIES."""

    @pytest.mark.asyncio
    async def test_max_retries_bounded(self, monkeypatch):
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="Bad Gateway")

        result = await resilient_request(
            "GET",
            "http://test/fail",
            deadline_s=10.0,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.attempts == 3
        assert result.error_class == "endpoint_error"

    @pytest.mark.asyncio
    async def test_deadline_is_a_hard_cap_across_retries(self):
        """Total wall time must respect ``deadline_s`` even with the real
        backoff: retries stop once the budget is spent, rather than each
        retry getting a fresh full timeout so the total runs to several ×
        the deadline (#72)."""
        import time as _time

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="Bad Gateway")

        start = _time.monotonic()
        # Real backoff (1s, 2s, …); deadline 1.5s. Without the cap the three
        # 502 attempts + backoffs would run ~3s+. With it, elapsed stays near
        # the deadline.
        result = await resilient_request(
            "GET",
            "http://test/fail",
            deadline_s=1.5,
            max_retries=5,
            _transport=_mock_transport(handler),
        )
        elapsed = _time.monotonic() - start
        assert elapsed < 3.0, f"deadline not enforced: took {elapsed:.2f}s"
        assert result.error_class == "endpoint_error"


class TestRetryableBackoff:
    """Retryable codes trigger retry with backoff."""

    @pytest.mark.asyncio
    async def test_429_then_success(self, monkeypatch):
        # 429 -> learn the host pace; _pace (no-op here) gates the retry, which
        # then succeeds.
        pace_calls = []

        async def track_pace(host, deadline_monotonic):
            pace_calls.append(host)

        monkeypatch.setattr("vlm_feedback_loop.services.http_client._pace", track_pace)

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(
                    429, text="Rate limited", headers={"Retry-After": "1"}
                )
            return httpx.Response(200, json={"ok": True})

        result = await resilient_request(
            "GET",
            "http://test/retry",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )
        assert result.attempts == 2
        assert result.error_class is None
        assert len(pace_calls) >= 1

    @pytest.mark.asyncio
    async def test_500_then_success(self, monkeypatch):
        """A transient local-NIM 500 gets the same bounded retry as other 5xx."""
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"ok": True})

        result = await resilient_request(
            "POST",
            "http://test/transient-500",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )
        assert result.attempts == 2
        assert result.error_class is None


class TestRetryAfter:
    """429 honors the server's Retry-After — the short 5xx backoff would
    otherwise retry straight back into the hosted rate-limit cap."""

    def test_parse_delta_seconds(self):
        from vlm_feedback_loop.services.http_client import _parse_retry_after

        assert _parse_retry_after("30") == 30.0
        assert _parse_retry_after(" 5 ") == 5.0

    def test_parse_none_and_garbage(self):
        from vlm_feedback_loop.services.http_client import _parse_retry_after

        assert _parse_retry_after(None) is None
        assert _parse_retry_after("") is None
        assert _parse_retry_after("not-a-date") is None

    def test_parse_http_date(self):
        from datetime import datetime, timedelta
        from email.utils import format_datetime

        from vlm_feedback_loop.services.http_client import _parse_retry_after

        future = datetime.now(UTC) + timedelta(seconds=20)
        delay = _parse_retry_after(format_datetime(future))
        assert delay is not None and 10.0 <= delay <= 25.0

    def test_delay_honors_retry_after_capped(self):
        from vlm_feedback_loop.services.http_client import (
            MAX_RETRY_AFTER_S,
            _retry_after_delay,
        )

        assert (
            _retry_after_delay(httpx.Response(429, headers={"Retry-After": "5"}), 0)
            == 5.0
        )
        # an absurd header is capped, not honored verbatim
        assert (
            _retry_after_delay(httpx.Response(429, headers={"Retry-After": "9999"}), 0)
            == MAX_RETRY_AFTER_S
        )

    def test_delay_without_header_uses_heavier_backoff(self):
        from vlm_feedback_loop.services.http_client import _retry_after_delay

        # No Retry-After: heavier than the 1-4 s transient backoff (base*4).
        assert _retry_after_delay(httpx.Response(429), 0) >= 4.0


class TestNonRetryableImmediate:
    """Non-retryable codes fail immediately."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    async def test_immediate_fail(self, status_code):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status_code, text="error")

        result = await resilient_request(
            "GET",
            "http://test/fail",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )
        assert result.attempts == 1
        assert result.error_class == "endpoint_error"

    @pytest.mark.asyncio
    async def test_provider_error_message_surfaced_in_detail(self):
        """A 4xx body's error message is folded into error_detail.

        The bare status was uninformative; the hosted NIM's OpenAI-shaped
        400 body names the actual cause (here a guided-decoding grammar
        rejection), which downstream detectors and the UI both need.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": 'Grammar error: Unimplemented keys: ["uniqueItems"]',
                        "type": "BadRequestError",
                        "code": 400,
                    }
                },
            )

        result = await resilient_request(
            "POST",
            "http://test/chat",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )
        assert result.error_class == "endpoint_error"
        assert result.error_detail is not None
        assert "HTTP 400" in result.error_detail
        assert "uniqueItems" in result.error_detail

    @pytest.mark.asyncio
    async def test_tao_error_desc_surfaced_in_detail(self):
        """TAO FTMS returns validation failures under ``error_desc``; callers
        need that field to diagnose rejected workspace and job payloads."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error_desc": (
                        "{'cloud_specific_details': "
                        "{'cloud_type': ['Missing data for required field.']}}"
                    )
                },
            )

        result = await resilient_request(
            "POST",
            "http://test/api/v2/orgs/my-org/workspaces",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )

        assert result.error_class == "endpoint_error"
        assert result.error_detail is not None
        assert "HTTP 400" in result.error_detail
        assert "cloud_type" in result.error_detail
        assert "Missing data for required field" in result.error_detail

    @pytest.mark.asyncio
    async def test_error_detail_bounded_for_huge_body(self):
        """An oversized provider body is truncated in the detail string."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="x" * 5000)

        result = await resilient_request(
            "POST",
            "http://test/chat",
            deadline_s=10.0,
            _transport=_mock_transport(handler),
        )
        assert result.error_detail is not None
        assert len(result.error_detail) < 500


class TestExhaustedRetries:
    """Exhausted retries return an error without crashing."""

    @pytest.mark.asyncio
    async def test_returns_error_not_exception(self, monkeypatch):
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        result = await resilient_request(
            "GET",
            "http://test/down",
            deadline_s=10.0,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.error_class == "endpoint_error"
        assert result.attempts == 3
        assert "Exhausted" in (result.error_detail or "")


class TestConnectionErrorRetried:
    """Connection errors are retried."""

    @pytest.mark.asyncio
    async def test_connection_error_then_success(self, monkeypatch):
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise httpx.ConnectError("Connection refused")
            return httpx.Response(200, json={"ok": True})

        result = await resilient_request(
            "GET",
            "http://test/unstable",
            deadline_s=10.0,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.attempts == 3
        assert result.error_class is None


class TestNetworkErrorTypesCaught:
    """Regression: ``httpx.ReadError`` / ``WriteError`` / ``CloseError``
    must NOT escape ``resilient_request``.

    Catching only ``ConnectError`` (one subclass of ``NetworkError``) is
    not enough: when a connection is kernel-accepted but the downstream
    server closes without writing response headers (the exact cold-start
    race window of a freshly-launched NIM container), httpx raises
    ``ReadError`` — which would leak out of ``resilient_request`` and
    crash every fresh ``_poll_health`` task on local NIM deploys. The
    except clause therefore catches ``httpx.NetworkError`` (the parent),
    and these tests pin each ``NetworkError`` subclass.
    """

    @pytest.mark.asyncio
    async def test_read_error_returns_endpoint_error_not_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            # The cold-start race window: connect succeeded, then the
            # peer closed before sending headers.
            raise httpx.ReadError("")  # empty message — the realistic shape

        result = await resilient_request(
            "GET",
            "http://test/coldstart",
            deadline_s=5.0,
            max_retries=1,
            _transport=_mock_transport(handler),
        )
        # Must return a result, not raise.
        assert result.error_class == "endpoint_error"
        assert result.status_code is None
        # Detail should be type-named even though str(exc) is empty —
        # without this fallback, the failure log line ends with "— "
        # (silent) and the operator can't tell ReadError from
        # ConnectError from a network outage.
        assert result.error_detail and "ReadError" in result.error_detail

    @pytest.mark.asyncio
    async def test_write_error_returns_endpoint_error_not_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.WriteError("")

        result = await resilient_request(
            "GET",
            "http://test/coldstart",
            deadline_s=5.0,
            max_retries=1,
            _transport=_mock_transport(handler),
        )
        assert result.error_class == "endpoint_error"
        assert result.error_detail and "WriteError" in result.error_detail


class TestNetworkErrorRetry:
    """Every ``httpx.NetworkError`` subclass MUST be caught and
    retried, not allowed to escape and crash the caller. An except tuple
    listing only ``ConnectError`` + ``RemoteProtocolError`` + ``OSError``
    lets ``ReadError`` (connection reset / closed mid-response) escape
    and crash the local NIM health-poll task. NIM containers bind their
    host port before the HTTP server is fully accepting requests, so an
    early poll succeeds at TCP-connect but fails at read."""

    @pytest.mark.asyncio
    async def test_read_error_is_retried_not_raised(self, monkeypatch):
        """The exact NIM cold-start health-poll failure mode: connection
        accepted, then closed mid-response."""
        # Neutralize the real 1s/2s backoff (as the file's other retry tests
        # do): this test exercises retry-on-NetworkError, not the deadline
        # interaction — a real backoff would spend the whole deadline_s budget.
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        attempts = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise httpx.ReadError("connection reset")
            return httpx.Response(200, json={"ok": True})

        result = await resilient_request(
            "GET",
            "http://test/health",
            deadline_s=1,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.error_class is None
        assert result.status_code == 200
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_read_error_exhausts_retries_returns_error_not_raise(self):
        """When every attempt hits ReadError, return an error result
        (not crash the task). Caller MUST be able to retry on the next
        poll cycle."""

        def handler(req: httpx.Request) -> httpx.Response:
            raise httpx.ReadError("persistent read error")

        result = await resilient_request(
            "GET",
            "http://test/h",
            deadline_s=1,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.error_class is not None
        assert result.status_code is None

    @pytest.mark.asyncio
    async def test_write_error_also_caught(self, monkeypatch):
        """``WriteError`` is another ``NetworkError`` subclass; the client
        catches ``httpx.NetworkError`` (the base class) so every
        subclass is covered uniformly."""
        monkeypatch.setattr(
            "vlm_feedback_loop.services.http_client._backoff", _noop_backoff
        )
        attempts = 0

        def handler(req: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.WriteError("connection broken")
            return httpx.Response(200, json={"ok": True})

        result = await resilient_request(
            "GET",
            "http://test/h",
            deadline_s=1,
            max_retries=3,
            _transport=_mock_transport(handler),
        )
        assert result.error_class is None
        assert result.status_code == 200


class TestAdaptiveThrottle:
    """Adaptive per-host pacing learns the safe rate from 429s (build.nvidia.com
    RPM is undocumented + per-model) and stays dormant until rate-limited."""

    def test_note_rate_limited_seeds_from_retry_after(self):
        from vlm_feedback_loop.services import http_client as hc

        hc._note_rate_limited("h", 5.0)
        assert hc._HOST_MIN_INTERVAL["h"] == 5.0

    def test_note_rate_limited_floor_and_cap(self):
        from vlm_feedback_loop.services import http_client as hc

        hc._note_rate_limited("low", 0.0)  # below floor -> floor
        assert hc._HOST_MIN_INTERVAL["low"] == hc._THROTTLE_FLOOR_S
        hc._note_rate_limited("high", 9999.0)  # above cap -> cap
        assert hc._HOST_MIN_INTERVAL["high"] == hc.MAX_RETRY_AFTER_S

    def test_note_ok_decays_then_clears(self):
        from vlm_feedback_loop.services import http_client as hc

        hc._HOST_MIN_INTERVAL["h"] = 1.0
        hc._note_ok("h")
        assert hc._HOST_MIN_INTERVAL["h"] == pytest.approx(0.9)  # gentle 10% decay
        for _ in range(40):  # 0.9^n drops below 0.1 (~n=22) -> removed
            hc._note_ok("h")
        assert "h" not in hc._HOST_MIN_INTERVAL

    @pytest.mark.asyncio
    async def test_pace_dormant_when_not_throttled(self):
        from vlm_feedback_loop.services import http_client as hc

        t0 = hc.time.monotonic()
        await hc._pace("h", hc.time.monotonic() + 10)  # interval 0 -> no wait
        assert hc.time.monotonic() - t0 < 0.05

    @pytest.mark.asyncio
    async def test_pace_waits_when_throttled(self):
        from vlm_feedback_loop.services import http_client as hc

        hc._HOST_MIN_INTERVAL["h"] = 0.15
        hc._HOST_LAST_CALL["h"] = hc.time.monotonic()
        t0 = hc.time.monotonic()
        await hc._pace("h", hc.time.monotonic() + 100)
        assert 0.10 <= hc.time.monotonic() - t0 <= 0.6

    def test_throttle_key_separates_models(self):
        # Per-model rate limits: the Teacher and the embedding model share the
        # host but must get independent learned paces (no cross-throttle).
        from vlm_feedback_loop.services import http_client as hc

        ka = hc._throttle_key("https://h/v1/chat/completions", {"model": "teacher"})
        kb = hc._throttle_key("https://h/v1/embeddings", {"model": "embed"})
        assert ka != kb
        assert hc._throttle_key("https://h/v1/x", None) == "h"  # no model -> host only
