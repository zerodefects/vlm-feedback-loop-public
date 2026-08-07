# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async HTTP client with deadline enforcement and bounded retries.

Retryable: 429, 500, 502, 503, 504, transient connection failures.
Non-retryable: 400, 401, 403, 404, 422.
Timeout: returns immediately (no retry — the deadline is a hard cap).
``schema_invalid`` is NOT determined here — the caller validates response
content against SchemaCore.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Final, cast

import httpx

from vlm_feedback_loop.services import hosted_rate_limiter

logger = logging.getLogger("vlm_feedback_loop.services.http_client")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
BASE_BACKOFF_INTERVALS = [1.0, 2.0, 4.0]
# Cap on honoring a server ``Retry-After`` so a misbehaving header can't park a
# background worker for minutes. The deadline is still the hard upper bound.
MAX_RETRY_AFTER_S = 60.0


@dataclass
class HttpResult:
    """Result of an HTTP request with retry semantics."""

    status_code: int | None = None
    body: Any = None
    error_class: str | None = None  # "timeout" | "endpoint_error" | None
    error_detail: str | None = None
    attempts: int = 0


async def _backoff(attempt_index: int, max_wait: float | None = None) -> None:
    """Exponential backoff with jitter, capped at ``max_wait`` seconds so a
    backoff can't sleep past the caller's remaining deadline. Monkeypatched to
    no-op in tests."""
    base = BASE_BACKOFF_INTERVALS[min(attempt_index, len(BASE_BACKOFF_INTERVALS) - 1)]
    jitter = random.uniform(0, base * 0.5)
    wait = base + jitter
    if max_wait is not None:
        wait = min(wait, max_wait)
    if wait > 0:
        await asyncio.sleep(wait)


def _parse_retry_after(value: str | None) -> float | None:
    """Parse an HTTP ``Retry-After`` header into seconds.

    Accepts delta-seconds (``"30"``) or an HTTP-date; returns ``None`` when the
    header is absent or unparseable.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return max(0.0, (dt - datetime.now(UTC)).total_seconds())


def _retry_after_delay(response: httpx.Response, attempt_index: int) -> float:
    """Backoff delay for a 429 (pure). Honors the server's ``Retry-After``
    (capped at :data:`MAX_RETRY_AFTER_S`) when present; otherwise uses a
    heavier-than-transient exponential backoff — a rate limit needs a real
    pause, not the 1-4 s used for 5xx / connection blips. Honoring Retry-After
    also self-throttles outbound calls below the provider's cap, which is what
    actually lets a long background eval drain under a tight quota.
    """
    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
    if retry_after is not None:
        return min(retry_after, MAX_RETRY_AFTER_S)
    base = BASE_BACKOFF_INTERVALS[min(attempt_index, len(BASE_BACKOFF_INTERVALS) - 1)]
    return min(base * 4.0, MAX_RETRY_AFTER_S)


# ── Adaptive per-host throttle ──────────────────────────────────────────────
# build.nvidia.com RPM is undocumented and varies per model (NVIDIA forum
# 344567: "rate limits vary for each model, and we do not publish those"), so a
# static client-side cap is wrong. Learn the safe pace from 429s instead: raise
# a host's minimum inter-request interval on a 429 (seeded by Retry-After),
# decay it on each success. Dormant (interval 0) until a host actually rate-
# limits us, so healthy hosts, local NIMs, and mock-transport tests are
# unaffected. Shared module-level state because the limit is per-account.
_HOST_MIN_INTERVAL: dict[str, float] = {}  # host -> seconds required between calls
_HOST_LAST_CALL: dict[str, float] = {}  # host -> monotonic time of reserved send
_HOST_LOCKS: dict[str, asyncio.Lock] = {}  # host -> lock serializing slot reserve
_THROTTLE_FLOOR_S = 1.0  # floor for a learned interval after a 429
# Because the pace is learned, not hardcoded, it self-re-converges if the
# unpublished limit ever shifts. A LOWERED limit is caught fast — fresh 429s
# multiply the interval up (_THROTTLE_INCREASE). A RAISED limit is picked up
# gradually — the gentle 10%/success decay relaxes the interval back into the
# new headroom over a window of calls rather than re-probing aggressively
# (which would sawtooth 429s). Backend restarts reset to dormant and re-learn
# from scratch.
_THROTTLE_DECAY = 0.9  # multiplicative decrease applied on each success
_THROTTLE_INCREASE = 2.0  # multiplicative increase applied on each 429


def _host_key(url: str) -> str:
    try:
        return httpx.URL(url).host or url
    except Exception:
        return url


def _throttle_key(url: str, json_body: dict[str, Any] | None) -> str:
    """Throttle key = host plus the request's model when present. Rate limits
    are per-model (forum 344567), so the Teacher's learned pace must stay
    independent of the embedding/CLIP model's even though both share the
    integrate.api.nvidia.com host — a Teacher 429 must not throttle embeddings
    and vice versa."""
    host = _host_key(url)
    model = json_body.get("model") if isinstance(json_body, dict) else None
    return f"{host}|{model}" if model else host


def _note_rate_limited(host: str, retry_after: float) -> None:
    """Raise ``host``'s learned min-interval after a 429 (multiplicative
    increase, seeded by the server's Retry-After, capped)."""
    cur = _HOST_MIN_INTERVAL.get(host, 0.0)
    _HOST_MIN_INTERVAL[host] = min(
        max(cur * _THROTTLE_INCREASE, retry_after, _THROTTLE_FLOOR_S), MAX_RETRY_AFTER_S
    )


def _note_ok(host: str) -> None:
    """Decay ``host``'s min-interval toward zero on a success so a transient
    throttle doesn't pin the pace forever."""
    cur = _HOST_MIN_INTERVAL.get(host, 0.0)
    if cur <= 0.0:
        return
    nxt = cur * _THROTTLE_DECAY
    if nxt < 0.1:
        _HOST_MIN_INTERVAL.pop(host, None)
    else:
        _HOST_MIN_INTERVAL[host] = nxt


def _host_lock(host: str) -> asyncio.Lock:
    lock = _HOST_LOCKS.get(host)
    if lock is None:
        lock = asyncio.Lock()
        _HOST_LOCKS[host] = lock
    return lock


async def _pace(host: str, deadline_monotonic: float) -> None:
    """Reserve this call's send slot, spacing calls to ``host`` at least its
    learned min-interval apart.

    The reservation — read the last send time, advance it to the next slot —
    happens under a per-host lock so concurrent same-host calls each claim a
    DISTINCT staggered slot. The previous read-sleep-then-write-after-await
    let concurrent calls all read the same last-call, sleep the same interval,
    then burst together straight back into the rate limit.

    No-op until the host has been rate-limited (interval 0). Monkeypatched to
    no-op in tests."""
    interval = _HOST_MIN_INTERVAL.get(host, 0.0)
    if interval <= 0.0:
        return
    async with _host_lock(host):
        now = time.monotonic()
        # Next allowed send time; capped at the caller's deadline so a paced
        # call never sleeps past its budget. Advance _HOST_LAST_CALL to the
        # reserved (future) time so the next reserver stacks on top of it.
        send_at = min(
            max(now, _HOST_LAST_CALL.get(host, 0.0) + interval),
            deadline_monotonic,
        )
        _HOST_LAST_CALL[host] = send_at
        wait = send_at - now
    if wait > 0:
        await asyncio.sleep(wait)


def _parse_body(response: httpx.Response) -> Any:
    """Try JSON parse, fall back to text."""
    try:
        return response.json()
    except Exception:
        return response.text


#: Cap on the provider-message snippet folded into an endpoint-error detail.
#: Keeps an oversized HTML error page or a runaway provider message out of the
#: persisted OperationRecord error artifact while preserving the actionable head.
_MAX_ERROR_DETAIL_CHARS: Final[int] = 400


def _endpoint_error_detail(status_code: int, body: Any) -> str:
    """Format an endpoint-error detail, folding in the provider's message.

    A bare ``HTTP 400`` is uninformative: a hosted NIM's 4xx body carries the
    actionable reason (a guided-decoding ``Grammar error: Unimplemented keys:
    [...]``, a rejected generation control, a quota message). Surfacing a
    bounded snippet makes the failure diagnosable in the UI, logs, and the
    stored error artifact, and lets the capability-rejection detectors
    (``teacher_rejection``) match on the provider's own wording instead of a
    signal that never arrives.
    """
    message = _extract_provider_error_message(body)
    if not message:
        return f"HTTP {status_code}"
    return f"HTTP {status_code}: {message[:_MAX_ERROR_DETAIL_CHARS]}"


def _extract_provider_error_message(body: Any) -> str:
    """Best-effort pull of a human-readable error string from a parsed body.

    Handles the OpenAI-compatible ``{"error": {"message": ...}}`` shape the
    hosted NIM returns, a FastAPI ``{"detail": ...}``, TAO FTMS's
    ``{"error_desc": ...}``, a top-level ``{"message": ...}``, and a
    plain-text body. Returns ``""`` when nothing usable is present (caller
    falls back to the bare status line).
    """
    if isinstance(body, str):
        return body.strip()
    if not isinstance(body, dict):
        return ""
    obj = cast("dict[str, Any]", body)
    err: Any = obj.get("error")
    if isinstance(err, dict):
        msg: Any = cast("dict[str, Any]", err).get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    elif isinstance(err, str) and err.strip():
        return err.strip()
    for key in ("detail", "error_desc", "message"):
        val: Any = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


async def resilient_request(
    method: str,
    url: str,
    *,
    deadline_s: float,
    max_retries: int = 3,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> HttpResult:
    """HTTP request with deadline enforcement and bounded retries.

    ``deadline_s`` is a hard wall-clock cap across all attempts (each request
    is bounded by the time remaining, and the loop stops once it is spent).
    ``max_retries`` is the total ATTEMPT count — the loop runs
    ``range(max_retries)``, so ``max_retries=3`` means one initial send plus
    up to two retries.

    Parameters:
        _transport: Injected transport for testing (httpx.MockTransport).
    """
    attempts = 0
    last_error: str | None = None
    key = _throttle_key(url, json_body)
    deadline_monotonic = time.monotonic() + deadline_s

    client_kwargs: dict[str, Any] = {"timeout": httpx.Timeout(deadline_s)}
    if _transport is not None:
        client_kwargs["transport"] = _transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        for attempt in range(max_retries):
            # Enforce ``deadline_s`` as a true wall-clock cap across ALL
            # attempts: stop once the budget is spent, and bound each request
            # by the time remaining so N retries × a fresh full timeout can't
            # run the total to several × the deadline.
            remaining = deadline_monotonic - time.monotonic()
            if remaining <= 0:
                last_error = last_error or "deadline exceeded"
                break
            attempts += 1
            try:
                # Proactive global ceiling for hosted (build.nvidia.com) calls,
                # applied per attempt so retries count against the account RPM.
                # No-op for self-hosted/local-NIM/codex-agent URLs and when
                # HOSTED_GLOBAL_RPM is unset. Runs before the per-host adaptive
                # pacer so the cross-model budget gates first.
                await hosted_rate_limiter.acquire_if_hosted(url)
                await _pace(key, deadline_monotonic)
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=httpx.Timeout(
                        max(0.0, deadline_monotonic - time.monotonic())
                    ),
                )

                if response.status_code < 400:
                    _note_ok(key)
                    return HttpResult(
                        status_code=response.status_code,
                        body=_parse_body(response),
                        error_class=None,
                        error_detail=None,
                        attempts=attempts,
                    )

                if response.status_code in NON_RETRYABLE_STATUS_CODES:
                    body = _parse_body(response)
                    return HttpResult(
                        status_code=response.status_code,
                        body=body,
                        error_class="endpoint_error",
                        error_detail=_endpoint_error_detail(response.status_code, body),
                        attempts=attempts,
                    )

                if response.status_code in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code == 429:
                        # Rate limit: learn the host's pace (seeded by
                        # Retry-After). The next loop-top _pace enforces the wait
                        # and throttles concurrent/subsequent calls too, so we
                        # stop bursting straight back into the cap.
                        _note_rate_limited(key, _retry_after_delay(response, attempt))
                    elif (
                        attempt < max_retries - 1
                        and deadline_monotonic - time.monotonic() > 0
                    ):
                        await _backoff(
                            attempt, max_wait=deadline_monotonic - time.monotonic()
                        )
                    continue

                # Other status codes — non-retryable
                body = _parse_body(response)
                return HttpResult(
                    status_code=response.status_code,
                    body=body,
                    error_class="endpoint_error",
                    error_detail=_endpoint_error_detail(response.status_code, body),
                    attempts=attempts,
                )

            except httpx.TimeoutException:
                return HttpResult(
                    status_code=None,
                    body=None,
                    error_class="timeout",
                    error_detail="Request timed out",
                    attempts=attempts,
                )

            except (
                # Catch every ``httpx.NetworkError`` subclass — a
                # narrower ``ConnectError``/``RemoteProtocolError`` tuple lets
                # ``ReadError`` (connection reset / closed mid-response)
                # escape and crash the caller. The local NIM health-poll is
                # the evidence: NIM containers bind their host port before
                # the HTTP server is fully accepting requests, so an early
                # poll succeeds at TCP-connect but the read fails; if that
                # error escapes, the poll task crashes and the deploy times
                # out waiting for a container that is already running.
                httpx.NetworkError,
                httpx.RemoteProtocolError,
                OSError,
            ) as exc:
                last_error = str(exc) or repr(exc)
                if (
                    attempt < max_retries - 1
                    and deadline_monotonic - time.monotonic() > 0
                ):
                    await _backoff(
                        attempt, max_wait=deadline_monotonic - time.monotonic()
                    )
                continue

    # Exhausted the attempt budget (or the deadline). ``max_retries`` is the
    # total ATTEMPT count (the loop runs range(max_retries)), so report the
    # attempts actually made rather than mislabeling it as a retry count.
    return HttpResult(
        status_code=None,
        body=None,
        error_class="endpoint_error",
        error_detail=f"Exhausted {attempts} attempt(s). Last: {last_error}",
        attempts=attempts,
    )
