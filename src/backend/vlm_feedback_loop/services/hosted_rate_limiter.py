# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-global proactive RPM ceiling for HOSTED (build.nvidia.com) NIM calls.

The per-host adaptive pacer in ``http_client`` (``_pace`` / ``_note_rate_limited``)
*reacts* to 429s after they happen, and its throttle key is per-(host, model)
because build.nvidia.com's documented limits are per-model. But some accounts
enforce a **global, cross-model** RPM cap (e.g. a personal key shared across all
models). Under such a cap, several concurrent eval streams of *different* models
collectively exceed the account RPM, and the slow models get starved into
429/timeout storms before the per-model reactive pacer can learn.

This module adds a single **proactive** token ceiling shared across every hosted
request in the process, so concurrent multi-model work stays under one account
budget. It gates only URLs on the configured hosted host; self-hosted and local
NIM requests are never throttled. Disabled by default
(``HOSTED_GLOBAL_RPM=0``); opt in by setting the RPM at startup.

``SlidingWindowRpmLimiter`` is the single sliding-window implementation, also
imported by ``cli_autorun`` and scripts (the CLI imports from services, never
the reverse).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from urllib.parse import urlsplit

logger = logging.getLogger("vlm_feedback_loop.hosted_rate_limiter")


class SlidingWindowRpmLimiter:
    """Sliding-window RPM limiter. ``rpm <= 0`` disables it."""

    def __init__(self, rpm: int, *, wait_log_level: int = logging.DEBUG) -> None:
        self.rpm = int(rpm or 0)
        self._wait_log_level = wait_log_level
        self._timestamps: deque[float] = deque()
        # asyncio.Lock is created lazily in ``acquire`` so the limiter can
        # be constructed before an event loop exists (e.g., at argparse
        # time or in a sync module-level scope).
        self._lock: asyncio.Lock | None = None

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    async def acquire(self) -> None:
        """Block until the next call would not exceed the RPM budget."""
        if not self.enabled:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.rpm:
                wait_s = max(0.0, self._timestamps[0] + 60.0 - time.monotonic())
                if wait_s > 0:
                    logger.log(
                        self._wait_log_level,
                        "RPM limiter waiting %.2fs (budget=%d/min, in_window=%d)",
                        wait_s,
                        self.rpm,
                        len(self._timestamps),
                    )
                    await asyncio.sleep(wait_s)
                # Re-prune after sleep
                now2 = time.monotonic()
                cutoff2 = now2 - 60.0
                while self._timestamps and self._timestamps[0] < cutoff2:
                    self._timestamps.popleft()
            self._timestamps.append(time.monotonic())


_limiter: SlidingWindowRpmLimiter | None = None
_hosted_host: str = ""


def configure(rpm: int, hosted_base_url: str) -> None:
    """Set the global hosted RPM ceiling and the hosted host to gate on.

    ``rpm <= 0`` disables the limiter. Called once at backend startup; safe to
    call before an event loop exists (the lock is created lazily on first use).
    """
    global _limiter, _hosted_host
    _limiter = SlidingWindowRpmLimiter(rpm) if int(rpm or 0) > 0 else None
    _hosted_host = urlsplit(hosted_base_url or "").netloc
    if _limiter is not None:
        logger.info(
            "hosted global RPM limiter active: %d req/min on host '%s'",
            _limiter.rpm,
            _hosted_host or "(unset)",
        )


def _is_hosted(url: str) -> bool:
    return bool(_hosted_host) and urlsplit(url).netloc == _hosted_host


async def acquire_if_hosted(url: str) -> None:
    """Block until a global token frees, but only for hosted-host URLs.

    No-op when disabled or when the URL is not the hosted host (self-hosted NIM,
    local NIM, codex-agent/Opus). Every call — including retries — consumes one
    token, so retry attempts count against the account budget.
    """
    if _limiter is None or not _is_hosted(url):
        return
    await _limiter.acquire()
