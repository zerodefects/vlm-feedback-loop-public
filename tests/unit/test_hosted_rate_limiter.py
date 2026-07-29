# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared RPM limiter and the hosted facade around it.

Two contracts are pinned here:

1. ``SlidingWindowRpmLimiter`` — the single sliding-window implementation
   (also used by AutoRun and scripts): ``rpm <= 0`` disables, under-budget
   acquires never block, over-budget acquires sleep until the oldest
   timestamp leaves the 60-s window, and the window slides forward.
   These tests use a monkeypatched ``time.monotonic`` and ``asyncio.sleep``
   so the contract can be verified deterministically without waiting 60 s.

2. The process-global hosted facade (the reason this module exists): a single
   cross-model token budget that gates only hosted-host URLs, counts every
   call, and paces calls beyond the budget — so concurrent multi-model eval
   streams stay under a global account RPM cap instead of storming into 429s.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from vlm_feedback_loop.services import hosted_rate_limiter as hrl
from vlm_feedback_loop.services.hosted_rate_limiter import SlidingWindowRpmLimiter

HOSTED = "https://integrate.api.nvidia.com/v1"
HOSTED_CALL = "https://integrate.api.nvidia.com/v1/chat/completions"
LOCAL_CALL = "http://localhost:4567/v1/chat/completions"  # codex-agent / Opus
SELF_HOSTED = "http://127.0.0.1:8000/v1/chat/completions"  # local NIM


# ── SlidingWindowRpmLimiter: the shared window contract ─────────────────────


@pytest.mark.asyncio
async def test_disabled_limiter_is_noop() -> None:
    lim = SlidingWindowRpmLimiter(rpm=0)
    assert lim.enabled is False
    # 1000 acquires must complete instantly (no sleep, no error).
    for _ in range(1000):
        await lim.acquire()


@pytest.mark.asyncio
async def test_negative_rpm_treated_as_disabled() -> None:
    lim = SlidingWindowRpmLimiter(rpm=-1)
    assert lim.enabled is False
    await lim.acquire()  # must not raise


@pytest.mark.asyncio
async def test_under_budget_acquires_immediately() -> None:
    """At 4 RPM, the first 4 calls within 60 s must not block."""
    lim = SlidingWindowRpmLimiter(rpm=4)
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    sleeps: list[float] = []

    async def _fake_sleep(s: float) -> None:
        sleeps.append(s)
        # Advance the fake clock past the sleep so the next acquire sees
        # the window expire.
        fake_now[0] += s

    with (
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.time.monotonic",
            side_effect=_now,
        ),
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.asyncio.sleep",
            side_effect=_fake_sleep,
        ),
    ):
        for _ in range(4):
            await lim.acquire()
            fake_now[0] += 1.0  # simulate ~1s call latency between acquires

    # No sleeps should have happened — we never exceeded the budget.
    assert sleeps == []


@pytest.mark.asyncio
async def test_over_budget_blocks_until_oldest_expires() -> None:
    """5th acquire at 4 RPM must block until 60 s after the 1st acquire."""
    lim = SlidingWindowRpmLimiter(rpm=4)
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    sleeps: list[float] = []

    async def _fake_sleep(s: float) -> None:
        sleeps.append(s)
        fake_now[0] += s

    with (
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.time.monotonic",
            side_effect=_now,
        ),
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.asyncio.sleep",
            side_effect=_fake_sleep,
        ),
    ):
        # Burn the budget: 4 acquires at t=0,1,2,3.
        for _ in range(4):
            await lim.acquire()
            fake_now[0] += 1.0

        # 5th acquire at t=4 should sleep until t=60 (oldest at 0 + 60).
        await lim.acquire()

    assert len(sleeps) == 1
    assert sleeps[0] == pytest.approx(56.0, abs=0.01)  # 60 - 4


@pytest.mark.asyncio
async def test_window_slides_forward() -> None:
    """After 60s elapse, old timestamps are pruned and budget is freed."""
    lim = SlidingWindowRpmLimiter(rpm=2)
    fake_now = [0.0]

    def _now() -> float:
        return fake_now[0]

    sleeps: list[float] = []

    async def _fake_sleep(s: float) -> None:
        sleeps.append(s)
        fake_now[0] += s

    with (
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.time.monotonic",
            side_effect=_now,
        ),
        patch(
            "vlm_feedback_loop.services.hosted_rate_limiter.asyncio.sleep",
            side_effect=_fake_sleep,
        ),
    ):
        # Two acquires at t=0,1 fill the budget.
        await lim.acquire()
        fake_now[0] = 1.0
        await lim.acquire()

        # Jump to t=70: both timestamps are out of the 60-s window.
        fake_now[0] = 70.0

        # Two more acquires should NOT block — the window is empty.
        await lim.acquire()
        await lim.acquire()

    assert sleeps == []  # no sleeps required


# ── Hosted facade: process-global gating on the hosted host only ────────────


def test_disabled_by_default_is_noop():
    hrl.configure(0, HOSTED)
    # acquire returns immediately even for a hosted URL
    t0 = time.monotonic()
    asyncio.run(hrl.acquire_if_hosted(HOSTED_CALL))
    assert time.monotonic() - t0 < 0.1


def test_only_hosted_host_is_gated():
    hrl.configure(1, HOSTED)

    async def scenario():
        # First hosted call consumes the single token instantly.
        await hrl.acquire_if_hosted(HOSTED_CALL)
        # Non-hosted URLs must NOT be throttled even though the budget is spent
        # (codex-agent/Opus + local NIM paths stay free).
        t0 = time.monotonic()
        await hrl.acquire_if_hosted(LOCAL_CALL)
        await hrl.acquire_if_hosted(SELF_HOSTED)
        return time.monotonic() - t0

    assert asyncio.run(scenario()) < 0.1


def test_budget_paces_excess_hosted_calls(monkeypatch):
    # Spend the 2-token/min budget, then the 3rd hosted call must pace ~60s.
    # Mock only asyncio.sleep (capture the requested wait without blocking);
    # real clock => the 3 acquires land within microseconds so the wait ≈ 60s.
    hrl.configure(2, HOSTED)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(s):
        slept.append(s)
        await real_sleep(0)  # yield, don't actually block

    monkeypatch.setattr(hrl.asyncio, "sleep", fake_sleep)

    async def scenario():
        await hrl.acquire_if_hosted(HOSTED_CALL)  # token 1
        await hrl.acquire_if_hosted(HOSTED_CALL)  # token 2 (budget full)
        await hrl.acquire_if_hosted(HOSTED_CALL)  # 3rd -> must pace ~60s

    asyncio.run(scenario())
    assert slept and 55.0 < slept[0] <= 60.0
