# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreground priority dispatch — holds background while foreground is active.

Foreground endpoints (interactive_proposal, retry, rationale_regeneration)
call enter/exit around their handler. Background HTTP callers await
``wait_for_background()`` before dispatching. Already-in-flight background
requests complete normally — this is a dispatch hold, not preemption.
"""

from __future__ import annotations

import asyncio
from types import TracebackType


class _ForegroundScope:
    """Async context manager returned by ``ForegroundPriorityDispatch.foreground``."""

    def __init__(self, dispatch: ForegroundPriorityDispatch) -> None:
        self._dispatch = dispatch

    async def __aenter__(self) -> None:
        await self._dispatch.enter_foreground()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._dispatch.exit_foreground()


class ForegroundPriorityDispatch:
    """Holds NEW background HTTP dispatches while foreground is active."""

    def __init__(self) -> None:
        self._foreground_count: int = 0
        self._lock = asyncio.Lock()
        self._bg_gate = asyncio.Event()
        self._bg_gate.set()  # open — no foreground active
        self._enabled: bool = True

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the hold (``FOREGROUND_HOLD_ENABLED``, lifespan-set).

        The gate exists so a human SME's interactive proposal is never queued
        behind background evaluation traffic. Long AutoRun runs issue
        proposals continuously through the same global endpoint, which turns
        the hold into indefinite background-eval starvation (observed
        2026-07-14: one run's continuous proposal stream starved another
        run's evaluations on a different GPU/teacher entirely). Operators
        running long AutoRun/batch workloads can disable the hold; the
        default keeps the shipped interactive behavior.
        """
        self._enabled = enabled
        if not enabled:
            self._bg_gate.set()

    async def enter_foreground(self) -> None:
        """Call when a foreground request begins."""
        async with self._lock:
            self._foreground_count += 1
            if self._enabled:
                self._bg_gate.clear()

    async def exit_foreground(self) -> None:
        """Call when a foreground request completes."""
        async with self._lock:
            self._foreground_count = max(0, self._foreground_count - 1)
            if self._foreground_count == 0 or not self._enabled:
                self._bg_gate.set()

    def foreground(self) -> _ForegroundScope:
        """Scope a foreground request; holds background dispatch for its duration.

        Wrap a foreground handler (interactive proposal / retry / rationale
        regeneration) with ``async with priority_dispatch.foreground():`` — the
        gate reopens on exit even if the handler raises.
        """
        return _ForegroundScope(self)

    async def wait_for_background(self) -> None:
        """Background callers await this before dispatching HTTP requests.

        Returns immediately when no foreground is active.
        """
        await self._bg_gate.wait()


# Module-level singleton
priority_dispatch = ForegroundPriorityDispatch()
