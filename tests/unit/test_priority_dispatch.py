# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreground-priority dispatch hold — including the batch-campaign switch."""

from __future__ import annotations

import asyncio

import pytest

from vlm_feedback_loop.services.priority import ForegroundPriorityDispatch


async def _background_dispatches_within(dispatch, timeout_s: float = 0.05) -> bool:
    try:
        await asyncio.wait_for(dispatch.wait_for_background(), timeout=timeout_s)
    except TimeoutError:
        return False
    return True


class TestForegroundHold:
    @pytest.mark.asyncio
    async def test_enabled_hold_blocks_background_while_foreground_active(self):
        """Shipped default: an in-flight foreground request holds new
        background dispatches, and the gate reopens when it exits."""
        dispatch = ForegroundPriorityDispatch()
        async with dispatch.foreground():
            assert not await _background_dispatches_within(dispatch)
        assert await _background_dispatches_within(dispatch)

    @pytest.mark.asyncio
    async def test_disabled_hold_lets_background_dispatch_during_foreground(self):
        """FOREGROUND_HOLD_ENABLED=false (batch/AutoRun campaigns): background
        evaluation dispatch proceeds even while proposals are in flight — a
        continuous AutoRun proposal stream must not starve background evals
        (observed live: one lane's warmup seeds held another lane's evals
        indefinitely on a different GPU/teacher)."""
        dispatch = ForegroundPriorityDispatch()
        dispatch.set_enabled(False)
        async with dispatch.foreground():
            assert await _background_dispatches_within(dispatch)

    @pytest.mark.asyncio
    async def test_disabling_mid_hold_releases_waiting_background(self):
        """Flipping the switch off while a foreground request is active opens
        the gate immediately for already-waiting background callers."""
        dispatch = ForegroundPriorityDispatch()
        async with dispatch.foreground():
            assert not await _background_dispatches_within(dispatch)
            dispatch.set_enabled(False)
            assert await _background_dispatches_within(dispatch)

    @pytest.mark.asyncio
    async def test_reenabling_restores_the_hold(self):
        dispatch = ForegroundPriorityDispatch()
        dispatch.set_enabled(False)
        dispatch.set_enabled(True)
        async with dispatch.foreground():
            assert not await _background_dispatches_within(dispatch)
