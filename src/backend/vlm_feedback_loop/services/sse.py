# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project-scoped SSE event broadcasting.

Event types are open strings — no enum, no validation — so new types
(TAO job progress, NIM benchmark) are added without modifying this module.
Wire format: ``event: {type}\\ndata: {json}\\n\\n``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from vlm_feedback_loop.db.base import utc_now

logger = logging.getLogger("vlm_feedback_loop.services.sse")

# Idle streams must emit bytes more often than the smallest proxy read
# timeout in front of them — the shipped edge config (nginx.conf) drops
# connections after 300 s of byte-silence.
SSE_KEEPALIVE_INTERVAL_S = 15.0

# SSE comment line: ignored by EventSource clients, but resets the
# proxy's read timer.
SSE_KEEPALIVE_MESSAGE = ": keepalive\n\n"


class SSEManager:
    """Per-project event broadcast to connected SSE clients."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[str | None]]] = {}

    def subscribe(self, project_id: str) -> asyncio.Queue[str | None]:
        """Create a subscription queue for the given project."""
        if project_id not in self._subscribers:
            self._subscribers[project_id] = []
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)
        self._subscribers[project_id].append(queue)
        return queue

    def unsubscribe(self, project_id: str, queue: asyncio.Queue[str | None]) -> None:
        """Remove a subscription queue. Safe to call multiple times."""
        subs = self._subscribers.get(project_id, [])
        with contextlib.suppress(ValueError):
            subs.remove(queue)
        if not subs:
            self._subscribers.pop(project_id, None)

    async def emit(
        self, project_id: str, event_type: str, data: dict[str, Any]
    ) -> None:
        """Broadcast an event to all subscribers of a project.

        ``event_type`` is an open string — no validation.
        ``data`` should contain ``run_id`` and ``timestamp`` per spec;
        ``timestamp`` is auto-injected if missing.
        """
        if "timestamp" not in data:
            data["timestamp"] = utc_now()

        message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        for queue in self._subscribers.get(project_id, []):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # Drop the OLDEST queued message, not this one: SSE is a hint
                # channel and the newest event (e.g. a terminal run.completed)
                # is the most important to deliver. Dropping the newest could
                # make a briefly-slow client miss a terminal event and hang;
                # the frontend reconciles stale intermediate state from REST.
                logger.warning(
                    "SSE queue full for project %s, dropping oldest event",
                    project_id,
                )
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(message)

    def stream(
        self,
        project_id: str,
        *,
        keepalive_interval_s: float = SSE_KEEPALIVE_INTERVAL_S,
    ) -> AsyncIterator[str]:
        """Subscribe and yield SSE messages, interleaving keepalive comments.

        Subscription happens eagerly (before the first ``anext``) so events
        emitted between response construction and first iteration are not
        lost. The generator unsubscribes when closed or exhausted.
        """
        queue = self.subscribe(project_id)

        async def _generate() -> AsyncIterator[str]:
            try:
                while True:
                    try:
                        message = await asyncio.wait_for(
                            queue.get(), timeout=keepalive_interval_s
                        )
                    except TimeoutError:
                        yield SSE_KEEPALIVE_MESSAGE
                        continue
                    if message is None:  # shutdown sentinel
                        break
                    yield message
            finally:
                self.unsubscribe(project_id, queue)

        return _generate()

    async def close_all(self) -> None:
        """Send shutdown sentinel to all subscribers."""
        for subs in self._subscribers.values():
            for queue in subs:
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(None)
        self._subscribers.clear()


# Module-level singleton
sse_manager = SSEManager()
