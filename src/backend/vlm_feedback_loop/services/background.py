# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process asyncio background task management.

No external task queue (Celery/Redis). Tasks run on the asyncio event loop.
CPU-bound work uses ``asyncio.to_thread()``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import traceback
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger("vlm_feedback_loop.services.background")


class BackgroundTaskManager:
    """Tracks in-process asyncio tasks with graceful shutdown."""

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._shutdown_event = asyncio.Event()

    def register(
        self, task_id: str, coro: Coroutine[Any, Any, Any]
    ) -> asyncio.Task[Any]:
        """Schedule a coroutine as a tracked background task.

        Raises RuntimeError if a task with the same ID is already running.
        """
        if task_id in self._tasks and not self._tasks[task_id].done():
            raise RuntimeError(f"Task {task_id} is already running")

        task = asyncio.create_task(coro, name=f"bg-{task_id}")
        self._tasks[task_id] = task
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        logger.info("Background task registered: %s", task_id)
        return task

    def try_register(
        self,
        task_id: str,
        coro: Coroutine[Any, Any, Any],
        *,
        no_loop_warning: str,
    ) -> asyncio.Task[Any] | None:
        """Register ``coro`` under ``task_id`` unless one is already running.

        Shared trigger policy for the pHash/CLIP sweepers: dedupe by task
        ID, close the coroutine whenever it will not be scheduled (avoiding
        the "never awaited" warning), and classify ``RuntimeError``:

        1. Task already running — race between the ``active_task_ids``
           check and ``register``. Expected, swallow.
        2. "no running event loop" — caller is on a thread without an
           asyncio loop (e.g. a sync FastAPI route handler running in the
           threadpool). The worker would silently never start, so log
           ``no_loop_warning`` to make regressions visible; startup
           recovery re-runs the work on the next backend restart.

        Returns the scheduled task, or ``None`` when nothing was scheduled.
        """
        if task_id in self.active_task_ids:
            coro.close()
            return None
        try:
            return self.register(task_id, coro)
        except RuntimeError as exc:
            coro.close()
            msg = str(exc).lower()
            if "running event loop" in msg or "no current event loop" in msg:
                logger.warning("%s (%s)", no_loop_warning, exc)
            return None

    def _on_task_done(self, task_id: str, task: asyncio.Task[Any]) -> None:
        # A stable ID may have been rebound after this task became done but
        # before its callback ran. Only that ID's current owner may remove it.
        if self._tasks.get(task_id) is task:
            self._tasks.pop(task_id)
        if task.cancelled():
            logger.info("Background task cancelled: %s", task_id)
            return
        exc = task.exception()
        if exc is None:
            logger.info("Background task completed: %s", task_id)
            return
        # ``str(exc)`` is empty for several common exception types (custom
        # asyncio cancellation, RuntimeError() with no args, etc.), leaving
        # operators with "Background task failed: <id> — " and nothing
        # after the dash. Log ``repr(exc)`` so the type name is always
        # present, plus the formatted traceback so downstream JSON-log
        # consumers can surface the call site without spelunking through
        # process logs.
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        # ``exc_info=`` populates ``LogRecord.exc_info`` so structured
        # consumers (e.g. ``StructuredJsonFormatter``) can serialize the
        # exception payload independently of the message format string.
        # The ``%r`` + traceback in the message remains the human-readable
        # path; both channels carry the same root cause.
        logger.error(
            "Background task failed: %s — %r\n%s",
            task_id,
            exc,
            tb_text,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def is_shutting_down(self) -> bool:
        """Check if shutdown has been requested. Tasks should poll this."""
        return self._shutdown_event.is_set()

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel one tracked task and wait for its cancellation handler.

        Returns ``True`` when an active task was found. The task remains
        responsible for any domain-specific cleanup in its
        ``CancelledError`` handler.
        """
        task = self._tasks.get(task_id)
        if task is None or task.done():
            return False

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return True

    async def cancel_all(self, grace_seconds: float = 5.0) -> None:
        """Two-phase shutdown: signal → wait → force cancel."""
        self._shutdown_event.set()

        running = [t for t in self._tasks.values() if not t.done()]
        if not running:
            return

        logger.info(
            "Graceful shutdown: waiting %ss for %d tasks",
            grace_seconds,
            len(running),
        )

        # Wait for voluntary exit
        _, pending = await asyncio.wait(
            running, timeout=grace_seconds, return_when=asyncio.ALL_COMPLETED
        )

        # Cancel stragglers
        for task in pending:
            task.cancel()
            logger.info("Force-cancelling task: %s", task.get_name())

        if pending:
            await asyncio.wait(pending, timeout=2.0)

        self._tasks.clear()

    @property
    def active_task_ids(self) -> list[str]:
        return [tid for tid, t in self._tasks.items() if not t.done()]


async def run_in_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run a sync function in the default thread pool.

    Use for CPU-bound work (pHash, file I/O) to keep the event loop responsive.
    """
    return await asyncio.to_thread(fn, *args)


# ── Low-priority pool for background image processing ───────────────────────

# Heavy background PIL work — the pHash sweep's decode and the CLIP
# embedding worker's read/resize/normalize — runs here instead of the
# default pool, on threads pinned to a low OS scheduler priority (nice).
# Under CPU contention these decodes then yield to the event loop and
# interactive requests (image serving, guidance saves, proposals) instead
# of competing with them for a core. The worker count is capped well below
# the core count so this bulk work can never monopolize the box.
_IMAGE_PROC_NICE = 10


def _lower_thread_priority() -> None:
    # Per-thread nice is Linux-specific (who=0 targets the calling task);
    # best-effort no-op where the platform does not support it.
    with contextlib.suppress(OSError, AttributeError):  # pragma: no cover
        os.setpriority(os.PRIO_PROCESS, 0, _IMAGE_PROC_NICE)


_low_priority_image_pool = ThreadPoolExecutor(
    max_workers=2,
    thread_name_prefix="img-lowpri",
    initializer=_lower_thread_priority,
)


async def run_in_low_priority_thread(fn: Callable[..., Any], *args: Any) -> Any:
    """Run CPU-bound background image work (PIL decode/resize, pHash) off the
    event loop on a low-OS-priority thread pool, so it does not compete with
    interactive requests for CPU. Same call shape as ``run_in_thread``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_low_priority_image_pool, fn, *args)


# Module-level singleton
background_manager = BackgroundTaskManager()
