# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared lifecycle handling for direct asyncio child processes."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys

_PROCESS_CLEANUP_TIMEOUT_S = 5.0


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    """Best-effort terminate a child and consume its process/pipe state."""
    killed_group = False
    process_id = getattr(process, "pid", None)
    if os.name == "posix" and isinstance(process_id, int):
        try:
            # Production callers start a new session, so the child PID is
            # also its process-group ID. Signal the group even if its leader
            # has exited: a worker can retain the leader's stdout/stderr pipes
            # and otherwise keep communicate() blocked indefinitely.
            os.killpg(process_id, signal.SIGKILL)
            killed_group = True
        except OSError:
            pass
    if process.returncode is None and not killed_group:
        with contextlib.suppress(Exception):
            process.kill()

    try:
        await process.communicate()
    except Exception:
        # A broken PIPE transport can make communicate fail even after the
        # child exits. Direct wait is the remaining way to reap it.
        with contextlib.suppress(Exception):
            await process.wait()


async def communicate_with_timeout(
    process: asyncio.subprocess.Process,
    *,
    timeout_s: float,
    stdin: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Communicate with a child, killing and reaping it on every interruption.

    Timeout, task cancellation, and unexpected pipe failures propagate to the
    caller after cleanup. Domain-specific callers remain responsible for
    decoding and redacting output.
    """
    completed = False
    try:
        communication = (
            process.communicate(input=stdin)
            if stdin is not None
            else process.communicate()
        )
        stdout, stderr = await asyncio.wait_for(
            communication,
            timeout=timeout_s,
        )
        completed = True
        return stdout or b"", stderr or b""
    finally:
        if not completed:
            original_exception = sys.exception()
            # SIGKILL plus pipe draining should complete promptly. Bound this
            # last-resort cleanup so an OS-level kill failure cannot make the
            # caller permanently uncancellable.
            cleanup_task = asyncio.create_task(
                asyncio.wait_for(
                    _kill_and_reap(process),
                    timeout=_PROCESS_CLEANUP_TIMEOUT_S,
                )
            )
            later_cancellation: asyncio.CancelledError | None = None

            # Shield the child cleanup from this task's cancellation. A caller
            # may cancel more than once during shutdown, so keep joining the
            # same cleanup task until the child has actually been reaped.
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError as exc:
                    later_cancellation = exc
                except Exception:
                    break

            # Cleanup is best-effort and must not replace the communication
            # failure. _kill_and_reap normally consumes its own errors, but
            # retrieving the result also prevents an unobserved-task warning.
            with contextlib.suppress(BaseException):
                cleanup_task.result()

            # Cancellation that arrived only while cleaning up should still
            # be honored. If cancellation was already the original outcome,
            # completing this finally block lets that original instance pass.
            if later_cancellation is not None and not isinstance(
                original_exception, asyncio.CancelledError
            ):
                raise later_cancellation
