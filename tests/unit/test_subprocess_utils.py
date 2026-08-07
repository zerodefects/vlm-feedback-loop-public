# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Child-process cleanup on completion, timeout, and cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import sys
from typing import cast

import pytest

from vlm_feedback_loop.services.subprocess_utils import communicate_with_timeout


class _SuccessfulProcess:
    returncode = 0

    def __init__(self) -> None:
        self.events: list[str] = []

    async def communicate(self, input=None):
        self.events.append(f"communicate:{input!r}")
        return b"out", b"err"

    def kill(self) -> None:
        self.events.append("kill")


class _BlockingProcess:
    returncode: int | None = None

    def __init__(self) -> None:
        self.events: list[str] = []
        self.communicate_count = 0

    async def communicate(self, input=None):
        self.communicate_count += 1
        self.events.append(f"communicate:{self.communicate_count}")
        if self.communicate_count == 1:
            await asyncio.Future()
        return b"", b""

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9


class _FailingProcess:
    returncode: int | None = None

    def __init__(self) -> None:
        self.events: list[str] = []
        self.communicate_count = 0

    async def communicate(self, input=None):
        self.communicate_count += 1
        self.events.append(f"communicate:{self.communicate_count}")
        if self.communicate_count == 1:
            raise BrokenPipeError("pipe failed")
        raise RuntimeError("cleanup failed")

    def kill(self) -> None:
        self.events.append("kill")
        self.returncode = -9

    async def wait(self) -> int:
        self.events.append("wait")
        return -9


class _SlowCleanupProcess(_BlockingProcess):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.reaped = False

    async def communicate(self, input=None):
        self.communicate_count += 1
        self.events.append(f"communicate:{self.communicate_count}")
        if self.communicate_count == 1:
            await asyncio.Future()
        self.cleanup_started.set()
        await self.allow_cleanup.wait()
        self.reaped = True
        return b"", b""


class _KillFailureProcess(_FailingProcess):
    def kill(self) -> None:
        self.events.append("kill")
        raise PermissionError("kill failed")


class _HangingKillFailureProcess(_BlockingProcess):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_cancelled = False

    async def communicate(self, input=None):
        self.communicate_count += 1
        self.events.append(f"communicate:{self.communicate_count}")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            if self.communicate_count == 2:
                self.cleanup_cancelled = True
            raise

    def kill(self) -> None:
        self.events.append("kill")
        raise PermissionError("kill failed")


class _ProcessGroupMember(_BlockingProcess):
    pid = 43210


@pytest.mark.asyncio
async def test_success_returns_output_without_killing():
    process = _SuccessfulProcess()

    result = await communicate_with_timeout(
        cast("asyncio.subprocess.Process", process),
        timeout_s=1.0,
        stdin=b"input",
    )

    assert result == (b"out", b"err")
    assert process.events == ["communicate:b'input'"]


@pytest.mark.asyncio
async def test_timeout_kills_and_reaps_before_reraising():
    process = _BlockingProcess()

    with pytest.raises(TimeoutError):
        await communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=0.001,
        )

    assert process.events == ["communicate:1", "kill", "communicate:2"]


@pytest.mark.asyncio
async def test_cancellation_kills_and_reaps_before_propagating():
    process = _BlockingProcess()
    task = asyncio.create_task(
        communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=10.0,
        )
    )
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.events == ["communicate:1", "kill", "communicate:2"]


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_reaping():
    process = _SlowCleanupProcess()
    task = asyncio.create_task(
        communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=10.0,
        )
    )
    await asyncio.sleep(0)

    task.cancel()
    await process.cleanup_started.wait()
    task.cancel()
    process.allow_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.reaped is True
    assert process.events == ["communicate:1", "kill", "communicate:2"]


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_mask_pipe_failure():
    process = _FailingProcess()

    with pytest.raises(BrokenPipeError, match="pipe failed"):
        await communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=1.0,
        )

    assert process.events == ["communicate:1", "kill", "communicate:2", "wait"]


@pytest.mark.asyncio
async def test_kill_failure_does_not_mask_pipe_failure_or_skip_reap():
    process = _KillFailureProcess()

    with pytest.raises(BrokenPipeError, match="pipe failed"):
        await communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=1.0,
        )

    assert process.events == ["communicate:1", "kill", "communicate:2", "wait"]


@pytest.mark.asyncio
async def test_kill_failure_cannot_block_cancellation_forever(monkeypatch):
    from vlm_feedback_loop.services import subprocess_utils

    monkeypatch.setattr(subprocess_utils, "_PROCESS_CLEANUP_TIMEOUT_S", 0.01)
    process = _HangingKillFailureProcess()
    task = asyncio.create_task(
        communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=10.0,
        )
    )
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)

    assert process.cleanup_cancelled is True
    assert process.events == ["communicate:1", "kill", "communicate:2"]


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_timeout_kills_the_managed_process_group(monkeypatch):
    from vlm_feedback_loop.services import subprocess_utils

    process = _ProcessGroupMember()
    killed: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        subprocess_utils.os,
        "killpg",
        lambda process_id, sig: killed.append((process_id, sig)),
    )

    with pytest.raises(TimeoutError):
        await communicate_with_timeout(
            cast("asyncio.subprocess.Process", process),
            timeout_s=0.001,
        )

    assert killed == [(process.pid, signal.SIGKILL)]
    assert process.events == ["communicate:1", "communicate:2"]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc state")
@pytest.mark.asyncio
async def test_exited_session_leader_cannot_leave_a_pipe_holding_worker():
    parent_code = """
import subprocess
import sys
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
print(child.pid, flush=True)
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        parent_code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    descendant_pid = int((await process.stdout.readline()).decode().strip())

    try:
        for _ in range(100):
            if process.returncode is not None:
                break
            await asyncio.sleep(0.01)
        assert process.returncode == 0

        with pytest.raises(TimeoutError):
            await communicate_with_timeout(process, timeout_s=0.05)

        for _ in range(100):
            proc_stat = f"/proc/{descendant_pid}/stat"
            if not os.path.exists(proc_stat):
                break
            try:
                with open(proc_stat, encoding="utf-8") as stat_file:
                    if stat_file.read().split()[2] == "Z":
                        break
            except FileNotFoundError:
                # The cleanup succeeded between exists() and open().
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("descendant survived cleanup of its exited session leader")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.communicate(), timeout=1.0)
        with contextlib.suppress(ProcessLookupError):
            os.kill(descendant_pid, signal.SIGKILL)
