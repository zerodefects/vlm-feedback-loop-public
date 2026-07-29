# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``BackgroundTaskManager._on_task_done`` failure logging.

Formatting the failure as ``"Background task failed: %s — %s"`` with the
exception object's ``__str__`` alone is not enough: when the underlying
exception has an empty ``__str__`` (e.g. a bare ``Exception()``,
``httpx.ReadError("")``, or a SQLAlchemy ``IntegrityError`` whose
detail lives only in ``__cause__``), the log reads literally
``"Background task failed: <task_id> — "`` — useless for debugging.
Evidence: a local-NIM ``_poll_health`` background task failed at
register time with exactly this shape and the empty error message hid
the cause for hours.

These tests pin the contract that ``_on_task_done`` logs the full
traceback via ``exc_info`` so a background task failing with an
empty-str exception still shows the operator the call site and
exception type immediately.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from vlm_feedback_loop.services.background import BackgroundTaskManager


@pytest.mark.asyncio
async def test_failed_task_logs_full_traceback_when_exception_str_is_empty(caplog):
    # The bug surface: an exception with empty __str__ produced an
    # empty-after-the-dash log line that silently swallowed the cause.
    class _EmptyMessageError(Exception):
        pass

    async def _failing_coro():
        raise _EmptyMessageError()  # str(exc) == ""

    manager = BackgroundTaskManager()
    with caplog.at_level(logging.ERROR, logger="vlm_feedback_loop.services.background"):
        task = manager.register("test-empty-failure", _failing_coro())
        # Wait for the task to settle.
        await asyncio.gather(task, return_exceptions=True)

    failure_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert failure_records, "expected at least one error log for the failed task"
    record = failure_records[-1]
    assert "test-empty-failure" in record.getMessage()
    # Invariant: exc_info is populated so the formatted record
    # carries the exception type + traceback even when str(exc) is empty.
    assert record.exc_info is not None, (
        "background task failure must log full exc_info so empty-str "
        "exceptions don't silently swallow root cause"
    )
    exc_type, exc, tb = record.exc_info
    assert exc_type is _EmptyMessageError
    assert tb is not None
    # The Python logging machinery embeds the traceback when exc_info is
    # set; verify the type name is present in the formatted output.
    formatted_full = (
        record.getMessage()
        + "\n"
        + ("".join(__import__("traceback").format_exception(exc_type, exc, tb)))
    )
    assert "_EmptyMessageError" in formatted_full
    assert "_failing_coro" in formatted_full


@pytest.mark.asyncio
async def test_succeeded_task_does_not_log_traceback(caplog):
    """Regression guard: success path remains a single info line, no exc_info."""

    async def _ok_coro():
        return None

    manager = BackgroundTaskManager()
    with caplog.at_level(logging.INFO, logger="vlm_feedback_loop.services.background"):
        task = manager.register("test-success", _ok_coro())
        await asyncio.gather(task, return_exceptions=True)

    completed_records = [
        r
        for r in caplog.records
        if "Background task completed" in r.getMessage()
        and "test-success" in r.getMessage()
    ]
    assert completed_records, (
        "expected exactly one 'Background task completed' info line"
    )
    assert all(r.exc_info is None for r in completed_records)


def test_structured_json_formatter_includes_exc_info_in_details():
    """Empty-message exceptions must surface the traceback in the JSON output.

    If ``StructuredJsonFormatter`` strips ``exc_info``, then even when
    ``_on_task_done`` correctly passes it, the JSON line that lands on
    stdout is identical to a no-op — the empty-message failure mode
    stays invisible on the structured-logging path.
    """
    import json
    import sys

    from vlm_feedback_loop.services.logging_config import StructuredJsonFormatter

    class _Empty(Exception):
        pass

    try:
        raise _Empty()
    except _Empty:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test.module",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Background task failed: %s — %s",
        args=("test-id", _Empty()),
        exc_info=exc_info,
    )

    rendered = StructuredJsonFormatter().format(record)
    parsed = json.loads(rendered)
    assert parsed["level"] == "error"
    assert "Background task failed" in parsed["message"]
    # Invariant: exc_info renders into details.exc_info.
    assert parsed["details"] is not None
    assert "exc_info" in parsed["details"]
    assert "_Empty" in parsed["details"]["exc_info"]
    assert "Traceback" in parsed["details"]["exc_info"]


def test_structured_json_formatter_no_exc_info_when_none():
    """Regression: no exc_info → details stays as the caller set it."""
    import json

    from vlm_feedback_loop.services.logging_config import StructuredJsonFormatter

    record = logging.LogRecord(
        name="test.module",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="all clear",
        args=None,
        exc_info=None,
    )
    record.details = {"some_field": 42}
    rendered = StructuredJsonFormatter().format(record)
    parsed = json.loads(rendered)
    assert parsed["details"] == {"some_field": 42}
