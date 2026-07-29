# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-project warning throttling on project-open failures.

Without throttling, the polling tick emits an identical WARN line every
cadence interval (30–60 s) for every project the backend can't open —
most commonly the ``ProjectLockedError`` race against an in-process
``scripts/rps_e2e.py`` BUILD-mode invocation. Over a multi-hour session
with several orphan projects that produces thousands of duplicate WARN
lines that drown out genuine progress signal.

:func:`_log_project_open_failure` collapses identical recurrences
into a single WARN per ``_PROJECT_OPEN_WARNING_THROTTLE_S`` window
(default 300 s). DEBUG-level logging still fires every tick so
``LOG_LEVEL=debug`` operators retain full granularity.
"""

from __future__ import annotations

import logging

import pytest

from vlm_feedback_loop.services import tao_polling_service
from vlm_feedback_loop.services.tao_polling_service import (
    _PROJECT_OPEN_WARNING_THROTTLE_S,
    _log_project_open_failure,
)


@pytest.fixture(autouse=True)
def _reset_throttle_state():
    """Each test starts with an empty throttle dict — isolation matters
    because the module-level dict persists across calls in production."""
    tao_polling_service._project_open_warning_last_at.clear()
    yield
    tao_polling_service._project_open_warning_last_at.clear()


class _FakeProjectLockedError(Exception):
    """Stand-in for the ``ProjectLockedError`` we see in production."""


class TestProjectOpenWarningThrottle:
    def test_first_occurrence_emits_warn(self, caplog) -> None:
        caplog.set_level(
            logging.WARNING, logger="vlm_feedback_loop.tao_polling_service"
        )
        exc = _FakeProjectLockedError("project busy")

        _log_project_open_failure("proj-A", exc, now=0.0)

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        assert "proj-A" in warns[0].getMessage()
        assert "_FakeProjectLockedError" in warns[0].getMessage()
        assert "project busy" in warns[0].getMessage()

    def test_second_occurrence_within_window_is_debug(self, caplog) -> None:
        """Repeated within throttle window → DEBUG (not WARN)."""
        caplog.set_level(logging.DEBUG, logger="vlm_feedback_loop.tao_polling_service")
        exc = _FakeProjectLockedError("project busy")

        _log_project_open_failure("proj-A", exc, now=0.0)
        # 30s later — well inside the 300s default window.
        _log_project_open_failure("proj-A", exc, now=30.0)

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        debugs = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(warns) == 1, "only first occurrence is WARN"
        assert len(debugs) == 1, "second occurrence is DEBUG"
        # DEBUG message must carry the same diagnostic payload as WARN —
        # operators with LOG_LEVEL=debug must still get full granularity.
        assert "proj-A" in debugs[0].getMessage()
        assert "_FakeProjectLockedError" in debugs[0].getMessage()

    def test_occurrence_after_window_re_warns(self, caplog) -> None:
        """First WARN; quiet during window; then re-WARN when window expires.

        This is the key behavior — the operator gets periodic re-confirmation
        the issue is ongoing without being drowned by per-tick noise.
        """
        caplog.set_level(
            logging.WARNING, logger="vlm_feedback_loop.tao_polling_service"
        )
        exc = _FakeProjectLockedError("locked")

        _log_project_open_failure("proj-A", exc, now=0.0)
        # Within window — suppressed at WARN level.
        _log_project_open_failure("proj-A", exc, now=100.0)
        # After window expiry — re-WARN.
        _log_project_open_failure("proj-A", exc, now=_PROJECT_OPEN_WARNING_THROTTLE_S)

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 2, "first + post-window WARN"

    def test_different_projects_independently_throttled(self, caplog) -> None:
        """Throttle state is per-project — proj-A's WARN must not suppress proj-B's."""
        caplog.set_level(
            logging.WARNING, logger="vlm_feedback_loop.tao_polling_service"
        )
        exc = _FakeProjectLockedError("locked")

        _log_project_open_failure("proj-A", exc, now=0.0)
        _log_project_open_failure("proj-B", exc, now=0.0)
        _log_project_open_failure("proj-A", exc, now=10.0)  # suppressed
        _log_project_open_failure("proj-B", exc, now=10.0)  # suppressed

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        ids_warned = [r.getMessage() for r in warns]
        assert len(warns) == 2
        assert any("proj-A" in m for m in ids_warned)
        assert any("proj-B" in m for m in ids_warned)

    def test_empty_string_exception_still_produces_signal(self, caplog) -> None:
        """Observability invariant: empty str(exc) falls back to
        ``(no message)`` so the operator still sees the type."""
        caplog.set_level(
            logging.WARNING, logger="vlm_feedback_loop.tao_polling_service"
        )

        class _SilentError(Exception):
            def __str__(self) -> str:
                return ""

        _log_project_open_failure("proj-A", _SilentError(), now=0.0)

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 1
        msg = warns[0].getMessage()
        assert "_SilentError" in msg
        assert "(no message)" in msg
