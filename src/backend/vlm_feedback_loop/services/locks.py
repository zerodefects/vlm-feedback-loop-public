# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-process file lock per project.

Uses ``fcntl.flock(LOCK_EX | LOCK_NB)`` on a ``project.lock`` file
separate from ``project.db`` to avoid interfering with SQLite locking.
"""

from __future__ import annotations

import fcntl
import logging
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger("vlm_feedback_loop.services.locks")


class ProjectLockedError(Exception):
    """Raised when a project is already locked by another process."""

    pass


# Track acquired locks: project_id → (fd, file_object).
# The file object MUST be kept alive or GC closes the FD, releasing the lock.
_active_locks: dict[str, tuple[int, object]] = {}


def acquire_project_lock(project_dir: Path) -> int:
    """Acquire an exclusive file lock on ``{project_dir}/project.lock``.

    Non-blocking — fails immediately if another process holds the lock.

    Returns:
        The file descriptor of the lock file.

    Raises:
        ProjectLockedError: if the lock is already held.
    """
    lock_path = project_dir / "project.lock"
    project_id = project_dir.name

    # Skip if we already hold this lock
    if project_id in _active_locks:
        return _active_locks[project_id][0]

    # NOTE: intentionally not using `with open(...)` — the file handle's lifetime
    # is tied to the process (the advisory lock is released only when the fd closes),
    # and we transfer ownership of the handle to the module-level `_active_locks`
    # dict below.  SIM115 is suppressed on the open() line.
    f = open(lock_path, "w")  # noqa: SIM115
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        f.close()
        raise ProjectLockedError(
            "This project is already open in another process."
        ) from exc

    _active_locks[project_id] = (f.fileno(), f)
    logger.info("Project lock acquired: %s", project_dir)
    return f.fileno()


def release_project_lock(project_id: str) -> None:
    """Release a previously acquired project lock."""
    entry = _active_locks.pop(project_id, None)
    if entry is None:
        return
    fd, f = entry
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        cast("Any", f).close()
    except OSError:
        pass
    logger.info("Project lock released: %s", project_id)


def release_all_locks() -> None:
    """Release all project locks. Called during shutdown."""
    for project_id in list(_active_locks.keys()):
        release_project_lock(project_id)


def clear_lock_state() -> None:
    """Clear lock tracking without releasing OS locks. For testing only."""
    _active_locks.clear()
