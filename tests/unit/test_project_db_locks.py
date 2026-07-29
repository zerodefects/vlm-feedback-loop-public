# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-project SQLite write lock registry.

The three writers that share one project.db (the :ingest handler, the
pHash sweeper, the CLIP embedding worker) only serialize correctly if
``get_project_write_lock`` hands them the *same* lock object. None of
their individual worker tests can catch a registry regression — each
worker runs fine holding a private lock; the failure only appears as
``database is locked`` errors under concurrent load.
"""

from vlm_feedback_loop.services.project_db_locks import get_project_write_lock


async def test_same_project_returns_same_lock_instance():
    """Every caller asking for a project's write lock must get the one
    shared lock object — otherwise the ingest handler, pHash sweeper,
    and CLIP worker each hold a private lock and stop serializing."""
    lock = get_project_write_lock("proj-lock-identity")

    async with lock:
        # A second caller during the first caller's write window must
        # see the held lock, not a fresh acquirable one.
        second = get_project_write_lock("proj-lock-identity")
        assert second is lock
        assert second.locked()


async def test_different_projects_get_independent_locks():
    """Writes to one project must not block writes to another: distinct
    project ids map to distinct locks, and holding one leaves the other
    acquirable."""
    lock_a = get_project_write_lock("proj-lock-indep-a")
    lock_b = get_project_write_lock("proj-lock-indep-b")

    assert lock_a is not lock_b
    async with lock_a:
        assert not lock_b.locked()
