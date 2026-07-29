# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for per-project lifespan recovery isolation.

One corrupt project DB (Alembic state drifted such that
``open_project_db`` raises ``DatabaseMigrationError``) must not
propagate up through the FastAPI lifespan and prevent backend startup
for every other project in the workspace. These tests pin the local-NIM,
embedding, and provisional Training Suite recovery scans to skip the
broken project with a logged warning and continue with healthy ones.

The other two lifespan recovery functions
(``main.py::_recover_interrupted_runs`` and
``services/tao_job_service.py::recover_interrupted_tao_jobs``) catch
``open_project_db`` failures explicitly themselves, so they are not
covered here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def workspace_with_two_projects(tmp_path: Path) -> tuple[Path, str, str]:
    """Stage a tmp workspace with two project directories.

    The directory structure mimics the real one, but neither
    DB is initialized. The recovery functions iterate by directory; we
    monkeypatch ``get_project_engine`` to control which one fails.
    """
    workspace = tmp_path / "vlm_workspace"
    healthy_id = "11111111-aaaa-bbbb-cccc-000000000001"
    broken_id = "22222222-aaaa-bbbb-cccc-000000000002"
    for pid in (healthy_id, broken_id):
        proj = workspace / "projects" / pid
        proj.mkdir(parents=True, exist_ok=True)
        # Empty file is enough — the iteration check is ``project.db.exists()``.
        (proj / "project.db").touch()
    return workspace, healthy_id, broken_id


# ── recover_local_deployments ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_local_deployments_isolates_broken_project(
    workspace_with_two_projects, caplog
):
    """One broken project DB MUST NOT prevent the other project from being
    inspected. Without per-project isolation, ``DatabaseMigrationError``
    from ``get_project_engine`` propagates up through lifespan and the
    entire backend fails to start."""
    workspace, healthy_id, broken_id = workspace_with_two_projects
    from vlm_feedback_loop.db.engine import DatabaseMigrationError
    from vlm_feedback_loop.services import local_nim_service

    visited: list[str] = []

    def _fakeget_project_engine(project_id: str, _workspace_root: str):
        visited.append(project_id)
        if project_id == broken_id:
            raise DatabaseMigrationError(
                "Migration failed for /tmp/.../project.db. Error: "
                "(sqlite3.OperationalError) table projects already exists"
            )
        return None  # healthy path returns engine; None short-circuits the rest cleanly

    settings = type("S", (), {"WORKSPACE_ROOT": str(workspace)})()

    with patch.object(
        local_nim_service, "get_project_engine", side_effect=_fakeget_project_engine
    ):
        with caplog.at_level(
            logging.WARNING, logger="vlm_feedback_loop.services.local_nim_service"
        ):
            # MUST NOT raise.
            await local_nim_service.recover_local_deployments(
                str(workspace),
                settings,  # type: ignore[arg-type]
            )

    # Both projects were attempted — broken one didn't short-circuit healthy.
    assert healthy_id in visited
    assert broken_id in visited
    # The broken project surfaced a warning naming the project_id and the
    # exception class, so an operator immediately sees which DB to fix.
    warning_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "Skipping local-NIM recovery" in r.getMessage()
    ]
    assert any(broken_id in m for m in warning_msgs), (
        f"expected a warning citing the broken project {broken_id}; got: {warning_msgs}"
    )
    assert any("DatabaseMigrationError" in m for m in warning_msgs), (
        f"expected exception class name in the warning; got: {warning_msgs}"
    )


# ── recover_interrupted_training_suite_setups ───────────────────────────────


def test_recover_training_suite_setups_isolates_broken_project(
    workspace_with_two_projects, caplog
):
    """A legacy DB cannot abort startup during provisional-suite recovery."""
    workspace, healthy_id, broken_id = workspace_with_two_projects
    from vlm_feedback_loop.db.engine import DatabaseMigrationError
    from vlm_feedback_loop.services import training_suite_service

    visited: list[str] = []

    class _EmptyQuery:
        def filter(self, *_args):
            return self

        def all(self):
            return []

    class _EmptySession:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, *_args):
            return _EmptyQuery()

    def _fake_open_project_db(project_dir: Path):
        visited.append(project_dir.name)
        if project_dir.name == broken_id:
            raise DatabaseMigrationError("unsupported pre-public revision")
        return object()

    settings = type("S", (), {"WORKSPACE_ROOT": str(workspace)})()
    with (
        patch.object(
            training_suite_service,
            "open_project_db",
            side_effect=_fake_open_project_db,
        ),
        patch.object(training_suite_service, "Session", return_value=_EmptySession()),
        caplog.at_level(
            logging.WARNING,
            logger="vlm_feedback_loop.training_suite_service",
        ),
    ):
        recovered = training_suite_service.recover_interrupted_training_suite_setups(
            settings  # type: ignore[arg-type]
        )

    assert recovered == 0
    assert healthy_id in visited
    assert broken_id in visited
    warning_msgs = [
        record.getMessage()
        for record in caplog.records
        if "Skipping Training Suite setup recovery" in record.getMessage()
    ]
    assert any(broken_id in message for message in warning_msgs)
    assert any("DatabaseMigrationError" in message for message in warning_msgs)


# ── recover_embedding_tasks ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recover_embedding_tasks_isolates_broken_project(
    workspace_with_two_projects, caplog
):
    """Same isolation guarantee for the embedding-recovery path."""
    workspace, healthy_id, broken_id = workspace_with_two_projects
    from vlm_feedback_loop.db.engine import DatabaseMigrationError
    from vlm_feedback_loop.services import clip_embedding_service

    visited: list[str] = []

    def _fakeget_project_engine(project_id: str, _workspace_root: str):
        visited.append(project_id)
        if project_id == broken_id:
            raise DatabaseMigrationError("simulated corruption")
        return None  # healthy path: no project row → continue without triggering worker

    settings = type(
        "S",
        (),
        {
            "WORKSPACE_ROOT": str(workspace),
            "EMBEDDING_PROVIDER": "auto",
            "EMBEDDINGS_AUTO_COMPUTE": True,
        },
    )()

    with patch.object(
        clip_embedding_service,
        "get_project_engine",
        side_effect=_fakeget_project_engine,
    ):
        with caplog.at_level(
            logging.WARNING, logger="vlm_feedback_loop.clip_embedding"
        ):
            await clip_embedding_service.recover_embedding_tasks(
                settings  # type: ignore[arg-type]
            )

    assert healthy_id in visited
    assert broken_id in visited
    warning_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "Skipping embedding recovery" in r.getMessage()
    ]
    assert any(broken_id in m for m in warning_msgs), (
        f"expected a warning citing the broken project {broken_id}; got: {warning_msgs}"
    )
