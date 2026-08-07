# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ingest pHash sweeper service.

Mirrors ``tests/unit/test_clip_embedding.py`` structure — same workspace
+ project + image-on-disk fixtures, no external HTTP since the sweeper
has no NIM dependency.

Six scenarios cover the canonical worker contract:
  1. Happy-path single pass — N rows get pHash, ``ingest_completed`` SSE
  2. Idempotent re-run — running the worker twice does no extra work
  3. Restart recovery — ``recover_ingest_tasks`` triggers for projects
     with at least one null-pHash row
  4. Per-row error tolerance — unreadable file leaves row at NULL but
     sweep terminates cleanly with no ``run_failed``
  5. Dedup — ``trigger_ingest_processing`` twice in a row is a no-op
  6. Shutdown mid-pass — ``is_shutting_down`` short-circuits without
     emitting ``ingest_completed``
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, PropertyMock, patch

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from conftest import (
    create_project_via_api,
    make_api_client,
    make_settings,
    make_test_image,
)
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.services.background import background_manager
from vlm_feedback_loop.services.ingest_sweeper_service import (
    _ingest_worker,
    recover_ingest_tasks,
    trigger_ingest_processing,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _seed_skeleton_rows(
    project_id: str,
    workspace_root: str,
    count: int,
    image_dir: Path,
    prefix: str = "img",
) -> list[Path]:
    """Insert ``count`` skeleton Example rows directly with ``phash=None``.

    Returns the list of created image paths so the test can also assert
    on filesystem state if it wants. Bypasses the HTTP ingest endpoint
    so the test isolates the sweeper from the upstream endpoint.
    """
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(project_id, workspace_root)
    assert engine is not None
    paths: list[Path] = []
    with Session(engine) as session:
        for i in range(count):
            img_path = make_test_image(image_dir / f"{prefix}_{i}.jpg")
            paths.append(img_path)
            session.add(
                Example(
                    example_key=f"{prefix}_{i}--abc",
                    project_id=project_id,
                    storage_ref=str(img_path),
                    ingested_at=utc_now(),
                    source_metadata={},
                    state="Unlabeled",
                    phash=None,
                )
            )
        session.commit()
    return paths


def _count_null_phash(project_id: str, workspace_root: str) -> int:
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(project_id, workspace_root)
    assert engine is not None
    with Session(engine) as session:
        return len(
            session.execute(
                select(Example.example_key).where(
                    Example.project_id == project_id,
                    Example.phash.is_(None),
                )
            )
            .scalars()
            .all()
        )


# ── Tests ───────────────────────────────────────────────────────────────────


class TestSweeperHappyPath:
    """Single-pass happy path: N rows get pHash, ``ingest_completed`` fires."""

    @pytest.mark.asyncio
    async def test_single_pass_populates_all_rows(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]

        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=10, image_dir=tmp_path)

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.sse_manager.emit",
            new=AsyncMock(),
        ) as mock_emit:
            await _ingest_worker(pid, settings.WORKSPACE_ROOT, settings)

        assert _count_null_phash(pid, settings.WORKSPACE_ROOT) == 0
        # ingest_completed fires exactly once with processed=total=10
        completed_calls = [
            c for c in mock_emit.call_args_list if c.args[1] == "ingest_completed"
        ]
        assert len(completed_calls) == 1
        assert completed_calls[0].args[2] == {"processed": 10, "total": 10}


class TestSweeperIdempotency:
    """Running the worker twice on the same data is a no-op the second time."""

    @pytest.mark.asyncio
    async def test_second_run_does_no_work(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]
        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=5, image_dir=tmp_path)

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.sse_manager.emit",
            new=AsyncMock(),
        ):
            await _ingest_worker(pid, settings.WORKSPACE_ROOT, settings)

        assert _count_null_phash(pid, settings.WORKSPACE_ROOT) == 0

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.sse_manager.emit",
            new=AsyncMock(),
        ) as mock_emit:
            await _ingest_worker(pid, settings.WORKSPACE_ROOT, settings)

        # Second run finds no remaining rows → exits without dispatching
        # any progress batch. Only the terminal ``ingest_completed`` fires
        # with processed=total=0.
        progress_calls = [
            c for c in mock_emit.call_args_list if c.args[1] == "ingest_progress"
        ]
        assert progress_calls == []
        completed_calls = [
            c for c in mock_emit.call_args_list if c.args[1] == "ingest_completed"
        ]
        assert len(completed_calls) == 1
        assert completed_calls[0].args[2] == {"processed": 0, "total": 0}


class TestSweeperRecovery:
    """``recover_ingest_tasks`` triggers the sweeper for projects with null pHash."""

    @pytest.mark.asyncio
    async def test_recovery_triggers_for_null_phash_rows(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]
        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=4, image_dir=tmp_path)

        task_id = f"ingest-sweep-{pid}"
        # Pre-condition: no task registered.
        assert task_id not in background_manager.active_task_ids

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.trigger_ingest_processing"
        ) as mock_trigger:
            await recover_ingest_tasks(settings)
        # Recovery saw the null-pHash rows and called trigger.
        mock_trigger.assert_called_once_with(pid, settings.WORKSPACE_ROOT, settings)

    @pytest.mark.asyncio
    async def test_recovery_skips_when_no_null_phash(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        create_project_via_api(client)
        # No skeleton rows — project has zero examples → nothing to recover.

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.trigger_ingest_processing"
        ) as mock_trigger:
            await recover_ingest_tasks(settings)
        mock_trigger.assert_not_called()

    @pytest.mark.asyncio
    async def test_recovery_skips_archived_projects(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]
        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=2, image_dir=tmp_path)

        # Drop the .archived marker file the way the soft-archive endpoint
        # does at runtime.
        archived_marker = Path(settings.WORKSPACE_ROOT) / "projects" / pid / ".archived"
        archived_marker.touch()

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.trigger_ingest_processing"
        ) as mock_trigger:
            await recover_ingest_tasks(settings)
        mock_trigger.assert_not_called()


class TestSweeperPerRowErrorTolerance:
    """An unreadable file leaves the row at NULL but the sweep terminates."""

    @pytest.mark.asyncio
    async def test_unreadable_file_does_not_abort_sweep(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]
        # Seed 4 good rows + 1 row pointing at a path that doesn't exist.
        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=4, image_dir=tmp_path)

        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            session.add(
                Example(
                    example_key="img_bad--bad",
                    project_id=pid,
                    storage_ref=str(tmp_path / "does_not_exist.jpg"),
                    ingested_at=utc_now(),
                    source_metadata={},
                    state="Unlabeled",
                    phash=None,
                )
            )
            session.commit()

        with patch(
            "vlm_feedback_loop.services.ingest_sweeper_service.sse_manager.emit",
            new=AsyncMock(),
        ) as mock_emit:
            await _ingest_worker(pid, settings.WORKSPACE_ROOT, settings)

        # Bad row stays at NULL; the 4 good rows are populated.
        assert _count_null_phash(pid, settings.WORKSPACE_ROOT) == 1
        # No run_failed event — per-row failures don't escalate.
        run_failed_calls = [
            c for c in mock_emit.call_args_list if c.args[1] == "run_failed"
        ]
        assert run_failed_calls == []
        # ingest_completed still fires.
        completed_calls = [
            c for c in mock_emit.call_args_list if c.args[1] == "ingest_completed"
        ]
        assert len(completed_calls) == 1


class TestSweeperDedup:
    """``trigger_ingest_processing`` twice in a row is a no-op."""

    def test_second_trigger_is_noop_when_task_running(self, tmp_path: Path):
        """When a sweeper task is already registered, a second trigger does nothing.

        Synchronous test — uses a fake ``active_task_ids`` so we don't
        need to actually run an asyncio task. The contract is:
        ``task_id in active_task_ids → return without registering``.
        """
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]

        task_id = f"ingest-sweep-{pid}"

        # ``active_task_ids`` is a property — patch it via PropertyMock on
        # the class, not the instance attribute (which has no setter).
        # The trigger should see the task as "already running" and bail
        # before calling ``register``.
        from vlm_feedback_loop.services.background import BackgroundTaskManager

        with (
            patch.object(
                BackgroundTaskManager,
                "active_task_ids",
                new_callable=PropertyMock,
                return_value=[task_id],
            ),
            patch(
                "vlm_feedback_loop.services.ingest_sweeper_service.background_manager.register"
            ) as mock_register,
        ):
            trigger_ingest_processing(pid, settings.WORKSPACE_ROOT, settings)

        mock_register.assert_not_called()


class TestSweeperShutdown:
    """``is_shutting_down`` short-circuits without emitting ``ingest_completed``."""

    @pytest.mark.asyncio
    async def test_no_completed_event_on_shutdown(self, tmp_path: Path):
        settings = make_settings(tmp_path / "workspace")
        client = make_api_client(tmp_path, settings=settings)
        project = create_project_via_api(client)
        pid = project["project_id"]
        _seed_skeleton_rows(pid, settings.WORKSPACE_ROOT, count=3, image_dir=tmp_path)

        # Pretend the backend started shutting down before the worker
        # even queried for pending rows. The worker exits at the top of
        # the outer loop without firing any SSE.
        with (
            patch(
                "vlm_feedback_loop.services.ingest_sweeper_service.background_manager.is_shutting_down",
                return_value=True,
            ),
            patch(
                "vlm_feedback_loop.services.ingest_sweeper_service.sse_manager.emit",
                new=AsyncMock(),
            ) as mock_emit,
        ):
            await _ingest_worker(pid, settings.WORKSPACE_ROOT, settings)

        assert mock_emit.call_args_list == []
        # Rows still null — recovery on next startup picks them up.
        assert _count_null_phash(pid, settings.WORKSPACE_ROOT) == 3
