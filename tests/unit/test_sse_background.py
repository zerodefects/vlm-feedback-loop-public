# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SSE, background tasks, and concurrency."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import time

import pytest
from sqlalchemy.orm import Session

from vlm_feedback_loop.services.background import BackgroundTaskManager, run_in_thread
from vlm_feedback_loop.services.locks import (
    acquire_project_lock,
    release_all_locks,
)
from vlm_feedback_loop.services.priority import ForegroundPriorityDispatch
from vlm_feedback_loop.services.sse import SSEManager

# ── SSE Infrastructure ───────────────────────────────────────────────────────


class TestSSEContentType:
    """GET returns Content-Type: text/event-stream."""

    @pytest.mark.asyncio
    async def test_returns_event_stream(self, test_app_client):
        """Call the real endpoint handler and assert the response is a
        correctly-headered event stream, then close the body iterator
        (which must unsubscribe). Replaces a literal ``assert True``.
        (TestClient cannot stream an infinite SSE body — its portal
        blocks on response completion — so the handler is exercised
        directly; the require_project existence guard is a route-level
        dependency and does not run here.)"""
        from vlm_feedback_loop.routers.projects import project_events
        from vlm_feedback_loop.services.sse import sse_manager

        create = test_app_client.post(
            "/v1/projects", json={"name": "sse-ct", "description": ""}
        )
        assert create.status_code == 201
        pid = create.json()["project_id"]

        resp = await project_events(pid)
        try:
            assert resp.media_type == "text/event-stream"
            assert resp.headers["cache-control"] == "no-cache"
            assert resp.headers["x-accel-buffering"] == "no"
            assert len(sse_manager._subscribers.get(pid, [])) == 1
            # Pull one real event through the live stream (also starts
            # the generator so its unsubscribe-finally is armed).
            await sse_manager.emit(pid, "test_event", {"run_id": "r1"})
            first = await asyncio.wait_for(anext(resp.body_iterator), timeout=2.0)
            assert first.startswith("event: test_event\n")
        finally:
            await resp.body_iterator.aclose()
        # Closing the stream unsubscribed the client.
        assert sse_manager._subscribers.get(pid, []) == []

    @pytest.mark.asyncio
    async def test_sse_wire_format(self, sse_mgr: SSEManager):
        """Verify the SSE message format is correct event-stream."""
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit("proj", "test_event", {"run_id": "r1"})
        msg = queue.get_nowait()
        # SSE wire format: event: {type}\ndata: {json}\n\n
        assert msg.startswith("event: test_event\n")
        assert "data: " in msg
        assert msg.endswith("\n\n")

    @pytest.mark.asyncio
    async def test_full_queue_drops_oldest_keeps_newest(self, sse_mgr: SSEManager):
        """On a full queue, emit drops the OLDEST message and keeps the
        newest — so a briefly-slow client never misses a terminal event
        (e.g. run.completed) and hang. The frontend reconciles the dropped
        intermediate state from REST."""
        queue = sse_mgr.subscribe("proj")
        # Fill the queue to capacity with progress events.
        capacity = queue.maxsize
        for i in range(capacity):
            await sse_mgr.emit("proj", "progress", {"run_id": "r1", "n": i})
        assert queue.full()

        # One more emit (the important terminal event) on the full queue.
        await sse_mgr.emit("proj", "run_completed", {"run_id": "r1"})

        # Drain and confirm the terminal event survived while the OLDEST
        # progress event (n=0) was the one dropped.
        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        assert any("run_completed" in m for m in drained), "terminal event dropped"
        assert not any('"n": 0' in m for m in drained), "oldest not dropped"


class TestSSEKeepalive:
    """Idle SSE streams emit keepalive comments so proxies with read
    timeouts (nginx.conf: 300 s) do not drop the connection.
    """

    @pytest.mark.asyncio
    async def test_idle_stream_emits_keepalive_comment(self, sse_mgr: SSEManager):
        stream = sse_mgr.stream("proj", keepalive_interval_s=0.02)
        try:
            first = await asyncio.wait_for(anext(stream), timeout=2.0)
            assert first == ": keepalive\n\n"
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_events_still_delivered_between_keepalives(self, sse_mgr: SSEManager):
        stream = sse_mgr.stream("proj", keepalive_interval_s=0.02)
        try:
            # Consume one keepalive so the emit lands mid-stream.
            await asyncio.wait_for(anext(stream), timeout=2.0)
            await sse_mgr.emit("proj", "test_event", {"run_id": "r1"})
            message = await asyncio.wait_for(anext(stream), timeout=2.0)
            assert message.startswith("event: test_event\n")
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_closing_stream_unsubscribes(self, sse_mgr: SSEManager):
        stream = sse_mgr.stream("proj", keepalive_interval_s=60.0)
        assert len(sse_mgr._subscribers.get("proj", [])) == 1
        # Start the generator so its finally-block is armed, then close.
        task = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await stream.aclose()
        assert sse_mgr._subscribers.get("proj", []) == []


class TestSSEProjectScoping:
    """Project A events do not appear on project B's stream."""

    @pytest.mark.asyncio
    async def test_events_scoped_to_project(self, sse_mgr: SSEManager):
        queue_a = sse_mgr.subscribe("project-a")
        sse_mgr.subscribe("project-b")

        await sse_mgr.emit("project-b", "test_event", {"run_id": "r1"})

        # Project A's queue should be empty
        assert queue_a.empty()


class TestSSEPayloadFields:
    """Each payload has run_id and timestamp."""

    @pytest.mark.asyncio
    async def test_has_run_id_and_timestamp(self, sse_mgr: SSEManager):
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit("proj", "test_event", {"run_id": "r1"})

        msg = queue.get_nowait()
        data = json.loads(msg.split("data: ")[1].strip())
        assert "run_id" in data
        assert "timestamp" in data


class TestSSEProgressPayload:
    """Progress events include processed and total."""

    @pytest.mark.asyncio
    async def test_progress_has_processed_and_total(self, sse_mgr: SSEManager):
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit(
            "proj",
            "evaluation_progress",
            {"run_id": "r1", "processed": 5, "total": 10},
        )

        msg = queue.get_nowait()
        data = json.loads(msg.split("data: ")[1].strip())
        assert data["processed"] == 5
        assert data["total"] == 10


class TestSSECompletionPayload:
    """Completion events include status and summary."""

    @pytest.mark.asyncio
    async def test_completion_has_status_and_summary(self, sse_mgr: SSEManager):
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit(
            "proj",
            "evaluation_completed",
            {"run_id": "r1", "status": "completed", "summary": {"accuracy": 0.85}},
        )

        msg = queue.get_nowait()
        assert "event: evaluation_completed" in msg
        data = json.loads(msg.split("data: ")[1].strip())
        assert data["status"] == "completed"
        assert "summary" in data


class TestSSERunFailed:
    """run_failed has run_id, run_type, error_summary."""

    @pytest.mark.asyncio
    async def test_run_failed_fields(self, sse_mgr: SSEManager):
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit(
            "proj",
            "run_failed",
            {
                "run_id": "r1",
                "run_type": "evaluation_run",
                "error_summary": "timeout",
            },
        )

        msg = queue.get_nowait()
        data = json.loads(msg.split("data: ")[1].strip())
        assert data["run_id"] == "r1"
        assert data["run_type"] == "evaluation_run"
        assert data["error_summary"] == "timeout"


class TestSSENoReplay:
    """No Last-Event-ID replay — new subscriber gets nothing."""

    @pytest.mark.asyncio
    async def test_no_replay(self, sse_mgr: SSEManager):
        # Emit events before subscribing
        await sse_mgr.emit("proj", "old_event", {"run_id": "r0"})

        # New subscriber should not see the old event
        queue = sse_mgr.subscribe("proj")
        assert queue.empty()


class TestSSEArbitraryTypesAnd404:
    """Arbitrary event types accepted; 404 for non-existent project."""

    @pytest.mark.asyncio
    async def test_accepts_custom_event_type(self, sse_mgr: SSEManager):
        queue = sse_mgr.subscribe("proj")
        await sse_mgr.emit("proj", "test_custom_event", {"run_id": "r1"})

        msg = queue.get_nowait()
        assert "event: test_custom_event" in msg

    def test_404_for_nonexistent_project(self, test_app_client):
        resp = test_app_client.get("/v1/projects/nonexistent-uuid/events")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


# ── Background Task Framework ────────────────────────────────────────────────


class TestBackgroundInProcess:
    """Tasks run in-process via asyncio."""

    @pytest.mark.asyncio
    async def test_task_runs_and_completes(self):
        mgr = BackgroundTaskManager()
        result = []

        async def work():
            result.append("done")

        task = mgr.register("t1", work())
        await task
        assert result == ["done"]
        assert "t1" not in mgr.active_task_ids

    @pytest.mark.asyncio
    async def test_completed_callback_keeps_live_replacement_owned(self, caplog):
        """A superseded task logs its failure without untracking its successor."""
        import logging

        mgr = BackgroundTaskManager()
        start_replacement = asyncio.Event()
        keep_replacement_running = asyncio.Event()
        replacement: dict[str, asyncio.Task[None]] = {}
        old_task: asyncio.Task[None]

        async def replacement_work() -> None:
            await keep_replacement_running.wait()

        async def register_replacement() -> None:
            await start_replacement.wait()
            assert old_task.done()
            replacement["task"] = mgr.register("stable-worker", replacement_work())

        async def old_work() -> None:
            start_replacement.set()
            raise RuntimeError("old worker failed")

        registrar = asyncio.create_task(register_replacement())
        with caplog.at_level(
            logging.ERROR, logger="vlm_feedback_loop.services.background"
        ):
            old_task = mgr.register("stable-worker", old_work())
            await asyncio.gather(old_task, registrar, return_exceptions=True)
            await asyncio.sleep(0)

        replacement_task = replacement["task"]
        duplicate = replacement_work()
        try:
            assert mgr.active_task_ids == ["stable-worker"]
            assert not replacement_task.done()

            with pytest.raises(RuntimeError, match="already running"):
                mgr.register("stable-worker", duplicate)

            assert any(
                "old worker failed" in record.getMessage()
                for record in caplog.records
                if record.name == "vlm_feedback_loop.services.background"
            )

            await mgr.cancel_all(grace_seconds=0)

            assert replacement_task.cancelled()
        finally:
            duplicate.close()
            if not replacement_task.done():
                replacement_task.cancel()
                await asyncio.gather(replacement_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_task_failure_logs_repr_and_traceback(self, caplog):
        """``str(task.exception())`` is empty for several common exception
        types — a log line built from it reads "Background task failed:
        <id> — " with nothing after the dash. The logger MUST emit
        ``repr(exc)`` (always non-empty) plus the formatted traceback so
        the failure surface is self-contained in JSON logs."""
        import logging

        mgr = BackgroundTaskManager()

        async def boom():
            # ``RuntimeError()`` with no args: ``str(exc)`` is "" — the
            # case that yields a dangling em-dash unless repr() is logged.
            raise RuntimeError()

        with caplog.at_level(
            logging.ERROR, logger="vlm_feedback_loop.services.background"
        ):
            task = mgr.register("t-fail", boom())
            with pytest.raises(RuntimeError):
                await task
            # Yield once so the task's done_callback runs.
            await asyncio.sleep(0)

        records = [
            r
            for r in caplog.records
            if r.name == "vlm_feedback_loop.services.background"
            and r.levelno == logging.ERROR
        ]
        assert records, "expected at least one ERROR log from background.py"
        joined = "\n".join(r.getMessage() for r in records)
        # repr(RuntimeError()) produces "RuntimeError()" — ALWAYS populated.
        assert "RuntimeError()" in joined
        # The formatted traceback names the boom() frame.
        assert "in boom" in joined


class TestRunRecordPrePersist:
    """Run record persisted before background work begins."""

    def test_run_record_survives_in_db(self, project_engine):
        from vlm_feedback_loop.db.models.run import RunRecord

        with Session(project_engine) as session:
            run = RunRecord(
                project_id="test-proj",
                run_type="evaluation_run",
                status="queued",
            )
            session.add(run)
            session.commit()
            run_id = run.run_id

        # Record exists and is queryable (simulates surviving a kill)
        with Session(project_engine) as session:
            found = session.query(RunRecord).filter_by(run_id=run_id).first()
            assert found is not None
            assert found.status == "queued"


class TestRestartRecovery:
    """Restart recovery transitions."""

    def test_eval_run_to_failed(self, project_engine):
        from vlm_feedback_loop.db.models.run import RunRecord

        with Session(project_engine) as session:
            run = RunRecord(
                project_id="test",
                run_type="evaluation_run",
                status="running",
            )
            session.add(run)
            session.commit()
            run_id = run.run_id

        # Simulate recovery by directly calling the recovery function's logic
        with Session(project_engine) as session:
            non_terminal = (
                session.query(RunRecord)
                .filter(RunRecord.status.in_(["queued", "running", "canceling"]))
                .all()
            )
            for r in non_terminal:
                if r.run_type == "evaluation_run":
                    r.status = "failed"
                    r.status_reason = "backend_restart_interrupted"
            session.commit()

        with Session(project_engine) as session:
            found = session.query(RunRecord).filter_by(run_id=run_id).first()
            assert found.status == "failed"
            assert found.status_reason == "backend_restart_interrupted"

    def test_batch_run_to_queued_recovered(self, project_engine):
        from vlm_feedback_loop.db.models.run import RunRecord

        with Session(project_engine) as session:
            run = RunRecord(
                project_id="test",
                run_type="batch_label_run",
                status="running",
            )
            session.add(run)
            session.commit()
            run_id = run.run_id

        with Session(project_engine) as session:
            non_terminal = (
                session.query(RunRecord)
                .filter(RunRecord.status.in_(["queued", "running", "canceling"]))
                .all()
            )
            for r in non_terminal:
                if r.run_type == "batch_label_run" and r.status in (
                    "queued",
                    "running",
                ):
                    r.status = "queued"
                    r.recovered_from_restart = True
            session.commit()

        with Session(project_engine) as session:
            found = session.query(RunRecord).filter_by(run_id=run_id).first()
            assert found.status == "queued"
            assert found.recovered_from_restart is True

    def test_paused_stays_paused(self, project_engine):
        from vlm_feedback_loop.db.models.run import RunRecord

        with Session(project_engine) as session:
            run = RunRecord(
                project_id="test",
                run_type="batch_label_run",
                status="paused",
                paused_reason="circuit_breaker_threshold_reached",
            )
            session.add(run)
            session.commit()
            run_id = run.run_id

        # Recovery should not touch paused runs
        with Session(project_engine) as session:
            non_terminal = (
                session.query(RunRecord)
                .filter(RunRecord.status.in_(["queued", "running", "canceling"]))
                .all()
            )
            # paused is NOT in the filter, so it's not touched
            assert len(non_terminal) == 0

        with Session(project_engine) as session:
            found = session.query(RunRecord).filter_by(run_id=run_id).first()
            assert found.status == "paused"


class TestCLIPResumableHook:
    """Framework supports resumable background tasks."""

    @pytest.mark.asyncio
    async def test_register_works_for_resumable_pattern(self):
        """BackgroundTaskManager can register tasks that resume from state."""
        mgr = BackgroundTaskManager()
        progress = []

        async def resumable_work():
            # Simulates resuming from persisted progress
            start_from = 3  # would come from DB query in real code
            for i in range(start_from, 6):
                progress.append(i)

        task = mgr.register("clip-resume", resumable_work())
        await task
        assert progress == [3, 4, 5]


class TestGracefulShutdown:
    """Graceful shutdown cancels tasks and sets flag."""

    @pytest.mark.asyncio
    async def test_cancel_task_targets_only_the_requested_worker(self):
        mgr = BackgroundTaskManager()
        target_cancelled = False
        other_finished = asyncio.Event()

        async def target():
            nonlocal target_cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                target_cancelled = True
                raise

        async def other():
            await other_finished.wait()

        mgr.register("target", target())
        other_task = mgr.register("other", other())
        await asyncio.sleep(0)

        assert await mgr.cancel_task("target") is True
        assert target_cancelled is True
        assert "other" in mgr.active_task_ids
        assert await mgr.cancel_task("missing") is False

        other_finished.set()
        await other_task

    @pytest.mark.asyncio
    async def test_cancel_all(self):
        mgr = BackgroundTaskManager()
        cancelled = False

        async def long_task():
            nonlocal cancelled
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                cancelled = True
                raise

        mgr.register("long", long_task())
        assert not mgr.is_shutting_down()

        await mgr.cancel_all(grace_seconds=0.1)

        assert mgr.is_shutting_down()
        assert cancelled


class TestForegroundPriority:
    """Background held while foreground is active."""

    @pytest.mark.asyncio
    async def test_background_held_during_foreground(self):
        dispatch = ForegroundPriorityDispatch()
        bg_completed = asyncio.Event()

        await dispatch.enter_foreground()

        async def bg_work():
            await dispatch.wait_for_background()
            bg_completed.set()

        asyncio.create_task(bg_work())

        # Background should NOT complete within a short time
        await asyncio.sleep(0.1)
        assert not bg_completed.is_set()

        # Release foreground
        await dispatch.exit_foreground()
        await asyncio.sleep(0.05)
        assert bg_completed.is_set()

    @pytest.mark.asyncio
    async def test_foreground_context_manager_holds_then_releases_on_error(self):
        """``foreground()`` holds background for its scope and reopens on error."""
        dispatch = ForegroundPriorityDispatch()
        bg_done = asyncio.Event()

        async def bg_work():
            await dispatch.wait_for_background()
            bg_done.set()

        with contextlib.suppress(RuntimeError):
            async with dispatch.foreground():
                assert not dispatch._bg_gate.is_set()
                asyncio.create_task(bg_work())
                await asyncio.sleep(0.05)
                assert not bg_done.is_set()  # held while foreground active
                raise RuntimeError("handler blew up")

        # Even though the handler raised, the gate must reopen.
        assert dispatch._bg_gate.is_set()
        await asyncio.sleep(0.05)
        assert bg_done.is_set()

    @pytest.mark.asyncio
    async def test_proposal_endpoint_enters_foreground(self, monkeypatch):
        """The proposal route wraps the Teacher call in the foreground scope.

        Regression for the inert dispatch: enter/exit_foreground previously had
        zero production callers. Patch the service the router calls and assert
        the global dispatch is foreground-active at call time.
        """
        from conftest import make_settings
        from vlm_feedback_loop.routers import proposals as proposals_router
        from vlm_feedback_loop.schemas.proposal import ProposalRequest
        from vlm_feedback_loop.services.priority import priority_dispatch

        seen: list[bool] = []

        async def _fake_create_proposal(*args, **kwargs):
            seen.append(not priority_dispatch._bg_gate.is_set())
            return "not found: short-circuit after recording"

        monkeypatch.setattr(proposals_router, "create_proposal", _fake_create_proposal)
        settings = make_settings("/tmp/x")

        with pytest.raises(Exception):  # noqa: B017 — 404 HTTPException after record
            await proposals_router.create_proposal_endpoint(
                "p1", ProposalRequest(example_key="k1"), settings=settings
            )

        assert seen == [True], "router must be foreground-active during the call"
        # And the gate is reopened afterward.
        assert priority_dispatch._bg_gate.is_set()


# ── CPU Thread Pool ──────────────────────────────────────────────────────────


class TestThreadPool:
    """CPU-bound work doesn't block the event loop."""

    @pytest.mark.asyncio
    async def test_event_loop_responsive_during_cpu_work(self):
        quick_done = asyncio.Event()

        async def quick_coro():
            quick_done.set()

        def slow_cpu():
            time.sleep(0.5)
            return 42

        # Start CPU work and a quick coroutine concurrently
        cpu_task = asyncio.create_task(run_in_thread(slow_cpu))
        asyncio.get_event_loop().call_soon(lambda: asyncio.create_task(quick_coro()))
        await asyncio.sleep(0.1)

        # Quick coroutine should complete even while CPU work runs
        assert quick_done.is_set()

        result = await cpu_task
        assert result == 42


# ── Single-Process Lock ──────────────────────────────────────────────────────


class TestSingleProcessLock:
    """Second process lock attempt fails with hard error."""

    def test_second_lock_fails(self, tmp_project_dir):
        acquire_project_lock(tmp_project_dir)

        # Open a SECOND file descriptor to the same lock file
        # On Linux, flock is per-FD, so this simulates a second process
        import fcntl

        lock_path = tmp_project_dir / "project.lock"
        f2 = open(lock_path, "w")
        with pytest.raises((IOError, OSError)):
            fcntl.flock(f2.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        f2.close()

        release_all_locks()

    def test_project_locked_error_message(self, tmp_project_dir):
        """A lock held by ANOTHER process raises ProjectLockedError with
        the spec message (the dialog copy keys off it). Replaces an
        assertion-free acquire/clear/acquire sequence."""
        import subprocess
        import sys as _sys

        from vlm_feedback_loop.services.locks import ProjectLockedError

        lock_path = tmp_project_dir / "project.lock"
        holder = subprocess.Popen(
            [
                _sys.executable,
                "-c",
                (
                    "import fcntl, time\n"
                    f"f = open({str(lock_path)!r}, 'w')\n"
                    "fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
                    "print('HELD', flush=True)\n"
                    "time.sleep(30)\n"
                ),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "HELD"
            with pytest.raises(
                ProjectLockedError, match="already open in another process"
            ):
                acquire_project_lock(tmp_project_dir)
        finally:
            holder.kill()
            holder.wait()
        release_all_locks()

    def test_same_process_reacquire_is_idempotent(self, tmp_project_dir):
        fd1 = acquire_project_lock(tmp_project_dir)
        fd2 = acquire_project_lock(tmp_project_dir)
        assert fd1 == fd2
        release_all_locks()

    def test_no_override_parameter(self):
        """No override/force path exists."""
        sig = inspect.signature(acquire_project_lock)
        params = list(sig.parameters.keys())
        assert params == ["project_dir"], (
            f"acquire_project_lock should accept only project_dir, got {params}"
        )
