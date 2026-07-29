# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract of the shared RunRecord lookup guard (services.run_queries).

find_run fronts every get/cancel/resume entry point in evaluation_service
and batch_label_service. These tests pin the guard's own contract: error
strings must classify as 404 through map_service_error (not 400/500), a
run id of the wrong run_type is never returned across the eval/batch
endpoint boundary, and lookups are scoped to the requesting project.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from sqlalchemy import event, text
from sqlalchemy.orm import Session, object_session

from conftest import PID, open_project_workspace, setup_project_db
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.errors import map_service_error
from vlm_feedback_loop.services.run_queries import find_run, update_run_if_not_terminal

EVAL_ID = "run-eval-1"
BATCH_ID = "run-batch-1"


def _seed_runs(tmp_path):
    engine, _, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        session.add(
            RunRecord(
                run_id=EVAL_ID,
                project_id=PID,
                run_type="evaluation_run",
                status="completed",
            )
        )
        session.add(
            RunRecord(
                run_id=BATCH_ID,
                project_id=PID,
                run_type="batch_label_run",
                status="completed",
            )
        )
        session.commit()
    return engine


def test_returns_session_attached_record_for_matching_type(tmp_path):
    """A matching id+type lookup returns the caller's session-attached record.

    Cancel/resume mutate the returned record and commit on the same
    session; a detached copy would make those transitions silent no-ops.
    """
    engine = _seed_runs(tmp_path)
    with Session(engine) as session:
        run = find_run(session, PID, EVAL_ID, run_type="evaluation_run")
        assert isinstance(run, RunRecord)
        assert run.run_id == EVAL_ID
        assert object_session(run) is session


@pytest.mark.parametrize(
    ("run_id", "requested_type"),
    [
        (EVAL_ID, "batch_label_run"),
        (BATCH_ID, "evaluation_run"),
    ],
)
def test_wrong_run_type_is_a_404_not_the_record(tmp_path, run_id, requested_type):
    """A run id that exists under a different run_type reports 404, in both
    directions.

    This is the guard against eval/batch endpoint type confusion, and the
    error string must keep the marker map_service_error classifies as 404 —
    losing it would silently flip these endpoints to 400.
    """
    engine = _seed_runs(tmp_path)
    with Session(engine) as session:
        result = find_run(session, PID, run_id, run_type=requested_type)
    assert isinstance(result, str)
    assert run_id in result
    assert map_service_error(result).status_code == 404


def test_run_of_another_project_is_not_found(tmp_path):
    """Lookups are scoped to the requesting project: a run id recorded under
    a different project_id is reported 404, never returned."""
    engine, _, _ = open_project_workspace(tmp_path, PID)
    with Session(engine) as session:
        session.add(
            RunRecord(
                run_id=EVAL_ID,
                project_id="other-project",
                run_type="evaluation_run",
                status="completed",
            )
        )
        session.commit()
        result = find_run(session, PID, EVAL_ID, run_type="evaluation_run")
    assert isinstance(result, str)
    assert map_service_error(result).status_code == 404


class TestUpdateRunIfNotTerminal:
    """The guarded update is the atomic core of the terminal-row
    invariant: a write must not land after a concurrent transaction
    terminalizes the run, even though WAL readers still see 'running'."""

    def _seed(self, engine, status="running"):
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id="r1",
                    project_id="p1",
                    run_type="batch_label_run",
                    status=status,
                )
            )
            s.commit()

    def test_updates_live_run(self, tmp_path):
        engine, _ = setup_project_db(tmp_path)
        self._seed(engine)
        with Session(engine) as s:
            ok = update_run_if_not_terminal(
                s,
                "r1",
                {"status": "paused"},
                terminal_statuses=frozenset({"completed", "failed", "canceled"}),
            )
            s.commit()
        assert ok is True
        with Session(engine) as s:
            assert s.get(RunRecord, "r1").status == "paused"

    def test_refuses_terminal_run(self, tmp_path):
        engine, _ = setup_project_db(tmp_path)
        self._seed(engine, status="failed")
        with Session(engine) as s:
            ok = update_run_if_not_terminal(
                s,
                "r1",
                {"status": "paused"},
                terminal_statuses=frozenset({"completed", "failed", "canceled"}),
            )
            s.commit()
        assert ok is False
        with Session(engine) as s:
            assert s.get(RunRecord, "r1").status == "failed"

    def test_concurrent_terminalization_blocks_the_write(self, tmp_path):
        """The not-terminal predicate is evaluated at write time, not at a
        prior read: while another connection holds an uncommitted
        terminalization, the guarded update waits on the write lock; the
        terminalization commits mid-wait; the guarded update must then see
        the terminal status and refuse (False). A check-then-act
        implementation fails here — its pre-read sees 'running' and its
        blind write lands 'paused' over the committed 'failed'."""
        engine, pdir = setup_project_db(tmp_path)
        self._seed(engine)
        db_path = str(Path(pdir) / "project.db")

        # Connection A: begin a terminalizing write and hold the lock open.
        conn_a = sqlite3.connect(db_path, timeout=5)
        conn_a.execute("PRAGMA journal_mode=WAL")
        conn_a.execute("BEGIN IMMEDIATE")
        conn_a.execute("UPDATE run_records SET status='failed' WHERE run_id='r1'")

        # Fires when the worker issues its UPDATE — by then a check-then-act
        # pre-read would already have seen 'running' (A is uncommitted).
        update_issued = threading.Event()

        def _note_update(_conn, _cursor, statement, *_args):
            if statement.lstrip().upper().startswith("UPDATE"):
                update_issued.set()

        event.listen(engine, "before_cursor_execute", _note_update)

        outcome: list[bool] = []
        errors: list[Exception] = []

        def _guarded_update():
            with Session(engine) as s:
                # Generous timeout: the worker's write waits for A's lock
                # instead of erroring out before A commits.
                s.execute(text("PRAGMA busy_timeout=10000"))
                try:
                    ok = update_run_if_not_terminal(
                        s,
                        "r1",
                        {"status": "paused"},
                        terminal_statuses=frozenset(
                            {"completed", "failed", "canceled"}
                        ),
                    )
                    s.commit()
                    outcome.append(ok)
                except Exception as exc:  # pragma: no cover - fails the test
                    errors.append(exc)
                    s.rollback()

        try:
            worker = threading.Thread(target=_guarded_update)
            worker.start()
            assert update_issued.wait(timeout=10), "worker never issued its UPDATE"
            # Terminalize-and-commit lands while the worker's write is
            # contending for A's lock.
            conn_a.commit()
            worker.join(timeout=15)
            assert not worker.is_alive(), "guarded update never finished"
        finally:
            event.remove(engine, "before_cursor_execute", _note_update)
            conn_a.close()

        assert errors == []
        # The guard re-evaluated at write time: refused, no resurrection.
        assert outcome == [False]
        with Session(engine) as s:
            assert s.get(RunRecord, "r1").status == "failed"
