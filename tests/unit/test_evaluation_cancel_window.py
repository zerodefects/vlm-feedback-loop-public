# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operation Records persisted during the evaluation canceling window
must be marked ``ignored_due_to_run_cancellation=True``.

The cancellation flow is two-phase:
  1. Run transitions to ``canceling``; new per-example dispatches stop.
  2. Already-in-flight inferences are allowed to settle. Some of them
     will complete and persist their Operation Records *during* the
     canceling window — the run is canceled but the records are real.

The spec says these records MUST NOT contribute to authoritative
metrics, but MAY be retained for audit. The implementation flags them
on transition to ``canceled`` via ``_finalize_canceled``.

A regression that loses the ``ignored_*`` flag would silently corrupt
the next evaluation's diff against this run, because canceled records
would be treated as authoritative.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import add_project_row, make_settings, open_project_workspace
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services.evaluation_service import _finalize_canceled

PID = "proj-cancel"
GID = "g-cancel"
MCID = "mc-cancel"


def _setup_project(tmp_path: Path):
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=()
    )
    with Session(engine) as s:
        add_project_row(s, PID, str(project_dir), name="Cancel Test")
        s.commit()
    return engine, make_settings(workspace)


def _add_run(engine, status: str = "canceling") -> str:
    rid = generate_uuid4()
    with Session(engine) as s:
        s.add(
            RunRecord(
                run_id=rid,
                project_id=PID,
                run_type="evaluation_run",
                status=status,
                guidance_id=GID,
                model_config_id=MCID,
                generation_preset_key="precise",
                thinking_mode_effective="on",
                visual_budget_preset_key="balanced",
                structured_generation_mode_effective="auto",
                created_at=utc_now(),
            )
        )
        s.commit()
    return rid


def _add_op_record(
    engine,
    *,
    evaluation_run_id: str,
    ignored: bool = False,
    example_key: str | None = None,
) -> str:
    inv_id = generate_uuid4()
    with Session(engine) as s:
        s.add(
            OperationRecord(
                inference_invocation_id=inv_id,
                project_id=PID,
                purpose="evaluation",
                example_key=example_key or generate_uuid4(),
                evaluation_run_id=evaluation_run_id,
                model_config_id=MCID,
                guidance_id=GID,
                model_name="cosmos-r2-test",
                invocation_status="success",
                ignored_due_to_run_cancellation=ignored,
            )
        )
        s.commit()
    return inv_id


# ── The invariant ───────────────────────────────────────────────────────────


class TestIgnoredDueToRunCancellation:
    """``_finalize_canceled`` must flag late-completing OperationRecords."""

    @pytest.mark.asyncio
    async def test_records_inflight_during_cancel_get_flagged(self, tmp_path):
        engine, _ = _setup_project(tmp_path)
        rid = _add_run(engine, status="canceling")
        # Two records that completed during the canceling window —
        # neither was flagged at insert time.
        inv_a = _add_op_record(engine, evaluation_run_id=rid, ignored=False)
        inv_b = _add_op_record(engine, evaluation_run_id=rid, ignored=False)

        await _finalize_canceled(engine, PID, rid)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
            recs = (
                s.query(OperationRecord)
                .filter(OperationRecord.evaluation_run_id == rid)
                .all()
            )
        assert run.status == "canceled"
        assert run.completed_at is not None
        # Both late-completing records are now marked non-authoritative.
        assert all(r.ignored_due_to_run_cancellation for r in recs)
        flagged = {r.inference_invocation_id for r in recs}
        assert {inv_a, inv_b}.issubset(flagged)

    @pytest.mark.asyncio
    async def test_already_flagged_records_are_idempotent(self, tmp_path):
        # Pre-flagged records aren't re-touched (the .is_(False) filter
        # in the bulk update means the second cancel pass is a no-op,
        # which keeps repeated cancels safe).
        engine, _ = _setup_project(tmp_path)
        rid = _add_run(engine, status="canceling")
        already_flagged = _add_op_record(engine, evaluation_run_id=rid, ignored=True)

        await _finalize_canceled(engine, PID, rid)

        with Session(engine) as s:
            rec = (
                s.query(OperationRecord)
                .filter_by(inference_invocation_id=already_flagged)
                .one()
            )
        # Still flagged (idempotent), no exception raised.
        assert rec.ignored_due_to_run_cancellation is True

    @pytest.mark.asyncio
    async def test_records_from_other_runs_are_not_touched(self, tmp_path):
        # Cancellation of run A MUST NOT flag records from run B.
        engine, _ = _setup_project(tmp_path)
        rid_a = _add_run(engine, status="canceling")
        rid_b = _add_run(engine, status="running")
        inv_b = _add_op_record(engine, evaluation_run_id=rid_b, ignored=False)

        await _finalize_canceled(engine, PID, rid_a)

        with Session(engine) as s:
            rec = (
                s.query(OperationRecord).filter_by(inference_invocation_id=inv_b).one()
            )
        # Run B's record untouched.
        assert rec.ignored_due_to_run_cancellation is False

    @pytest.mark.asyncio
    async def test_terminal_run_is_a_noop(self, tmp_path):
        # Calling _finalize_canceled on an already-terminal run MUST NOT
        # re-modify state (idempotency / replay safety).
        engine, _ = _setup_project(tmp_path)
        rid = _add_run(engine, status="completed")
        # Records on a successfully completed run shouldn't be flagged.
        inv = _add_op_record(engine, evaluation_run_id=rid, ignored=False)

        await _finalize_canceled(engine, PID, rid)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
            rec = s.query(OperationRecord).filter_by(inference_invocation_id=inv).one()
        # Run stays completed; record stays unflagged.
        assert run.status == "completed"
        assert rec.ignored_due_to_run_cancellation is False
