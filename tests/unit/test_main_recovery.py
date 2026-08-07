# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the startup-recovery scan in ``main.py``.

The recovery scan runs on every backend boot, before requests are
served, and must transition any non-terminal RunRecord left over from
a crashed prior process to a state consistent with the spec:

* Evaluation runs in queued / running / canceling → ``failed`` with
  ``status_reason="backend_restart_interrupted"`` (the SME re-triggers).
* Batch runs in queued / running → ``queued`` with
  ``recovered_from_restart=True``, and the ``(project_id, run_id)`` pair is
  returned so the async lifespan can dispatch the executor for auto-resume
  (idempotent persistence prevents duplicate per-example outcomes).
* Batch runs in ``canceling`` with a persisted cancellation timestamp become
  ``canceled``; without that durable intent marker they become ``failed``.
* Batch runs in ``paused`` → unchanged (don't auto-resume after a
  circuit-breaker pause; the SME explicitly resumed or canceled it).
* Terminal runs (``succeeded`` / ``failed`` / ``completed`` / ``canceled``
  / ``incomplete``) → unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_and_model_rows,
    add_example_row,
    add_fixture_guidance_row,
    add_standard_project_row,
    make_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.main import _recover_interrupted_runs
from vlm_feedback_loop.services.run_config import snapshot_run_config

PID = "proj-recovery"
GID = "g-recovery"
MCID = "mc-recovery"
EID = "endpoint-recovery"


# ── Fixtures and helpers ────────────────────────────────────────────────────


def _setup_project_with_runs(tmp_path: Path, archived: bool = False):
    """Create a workspace with one project DB and return engine+settings."""
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=()
    )
    if archived:
        (project_dir / ".archived").touch()
    with Session(engine) as s:
        add_standard_project_row(
            s,
            PID,
            str(project_dir),
            name="Recovery Test",
            active_guidance_id=GID,
            teacher_model_config_id=MCID,
        )
        add_fixture_guidance_row(s, PID, GID)
        add_endpoint_and_model_rows(
            s,
            PID,
            endpoint_id=EID,
            model_config_id=MCID,
        )
        s.commit()
    return engine, make_settings(workspace)


def _add_run(
    engine,
    *,
    run_type: str,
    status: str,
    run_id: str | None = None,
    recovered_from_restart: bool | None = None,
    paused_reason: str | None = None,
    cancel_requested_at: str | None = None,
    input_keys: list[str] | None = None,
) -> str:
    """Insert a RunRecord and return its ID."""
    rid = run_id or generate_uuid4()
    batch_input_keys = input_keys or []
    with Session(engine) as s:
        s.add(
            RunRecord(
                run_id=rid,
                project_id=PID,
                run_type=run_type,
                status=status,
                guidance_id=GID,
                model_config_id=MCID,
                generation_preset_key="precise",
                thinking_mode_effective="on",
                visual_budget_preset_key="balanced",
                structured_generation_mode_effective="auto",
                created_at=utc_now(),
                recovered_from_restart=recovered_from_restart,
                paused_reason=paused_reason,
                cancel_requested_at=cancel_requested_at,
                examples_total=(
                    len(batch_input_keys) if run_type == "batch_label_run" else 0
                ),
                metrics=(
                    {
                        "input_keys": batch_input_keys,
                        "circuit_breaker_threshold": 10,
                        "circuit_breaker_consecutive": 0,
                        "circuit_breaker_tripped": False,
                    }
                    if run_type == "batch_label_run"
                    else None
                ),
            )
        )
        s.commit()
    return rid


# ── Evaluation run recovery ──────────────────────────────────────────


class TestEvaluationRunRecovery:
    """Eval runs in non-terminal status → failed, status_reason set."""

    @pytest.mark.parametrize("status", ["queued", "running", "canceling"])
    def test_non_terminal_eval_run_transitions_to_failed(self, tmp_path, status):
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(engine, run_type="evaluation_run", status=status)

        _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        assert run.status == "failed"
        assert run.status_reason == "backend_restart_interrupted"
        assert run.completed_at is not None

    @pytest.mark.parametrize(
        "status", ["completed", "failed", "incomplete", "canceled"]
    )
    def test_terminal_eval_run_is_untouched(self, tmp_path, status):
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(engine, run_type="evaluation_run", status=status)

        _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        # Terminal status preserved; no status_reason added.
        assert run.status == status
        assert run.status_reason is None


# ── Batch labeling run recovery ─────────────────────────────


class TestBatchLabelRunRecovery:
    """Batch runs in queued/running → queued + recovered_from_restart=True."""

    @pytest.mark.parametrize("status", ["queued", "running"])
    def test_active_batch_run_transitions_to_queued_recovered(self, tmp_path, status):
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(engine, run_type="batch_label_run", status=status)

        resume_targets = _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        # queued/running both end up queued with the recovery flag set so
        # the executor knows to resume from the next unprocessed example.
        assert run.status == "queued"
        assert run.recovered_from_restart is True
        # batch runs do NOT inherit the eval-run "backend_restart_interrupted"
        # status_reason because they auto-resume rather than fail.
        assert run.status_reason is None
        # The run is handed back for the lifespan to dispatch (auto-resume);
        # a flag with no dispatch would strand the run in queued forever.
        assert (PID, rid) in resume_targets

    def test_paused_batch_run_stays_paused(self, tmp_path):
        # Paused batch runs (circuit breaker tripped) MUST NOT auto-resume —
        # the SME paused intentionally and must explicitly Resume / Cancel.
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(
            engine,
            run_type="batch_label_run",
            status="paused",
            paused_reason="circuit_breaker_threshold_reached",
        )

        _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
            assert run.status == "paused"
            assert run.paused_reason == "circuit_breaker_threshold_reached"
            assert run.runtime_config_snapshot is not None
            # No recovered flag: only auto-resumed runs get it.
            assert (
                run.recovered_from_restart is None
                or run.recovered_from_restart is False
            )
            model_config = s.get(ModelConfig, MCID)
            endpoint = s.get(NimEndpoint, EID)
            assert model_config is not None
            assert endpoint is not None
            model_config.context_window_tokens = 16384
            endpoint.base_url = "https://changed.example/v1"
            s.commit()

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
            resumed = snapshot_run_config(
                s,
                PID,
                run,
                example_keys=[],
                settings=settings,
            )

        assert resumed["mc_input"].context_window_tokens == 256000
        assert resumed["endpoint_base_url"] == "https://test.nvidia.com/v1"

    def test_canceling_batch_run_finalized_to_canceled(self, tmp_path):
        # `canceling` cannot survive a restart: the in-memory cancel event is
        # gone, so re-dispatching would never observe the cancel and the run
        # would wedge in a non-terminal state that blocks project archive.
        # Recovery honors the operator's cancel intent by finalizing it.
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(
            engine,
            run_type="batch_label_run",
            status="canceling",
            cancel_requested_at=utc_now(),
            input_keys=["authoritative", "crossing"],
        )
        with Session(engine) as session:
            add_example_row(
                session,
                PID,
                "authoritative",
                state="Auto-Labeled",
            )
            add_example_row(session, PID, "crossing")
            authoritative_invocation_id = generate_uuid4()
            session.add_all(
                [
                    OperationRecord(
                        inference_invocation_id=authoritative_invocation_id,
                        project_id=PID,
                        purpose="batch_label",
                        example_key="authoritative",
                        guidance_id=GID,
                        model_config_id=MCID,
                        endpoint_id=EID,
                        model_name="test-model",
                        batch_label_run_id=rid,
                        invocation_status="success",
                        schema_valid_core=True,
                        label_tier="auto_labeled",
                    ),
                    OperationRecord(
                        inference_invocation_id=generate_uuid4(),
                        project_id=PID,
                        purpose="batch_label",
                        example_key="crossing",
                        guidance_id=GID,
                        model_config_id=MCID,
                        endpoint_id=EID,
                        model_name="test-model",
                        batch_label_run_id=rid,
                        invocation_status="timeout",
                        label_tier="auto_labeled",
                        ignored_due_to_run_cancellation=True,
                    ),
                    Label(
                        label_id=generate_uuid4(),
                        project_id=PID,
                        example_key="authoritative",
                        label_status="auto_labeled",
                        guidance_id=GID,
                        inference_invocation_id=authoritative_invocation_id,
                        label_json={"severity": "high", "damaged": True},
                        labeled_at=utc_now(),
                        batch_label_run_id=rid,
                    ),
                ]
            )
            session.commit()

        resume_targets = _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        assert run.status == "canceled"
        assert run.status_reason == "canceled_on_restart"
        assert run.completed_at is not None
        assert run.examples_succeeded == 1
        assert run.examples_schema_invalid == 0
        assert run.examples_timeout == 0
        assert run.examples_endpoint_error == 0
        # A canceled run is terminal — it must NOT be handed back for resume.
        assert (PID, rid) not in resume_targets

    def test_canceling_batch_without_durable_intent_fails(self, tmp_path):
        """Recovery does not invent a user cancellation request."""
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(engine, run_type="batch_label_run", status="canceling")

        resume_targets = _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        assert run.status == "failed"
        assert run.status_reason == "backend_restart_interrupted"
        assert run.completed_at is not None
        assert (PID, rid) not in resume_targets

    def test_cancel_recovery_does_not_count_or_rewrite_a_torn_success(self, tmp_path):
        """Canceled recovery preserves a torn audit row but excludes it."""
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(
            engine,
            run_type="batch_label_run",
            status="canceling",
            cancel_requested_at=utc_now(),
            input_keys=["torn"],
        )
        invocation_id = generate_uuid4()
        with Session(engine) as session:
            add_example_row(session, PID, "torn")
            session.add(
                OperationRecord(
                    inference_invocation_id=invocation_id,
                    project_id=PID,
                    purpose="batch_label",
                    example_key="torn",
                    guidance_id=GID,
                    model_config_id=MCID,
                    endpoint_id=EID,
                    model_name="test-model",
                    batch_label_run_id=rid,
                    invocation_status="success",
                    schema_valid_core=True,
                    label_tier="auto_labeled",
                )
            )
            session.commit()

        _recover_interrupted_runs(settings)

        with Session(engine) as session:
            run = session.query(RunRecord).filter_by(run_id=rid).one()
            operation = session.get(OperationRecord, invocation_id)
            assert run.status == "canceled"
            assert run.examples_succeeded == 0
            assert operation is not None
            assert operation.invocation_status == "success"

    def test_cancel_recovery_fails_closed_on_foreign_lineage(self, tmp_path):
        """A canceled restart never derives counters from a foreign operation."""
        engine, settings = _setup_project_with_runs(tmp_path)
        rid = _add_run(
            engine,
            run_type="batch_label_run",
            status="canceling",
            cancel_requested_at=utc_now(),
            input_keys=["foreign"],
        )
        with Session(engine) as session:
            add_example_row(session, PID, "foreign")
            session.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="foreign",
                    guidance_id="other-guidance",
                    model_config_id=MCID,
                    endpoint_id=EID,
                    model_name="test-model",
                    batch_label_run_id=rid,
                    invocation_status="timeout",
                    label_tier="auto_labeled",
                )
            )
            session.commit()

        _recover_interrupted_runs(settings)

        with Session(engine) as session:
            run = session.query(RunRecord).filter_by(run_id=rid).one()
            assert run.status == "failed"
            assert run.status_reason == "batch_recovery_state_invalid"

    def test_malformed_cancel_recovery_does_not_abort_later_runs(self, tmp_path):
        """One invalid snapshot fails closed without aborting startup recovery."""
        engine, settings = _setup_project_with_runs(tmp_path)
        malformed = _add_run(
            engine,
            run_type="batch_label_run",
            status="canceling",
            cancel_requested_at=utc_now(),
            input_keys=["bad"],
        )
        recoverable = _add_run(
            engine,
            run_type="batch_label_run",
            status="running",
            input_keys=[],
        )
        with Session(engine) as session:
            session.query(RunRecord).filter_by(run_id=malformed).update(
                {"inference_contract": ["bad"]}
            )
            session.commit()

        resume_targets = _recover_interrupted_runs(settings)

        with Session(engine) as session:
            malformed_run = session.get(RunRecord, malformed)
            recoverable_run = session.get(RunRecord, recoverable)
            assert malformed_run is not None
            assert malformed_run.status == "failed"
            assert malformed_run.status_reason == "batch_recovery_state_invalid"
            assert recoverable_run is not None
            assert recoverable_run.status == "queued"
            assert recoverable_run.recovered_from_restart is True
        assert (PID, recoverable) in resume_targets


# ── Workspace-level edge cases ──────────────────────────────────────────────


class TestRecoverySkipping:
    """Recovery handles missing / archived workspaces gracefully."""

    def test_missing_workspace_directory_is_a_noop(self, tmp_path):
        # Nothing under WORKSPACE_ROOT/projects/* — recovery returns
        # without raising or creating anything.
        settings = make_settings(tmp_path / "no_such_workspace")
        # Should not raise.
        _recover_interrupted_runs(settings)

    def test_archived_project_is_skipped(self, tmp_path):
        # Projects with the .archived marker file are paused by design;
        # the recovery scan must skip them so archived state is preserved.
        engine, settings = _setup_project_with_runs(tmp_path, archived=True)
        rid = _add_run(engine, run_type="evaluation_run", status="running")

        _recover_interrupted_runs(settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=rid).one()
        # Archived → not touched even though it would otherwise transition.
        assert run.status == "running"
        assert run.status_reason is None

    def test_directory_without_project_db_is_skipped(self, tmp_path):
        # A stray directory under projects/ without a project.db file
        # MUST NOT crash recovery.
        workspace = tmp_path / "workspace"
        (workspace / "projects" / "stray").mkdir(parents=True)
        settings = make_settings(workspace)
        # Should not raise.
        _recover_interrupted_runs(settings)


# ── Mixed states in one project ─────────────────────────────────────────────


class TestMixedRecoveryStates:
    """Real-world recovery: one project with several runs in different states."""

    def test_only_non_terminal_runs_are_modified(self, tmp_path):
        engine, settings = _setup_project_with_runs(tmp_path)
        # Interleave terminal + non-terminal runs of various types.
        eval_running = _add_run(engine, run_type="evaluation_run", status="running")
        eval_done = _add_run(engine, run_type="evaluation_run", status="completed")
        batch_running = _add_run(engine, run_type="batch_label_run", status="running")
        batch_paused = _add_run(
            engine,
            run_type="batch_label_run",
            status="paused",
            paused_reason="circuit_breaker_threshold_reached",
        )
        batch_canceled = _add_run(engine, run_type="batch_label_run", status="canceled")

        _recover_interrupted_runs(settings)

        with Session(engine) as s:
            statuses = {r.run_id: r.status for r in s.query(RunRecord).all()}
        # eval running → failed; eval completed → unchanged
        assert statuses[eval_running] == "failed"
        assert statuses[eval_done] == "completed"
        # batch running → queued (recovered); paused → paused; canceled → canceled
        assert statuses[batch_running] == "queued"
        assert statuses[batch_paused] == "paused"
        assert statuses[batch_canceled] == "canceled"
