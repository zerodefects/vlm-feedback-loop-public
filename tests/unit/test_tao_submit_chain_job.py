# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_job_service.submit_chain_job.

Exercises the chain-submission helper that wraps the 4-step submission
protocol for a pre-created ``not_started`` TAOJob, and confirms SSE
emission happens only after the terminal state is durably persisted.

Also covers the export-field-mode lineage guard that runs inside
``submit_chain_job``: the training-suite endpoint creates both training
and evaluation DatasetExports with the same ``export_field_mode``, so a
well-formed chain never trips it. The guard exists to catch mode-drift
in ad-hoc / tampered / manually-seeded evaluate jobs — the guard tests
construct adversarial inputs to prove it rejects them cleanly before any
POST reaches TAO.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import add_tao_job_row, make_tao_settings, seed_tao_chain_project
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import tao_job_service
from vlm_feedback_loop.services.project_service import (
    set_project_engine,
)

PID = "proj-chain"


def _make_settings(workspace: Path, **overrides):
    return make_tao_settings(workspace, TAO_API_KEY="jwt-test", **overrides)


def _setup(tmp_path, *, train_mode="all", eval_mode="all"):
    workspace = tmp_path / "workspace"
    pdir = workspace / "projects" / PID
    pdir.mkdir(parents=True, exist_ok=True)
    engine = open_project_db(pdir)
    set_project_engine(PID, engine)
    with Session(engine) as s:
        seed_tao_chain_project(
            s,
            PID,
            str(pdir),
            train_export_mode=train_mode,
            eval_export_mode=eval_mode,
        )
        s.commit()
    return engine, workspace


def _add_chain_job(engine, *, status="not_started"):
    job_id = generate_uuid4()
    with Session(engine) as s:
        add_tao_job_row(
            s,
            PID,
            job_id,
            action="train",
            status=status,
            dataset_export_ids=["de-train"],
            chain_id="chain-1",
            chain_sequence=1,
        )
        s.commit()
    return job_id


def _seed_guard_chain(engine):
    """Seed a succeeded train + not_started dependent evaluate on one chain.

    Returns ``(train_id, eval_id)``. The evaluate is the job the
    export-field-mode lineage guard inspects on submission.
    """
    chain_id = generate_uuid4()
    train_id = generate_uuid4()
    eval_id = generate_uuid4()
    with Session(engine) as s:
        add_tao_job_row(
            s,
            PID,
            train_id,
            action="train",
            status="succeeded",
            dataset_export_ids=["de-train"],
            chain_id=chain_id,
            chain_sequence=1,
        )
        add_tao_job_row(
            s,
            PID,
            eval_id,
            action="evaluate",
            dataset_export_ids=["de-eval"],
            parent_tao_job_id=train_id,
            chain_id=chain_id,
            chain_sequence=2,
        )
        s.commit()
    return train_id, eval_id


# ══════════════════════════════════════════════════════════════════════════
# Happy + failure paths
# ══════════════════════════════════════════════════════════════════════════


class TestSubmitChainJob:
    @pytest.mark.asyncio
    async def test_transitions_not_started_to_submitted_on_success(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        job_id = _add_chain_job(engine)
        mock_submit = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-xyz",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)

        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(PID, job_id, settings=settings)
        assert result == "submitted"
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=job_id).first()
        assert job.status == "submitted"
        assert job.tao_external_job_id == "ext-xyz"
        assert job.started_at is not None

    @pytest.mark.asyncio
    async def test_marks_failed_with_sanitized_error_on_submission_failure(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        job_id = _add_chain_job(engine)
        mock_submit = AsyncMock(
            return_value={
                "success": False,
                "tao_external_job_id": None,
                "error": "TAO rejected request. Bearer nvapi-xyz leaked",
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)
        settings = _make_settings(workspace)

        result = await tao_job_service.submit_chain_job(PID, job_id, settings=settings)
        assert result == "failed"
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=job_id).first()
        assert job.status == "failed"
        assert job.completed_at is not None
        assert job.error_ref is not None
        # Sanitization redacts both Bearer and nvapi- patterns.
        assert "Bearer " not in job.error_ref
        assert "nvapi-xyz" not in job.error_ref

    @pytest.mark.asyncio
    async def test_rejects_non_not_started_jobs(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        job_id = _add_chain_job(engine, status="running")
        mock_submit = AsyncMock()
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)
        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(PID, job_id, settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()
        mock_submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_job_returns_not_found(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        mock_submit = AsyncMock()
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)
        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(
            PID, "does-not-exist", settings=settings
        )
        assert isinstance(result, str)
        assert "not found" in result.lower()
        mock_submit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_midchain_submission_failure_halts_chain_and_rolls_up_suite(
        self, tmp_path, monkeypatch
    ):
        """A mid-chain submission failure must not strand the suite in running.

        Regression for the poller-advance path: submit_chain_job's callers
        ignore its return value, so before the fix a failed mid-chain
        submission left the failed job's dependents not_started and the suite
        stuck in "running" forever. With advance_on_failure (default), the
        failure now halts dependents and rolls the suite up to a terminal
        status, exactly like a poll-detected failure.
        """
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(engine)  # chain-1, seq 1, action=train

        # Dependent evaluate job (seq 2) parented on the train job.
        eval_id = generate_uuid4()
        with Session(engine) as s:
            add_tao_job_row(
                s,
                PID,
                eval_id,
                action="evaluate",
                dataset_export_ids=["de-eval"],
                parent_tao_job_id=train_id,
                chain_id="chain-1",
                chain_sequence=2,
            )
            s.add(
                TrainingSuite(
                    training_suite_id=generate_uuid4(),
                    project_id=PID,
                    idempotency_key=generate_uuid4(),
                    guidance_id=generate_uuid4(),
                    training_preset="standard",
                    export_field_mode="all",
                    include_auto_labeled=False,
                    training_dataset_export_id="de-train",
                    evaluation_dataset_export_id="de-eval",
                    selected_student_base_model_config_ids=["mc-1"],
                    quantization_schemes=[],
                    chain_ids_ordered=["chain-1"],
                    status="running",
                    started_at="2026-07-10T00:00:00Z",
                )
            )
            s.commit()

        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": False,
                    "tao_external_job_id": None,
                    "error": "FTMS unreachable",
                }
            ),
        )
        settings = _make_settings(workspace)

        result = await tao_job_service.submit_chain_job(
            PID, train_id, settings=settings
        )

        assert result == "failed"
        with Session(engine) as s:
            train = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            ev = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            suite = s.query(TrainingSuite).filter_by(project_id=PID).one()
        assert train.status == "failed"
        # Dependent evaluate is halted, not left dangling as not_started.
        assert ev.status == "failed"
        assert ev.chain_halted_reason is not None
        # All chain jobs terminal + a failure → suite finalizes failed, not
        # stranded in "running".
        assert suite.status == "failed"
        assert suite.completed_at is not None

    @pytest.mark.asyncio
    async def test_emits_sse_after_terminal_write(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        job_id = _add_chain_job(engine)

        mock_submit = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-1",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)

        emitted: list[tuple[str, str, dict]] = []

        async def fake_emit(project_id, event_type, data):
            emitted.append((project_id, event_type, data))

        monkeypatch.setattr(tao_job_service.sse_manager, "emit", fake_emit)
        settings = _make_settings(workspace)
        await tao_job_service.submit_chain_job(PID, job_id, settings=settings)
        assert any(evt == "tao_job_progress" for _, evt, _ in emitted)


# ══════════════════════════════════════════════════════════════════════════
# Export-field-mode lineage guard: matching modes submit normally
# ══════════════════════════════════════════════════════════════════════════


class TestGuardMatchingModes:
    @pytest.mark.asyncio
    async def test_matching_modes_submits_normally(self, tmp_path, monkeypatch):
        """The guard's pass path: an evaluate whose export shares the train
        export's ``export_field_mode`` submits normally — the guard must
        never block a well-formed chain."""
        engine, workspace = _setup(tmp_path, train_mode="all", eval_mode="all")
        _, eval_id = _seed_guard_chain(engine)
        # Mock the actual TAO POST so we isolate the guard behavior.
        mock_submit = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-xyz",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)

        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(PID, eval_id, settings=settings)
        assert result == "submitted"
        mock_submit.assert_awaited_once()

        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert job.status == "submitted"
            assert job.tao_external_job_id == "ext-xyz"
            assert job.error_ref is None


# ══════════════════════════════════════════════════════════════════════════
# Export-field-mode lineage guard: mismatches reject pre-submission
# ══════════════════════════════════════════════════════════════════════════


class TestGuardMismatchRejection:
    @pytest.mark.asyncio
    async def test_mismatched_modes_fails_before_posting_to_tao(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path, train_mode="all", eval_mode="core_only")
        _, eval_id = _seed_guard_chain(engine)
        # Guard any TAO POST so the test fails loudly if submission still runs.
        submitted = {"count": 0}

        async def _fail_if_called(*args, **kwargs):
            submitted["count"] += 1
            return {
                "success": True,
                "tao_external_job_id": "ext-ZZZ",
                "error": None,
            }

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", _fail_if_called)

        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(PID, eval_id, settings=settings)
        assert result == "failed"
        # Crucially: no POST to TAO was made.
        assert submitted["count"] == 0

        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert job.status == "failed"
            assert "export_field_mode_mismatch" in (job.error_ref or "")
            assert "all" in (job.error_ref or "")
            assert "core_only" in (job.error_ref or "")
            assert job.tao_external_job_id is None

    @pytest.mark.asyncio
    async def test_mismatched_modes_emits_run_failed_sse(self, tmp_path, monkeypatch):
        engine, workspace = _setup(
            tmp_path, train_mode="aux_and_core", eval_mode="core_only"
        )
        _, eval_id = _seed_guard_chain(engine)
        emitted: list[tuple[str, dict]] = []

        async def capture(proj, event, payload):
            emitted.append((event, payload))

        monkeypatch.setattr(tao_job_service.sse_manager, "emit", capture)

        async def _never(*args, **kwargs):
            raise AssertionError("_submit_to_tao must not be called on mismatch")

        monkeypatch.setattr(tao_job_service, "_submit_to_tao", _never)

        settings = _make_settings(workspace)
        await tao_job_service.submit_chain_job(PID, eval_id, settings=settings)
        # Exactly one run_failed with the mismatch message.
        events = [e for e in emitted if e[0] == "run_failed"]
        assert len(events) == 1
        assert "export_field_mode_mismatch" in events[0][1]["error_summary"]


# ══════════════════════════════════════════════════════════════════════════
# Guard is scoped to evaluate action — train/quantize are unaffected
# ══════════════════════════════════════════════════════════════════════════


class TestGuardScope:
    @pytest.mark.asyncio
    async def test_train_job_unaffected_by_mismatched_exports(
        self, tmp_path, monkeypatch
    ):
        """Train action never triggers the lineage guard — it submits normally.

        Uses the same seeded chain but flips the `train` job back to
        ``not_started`` so ``submit_chain_job`` will process it.
        """
        engine, workspace = _setup(tmp_path, train_mode="all", eval_mode="core_only")
        train_id, _ = _seed_guard_chain(engine)
        # Reset the train job to not_started.
        with Session(engine) as s:
            train_job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            train_job.status = "not_started"
            s.commit()

        mock_submit = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-train",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)

        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(
            PID, train_id, settings=settings
        )
        # Train submits even when there's a downstream eval mismatch —
        # the guard checks evaluate jobs only.
        assert result == "submitted"
        mock_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_evaluate_without_chain_bypasses_guard(self, tmp_path, monkeypatch):
        """An evaluate job with no chain_id has nothing to compare against.

        Such jobs are ad-hoc and not covered by the invariant — the guard
        returns None (passes) and submission proceeds.
        """
        engine, workspace = _setup(tmp_path, train_mode="all", eval_mode="core_only")
        _, eval_id = _seed_guard_chain(engine)
        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            ej.chain_id = None
            ej.chain_sequence = None
            ej.parent_tao_job_id = None
            s.commit()

        mock_submit = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-eval",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", mock_submit)

        settings = _make_settings(workspace)
        result = await tao_job_service.submit_chain_job(PID, eval_id, settings=settings)
        # With no chain to compare against, the guard cannot fire — submit
        # normally.
        assert result == "submitted"
