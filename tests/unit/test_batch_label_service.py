# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for batch_label_service.

Covers: start / gate verification, config snapshot, input selection, background
execution, circuit breaker, resume, cancel, get/list, SSE events, restart
recovery, idempotency, and include_auto_labeled re-labeling.

All inference calls are mocked via monkeypatch on ``_invoke_for_batch_label``.
Gate checks are mocked via patch on ``compute_scaleup_gate``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.orm import Session

from conftest import (
    EID,
    GID,
    MCID,
    PID,
    add_endpoint_and_model_rows,
    add_endpoint_row,
    add_example_row,
    add_fixture_guidance_row,
    add_model_config_row,
    add_standard_project_row,
    make_stub_settings,
    patched_register,
    setup_project_db,
)
from support import fake_nim_failure, fake_nim_success, fake_prepare_result
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.services import batch_label_service as batch_label_service_module
from vlm_feedback_loop.services.batch_label_service import (
    BatchExampleResult,
    _execute_batch_label,
    cancel_batch_label_run,
    get_batch_label_run,
    list_batch_label_runs,
    resume_batch_label_run,
    start_batch_label_run,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _add_example(
    session, project_id, key, state="Unlabeled", phash=None, ingested_at=None
):
    add_example_row(
        session,
        project_id,
        key,
        state=state,
        phash=phash or "a" * 16,
        ingested_at=ingested_at or utc_now(),
    )


def _add_auto_label(session, project_id, key, guidance_id=GID, run_id="old-run"):
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=project_id,
            example_key=key,
            label_status="auto_labeled",
            guidance_id=guidance_id,
            inference_invocation_id=generate_uuid4(),
            label_json={"rationale_note": "old", "severity": "low", "damaged": False},
            labeled_at=utc_now(),
            batch_label_run_id=run_id,
        )
    )


def _setup_batch_project(tmp_path, n_unlabeled=5, n_auto_labeled=0):
    """Create project with Guidance, ModelConfig, examples, ready for batch."""
    engine, pdir = setup_project_db(tmp_path)
    settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))

    with Session(engine) as s:
        add_standard_project_row(s, PID, pdir)
        add_fixture_guidance_row(s)
        add_endpoint_and_model_rows(s)
        for i in range(n_unlabeled):
            key = f"ex_{i:03d}"
            _add_example(s, PID, key, state="Unlabeled")
        for i in range(n_auto_labeled):
            key = f"auto_{i:03d}"
            _add_example(s, PID, key, state="Auto-Labeled")
            _add_auto_label(s, PID, key)
        s.commit()

    return engine, pdir, settings


def _make_batch_success(example_key):
    return BatchExampleResult(
        example_key=example_key,
        invocation_id=generate_uuid4(),
        invocation_status="success",
        proposal_json={"rationale_note": "test", "severity": "high", "damaged": True},
        schema_valid_core=True,
    )


def _make_batch_failure(example_key, status="timeout"):
    return BatchExampleResult(
        example_key=example_key,
        invocation_id=generate_uuid4(),
        invocation_status=status,
        proposal_json=None,
        schema_valid_core=False,
    )


def _make_batch_schema_invalid(example_key):
    return BatchExampleResult(
        example_key=example_key,
        invocation_id=generate_uuid4(),
        invocation_status="success",
        proposal_json={"severity": "invalid_value"},
        schema_valid_core=False,
    )


def _terminalize_run(engine, run_id):
    """Concurrent-writer stand-in: fail the run the way the schema-evolution
    wipe does, once, so mocked invocations can terminalize it mid-flight."""
    with Session(engine) as s:
        run = s.query(RunRecord).filter_by(run_id=run_id).one()
        if run.status != "failed":
            run.status = "failed"
            run.status_reason = "schema_evolution_canceled"
            run.completed_at = utc_now()
            s.commit()


# Patch targets
_GATE_PATCH = "vlm_feedback_loop.services.evaluation_service.compute_scaleup_gate"
_INVOKE_PATCH = "vlm_feedback_loop.services.batch_label_service._invoke_for_batch_label"
_SSE_PATCH = "vlm_feedback_loop.services.batch_label_service.sse_manager"
# The shared run finalizers (teacher_rejection) emit through their own
# module's sse_manager binding — patch it there to observe those events.
_TR_SSE_PATCH = "vlm_feedback_loop.services.teacher_rejection.sse_manager"


def _mock_gate_ready(*args, **kwargs):
    return {"gate_status": "ready", "criteria": [], "evaluated_at": utc_now()}


def _mock_gate_not_ready(*args, **kwargs):
    return {"gate_status": "not_ready", "criteria": [], "evaluated_at": utc_now()}


def _seed_bl_run(
    engine,
    run_id: str,
    keys: list[str] | None,
    *,
    status: str = "queued",
    examples_total: int | None = None,
    include_auto_labeled: bool = False,
    **extra,
) -> None:
    """Seed a batch-label RunRecord with snapshotted config.

    ``keys=None`` seeds a run without the ``metrics`` input snapshot
    (``examples_total`` must then be given); ``extra`` passes through
    additional RunRecord columns (counters, ``paused_reason``, …).
    """
    row: dict = {
        "run_id": run_id,
        "project_id": PID,
        "run_type": "batch_label_run",
        "status": status,
        "guidance_id": GID,
        "model_config_id": MCID,
        "generation_preset_key": "precise",
        "thinking_mode_effective": "on",
        "visual_budget_preset_key": "balanced",
        "structured_generation_mode_effective": "auto",
        "examples_total": len(keys) if examples_total is None else examples_total,
    }
    if keys is not None:
        row["metrics"] = {
            "input_keys": keys,
            "include_auto_labeled": include_auto_labeled,
        }
    row.update(extra)
    with Session(engine) as s:
        s.add(RunRecord(**row))
        s.commit()


# ══════════════════════════════════════════════════════════════════════════════
# Section A: Start / Gate Verification
# ══════════════════════════════════════════════════════════════════════════════


class TestStart:
    @pytest.mark.asyncio
    async def test_start_creates_queued_run(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=5)
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module) as mock_reg,
        ):
            result = await start_batch_label_run(PID, settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "queued"
        assert result["run_type"] == "batch_label_run"
        assert result["examples_total"] == 5
        mock_reg.assert_called_once()

        # Verify RunRecord persisted
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run is not None
            assert run.run_type == "batch_label_run"
            assert run.status == "queued"

    @pytest.mark.asyncio
    async def test_start_rejects_when_batch_run_already_active(self, tmp_path):
        """One batch run per project — a second start returns a 409 conflict.

        Two overlapping runs share the same start-time Unlabeled snapshot and
        race to write duplicate auto_labeled Labels for the same key (a
        double-clicked "Start Batch Labeling" is the common trigger).
        """
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=5)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=generate_uuid4(),
                    project_id=PID,
                    run_type="batch_label_run",
                    status="running",
                    guidance_id=GID,
                    model_config_id=MCID,
                    examples_total=5,
                    metrics={"input_keys": [], "include_auto_labeled": False},
                )
            )
            s.commit()

        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module) as mock_reg,
        ):
            result = await start_batch_label_run(PID, settings=settings)

        assert isinstance(result, str)
        assert "conflict" in result.lower()
        mock_reg.assert_not_called()
        # No second run created.
        with Session(engine) as s:
            runs = (
                s.query(RunRecord)
                .filter_by(project_id=PID, run_type="batch_label_run")
                .all()
            )
            assert len(runs) == 1

    @pytest.mark.asyncio
    async def test_start_rejects_when_gate_not_ready(self, tmp_path):
        _setup_batch_project(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with patch(_GATE_PATCH, side_effect=_mock_gate_not_ready):
            result = await start_batch_label_run(PID, settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    @pytest.mark.asyncio
    async def test_start_no_guidance_returns_error(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir, active_guidance_id=None)
            s.commit()
        result = await start_batch_label_run(PID, settings=settings)
        assert isinstance(result, str)
        assert "guidance" in result.lower()

    @pytest.mark.asyncio
    async def test_start_no_teacher_returns_error(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir, teacher_model_config_id=None)
            add_fixture_guidance_row(s)
            s.commit()
        result = await start_batch_label_run(PID, settings=settings)
        assert isinstance(result, str)
        assert "teacher" in result.lower()

    @pytest.mark.asyncio
    async def test_start_zero_examples(self, tmp_path):
        """A run with zero matching examples should still create (completes immediately)."""
        _setup_batch_project(tmp_path, n_unlabeled=0)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(PID, settings=settings)
        assert not isinstance(result, str), result
        assert result["examples_total"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section B: Config Snapshot
# ══════════════════════════════════════════════════════════════════════════════


class TestConfigSnapshot:
    @pytest.mark.asyncio
    async def test_config_snapshot_persisted(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(PID, settings=settings)
        assert result["guidance_id"] == GID
        assert result["model_config_id"] == MCID
        assert result["generation_preset_key"] == "precise"
        assert result["thinking_mode_effective"] == "on"
        assert result["visual_budget_preset_key"] == "balanced"
        assert result["structured_generation_mode_effective"] == "auto"

    @pytest.mark.asyncio
    async def test_structured_generation_mode_override(self, tmp_path):
        _setup_batch_project(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(
                PID,
                structured_generation_mode="prompt_only",
                settings=settings,
            )
        assert result["structured_generation_mode_effective"] == "prompt_only"


# ══════════════════════════════════════════════════════════════════════════════
# Section C: Input Selection
# ══════════════════════════════════════════════════════════════════════════════


class TestInputSelection:
    @pytest.mark.asyncio
    async def test_default_selects_only_unlabeled(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(
            tmp_path, n_unlabeled=3, n_auto_labeled=2
        )
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(PID, settings=settings)
        assert result["examples_total"] == 3  # only Unlabeled

    @pytest.mark.asyncio
    async def test_include_auto_labeled(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(
            tmp_path, n_unlabeled=3, n_auto_labeled=2
        )
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(
                PID,
                include_auto_labeled=True,
                settings=settings,
            )
        assert result["examples_total"] == 5  # Unlabeled + Auto-Labeled

    @pytest.mark.asyncio
    async def test_ingested_after_filter(self, tmp_path):
        engine, pdir = setup_project_db(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with Session(engine) as s:
            add_standard_project_row(s, PID, pdir)
            add_fixture_guidance_row(s)
            add_endpoint_and_model_rows(s)
            _add_example(s, PID, "old", ingested_at="2025-01-01T00:00:00Z")
            _add_example(s, PID, "new", ingested_at="2026-06-01T00:00:00Z")
            s.commit()
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(
                PID,
                ingested_after="2026-01-01T00:00:00Z",
                settings=settings,
            )
        assert result["examples_total"] == 1

    @pytest.mark.asyncio
    async def test_run_limit_from_request(self, tmp_path):
        _setup_batch_project(tmp_path, n_unlabeled=10)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(
                PID,
                run_limit=3,
                settings=settings,
            )
        assert result["examples_total"] == 3

    @pytest.mark.asyncio
    async def test_run_limit_from_settings(self, tmp_path):
        _setup_batch_project(tmp_path, n_unlabeled=10)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_RUN_LIMIT=4,
        )
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(PID, settings=settings)
        assert result["examples_total"] == 4


# ══════════════════════════════════════════════════════════════════════════════
# Section D: Background Execution
# ══════════════════════════════════════════════════════════════════════════════


class TestExecution:
    @pytest.mark.asyncio
    async def test_terminalized_run_writes_no_labels_and_is_not_resurrected(
        self, tmp_path
    ):
        """An invocation already in flight when another writer terminalizes
        the run (schema evolution wiping this run's labels) must not land
        its auto-label after the wipe, and the completion finalizer must
        not resurrect the row — publishing a 'completed' run whose
        auto-labels were just deleted."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(2)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            # Terminalize the run while this invocation is "in flight".
            _terminalize_run(engine, rid)
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            labels = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .all()
            )
            assert labels == []
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"

    @pytest.mark.asyncio
    async def test_run_snapshotted_under_retired_guidance_fails_at_claim(
        self, tmp_path
    ):
        """A run created concurrently with a guidance edit can commit after
        the edit's cancel sweep ran — the one writer the sweep cannot see.
        Phase A re-checks the active Guidance under the claim's write lock
        and fails the run exactly as the sweep would have, before any
        Teacher call."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(2)]
        # The run is stamped with the (still existing) original version…
        _seed_bl_run(engine, run_id, keys)
        # …but by the time it executes, an edit has activated a new one.
        with Session(engine) as s:
            add_fixture_guidance_row(s, PID, "g2-post-edit", version_number=2)
            s.query(Project).filter_by(project_id=PID).update(
                {"active_guidance_id": "g2-post-edit"}
            )
            s.commit()

        with (
            patch(_INVOKE_PATCH, side_effect=AssertionError("must not invoke")),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "guidance_edited_during_run"
            labels = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .all()
            )
            assert labels == []

    @pytest.mark.asyncio
    async def test_circuit_breaker_pause_does_not_resurrect_terminalized_run(
        self, tmp_path
    ):
        """The pause write is the one status transition outside the
        finalizers: a run terminalized by another writer while the
        breaker was tripping must stay failed, not become a resumable
        'paused' row pointing at a wiped schema."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=3)
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(3)]
        _seed_bl_run(engine, run_id, keys)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=1,
        )

        async def _mock_invoke(pid, rid, ek, **kw):
            # Terminalize mid-flight, then return the endpoint error that
            # trips the breaker — the pause write follows the trip.
            _terminalize_run(engine, rid)
            return _make_batch_failure(ek, "endpoint_error")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "failed"
            assert run.status_reason == "schema_evolution_canceled"

    @pytest.mark.asyncio
    async def test_successful_run_completes(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=3)
        # Create run record directly
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(3)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_succeeded == 3
            assert run.completed_at is not None

            # Verify labels created
            labels = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .all()
            )
            assert len(labels) == 3

    @pytest.mark.asyncio
    async def test_labels_created_with_correct_fields(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        inv_id = generate_uuid4()

        async def _mock_invoke(pid, rid, ek, **kw):
            return BatchExampleResult(
                example_key=ek,
                invocation_id=inv_id,
                invocation_status="success",
                proposal_json={
                    "rationale_note": "batch",
                    "severity": "high",
                    "damaged": True,
                },
                schema_valid_core=True,
            )

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            label = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="ex_000",
                )
                .first()
            )
            assert label is not None
            assert label.label_status == "auto_labeled"
            assert label.batch_label_run_id == run_id
            assert label.guidance_id == GID
            assert label.inference_invocation_id == inv_id

    @pytest.mark.asyncio
    async def test_example_state_transitions_to_auto_labeled(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            ex = (
                s.query(Example)
                .filter_by(
                    project_id=PID,
                    example_key="ex_000",
                )
                .first()
            )
            assert ex.state == "Auto-Labeled"

    @pytest.mark.asyncio
    async def test_does_not_clobber_sme_label_when_state_changed_mid_run(
        self, tmp_path
    ):
        """A successful invocation must not overwrite SME work done mid-run.

        The input snapshot is taken at run start; an SME can Verify an example
        while the run is in flight. The write path re-checks the live
        example state and must skip persisting the auto-label — otherwise the
        verified label is shadowed and the image re-enters the review queue.
        """
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])
        with Session(engine) as s:
            # The SME verifies ex_000 after the run's snapshot was taken.
            ex = s.query(Example).filter_by(project_id=PID, example_key="ex_000").one()
            ex.state = "Verified"
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=PID,
                    example_key="ex_000",
                    label_status="verified",
                    guidance_id=GID,
                    inference_invocation_id=generate_uuid4(),
                    label_json={"severity": "high", "damaged": True},
                    labeled_at=utc_now(),
                    verified_outcome="Accept",
                    verified_at=utc_now(),
                )
            )
            s.commit()

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            # No auto_labeled Label was written for the verified example.
            auto = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="ex_000",
                    label_status="auto_labeled",
                )
                .all()
            )
            assert auto == [], "auto-label clobbered the SME's verified label"
            # SME state and verified label are untouched.
            ex = s.query(Example).filter_by(project_id=PID, example_key="ex_000").one()
            assert ex.state == "Verified"
            verified = (
                s.query(Label)
                .filter_by(
                    project_id=PID, example_key="ex_000", label_status="verified"
                )
                .all()
            )
            assert len(verified) == 1

    @pytest.mark.asyncio
    async def test_schema_invalid_does_not_create_label(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_schema_invalid(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_schema_invalid == 1
            assert run.examples_succeeded == 0
            label = (
                s.query(Label).filter_by(project_id=PID, example_key="ex_000").first()
            )
            assert label is None
            # Example state should NOT change
            ex = (
                s.query(Example).filter_by(project_id=PID, example_key="ex_000").first()
            )
            assert ex.state == "Unlabeled"

    @pytest.mark.asyncio
    async def test_timeout_increments_counter(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.examples_timeout == 1

    @pytest.mark.asyncio
    async def test_unhandled_exception_transitions_to_failed(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            raise RuntimeError("Simulated crash")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "failed"
            assert run.status_reason == "unhandled_exception"


# ══════════════════════════════════════════════════════════════════════════════
# Section E: Circuit Breaker
# ══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_pauses_on_consecutive_timeouts(self, tmp_path):
        n = 12  # more than threshold of 10
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.paused_reason == "circuit_breaker_threshold_reached"
            assert run.examples_timeout == 10
            assert run.completed_at is None  # paused is NOT terminal

    @pytest.mark.asyncio
    async def test_pauses_on_consecutive_endpoint_errors(self, tmp_path):
        n = 12
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_failure(ek, "endpoint_error")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.examples_endpoint_error == 10

    @pytest.mark.asyncio
    async def test_success_resets_circuit_breaker(self, tmp_path):
        """9 failures then a success then 9 more failures should NOT pause."""
        n = 19
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        call_count = 0

        async def _mock_invoke(pid, rid, ek, **kw):
            nonlocal call_count
            call_count += 1
            # 10th call succeeds (index 9), all others fail
            if call_count == 10:
                return _make_batch_success(ek)
            return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"  # no pause, completed normally
            assert run.examples_succeeded == 1
            assert run.examples_timeout == 18

    @pytest.mark.asyncio
    async def test_schema_invalid_does_not_increment_breaker(self, tmp_path):
        """5 timeouts, then 5 schema_invalid, then 5 more timeouts → total consecutive only 10."""
        n = 15
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        call_count = 0

        async def _mock_invoke(pid, rid, ek, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                return _make_batch_failure(ek, "timeout")
            elif call_count <= 10:
                return _make_batch_schema_invalid(ek)
            else:
                return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            # 5 timeouts + 5 schema_invalid (no change) + 5 more timeouts = 10 consecutive
            assert run.status == "paused"
            assert run.examples_timeout == 10
            assert run.examples_schema_invalid == 5

    @pytest.mark.asyncio
    async def test_schema_invalid_does_not_reset_breaker(self, tmp_path):
        """8 timeouts, then 1 schema_invalid, then 2 more timeouts → consecutive is 10 (not 2)."""
        n = 11
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        call_count = 0

        async def _mock_invoke(pid, rid, ek, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 8:
                return _make_batch_failure(ek, "timeout")
            elif call_count == 9:
                return _make_batch_schema_invalid(ek)
            else:
                return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.examples_timeout == 10  # 8 + 2

    @pytest.mark.asyncio
    async def test_threshold_configurable(self, tmp_path):
        n = 5
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=3,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.examples_timeout == 3


# ══════════════════════════════════════════════════════════════════════════════
# Section F: Resume
# ══════════════════════════════════════════════════════════════════════════════


class TestResume:
    @pytest.mark.asyncio
    async def test_resume_paused_run(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=3)
        run_id = generate_uuid4()
        _seed_bl_run(
            engine,
            run_id,
            ["ex_000", "ex_001", "ex_002"],
            status="paused",
            paused_reason="circuit_breaker_threshold_reached",
        )

        with patched_register(batch_label_service_module) as mock_reg:
            result = await resume_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "queued"
        mock_reg.assert_called_once()

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.paused_reason is None

    @pytest.mark.asyncio
    async def test_resume_non_paused_returns_conflict(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, None, status="completed", examples_total=3)
        result = await resume_batch_label_run(PID, run_id, settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    @pytest.mark.asyncio
    async def test_resume_skips_already_processed(self, tmp_path):
        """On resume, examples with existing OperationRecords are skipped."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=3)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001", "ex_002"]
        _seed_bl_run(engine, run_id, keys)
        with Session(engine) as s:
            # Simulate ex_000 already processed
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_000",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=True,
                    model_name="test-model",
                )
            )
            s.commit()

        invoke_calls = []

        async def _mock_invoke(pid, rid, ek, **kw):
            invoke_calls.append(ek)
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # ex_000 should have been skipped
        assert "ex_000" not in invoke_calls
        assert "ex_001" in invoke_calls
        assert "ex_002" in invoke_calls

    async def test_resume_reinvokes_pending_record(self, tmp_path):
        """A ``pending`` OperationRecord (written before the NIM call, then
        orphaned by a crash) is NOT done: resume must re-invoke that example,
        not strand it forever and miscount it as schema_invalid."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        _seed_bl_run(engine, run_id, keys)
        with Session(engine) as s:
            # ex_000 crashed mid-invoke: a pending record with no terminal
            # status. It must be retried, not treated as already-done.
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_000",
                    batch_label_run_id=run_id,
                    invocation_status="pending",
                    model_name="test-model",
                )
            )
            s.commit()

        invoke_calls = []

        async def _mock_invoke(pid, rid, ek, **kw):
            invoke_calls.append(ek)
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # The crashed-pending example is re-invoked, not stranded.
        assert "ex_000" in invoke_calls
        assert "ex_001" in invoke_calls
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_succeeded == 2
            assert run.examples_schema_invalid == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section G: Cancel
# ══════════════════════════════════════════════════════════════════════════════


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_running_sets_canceling(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, None, status="running", examples_total=5)

        result = await cancel_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "canceling"
        assert result["cancel_requested_at"] is not None

    @pytest.mark.asyncio
    async def test_cancel_paused_sets_canceled_directly(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(
            engine,
            run_id,
            None,
            status="paused",
            examples_total=5,
            paused_reason="circuit_breaker_threshold_reached",
        )

        result = await cancel_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "canceled"

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_of_a_just_resumed_run_falls_through_to_canceling(
        self, tmp_path, monkeypatch
    ):
        """Cancel read the run as paused, but a concurrent resume made it
        live again before the paused->canceled write. The write must not
        cancel a run whose executor is back (it would bypass the live
        task), and must not claim the run is terminal — it falls through
        to the ordinary canceling transition the executor honors."""
        from sqlalchemy import text

        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(
            engine,
            run_id,
            None,
            status="paused",
            examples_total=5,
            paused_reason="circuit_breaker_threshold_reached",
        )

        real = batch_label_service_module.update_run_if_not_terminal
        resumed = {"done": False}

        def _resume_then_delegate(session, rid, values, **kw):
            # The interleaved resume commits just before cancel's paused
            # write reaches the row; the real helper then sees 'queued'.
            if not resumed["done"] and kw.get("only_status") == "paused":
                resumed["done"] = True
                with engine.begin() as c2:
                    c2.execute(
                        text(
                            "UPDATE run_records SET status = 'queued' WHERE run_id = :r"
                        ),
                        {"r": rid},
                    )
            return real(session, rid, values, **kw)

        monkeypatch.setattr(
            batch_label_service_module,
            "update_run_if_not_terminal",
            _resume_then_delegate,
        )

        result = await cancel_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["status"] == "canceling"
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).one()
            assert run.status == "canceling"

    @pytest.mark.asyncio
    async def test_cancel_terminal_returns_conflict(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, None, status="completed", examples_total=5)

        result = await cancel_batch_label_run(PID, run_id, settings=settings)
        assert isinstance(result, str)
        assert "conflict" in result.lower()

    @pytest.mark.asyncio
    async def test_cancel_during_execution_stops_loop(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=5)
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(5)]
        _seed_bl_run(engine, run_id, keys)

        invoke_count = 0

        async def _mock_invoke(pid, rid, ek, **kw):
            nonlocal invoke_count
            invoke_count += 1
            # After 2 invocations, simulate cancel
            if invoke_count >= 2:
                from vlm_feedback_loop.services.batch_label_service import (
                    _cancel_events,
                )

                evt = _cancel_events.get(run_id)
                if evt:
                    evt.set()
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # Should not have processed all 5
        assert invoke_count < 5

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "canceled"


# ══════════════════════════════════════════════════════════════════════════════
# Section H: Get / List
# ══════════════════════════════════════════════════════════════════════════════


class TestGetList:
    def test_get_returns_full_detail(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(
            engine,
            run_id,
            None,
            status="completed",
            examples_total=10,
            examples_succeeded=8,
            examples_schema_invalid=1,
            examples_timeout=1,
        )

        result = get_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str), result
        assert result["run_id"] == run_id
        assert result["run_type"] == "batch_label_run"
        assert result["examples_succeeded"] == 8
        assert result["examples_total"] == 10

    def test_get_nonexistent_returns_error(self, tmp_path):
        _setup_batch_project(tmp_path)
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        result = get_batch_label_run(PID, "no-such-run", settings=settings)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_get_wrong_run_type_returns_error(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="evaluation_run",
                    status="completed",
                    examples_total=5,
                )
            )
            s.commit()
        result = get_batch_label_run(PID, run_id, settings=settings)
        assert isinstance(result, str)
        assert "not a batch" in result.lower()

    def test_list_returns_newest_first(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        rids = []
        with Session(engine) as s:
            for i in range(3):
                rid = generate_uuid4()
                rids.append(rid)
                s.add(
                    RunRecord(
                        run_id=rid,
                        project_id=PID,
                        run_type="batch_label_run",
                        status="completed",
                        examples_total=i,
                    )
                )
            s.commit()
        # Manually update created_at via raw SQL to get distinct timestamps
        # (the before_insert hook overrides provided values with utc_now()).
        from sqlalchemy import text

        with Session(engine) as s:
            for i, rid in enumerate(rids):
                s.execute(
                    text("UPDATE run_records SET created_at = :ts WHERE run_id = :rid"),
                    {"ts": f"2026-04-{10 + i:02d}T00:00:00Z", "rid": rid},
                )
            s.commit()

        items, cursor = list_batch_label_runs(PID, settings=settings)
        assert len(items) == 3
        # Newest first: April 12 > April 11 > April 10
        assert items[0]["created_at"] > items[1]["created_at"]
        assert items[1]["created_at"] > items[2]["created_at"]

    def test_get_resolves_model_name_and_guidance_version(self, tmp_path):
        """The report's config line renders resolved model_name + Guidance v{N}."""
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        _seed_bl_run(
            engine,
            run_id,
            None,
            status="completed",
            examples_total=1,
            examples_succeeded=1,
        )

        result = get_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str)
        assert result["model_name"] == "test-model"
        assert result["guidance_version_number"] == 1

    def test_get_handles_missing_fks(self, tmp_path):
        """`model_name` and `guidance_version_number` fall back to None on dangling FKs."""
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    guidance_id="missing-guid",
                    model_config_id="missing-mc",
                    examples_total=1,
                )
            )
            s.commit()

        result = get_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str)
        assert result["model_name"] is None
        assert result["guidance_version_number"] is None
        assert result["model_config_id"] == "missing-mc"  # Raw ID still returned

    def test_get_aggregates_common_errors(self, tmp_path):
        """The report's common-errors block groups by error signature."""
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    examples_total=7,
                )
            )
            # Seed 3 schema_invalid on primary_damage, 2 timeouts, 1 endpoint, 1 success
            for i in range(3):
                s.add(
                    OperationRecord(
                        inference_invocation_id=generate_uuid4(),
                        project_id=PID,
                        example_key=f"ex-pd-{i}",
                        purpose="batch_label",
                        invocation_status="schema_invalid",
                        validation_errors_core=[
                            "primary_damage: value not in allowed values"
                        ],
                        batch_label_run_id=run_id,
                    )
                )
            for i in range(2):
                s.add(
                    OperationRecord(
                        inference_invocation_id=generate_uuid4(),
                        project_id=PID,
                        example_key=f"ex-to-{i}",
                        purpose="batch_label",
                        invocation_status="timeout",
                        batch_label_run_id=run_id,
                    )
                )
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    example_key="ex-ep-1",
                    purpose="batch_label",
                    invocation_status="endpoint_error",
                    batch_label_run_id=run_id,
                )
            )
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    example_key="ex-ok-1",
                    purpose="batch_label",
                    invocation_status="success",
                    batch_label_run_id=run_id,
                )
            )
            # Canceled record should NOT appear in common_errors
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    example_key="ex-cancel-1",
                    purpose="batch_label",
                    invocation_status="schema_invalid",
                    validation_errors_core=["severity: out of range"],
                    batch_label_run_id=run_id,
                    ignored_due_to_run_cancellation=True,
                )
            )
            s.commit()

        result = get_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str)
        errors = result["common_errors"]
        # Sorted by count desc: primary_damage(3) > timeout(2) > endpoint_error(1)
        assert len(errors) == 3
        assert errors[0]["code"] == "schema_invalid:primary_damage"
        assert errors[0]["count"] == 3
        assert errors[1]["code"] == "timeout"
        assert errors[1]["count"] == 2
        assert errors[2]["code"] == "endpoint_error"
        assert errors[2]["count"] == 1

    def test_get_common_errors_empty_when_no_failures(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        run_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=run_id,
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    examples_total=1,
                )
            )
            s.commit()
        result = get_batch_label_run(PID, run_id, settings=settings)
        assert not isinstance(result, str)
        assert result["common_errors"] == []

    def test_list_with_status_filter(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path)
        with Session(engine) as s:
            s.add(
                RunRecord(
                    run_id=generate_uuid4(),
                    project_id=PID,
                    run_type="batch_label_run",
                    status="completed",
                    examples_total=5,
                )
            )
            s.add(
                RunRecord(
                    run_id=generate_uuid4(),
                    project_id=PID,
                    run_type="batch_label_run",
                    status="paused",
                    paused_reason="circuit_breaker_threshold_reached",
                    examples_total=5,
                )
            )
            s.commit()

        items, _ = list_batch_label_runs(
            PID,
            status_filter="paused",
            settings=settings,
        )
        assert len(items) == 1
        assert items[0]["status"] == "paused"


# ══════════════════════════════════════════════════════════════════════════════
# Section I: SSE Events
# ══════════════════════════════════════════════════════════════════════════════


class TestSSEEvents:
    @pytest.mark.asyncio
    async def test_progress_events_emitted(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000", "ex_001"])

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

            # Should have progress events (initial + per-example + completion)
            event_types = [call.args[1] for call in mock_sse.emit.call_args_list]
            assert "batch_label_progress" in event_types

    @pytest.mark.asyncio
    async def test_completed_event_emitted(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

            event_types = [call.args[1] for call in mock_sse.emit.call_args_list]
            assert "batch_label_completed" in event_types

    @pytest.mark.asyncio
    async def test_failed_event_emitted(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"])

        async def _mock_invoke(pid, rid, ek, **kw):
            raise RuntimeError("boom")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
            patch(_TR_SSE_PATCH) as mock_tr_sse,
        ):
            mock_sse.emit = AsyncMock()
            mock_tr_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

            event_types = [call.args[1] for call in mock_tr_sse.emit.call_args_list]
            assert "run_failed" in event_types

    @pytest.mark.asyncio
    async def test_paused_event_emitted(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=12)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(12)]
        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_failure(ek, "timeout")

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

            # Check that a progress event with paused=True was emitted
            paused_events = [
                call
                for call in mock_sse.emit.call_args_list
                if call.args[1] == "batch_label_progress"
                and call.args[2].get("paused") is True
            ]
            assert len(paused_events) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# Section J: Restart Recovery
# ══════════════════════════════════════════════════════════════════════════════


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_no_duplicate_label_on_resume(self, tmp_path):
        """When an example already has an OperationRecord + Label, skip it."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        _seed_bl_run(engine, run_id, keys)
        with Session(engine) as s:
            # Simulate ex_000 already processed
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_000",
                    batch_label_run_id=run_id,
                    invocation_status="success",
                    schema_valid_core=True,
                    model_name="test-model",
                )
            )
            # And its label
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=PID,
                    example_key="ex_000",
                    label_status="auto_labeled",
                    guidance_id=GID,
                    inference_invocation_id=generate_uuid4(),
                    label_json={
                        "rationale_note": "t",
                        "severity": "high",
                        "damaged": True,
                    },
                    labeled_at=utc_now(),
                    batch_label_run_id=run_id,
                )
            )
            s.commit()

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_success(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            # ex_000 should have exactly 1 label (not duplicated)
            labels_000 = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="ex_000",
                )
                .all()
            )
            assert len(labels_000) == 1

            # ex_001 should have a new label
            labels_001 = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="ex_001",
                )
                .all()
            )
            assert len(labels_001) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Section L: include_auto_labeled Re-labeling
# ══════════════════════════════════════════════════════════════════════════════


class TestReLabeling:
    @pytest.mark.asyncio
    async def test_include_auto_labeled_replaces_existing_label(self, tmp_path):
        engine, pdir, settings = _setup_batch_project(
            tmp_path,
            n_unlabeled=0,
            n_auto_labeled=1,
        )
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["auto_000"], include_auto_labeled=True)

        new_inv_id = generate_uuid4()

        async def _mock_invoke(pid, rid, ek, **kw):
            return BatchExampleResult(
                example_key=ek,
                invocation_id=new_inv_id,
                invocation_status="success",
                proposal_json={
                    "rationale_note": "new",
                    "severity": "high",
                    "damaged": True,
                },
                schema_valid_core=True,
            )

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            labels = (
                s.query(Label)
                .filter_by(
                    project_id=PID,
                    example_key="auto_000",
                )
                .all()
            )
            assert len(labels) == 1
            # Should be the new label, not the old one
            assert labels[0].batch_label_run_id == run_id
            assert labels[0].inference_invocation_id == new_inv_id
            assert labels[0].label_json["rationale_note"] == "new"


# ══════════════════════════════════════════════════════════════════════════════
# Section M: Schema-invalid integration flow
# ══════════════════════════════════════════════════════════════════════════════
#
# Each link in the schema-invalid chain (evaluator rejection, no Label,
# circuit-breaker counter rules, dataset export filter) is tested in
# isolation (earlier sections here; the export filter in
# test_dataset_export_service.py); only this class runs a batch with intentionally
# malformed outputs and asserts the full chain.
#
# The end-to-end flow:
#   mixed outcomes → counters correct → only valid examples labeled →
#   circuit breaker NOT incremented by schema_invalid → dataset export
#   excludes schema-invalid examples.


class TestSchemaInvalidIntegrationFlow:
    """Schema-invalid outputs must not leak into training exports.

    The batch labeling pipeline has four downstream consumers for a
    schema-invalid model output:
      1. Label persistence: MUST NOT create a Label record.
      2. Circuit breaker: MUST NOT increment the counter
         (only timeout/endpoint_error increment).
      3. Counters on RunRecord: examples_schema_invalid increments, but
         examples_succeeded does not.
      4. Dataset export: tier filter `auto_labeled_only` queries
         Label records, so schema-invalid examples are naturally excluded
         (no Label → nothing to export).

    A single end-to-end test verifies all four links agree.
    """

    @pytest.mark.asyncio
    async def test_schema_invalid_does_not_leak_into_export(self, tmp_path):
        from vlm_feedback_loop.services.dataset_export_service import (
            create_dataset_export,
        )

        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=4)
        run_id = generate_uuid4()
        # 4 Unlabeled examples: 3 valid, 1 schema-invalid (ex_001).
        keys = ["ex_000", "ex_001", "ex_002", "ex_003"]

        _seed_bl_run(engine, run_id, keys)

        valid_keys = {"ex_000", "ex_002", "ex_003"}
        invalid_keys = {"ex_001"}

        async def _mock_invoke(pid, rid, ek, **kw):
            if ek in valid_keys:
                return _make_batch_success(ek)
            return _make_batch_schema_invalid(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # ── Link 1 & 3: Label persistence + counters ────────────────────
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_succeeded == len(valid_keys), (
                "Only valid outputs should increment examples_succeeded"
            )
            assert run.examples_schema_invalid == len(invalid_keys), (
                "Invalid outputs increment examples_schema_invalid"
            )
            assert run.examples_timeout == 0
            assert run.examples_endpoint_error == 0

            # Labels exist ONLY for valid examples.
            labels_in_run = (
                s.query(Label)
                .filter_by(project_id=PID, batch_label_run_id=run_id)
                .all()
            )
            labeled_keys = {lbl.example_key for lbl in labels_in_run}
            assert labeled_keys == valid_keys, (
                f"Only schema-valid outputs produce Label "
                f"records. Valid keys={valid_keys}, labeled={labeled_keys}"
            )
            # Schema-invalid example has NO Label under this run.
            for k in invalid_keys:
                invalid_labels = (
                    s.query(Label).filter_by(project_id=PID, example_key=k).all()
                )
                assert invalid_labels == [], (
                    f"Example {k} was schema-invalid; no Label should exist"
                )

            # Example states: valid keys became Auto-Labeled, invalid stays Unlabeled.
            for k in valid_keys:
                ex = (
                    s.query(Example)
                    .filter_by(
                        project_id=PID,
                        example_key=k,
                    )
                    .first()
                )
                assert ex.state == "Auto-Labeled"
            for k in invalid_keys:
                ex = (
                    s.query(Example)
                    .filter_by(
                        project_id=PID,
                        example_key=k,
                    )
                    .first()
                )
                assert ex.state == "Unlabeled", (
                    "Schema-invalid example must NOT advance to Auto-Labeled"
                )

        # ── Link 4: dataset export excludes schema-invalid ──────────────
        # The export ships only examples whose image exists on disk;
        # the fixture's /fake refs would be excluded wholesale,
        # so point every row at a real file first.
        imgs_dir = tmp_path / "imgs"
        imgs_dir.mkdir(exist_ok=True)
        with Session(engine) as s:
            for ex in s.query(Example).filter_by(project_id=PID).all():
                real = imgs_dir / f"{ex.example_key}.jpg"
                real.write_bytes(b"img")
                ex.storage_ref = str(real)
            s.commit()

        export = create_dataset_export(
            project_id=PID,
            dataset_intent="training",
            label_tier_filter="auto_labeled_only",
            export_field_mode="all",
            batch_label_run_id=run_id,
            selection_filters=None,
            settings=settings,
        )
        assert export["example_count"] == len(valid_keys)

        # Untar the archive and verify only valid examples appear in
        # annotations.json — schema-invalid examples MUST NOT leak into
        # training data.
        import tarfile

        archive_path = Path(export["artifact_refs"]["archive_path"])
        assert archive_path.exists()
        exported_keys: set[str] = set()
        with tarfile.open(archive_path, "r:gz") as tf:
            # Tar entries are namespaced under ``{export_id}/`` for the
            # cosmos-rl extraction-layout contract.
            ann_member = next(
                (m for m in tf.getmembers() if m.name.endswith("annotations.json")),
                None,
            )
            assert ann_member is not None
            member = tf.extractfile(ann_member)
            assert member is not None
            annotations = json.loads(member.read().decode("utf-8"))
        assert isinstance(annotations, list), (
            "annotations.json MUST be a top-level JSON array"
        )
        for item in annotations:
            exported_keys.add(item["id"])
        assert exported_keys == valid_keys, (
            f"Dataset export leaked schema-invalid examples. "
            f"Expected {valid_keys}, got {exported_keys}"
        )
        for invalid_key in invalid_keys:
            assert invalid_key not in exported_keys, (
                f"Schema-invalid example {invalid_key} leaked into training export"
            )

    @pytest.mark.asyncio
    async def test_schema_invalid_does_not_trip_circuit_breaker(self, tmp_path):
        """Schema-invalid must NOT increment the circuit-breaker counter.

        Run a batch where every outcome is schema-invalid — more than the
        circuit-breaker threshold of 10 — and verify the run completes
        normally without ever entering ``paused`` state.  This catches a
        regression where a refactor accidentally treats schema_invalid as a
        "failure" that advances the breaker (which would falsely halt batch
        runs on transient model schema-quirks and burn through the Scale-Up
        operator's investigation time).
        """
        # Build a project with threshold+5 examples so we cleanly exceed it.
        n_examples = 15  # > default BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=10
        engine, pdir, settings = _setup_batch_project(
            tmp_path,
            n_unlabeled=n_examples,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n_examples)]

        _seed_bl_run(engine, run_id, keys)

        async def _mock_invoke(pid, rid, ek, **kw):
            return _make_batch_schema_invalid(ek)

        with (
            patch(_INVOKE_PATCH, side_effect=_mock_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            # All schema_invalid, but run MUST complete — circuit breaker
            # only cares about timeout + endpoint_error.
            assert run.status == "completed", (
                f"schema_invalid MUST NOT trip the circuit breaker. "
                f"Got status={run.status} after {n_examples} schema-invalid outputs"
            )
            assert run.paused_reason is None
            assert run.examples_schema_invalid == n_examples
            assert run.examples_succeeded == 0
            assert run.examples_timeout == 0
            assert run.examples_endpoint_error == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section N: wire-mock e2e — real _invoke_for_batch_label against mocked NIM
# ══════════════════════════════════════════════════════════════════════════════
#
# Mirrors test_evaluation_service.py TestProfileBProductionPipeline. Lets
# the real ``_invoke_for_batch_label`` body run and mocks the lowest
# layer (``nim_client.chat_completions`` plus image-transport helpers),
# so regressions inside the invoke pipeline are caught here rather than
# hidden behind the fully-stubbed ``_invoke_for_batch_label`` used above.


def _bl_patch_nim_pipeline(chat_return):
    """Patch nim_client.chat_completions + prepare_images."""
    if callable(chat_return) or isinstance(chat_return, list):
        chat_mock = AsyncMock(side_effect=chat_return)
    else:
        chat_mock = AsyncMock(return_value=chat_return)

    return (
        patch(
            "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
            new=chat_mock,
        ),
        patch(
            "vlm_feedback_loop.services.prompt_service.prepare_images",
            new=AsyncMock(return_value=fake_prepare_result(1)),
        ),
    )


class TestRestartAutoResumeEndToEnd:
    """Auto-resume, end to end: interrupted run →
    REAL startup scan → the same executor main.py dispatches → terminal
    ``completed`` with exactly one Label per example (the pre-restart
    example is idempotently skipped, not re-invoked)."""

    @pytest.mark.asyncio
    async def test_interrupted_run_resumes_to_completed_without_duplicates(
        self, tmp_path
    ):
        from vlm_feedback_loop.db.models.operation import OperationRecord
        from vlm_feedback_loop.main import _recover_interrupted_runs

        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        _seed_bl_run(engine, run_id, keys)

        # Pre-restart state: the run was mid-flight and ex_000 was already
        # fully processed (OperationRecord + auto-label + state flip).
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            run.status = "running"
            s.add(
                OperationRecord(
                    inference_invocation_id=generate_uuid4(),
                    project_id=PID,
                    purpose="batch_label",
                    example_key="ex_000",
                    guidance_id=GID,
                    model_config_id=MCID,
                    endpoint_id=EID,
                    model_name="test-model",
                    invocation_status="success",
                    schema_valid_core=True,
                    label_tier="auto_labeled",
                    batch_label_run_id=run_id,
                )
            )
            _add_auto_label(s, PID, "ex_000", run_id=run_id)
            ex = s.query(Example).filter_by(example_key="ex_000").first()
            ex.state = "Auto-Labeled"
            s.commit()

        # The REAL startup scan (main.py) recovers it and hands it back
        # for dispatch.
        resume_targets = _recover_interrupted_runs(settings)
        assert (PID, run_id) in resume_targets
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "queued"
            assert run.recovered_from_restart is True

        # The executor main.py dispatches finishes the run (NIM mocked at
        # the wire level).
        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        chat_patch, prepare_patch = _bl_patch_nim_pipeline(fake_nim_success(gt_json))
        with chat_patch as chat_mock, prepare_patch, patch(_SSE_PATCH) as mock_sse:
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # Only the pending example hit the NIM.
        assert chat_mock.await_count == 1

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason or "(no reason)"
            for key in keys:
                labels = s.query(Label).filter_by(project_id=PID, example_key=key).all()
                assert len(labels) == 1, f"{key}: expected exactly 1 label"
                assert labels[0].label_status == "auto_labeled"


class TestProfileBProductionPipeline:
    """The real _invoke_for_batch_label body, with NIM mocked at the wire level."""

    @pytest.mark.asyncio
    async def test_happy_path_writes_auto_labeled_labels(self, tmp_path):
        """Two Unlabeled examples, NIM returns schema-valid JSON → run
        completes, 2 Label records with label_status='auto_labeled' are
        created, Example rows transition to 'Auto-Labeled', and 2
        OperationRecords persist with purpose='batch_label' +
        batch_label_run_id set."""
        from vlm_feedback_loop.db.models.operation import OperationRecord

        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        _seed_bl_run(engine, run_id, keys)

        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        patches = _bl_patch_nim_pipeline(fake_nim_success(gt_json))

        with (
            patches[0],
            patches[1],
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason or "(no reason)"
            assert run.examples_succeeded == 2
            assert run.examples_schema_invalid == 0
            assert run.examples_timeout == 0

            records = (
                s.query(OperationRecord)
                .filter_by(batch_label_run_id=run_id, purpose="batch_label")
                .all()
            )
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "success"
                assert r.schema_valid_core is True
                assert r.label_tier == "auto_labeled"
                assert r.model_name == "test-model"
                assert r.guidance_id == GID

            labels = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .all()
            )
            assert len(labels) == 2
            label_keys = sorted(lbl.example_key for lbl in labels)
            assert label_keys == ["ex_000", "ex_001"]
            for lbl in labels:
                assert lbl.batch_label_run_id == run_id
                assert lbl.label_json["severity"] == "high"
                assert lbl.label_json["damaged"] is True

            examples = (
                s.query(Example)
                .filter(Example.project_id == PID, Example.example_key.in_(keys))
                .all()
            )
            for ex in examples:
                assert ex.state == "Auto-Labeled"

    @pytest.mark.asyncio
    async def test_schema_invalid_creates_no_label(self, tmp_path):
        """NIM returns malformed JSON → OperationRecord persisted with
        schema_valid_core=False; NO Label record created (only
        schema_valid_core=true outputs produce Label records); circuit
        breaker NOT incremented (schema_invalid is ignored); run
        finalizes 'completed'."""
        from vlm_feedback_loop.db.models.operation import OperationRecord

        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=2)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001"]
        _seed_bl_run(engine, run_id, keys)

        patches = _bl_patch_nim_pipeline(fake_nim_success("not valid json"))

        with (
            patches[0],
            patches[1],
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason
            assert run.examples_succeeded == 0
            assert run.examples_schema_invalid == 2
            assert run.paused_reason is None

            records = (
                s.query(OperationRecord).filter_by(batch_label_run_id=run_id).all()
            )
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "schema_invalid"
                assert r.schema_valid_core is False

            labels = s.query(Label).filter_by(project_id=PID).all()
            assert len(labels) == 0

    @pytest.mark.asyncio
    async def test_timeout_increments_circuit_breaker_and_pauses(self, tmp_path):
        """NIM times out on every call → examples_timeout counter ticks;
        with circuit-breaker threshold 2 and 3 inputs, run pauses after
        the second consecutive timeout with paused_reason set."""
        from vlm_feedback_loop.db.models.operation import OperationRecord

        n = 3
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=n)
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=2,
        )
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(n)]
        _seed_bl_run(engine, run_id, keys)

        patches = _bl_patch_nim_pipeline(
            fake_nim_failure("Request timed out", status_code=504)
        )

        with (
            patches[0],
            patches[1],
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.paused_reason == "circuit_breaker_threshold_reached"
            # Exactly threshold timeouts → paused before the 3rd call
            assert run.examples_timeout == 2
            assert run.examples_succeeded == 0

            # Both timeouts left OperationRecords for audit
            records = (
                s.query(OperationRecord).filter_by(batch_label_run_id=run_id).all()
            )
            assert len(records) == 2
            for r in records:
                assert r.invocation_status == "timeout"

    @pytest.mark.asyncio
    async def test_structured_gen_rejection_fails_run_under_auto_mode(self, tmp_path):
        """Mid-run response_format 4xx rejection under ``auto`` mode
        MUST fail the whole run with
        ``status_reason="structured_generation_rejected"`` — distinct from
        the circuit-breaker ``paused`` path."""
        from vlm_feedback_loop.db.models.operation import OperationRecord

        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=3)
        run_id = generate_uuid4()
        keys = ["ex_000", "ex_001", "ex_002"]
        _seed_bl_run(engine, run_id, keys)

        rejection = fake_nim_failure(
            "HTTP 400: response_format json_schema not supported"
        )
        patches = _bl_patch_nim_pipeline(rejection)

        with (
            patches[0],
            patches[1],
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "failed"
            assert run.status_reason == "structured_generation_rejected"
            # The loop breaks after the first rejection fires cancel_event,
            # so only one OperationRecord should exist for the whole run.
            records = (
                s.query(OperationRecord).filter_by(batch_label_run_id=run_id).all()
            )
            assert len(records) == 1
            assert records[0].invocation_status == "endpoint_error"

            # No Labels created on a rejected run.
            labels = s.query(Label).filter_by(project_id=PID).all()
            assert len(labels) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section: Provider-aware concurrent dispatch
# ══════════════════════════════════════════════════════════════════════════════


def _setup_batch_project_with_mode(tmp_path, endpoint_mode, n_unlabeled):
    """``_setup_batch_project`` variant with an explicit Teacher endpoint mode."""
    engine, pdir = setup_project_db(tmp_path)
    settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))

    with Session(engine) as s:
        add_standard_project_row(s, PID, pdir)
        add_fixture_guidance_row(s)
        add_endpoint_row(s, PID, EID, endpoint_mode=endpoint_mode)
        add_model_config_row(s, PID, MCID, EID)
        for i in range(n_unlabeled):
            _add_example(s, PID, f"ex_{i:03d}", state="Unlabeled")
        s.commit()

    return engine, pdir, settings


def _make_in_flight_tracker(result_factory):
    """Async mock invoke + counters recording the max concurrent calls.

    Returns ``(async_invoke, state)``; a plain ``async def`` (not a class
    with async ``__call__``) so AsyncMock's side_effect awaits it.
    """
    state = {"in_flight": 0, "max_in_flight": 0, "calls": 0}

    async def _invoke(pid, rid, ek, **kw):
        state["calls"] += 1
        state["in_flight"] += 1
        state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
        # Yield long enough that every idle lane dispatches before the first
        # call returns — makes max_in_flight deterministic.
        await asyncio.sleep(0.02)
        state["in_flight"] -= 1
        return result_factory(ek)

    return _invoke, state


class TestConcurrentDispatch:
    """Batch dispatch width is provider-aware, like the evaluation worker."""

    @pytest.mark.asyncio
    async def test_self_hosted_endpoint_runs_parallel_lanes(self, tmp_path):
        """A self-hosted Teacher labels in N parallel lanes, not serially."""
        engine, pdir, _ = _setup_batch_project_with_mode(
            tmp_path, "self_hosted", n_unlabeled=8
        )
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CONCURRENCY_SELF_HOSTED=4,
        )
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, [f"ex_{i:03d}" for i in range(8)])

        tracker_invoke, tracker = _make_in_flight_tracker(_make_batch_success)
        with (
            patch(_INVOKE_PATCH, side_effect=tracker_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        assert tracker["max_in_flight"] == 4
        assert tracker["calls"] == 8
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_succeeded == 8
            labels = (
                s.query(Label)
                .filter_by(project_id=PID, label_status="auto_labeled")
                .all()
            )
            assert len(labels) == 8

    @pytest.mark.asyncio
    async def test_hosted_endpoint_stays_serial(self, tmp_path):
        """Hosted endpoints keep dispatch width 1 under shared RPM caps."""
        engine, pdir, _ = _setup_batch_project_with_mode(
            tmp_path, "hosted", n_unlabeled=6
        )
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CONCURRENCY_SELF_HOSTED=4,
        )
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, [f"ex_{i:03d}" for i in range(6)])

        tracker_invoke, tracker = _make_in_flight_tracker(_make_batch_success)
        with (
            patch(_INVOKE_PATCH, side_effect=tracker_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        assert tracker["max_in_flight"] == 1
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"
            assert run.examples_succeeded == 6

    @pytest.mark.asyncio
    async def test_concurrency_override_beats_provider_default(self, tmp_path):
        """A persisted per-run override wins over the provider default —
        including after restart recovery, which re-reads run metrics."""
        engine, pdir, _ = _setup_batch_project_with_mode(
            tmp_path, "hosted", n_unlabeled=6
        )
        settings = make_stub_settings(WORKSPACE_ROOT=str(tmp_path / "workspace"))
        run_id = generate_uuid4()
        keys = [f"ex_{i:03d}" for i in range(6)]
        _seed_bl_run(
            engine,
            run_id,
            None,
            examples_total=6,
            metrics={
                "input_keys": keys,
                "include_auto_labeled": False,
                "concurrency_override": 3,
            },
        )

        tracker_invoke, tracker = _make_in_flight_tracker(_make_batch_success)
        with (
            patch(_INVOKE_PATCH, side_effect=tracker_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        assert tracker["max_in_flight"] == 3
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed"

    @pytest.mark.asyncio
    async def test_start_persists_concurrency_override(self, tmp_path):
        """The start request's ``concurrency`` lands in run metrics, so a
        restart-recovered run resumes at the operator's chosen width."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=5)
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(PID, concurrency=5, settings=settings)
        assert not isinstance(result, str), result

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.metrics["concurrency_override"] == 5

    @pytest.mark.asyncio
    async def test_breaker_stops_new_dispatch_across_lanes(self, tmp_path):
        """When the breaker trips, lanes stop pulling new work; in-flight
        requests complete and are recorded before the run pauses."""
        n = 20
        engine, pdir, _ = _setup_batch_project_with_mode(
            tmp_path, "self_hosted", n_unlabeled=n
        )
        settings = make_stub_settings(
            WORKSPACE_ROOT=str(tmp_path / "workspace"),
            BATCH_LABEL_CONCURRENCY_SELF_HOSTED=4,
            BATCH_LABEL_CIRCUIT_BREAKER_THRESHOLD=3,
        )
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, [f"ex_{i:03d}" for i in range(n)])

        tracker_invoke, tracker = _make_in_flight_tracker(
            lambda ek: _make_batch_failure(ek, "endpoint_error")
        )
        with (
            patch(_INVOKE_PATCH, side_effect=tracker_invoke),
            patch(_SSE_PATCH) as mock_sse,
        ):
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        # Each pre-trip completion can let its lane pull one more key, so the
        # dispatch bound is threshold + lane count — far below the full run.
        assert tracker["calls"] <= 3 + 4
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "paused"
            assert run.paused_reason == "circuit_breaker_threshold_reached"
            assert run.completed_at is None
            # Every completed in-flight invocation is counted, none dropped.
            assert run.examples_endpoint_error == tracker["calls"]


# ══════════════════════════════════════════════════════════════════════════════
# Section O: per-run ICL mode for batch labeling
# ══════════════════════════════════════════════════════════════════════════════


def _add_icl_edit(session, key: str = "edit_000") -> None:
    """Seed one ICL-eligible verified Edit (§6.2: verified + no pool +
    Edit outcome + active guidance) with its Example row."""
    _add_example(session, PID, key, state="Verified")
    session.add(
        Label(
            label_id=generate_uuid4(),
            project_id=PID,
            example_key=key,
            label_status="verified",
            verified_outcome="Edit",
            pool_assignment=None,
            guidance_id=GID,
            inference_invocation_id=generate_uuid4(),
            label_json={
                "rationale_note": "corrected",
                "severity": "low",
                "damaged": False,
            },
            labeled_at=utc_now(),
        )
    )


class TestIclModeForBatch:
    """Batch labeling accepts the evaluation API's ``icl_mode``.
    ICL-negative teachers exist (freiburg/Omni: zero-shot 0.525 vs
    depth-1 0.217); ``"disabled"`` is the supported zero-shot batch path,
    replacing the process-global ``ICL_MAX_EXAMPLES=0`` workaround."""

    def _patch_pipeline(self, chat_return, n_images: int):
        return (
            patch(
                "vlm_feedback_loop.services.prompt_service.nim_client.chat_completions",
                new=AsyncMock(return_value=chat_return),
            ),
            patch(
                "vlm_feedback_loop.services.prompt_service.prepare_images",
                new=AsyncMock(return_value=fake_prepare_result(n_images)),
            ),
        )

    @pytest.mark.asyncio
    async def test_disabled_run_attaches_no_icl(self, tmp_path):
        """icl_mode='disabled' on the run → the eligible Edit is NOT
        selected; the OperationRecord shows zero ICL examples used."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        with Session(engine) as s:
            _add_icl_edit(s)
            s.commit()
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"], icl_mode="disabled")

        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        chat_patch, prep_patch = self._patch_pipeline(
            fake_nim_success(gt_json), n_images=1
        )
        with chat_patch, prep_patch, patch(_SSE_PATCH) as mock_sse:
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason or "(no reason)"
            record = (
                s.query(OperationRecord)
                .filter_by(batch_label_run_id=run_id, example_key="ex_000")
                .one()
            )
            assert not record.icl_example_keys_used
            assert (record.icl_images_attached_count or 0) == 0

    @pytest.mark.asyncio
    async def test_enabled_run_attaches_the_edit(self, tmp_path):
        """Control: the same project state with the default mode selects
        the eligible Edit — proving the disabled case above is the knob,
        not a broken pool."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        with Session(engine) as s:
            _add_icl_edit(s)
            s.commit()
        run_id = generate_uuid4()
        _seed_bl_run(engine, run_id, ["ex_000"], icl_mode="enabled")

        gt_json = '{"rationale_note":"ok","severity":"high","damaged":true}'
        chat_patch, prep_patch = self._patch_pipeline(
            fake_nim_success(gt_json), n_images=2
        )
        with chat_patch, prep_patch, patch(_SSE_PATCH) as mock_sse:
            mock_sse.emit = AsyncMock()
            await _execute_batch_label(PID, run_id, settings)

        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=run_id).first()
            assert run.status == "completed", run.status_reason or "(no reason)"
            record = (
                s.query(OperationRecord)
                .filter_by(batch_label_run_id=run_id, example_key="ex_000")
                .one()
            )
            assert record.icl_example_keys_used == ["edit_000"]

    @pytest.mark.asyncio
    async def test_icl_mode_persisted_and_echoed(self, tmp_path):
        """start_batch_label_run snapshots icl_mode on the RunRecord (so
        restart recovery resumes an ICL-off run as ICL-off) and echoes it
        on the create response."""
        engine, pdir, settings = _setup_batch_project(tmp_path, n_unlabeled=1)
        with (
            patch(_GATE_PATCH, side_effect=_mock_gate_ready),
            patched_register(batch_label_service_module),
        ):
            result = await start_batch_label_run(
                PID, icl_mode="disabled", settings=settings
            )
        assert not isinstance(result, str), result
        assert result["icl_mode"] == "disabled"
        with Session(engine) as s:
            run = s.query(RunRecord).filter_by(run_id=result["run_id"]).first()
            assert run.icl_mode == "disabled"
