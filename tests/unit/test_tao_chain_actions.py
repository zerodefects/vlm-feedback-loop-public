# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the chain-action dispatch in TAO polling.

Exercises the action-specific dispatch in ``_handle_succeeded`` and the
``evaluate``-failure handling in ``handle_terminal_failure``:

* ``train`` succeeded → registers a ``StudentModel`` with packaging validated.
* ``evaluate`` succeeded → dispatches the canonical-evaluator rescore.
* ``evaluate`` succeeded but rescore returns no parseable predictions → the
  paired ``StudentModel.quality_status`` flips to ``failed``.
* Downstream service exceptions during ``_handle_succeeded`` MUST NOT crash
  the polling loop.
* ``evaluate`` failed / canceled → paired ``StudentModel.quality_status``
  flips to ``failed``.
* ``train`` failed → halts the chain but MUST NOT touch a ``StudentModel``
  that was never registered (packaging never ran).

The polling loop itself is tested in ``test_tao_polling_service``; this
file calls the action-dispatch hooks directly with synthetic inputs so
each downstream service is asserted explicitly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_guidance_row,
    add_tao_job_row,
    make_settings,
    seed_tao_chain_project,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import (
    student_model_service,
    tao_polling_service,
    tao_rescoring_service,
)
from vlm_feedback_loop.services.project_service import (
    set_project_engine,
)

PID = "proj-chain-actions"
GID = "g-chain-actions"


def _seed_project(tmp_path: Path):
    workspace = tmp_path / "workspace"
    pdir = workspace / "projects" / PID
    pdir.mkdir(parents=True, exist_ok=True)
    engine = open_project_db(pdir)
    set_project_engine(PID, engine)
    with Session(engine) as s:
        add_guidance_row(
            s,
            PID,
            GID,
            {
                "fields": [
                    {
                        "field_id": "rn",
                        "field_name": "rationale_note",
                        "type": "string",
                        "role": "aux",
                        "display_order": 0,
                    },
                    {
                        "field_id": "dmg",
                        "field_name": "damage_type",
                        "type": "enum",
                        "role": "core",
                        "allowed_values": ["crush", "dent"],
                        "display_order": 1,
                    },
                ]
            },
        )
        seed_tao_chain_project(s, PID, str(pdir), guidance_id=GID)
        s.commit()
    return engine, workspace, pdir


def _seed_chain_baseline(engine, pdir: Path, chain_id: str):
    """Seed a minimal train→evaluate chain with both in 'running' state."""
    train_id = generate_uuid4()
    eval_id = generate_uuid4()
    suite_id = generate_uuid4()

    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key=f"idem-{chain_id}",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-eval",
                selected_student_base_model_config_ids=["mc-1"],
                quantization_schemes=[],
                chain_ids_ordered=[chain_id],
                status="running",
            )
        )
        cache = pdir / "artifacts" / "tao_jobs" / train_id
        best = cache / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "config.json").write_text("{}")
        (best / "model.safetensors").write_bytes(b"x")
        (best / "tokenizer.json").write_text("{}")
        add_tao_job_row(
            s,
            PID,
            train_id,
            action="train",
            status="running",
            dataset_export_ids=["de-train"],
            job_config={
                "training_preset": "standard",
                "lora_config": {"enable_lora": True},
                "resolved_training_fields": {
                    "policy": {"model_name_or_path": "nv/cosmos"}
                },
            },
            tao_create_job_request={},
            outputs={"artifact_cache_dir": str(cache)},
            tao_external_job_id="ext-tr",
            chain_id=chain_id,
            chain_sequence=1,
        )
        add_tao_job_row(
            s,
            PID,
            eval_id,
            action="evaluate",
            status="running",
            dataset_export_ids=["de-eval"],
            tao_create_job_request={},
            outputs={"artifact_cache_dir": "/nonexistent"},
            tao_external_job_id="ext-ev",
            parent_tao_job_id=train_id,
            chain_id=chain_id,
            chain_sequence=2,
        )
        s.commit()

    return train_id, eval_id


# ── Shared mocks for HTTP-side effects ────────────────────────────────────


@pytest.fixture
def patch_tao_http(monkeypatch):
    """Silence TAO artifact + logs HTTP — tests run without network."""
    monkeypatch.setattr(
        tao_polling_service,
        "_fetch_tao_artifacts",
        AsyncMock(return_value={"success": True, "artifacts": [], "error": None}),
    )
    monkeypatch.setattr(
        tao_polling_service,
        "_fetch_tao_logs",
        AsyncMock(return_value={"success": True, "logs_ref": None, "error": None}),
    )


# ═══════════════════════════════════════════════════════════════════════════
# _handle_succeeded dispatch by action
# ═══════════════════════════════════════════════════════════════════════════


class TestHandleSucceededDispatch:
    @pytest.mark.asyncio
    async def test_train_succeeded_registers_student_model(
        self, tmp_path, patch_tao_http
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-T")

        # Mark the job 'succeeded' before the hook, matching the real flow
        # (the poll loop commits the status transition first, then calls
        # _handle_succeeded).
        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "succeeded"
            s.commit()

        settings = make_settings(workspace)
        await tao_polling_service._handle_succeeded(
            PID,
            train_id,
            external_id="ext-tr",
            action="train",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            students = s.query(StudentModel).all()
            assert len(students) == 1
            assert students[0].tao_job_id == train_id
            assert students[0].checkpoint_packaging_status == "validated"
            assert students[0].quality_status == "pending"
            assert students[0].quantize_tao_job_id is None

    @pytest.mark.asyncio
    async def test_evaluate_succeeded_dispatches_rescore(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(engine, pdir, chain_id="chain-E")
        # Pre-register the StudentModel (simulates a prior train success).
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        # Mark eval succeeded + mock rescore_evaluate_job to return a run id
        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            ej.status = "succeeded"
            s.commit()

        mocked_run = "eval-run-id-xyz"
        mock_rescore = AsyncMock(return_value=mocked_run)
        monkeypatch.setattr(tao_rescoring_service, "rescore_evaluate_job", mock_rescore)

        await tao_polling_service._handle_succeeded(
            PID,
            eval_id,
            external_id="ext-ev",
            action="evaluate",
            engine=engine,
            settings=settings,
        )

        mock_rescore.assert_awaited_once()
        args, kwargs = mock_rescore.await_args
        assert args[0] == PID
        assert args[1] == eval_id
        # Student was not marked failed — rescore succeeded.
        with Session(engine) as s:
            student = s.query(StudentModel).one()
            assert (
                student.quality_status == "pending"
            )  # unchanged by hook; rescore updates directly

    @pytest.mark.asyncio
    async def test_evaluate_succeeded_but_rescore_returns_none_marks_failed(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """C2 widened: succeeded + empty predictions → quality_status=failed."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(engine, pdir, chain_id="chain-C2")
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            ej.status = "succeeded"
            s.commit()

        monkeypatch.setattr(
            tao_rescoring_service,
            "rescore_evaluate_job",
            AsyncMock(return_value=None),  # C2
        )

        await tao_polling_service._handle_succeeded(
            PID,
            eval_id,
            external_id="ext-ev",
            action="evaluate",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            student = s.query(StudentModel).one()
            assert student.quality_status == "failed"
            assert (
                student.nim_preflight_details.get("quality_failure_reason")
                == "no_parseable_predictions"
            )

    @pytest.mark.asyncio
    async def test_downstream_failure_does_not_crash_poller(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-crash")
        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "succeeded"
            s.commit()

        # Force the downstream service to raise.
        async def boom(*args, **kwargs):
            raise RuntimeError("synthetic packaging explosion")

        monkeypatch.setattr(student_model_service, "register_from_tao_terminal", boom)

        settings = make_settings(workspace)
        # Should NOT raise.
        await tao_polling_service._handle_succeeded(
            PID,
            train_id,
            external_id="ext-tr",
            action="train",
            engine=engine,
            settings=settings,
        )


# ═══════════════════════════════════════════════════════════════════════════
# handle_terminal_failure for evaluate action
# ═══════════════════════════════════════════════════════════════════════════


class TestHandleFailedEvaluateFlipsStudent:
    @pytest.mark.asyncio
    async def test_evaluate_failed_marks_student_quality_failed(
        self, tmp_path, patch_tao_http
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(engine, pdir, chain_id="chain-failE")
        settings = make_settings(workspace)
        # Pre-register baseline StudentModel.
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        # Mark evaluate as failed.
        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            ej.status = "failed"
            s.commit()

        await tao_polling_service.handle_terminal_failure(
            PID,
            eval_id,
            chain_id="chain-failE",
            chain_sequence=2,
            action="evaluate",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            student = s.query(StudentModel).one()
            assert student.quality_status == "failed"
            assert (
                "tao_evaluate_failed"
                in student.nim_preflight_details["quality_failure_reason"]
            )

    @pytest.mark.asyncio
    async def test_train_failed_does_not_mark_student_quality_failed(
        self, tmp_path, patch_tao_http
    ):
        """Train failure halts the chain; it does NOT touch a StudentModel
        that was never registered (packaging never ran).
        """
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-failT")
        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "failed"
            s.commit()

        settings = make_settings(workspace)
        await tao_polling_service.handle_terminal_failure(
            PID,
            train_id,
            chain_id="chain-failT",
            chain_sequence=1,
            action="train",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            assert s.query(StudentModel).count() == 0

    @pytest.mark.asyncio
    async def test_local_nim_operational_failure_preserves_pending_quality(
        self, tmp_path, patch_tao_http
    ):
        """A failed local serving attempt is not a measured quality result."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(
            engine, pdir, chain_id="chain-local-nim-fail"
        )
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        with Session(engine) as s:
            evaluate = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            evaluate.status = "failed"
            evaluate.training_backend = "student_nim_local"
            evaluate.error_ref = "student_nim_evaluation_failed"
            s.commit()

        await tao_polling_service.handle_terminal_failure(
            PID,
            eval_id,
            chain_id="chain-local-nim-fail",
            chain_sequence=2,
            action="evaluate",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            student = s.query(StudentModel).one()
            assert student.quality_status == "pending"
            assert not (student.nim_preflight_details or {}).get(
                "quality_failure_reason"
            )

    @pytest.mark.asyncio
    async def test_evaluate_canceled_flips_student_quality_failed(
        self, tmp_path, patch_tao_http
    ):
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(engine, pdir, chain_id="chain-cancel")
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        with Session(engine) as s:
            ej = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            ej.status = "canceled"
            s.commit()

        await tao_polling_service.handle_terminal_failure(
            PID,
            eval_id,
            chain_id="chain-cancel",
            chain_sequence=2,
            action="evaluate",
            terminal_status="canceled",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            student = s.query(StudentModel).one()
            assert student.quality_status == "failed"
            assert (
                "tao_evaluate_canceled"
                in student.nim_preflight_details["quality_failure_reason"]
            )


# ═══════════════════════════════════════════════════════════════════════════
# Failure-evidence log capture for train + quantize
# ═══════════════════════════════════════════════════════════════════════════
#
# Capturing TAO ``:logs`` only for ``evaluate`` failures is too narrow: an 8B train can
# fail within minutes with ``tao_status_raw="Error"`` while the Blueprint
# persists an empty ``error_ref`` — the operator then has to manually
# ``curl :logs`` against the external TAO job id to discover the root
# cause (e.g. an HF download stall on the TAO side). The capture therefore
# applies unconditionally to all three action types (train / evaluate /
# quantize).


class TestFailureLogCapture:
    """Train + quantize failures must persist ``:logs`` tail like evaluate does."""

    @pytest.mark.asyncio
    async def test_train_failed_captures_logs_tail(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """Train failures persist the ``:logs`` tail so operators
        don't need to ``curl :logs`` manually."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-F42-T")

        log_body = "ERROR: HuggingFace download stalled at 27.9%\nTraceback..."
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=log_body),
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "failed"
            s.commit()

        settings = make_settings(workspace)
        await tao_polling_service.handle_terminal_failure(
            PID,
            train_id,
            chain_id="chain-F42-T",
            chain_sequence=1,
            action="train",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert j.outputs is not None
            assert j.outputs.get("tao_logs_text") == log_body

    @pytest.mark.asyncio
    async def test_quantize_failed_captures_logs_tail_and_actionable_error(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """Quantize failures get the same treatment — races like the
        cosmos-rl-quantize SeaweedFS finalize race leave a diagnostic
        signature in :logs the operator should not have to fetch manually."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-F42-Q")

        # Add a quantize TAOJob as seq 3 (parents on train).
        quantize_id = generate_uuid4()
        with Session(engine) as s:
            s.add(
                TAOJob(
                    tao_job_id=quantize_id,
                    project_id=PID,
                    student_base_model_config_id="mc-1",
                    dataset_export_ids=["de-train"],
                    action="quantize",
                    status="failed",
                    training_backend="cosmos_rl_tao_vlm",
                    job_config={"quantization_method": "W8A16"},
                    tao_create_job_request={},
                    outputs={},
                    error_ref="quantize action failed for cosmos-rl",
                    tao_external_job_id="ext-qz",
                    parent_tao_job_id=train_id,
                    chain_id="chain-F42-Q",
                    chain_sequence=3,
                )
            )
            s.commit()

        log_body = (
            "Quantization failed: offset overflow while concatenating arrays, "
            "consider casting to large_list first."
        )
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=log_body),
        )

        settings = make_settings(workspace)
        await tao_polling_service.handle_terminal_failure(
            PID,
            quantize_id,
            chain_id="chain-F42-Q",
            chain_sequence=3,
            action="quantize",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=quantize_id).one()
            assert j.outputs is not None
            assert j.outputs.get("tao_logs_text") == log_body
            assert j.error_ref is not None
            assert "offset overflow while concatenating arrays" in j.error_ref
            assert "num_calibration_samples" in j.error_ref

    @pytest.mark.asyncio
    async def test_evaluate_failed_still_captures_logs_tail(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """Evaluate failures still capture :logs."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, eval_id = _seed_chain_baseline(engine, pdir, chain_id="chain-F42-E")
        settings = make_settings(workspace)
        await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        log_body = "qwen2_5_vl.py:1561 → AssertionError"
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=log_body),
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            j.status = "failed"
            s.commit()

        await tao_polling_service.handle_terminal_failure(
            PID,
            eval_id,
            chain_id="chain-F42-E",
            chain_sequence=2,
            action="evaluate",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            assert j.outputs is not None
            assert j.outputs.get("tao_logs_text") == log_body

    @pytest.mark.asyncio
    async def test_log_fetch_returning_none_does_not_inject_key(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """Best-effort capture — when :logs is unreachable / unauthenticated,
        the helper returns None and we MUST NOT inject the key with a None
        value (would mislead the failure classifier and other readers)."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-F42-N")

        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=None),
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "failed"
            s.commit()

        settings = make_settings(workspace)
        await tao_polling_service.handle_terminal_failure(
            PID,
            train_id,
            chain_id="chain-F42-N",
            chain_sequence=1,
            action="train",
            terminal_status="failed",
            engine=engine,
            settings=settings,
        )

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert j.outputs is not None
            assert "tao_logs_text" not in j.outputs

    @pytest.mark.asyncio
    async def test_log_fetch_skipped_on_canceled_status(
        self, tmp_path, patch_tao_http, monkeypatch
    ):
        """Only ``failed`` triggers the capture — ``canceled`` is an
        operator-initiated terminal and doesn't need server-side
        attribution — canceled
        evaluate jobs don't fetch logs."""
        engine, workspace, pdir = _seed_project(tmp_path)
        train_id, _ = _seed_chain_baseline(engine, pdir, chain_id="chain-F42-C")

        fetcher = AsyncMock(return_value="should not be called")
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_log_text", fetcher)

        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            j.status = "canceled"
            s.commit()

        settings = make_settings(workspace)
        await tao_polling_service.handle_terminal_failure(
            PID,
            train_id,
            chain_id="chain-F42-C",
            chain_sequence=1,
            action="train",
            terminal_status="canceled",
            engine=engine,
            settings=settings,
        )

        fetcher.assert_not_called()
        with Session(engine) as s:
            j = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert "tao_logs_text" not in (j.outputs or {})
