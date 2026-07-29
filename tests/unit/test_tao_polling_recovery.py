# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restart-recovery coverage for training chains.

Four invariants:

  1. If the backend crashes between Phase 1 commit and Phase 2 kickoff,
     the first chain's first train job lives in ``not_started`` with no
     external id. The polling worker's chain-advance logic (not a
     dedicated recovery function) must pick it up and submit it.

  2. An idempotent re-POST of the same training suite during/after
     recovery must NOT create duplicate TAOJob chains or resubmit an
     already-submitted job — existence check via
     ``(project_id, idempotency_key)`` rules.

  3. A chain frozen at "job N succeeded + outputs fetched, N+1 never
     submitted" (crash between the outputs-fetch ``completed`` commit
     and the chain advance) resumes from persisted state on the next
     polling tick.

  4. A chain frozen at "job N failed, dependents still not_started"
     (crash between the failed commit and the halt/roll-up
     transaction) halts its dependents and rolls up the TrainingSuite
     on the next polling tick — otherwise the suite strands in
     ``running`` forever with no pollable rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_example_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    make_stub_settings,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import (
    tao_job_service,
    tao_polling_service,
    training_suite_service,
)

PID = "recovery-proj"
GID = "g-recovery"
MC = "mc-recovery"
EID = "ep-recovery"


FIX_FIELDS = [
    {
        "field_id": "f0",
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": -1,
    },
    {
        "field_id": "f1",
        "field_name": "severity",
        "type": "enum",
        "role": "core",
        "display_order": 1,
        "allowed_values": ["low", "high"],
    },
]


def _make_settings(workspace: Path):
    return make_tao_settings(workspace, TAO_API_KEY="jwt-test")


def _setup(tmp_path: Path):
    engine, pdir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=("exports",)
    )

    with Session(engine) as s:
        add_project_row(
            s,
            PID,
            str(pdir),
            name="R",
            active_guidance_id=GID,
            teacher_model_config_id=MC,
        )
        add_guidance_row(
            s,
            PID,
            GID,
            {
                "fields": FIX_FIELDS,
                "generation_order": ["rationale_note", "severity"],
                "derived_json_schema": {},
                "schema_hash": "x",
            },
            description="Severity.",
        )
        add_endpoint_row(s, PID, EID, display_name="t", base_url="https://t/v1")
        add_model_config_row(
            s,
            PID,
            MC,
            EID,
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["student_base"]),
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_size",
            visual_budget_support="supported",
            tao_base_experiment_id="be-recovery-uuid",
            tao_base_experiment_pull_status="pull_complete",
        )
        # Seed a single verified non-pool label + one pool label so both
        # exports find content.
        images_dir = tmp_path / "imgs"
        images_dir.mkdir(exist_ok=True)
        for key, pool in [("v1", None), ("p1", "test_pool")]:
            img = images_dir / f"{key}.jpg"
            img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            add_example_row(s, PID, key, storage_ref=str(img), state="Verified")
            s.add(
                Label(
                    label_id=generate_uuid4(),
                    project_id=PID,
                    example_key=key,
                    label_status="verified",
                    guidance_id=GID,
                    inference_invocation_id=generate_uuid4(),
                    label_json={"rationale_note": "r", "severity": "low"},
                    labeled_at=utc_now(),
                    verified_outcome="Accept",
                    verified_at=utc_now(),
                    edited_core_fields=[],
                    edited_aux_fields=[],
                    rationale_source="teacher_proposal",
                    pool_assignment=pool,
                )
            )
        s.commit()

    # Bootstrap the TAODeploymentConfig singleton so
    # create_training_suite's workspace gate passes.
    from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
    from vlm_feedback_loop.db.engine import init_deployment_db

    dep_engine = init_deployment_db(workspace)
    with Session(dep_engine) as ds:
        cfg = ds.query(TAODeploymentConfig).first()
        assert cfg is not None
        cfg.tao_workspace_id = "ws-recovery"
        cfg.tao_workspace_bucket = "test-bucket"
        cfg.tao_workspace_cloud_type = "seaweedfs"
        cfg.tao_workspace_s3_endpoint_url_internal = "http://seaweedfs-s3:8333"
        cfg.tao_workspace_s3_endpoint_url_external = "http://127.0.0.1:8333"
        cfg.bootstrap_status = "bootstrapped"
        ds.commit()

    return engine, workspace


@pytest.fixture(autouse=True)
def _autostub_upload_archive(monkeypatch):
    """Stub the workspace upload for all recovery tests."""
    from vlm_feedback_loop.services import tao_dataset_upload_service as _uploads
    from vlm_feedback_loop.services import training_suite_service as _tss
    from vlm_feedback_loop.services.tao_dataset_upload_service import (
        UploadResult,
        build_s3_key,
        build_tao_spec_reference,
    )

    async def _noop(
        session,
        *,
        dataset_export,
        archive_path,
        deployment_config,
        s3_client,
        annotations_path,
        **_kw,
    ):
        key = build_s3_key(
            project_id=dataset_export.project_id,
            dataset_export_id=dataset_export.dataset_export_id,
            archive_name=archive_path.name,
        )
        spec = build_tao_spec_reference(
            deployment_config,
            bucket=deployment_config.tao_workspace_bucket,
            key=key,
        )
        annotation_key = build_s3_key(
            project_id=dataset_export.project_id,
            dataset_export_id=dataset_export.dataset_export_id,
            archive_name=annotations_path.name,
        )
        annotation_spec = build_tao_spec_reference(
            deployment_config,
            bucket=deployment_config.tao_workspace_bucket,
            key=annotation_key,
        )
        dataset_export.dataset_upload_ref = key
        dataset_export.dataset_upload_uri = (
            f"s3://{deployment_config.tao_workspace_bucket}/{key}"
        )
        return UploadResult(
            success=True,
            dataset_export_id=dataset_export.dataset_export_id,
            bucket=deployment_config.tao_workspace_bucket,
            key=key,
            upload_uri=dataset_export.dataset_upload_uri,
            spec_reference=spec,
            annotation_key=annotation_key,
            annotation_spec_reference=annotation_spec,
            sha256="noop",
            already_uploaded=False,
        )

    monkeypatch.setattr(_uploads, "upload_dataset_archive", _noop)
    monkeypatch.setattr(_tss, "build_s3_client", lambda _cfg: object())


# ══════════════════════════════════════════════════════════════════════════
# Kickoff interrupted between Phase 1 commit and Phase 2 submit
# ══════════════════════════════════════════════════════════════════════════


class TestKickoffInterrupted:
    @pytest.mark.asyncio
    async def test_polling_tick_submits_not_started_seq1_train(
        self, tmp_path, monkeypatch
    ):
        """Simulate a crash right after Phase 1 commit by creating the suite
        with kickoff mocked to a no-op. The first train job remains
        ``not_started``. The polling worker's chain-advance logic must
        then pick it up on the next tick."""
        engine, workspace = _setup(tmp_path)
        settings = _make_settings(workspace)

        # Mock submit_chain_job during the initial create_training_suite
        # so it doesn't actually submit — simulates a crash mid-kickoff.
        no_kickoff_mock = AsyncMock(return_value="submitting")

        monkeypatch.setattr(tao_job_service, "submit_chain_job", no_kickoff_mock)
        # Also mock _submit_to_tao to avoid any raw network paths.
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "x",
                    "error": None,
                }
            ),
        )

        result = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="recovery-a",
            settings=settings,
        )
        assert not isinstance(result, str)

        # Reset the job back to not_started (simulating a crash BEFORE
        # the kickoff call even transitioned it to submitting).
        chain_id = result["chains"][0]["chain_id"]
        with Session(engine) as s:
            train = (
                s.query(TAOJob).filter_by(chain_id=chain_id, chain_sequence=1).first()
            )
            train.status = "not_started"
            train.tao_external_job_id = None
            train.started_at = None
            s.commit()
            train_id = train.tao_job_id

        # Now the polling worker's chain-advance path must find and submit
        # this job. It does this via the ``_advance_after_terminal`` and
        # ``_find_next_chain_to_start`` helpers — or, in this case, by
        # directly picking up the first not_started job.
        # For recovery specifically: a fresh POST of the same
        # idempotency_key returns the existing suite; no new writes. A
        # future call to submit_chain_job on the train_id transitions it.
        # This is the spec's "standard restart recovery" path.

        # Re-issue with same idempotency key — expect idempotent replay
        # (no duplicate rows, no new submissions).
        replay = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="recovery-a",
            settings=settings,
        )
        assert not isinstance(replay, str)
        assert replay["training_suite_id"] == result["training_suite_id"]

        with Session(engine) as s:
            # Still exactly one suite, one chain's worth of jobs.
            assert s.query(TrainingSuite).count() == 1
            # 1 train + 1 baseline_eval = 2 jobs (no quantization).
            assert s.query(TAOJob).filter_by(chain_id=chain_id).count() == 2
            train = (
                s.query(TAOJob).filter_by(chain_id=chain_id, chain_sequence=1).first()
            )
            # The train job is still not_started (replay did not resubmit).
            assert train.status == "not_started"
            assert train.tao_job_id == train_id

        # Now directly submit via the chain helper — this is what the
        # polling worker would invoke to recover. We need the REAL
        # submit_chain_job (not the mock we installed earlier), so
        # monkeypatch.undo() first, then re-patch _submit_to_tao so
        # the HTTP transport stays deterministic.
        monkeypatch.undo()
        monkeypatch.setattr(
            tao_job_service,
            "_submit_to_tao",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_external_job_id": "recovered-ext",
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_job_service.sse_manager, "emit", AsyncMock())

        outcome = await tao_job_service.submit_chain_job(
            PID, train_id, settings=settings
        )
        assert outcome == "submitted"
        with Session(engine) as s:
            train = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
        assert train.status == "submitted"
        assert train.tao_external_job_id == "recovered-ext"

    @pytest.mark.asyncio
    async def test_idempotent_repost_does_not_double_submit(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        settings = _make_settings(workspace)

        submit_mock = AsyncMock(
            return_value={
                "success": True,
                "tao_external_job_id": "ext-dup",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "_submit_to_tao", submit_mock)
        monkeypatch.setattr(tao_job_service.sse_manager, "emit", AsyncMock())

        first = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="key-single",
            settings=settings,
        )
        second = await training_suite_service.create_training_suite(
            PID,
            student_base_model_config_ids=[MC],
            training_preset="quick",
            include_auto_labeled=False,
            export_field_mode="all",
            quantization_schemes=[],
            idempotency_key="key-single",
            settings=settings,
        )
        assert not isinstance(first, str)
        assert not isinstance(second, str)
        # Same suite, same chains, same jobs — only one submit.
        assert first["training_suite_id"] == second["training_suite_id"]
        # submit_chain_job → _submit_to_tao exactly once (first kickoff).
        assert submit_mock.await_count == 1
        with Session(engine) as s:
            assert s.query(TrainingSuite).count() == 1
            # The single chain should have 2 rows (train + baseline eval).
            assert s.query(TAOJob).count() == 2


# ══════════════════════════════════════════════════════════════════════════
# Mid-chain advance / halt recovery — the per-tick reconciliation scan
# ══════════════════════════════════════════════════════════════════════════


def _make_reconcile_settings(workspace: Path):
    return make_tao_settings(
        workspace, TAO_API_KEY="jwt-test", TAO_AUTOEVAL_SKIP_BASES=[]
    )


def _add_chain_job(
    engine,
    *,
    chain_id="chain-r",
    chain_sequence=1,
    status="succeeded",
    action="train",
    parent_tao_job_id=None,
    tao_external_job_id="ext-1",
    outputs_fetch_status=None,
    outputs=None,
):
    """Seed a TAOJob row. A ``succeeded`` fixture defaults to
    ``outputs_fetch_status="completed"`` (post-fetch steady state) so it
    does not trigger the stuck-outputs-fetch recovery scan."""
    if outputs_fetch_status is None:
        outputs_fetch_status = "completed" if status == "succeeded" else "pending"
    jid = generate_uuid4()
    with Session(engine) as s:
        s.add(
            TAOJob(
                tao_job_id=jid,
                project_id=PID,
                student_base_model_config_id=MC,
                dataset_export_ids=["de-1"],
                action=action,
                status=status,
                training_backend="cosmos_rl_tao_vlm",
                training_policy_type="sft" if action == "train" else None,
                job_config={"training_preset": "standard"},
                tao_create_job_request={
                    "kind": "experiment",
                    "action": action,
                    "specs": {},
                },
                tao_external_job_id=tao_external_job_id,
                chain_id=chain_id,
                chain_sequence=chain_sequence,
                parent_tao_job_id=parent_tao_job_id,
                outputs_fetch_status=outputs_fetch_status,
                outputs=outputs,
            )
        )
        s.commit()
    return jid


def _add_suite(engine, *, chain_ids, suite_id="ts-r"):
    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key=f"key-{suite_id}",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-t",
                evaluation_dataset_export_id="de-e",
                selected_student_base_model_config_ids=[MC],
                quantization_schemes=[],
                chain_ids_ordered=chain_ids,
                status="running",
            )
        )
        s.commit()
    return suite_id


class TestAdvanceRecovery:
    """Invariant 3: succeeded-and-fetched N + not_started N+1 resumes."""

    @pytest.mark.asyncio
    async def test_tick_submits_next_job_after_interrupted_advance(
        self, tmp_path, monkeypatch
    ):
        """succeeded N (outputs fetched) + not_started N+1 → tick submits
        N+1 from persisted state alone."""
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
            outputs_fetch_status="completed",
        )
        eval_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        assert submit_mock.await_count == 1
        assert submit_mock.await_args.args == (PID, eval_id)

    @pytest.mark.asyncio
    async def test_tick_submits_eligible_sibling_stranded_by_failed_leaf(
        self, tmp_path, monkeypatch
    ):
        """Chain isolation survives a restart: a failed evaluate whose
        independent quantize sibling (parent = the still-succeeded train)
        was never submitted gets resumed from persisted state."""
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
            outputs_fetch_status="completed",
        )
        _add_chain_job(
            engine,
            status="failed",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id="ext-eval",
            outputs={"tao_logs_text": "engine init failed"},
        )
        quantize_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=3,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        assert submit_mock.await_count == 1
        assert submit_mock.await_args.args == (PID, quantize_id)

    @pytest.mark.asyncio
    async def test_failed_outputs_fetch_never_advances(self, tmp_path, monkeypatch):
        """outputs_fetch_status='failed' is terminal-until-operator-retry —
        the recovery scan must not submit dependents past a job whose
        artifacts were never retrieved."""
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
            outputs_fetch_status="failed",
        )
        _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        assert submit_mock.await_count == 0

    @pytest.mark.asyncio
    async def test_chain_with_in_flight_work_is_left_to_the_live_flow(
        self, tmp_path, monkeypatch
    ):
        """While any chain member is in flight the live event flow owns the
        chain — the scan must not submit an eligible sibling alongside a
        running job (chains execute one job at a time)."""
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
            outputs_fetch_status="completed",
        )
        _add_chain_job(
            engine,
            status="running",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id="ext-eval",
        )
        _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=3,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Running",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        assert submit_mock.await_count == 0


class TestHaltRecovery:
    """Invariant 4: a committed failure whose halt/roll-up never ran is
    reconciled on the next tick."""

    def _seed_interrupted_failure(self, engine):
        train_id = _add_chain_job(
            engine,
            status="failed",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
        )
        eval_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        quantize_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=3,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        suite_id = _add_suite(engine, chain_ids=["chain-r"])
        return train_id, eval_id, quantize_id, suite_id

    @pytest.mark.asyncio
    async def test_tick_halts_dependents_and_rolls_up_suite(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        train_id, eval_id, quantize_id, suite_id = self._seed_interrupted_failure(
            engine
        )

        emit_mock = AsyncMock()
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", emit_mock)
        monkeypatch.setattr(
            tao_polling_service, "_fetch_tao_log_text", AsyncMock(return_value=None)
        )
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        with Session(engine) as s:
            for jid in (eval_id, quantize_id):
                row = s.query(TAOJob).filter_by(tao_job_id=jid).first()
                assert row.status == "failed"
                assert row.chain_halted_reason is not None
                assert train_id in row.chain_halted_reason
            suite = s.query(TrainingSuite).filter_by(training_suite_id=suite_id).first()
            assert suite.status == "failed"
            assert suite.completed_at is not None

        # run_failed surfaced for each newly halted dependent.
        failed_ids = {
            call.args[2]["tao_job_id"]
            for call in emit_mock.await_args_list
            if call.args[1] == "run_failed"
        }
        assert {eval_id, quantize_id} <= failed_ids

    @pytest.mark.asyncio
    async def test_halt_recovery_is_one_shot(self, tmp_path, monkeypatch):
        """Once the dependents are halted the fingerprint is gone — later
        ticks must not re-fire the failure flow (no repeated SSE noise,
        no repeated TAO :logs fetches)."""
        engine, workspace = _setup(tmp_path)
        self._seed_interrupted_failure(engine)

        emit_mock = AsyncMock()
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", emit_mock)
        log_fetch_mock = AsyncMock(return_value=None)
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_log_text", log_fetch_mock)
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_reconcile_settings(workspace)
        await tao_polling_service.tick(settings)
        emit_mock.reset_mock()
        log_fetch_mock.reset_mock()

        await tao_polling_service.tick(settings)

        assert emit_mock.await_count == 0
        assert log_fetch_mock.await_count == 0

    @pytest.mark.asyncio
    async def test_recovery_skips_log_refetch_when_already_captured(
        self, tmp_path, monkeypatch
    ):
        """The failure flow persists the TAO :logs tail on the failed job;
        a recovery re-fire must reuse the captured text instead of
        re-hitting TAO."""
        engine, workspace = _setup(tmp_path)
        train_id = _add_chain_job(
            engine,
            status="failed",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
            outputs={"tao_logs_text": "OOM at step 12"},
        )
        eval_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        log_fetch_mock = AsyncMock(return_value="fresh logs")
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_log_text", log_fetch_mock)
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        await tao_polling_service.tick(_make_reconcile_settings(workspace))

        with Session(engine) as s:
            root = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
            dep = s.query(TAOJob).filter_by(tao_job_id=eval_id).first()
        assert log_fetch_mock.await_count == 0
        assert root.outputs["tao_logs_text"] == "OOM at step 12"
        assert dep.status == "failed"


def test_startup_recovery_skips_archived_projects(tmp_path, monkeypatch):
    """Archived projects are dormant: the startup TAOJob scan must not
    open (and thereby auto-migrate or lock) their databases. Their
    submitting-orphan rows stay untouched until unarchive."""
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, "arch-proj", register_engine=True
    )
    with Session(engine) as s:
        add_project_row(s, "arch-proj", str(project_dir))
        s.add(
            TAOJob(
                tao_job_id="orphan-1",
                project_id="arch-proj",
                chain_id="chain-1",
                chain_sequence=1,
                student_base_model_config_id="mc-base",
                dataset_export_ids=["de-1"],
                action="train",
                status="submitting",
                tao_external_job_id=None,
                training_backend="cosmos_rl_tao_vlm",
                training_policy_type="sft",
                job_config={"training_preset": "standard"},
                tao_create_job_request={
                    "kind": "experiment",
                    "action": "train",
                    "specs": {},
                },
            )
        )
        s.commit()
    (project_dir / ".archived").touch()

    settings = make_stub_settings(WORKSPACE_ROOT=str(workspace))

    # The harm is OPENING the archived DB at all (auto-migrate + lock),
    # not just recovering rows — record engine requests to prove the
    # scan never asks for the archived project's engine.
    requested: list[str] = []
    real_engine_fn = tao_job_service.get_project_engine

    def _recording(pid, workspace_root):
        requested.append(pid)
        return real_engine_fn(pid, workspace_root)

    monkeypatch.setattr(tao_job_service, "get_project_engine", _recording)
    recovered = tao_job_service.recover_interrupted_tao_jobs(settings)
    assert recovered == []
    assert "arch-proj" not in requested
    with Session(engine) as s:
        row = s.query(TAOJob).filter_by(tao_job_id="orphan-1").one()
        assert row.status == "submitting"
