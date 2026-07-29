# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_polling_service.

Covers:
  * ``_should_poll`` cadence per status.
  * Status transitions persisted and ``last_polled_at`` updated.
  * ``poll_error_ref`` on transient failures without altering status.
  * ``tao_job_progress`` emitted on status or progress change.
  * Chain advancement on ``succeeded``: next job submitted.
  * Chain halt on ``failed``/``canceled``: remaining ``not_started`` rows
    marked ``failed`` + ``chain_halted_reason``.
  * Cross-chain advancement when a chain completes.
  * Artifact + logs retrieval on ``succeeded``.
  * Terminal jobs never re-polled; ``submitting`` and ``not_started`` skipped.
  * ``tao_job_completed`` on success, ``run_failed`` on failure/cancel.
  * Suite status rolled up to ``completed``/``failed``/``running``.
"""

from __future__ import annotations

import asyncio
import io
import json
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_model_config_row,
    add_project_row,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import tao_job_service, tao_polling_service

PID = "proj-poll"


# ── Setup helpers ───────────────────────────────────────────────────────────


def _make_settings(workspace: Path, **overrides):
    return make_tao_settings(
        workspace,
        TAO_API_KEY="jwt-test",
        TAO_AUTOEVAL_SKIP_BASES=[],
        **overrides,
    )


def _setup(tmp_path):
    engine, _pdir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=()
    )
    return engine, workspace


def _add_minimal_fixtures(engine, pdir):
    with Session(engine) as s:
        add_project_row(s, PID, str(pdir), name="T")
        add_endpoint_row(s, PID, "ep-1", display_name="t", base_url="https://test/v1")
        add_model_config_row(
            s,
            PID,
            "mc-1",
            "ep-1",
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["student_base"]),
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_size",
            visual_budget_support="supported",
        )
        s.commit()


def _add_chain_job(
    engine,
    *,
    chain_id="chain-a",
    chain_sequence=1,
    status="submitted",
    action="train",
    parent_tao_job_id=None,
    tao_external_job_id="ext-1",
    last_polled_at=None,
    outputs_fetch_status: str | None = None,
):
    """Seed a TAOJob row.

    ``outputs_fetch_status`` defaults mirror the migration backfill
    rule: a fixture seeded as ``status=succeeded`` is treated as
    already-completed (post-migration steady state) so it does NOT
    trigger the stuck-fetch recovery scan in ``tick()``. Recovery tests
    override this kwarg explicitly.
    """
    if outputs_fetch_status is None:
        outputs_fetch_status = "completed" if status == "succeeded" else "pending"

    jid = generate_uuid4()
    with Session(engine) as s:
        s.add(
            TAOJob(
                tao_job_id=jid,
                project_id=PID,
                student_base_model_config_id="mc-1",
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
                last_polled_at=last_polled_at,
                outputs_fetch_status=outputs_fetch_status,
            )
        )
        s.commit()
    return jid


# ══════════════════════════════════════════════════════════════════════════
# _list_tao_job_files — GET enumeration of workspace artifact keys
# ══════════════════════════════════════════════════════════════════════════


class TestListTaoJobFiles:
    """``_list_tao_job_files`` parses ``GET :list_files`` (FTMS 6.26.3)
    into ``{success, keys, error}``.

    The endpoint returns a JSON array of workspace-relative keys; these
    feed ``_select_hf_checkpoint_keys`` / ``_select_evaluate_results_key``
    in production. Without this helper the workspace-S3 retrieval path
    has nothing to enumerate.
    """

    @pytest.mark.asyncio
    async def test_parses_json_array_into_keys(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")

        captured: dict = {}

        async def fake_request(method, url, **kwargs):
            captured["method"] = method
            captured["url"] = url
            return SimpleNamespace(
                error_class=None,
                body=[
                    "results/job-1/safetensors/epoch_1/config.json",
                    "results/job-1/safetensors/epoch_1/00000.safetensors",
                    "results/job-1/status.json",
                ],
            )

        async def fake_preflight(_settings):
            return {"Authorization": "Bearer jwt-test"}, None

        monkeypatch.setattr(tao_polling_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_polling_service, "tao_preflight", fake_preflight)

        result = await tao_polling_service._list_tao_job_files(
            "ext-job-1", settings=settings
        )
        assert result["success"] is True
        assert result["error"] is None
        assert result["keys"] == [
            "results/job-1/safetensors/epoch_1/config.json",
            "results/job-1/safetensors/epoch_1/00000.safetensors",
            "results/job-1/status.json",
        ]
        # Endpoint shape: GET on ``:list_files`` (POST returns 405 on
        # this FTMS version; fixed at the endpoint layer).
        assert captured["method"] == "GET"
        assert captured["url"].endswith("/orgs/example-org/jobs/ext-job-1:list_files")

    @pytest.mark.asyncio
    async def test_http_failure_returns_success_false_with_error(
        self, tmp_path, monkeypatch
    ):
        from types import SimpleNamespace

        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")

        async def fake_request(method, url, **kwargs):
            return SimpleNamespace(
                error_class="http_error",
                error_detail="HTTP 500",
                body=None,
            )

        async def fake_preflight(_settings):
            return {"Authorization": "Bearer jwt-test"}, None

        monkeypatch.setattr(tao_polling_service, "resilient_request", fake_request)
        monkeypatch.setattr(tao_polling_service, "tao_preflight", fake_preflight)

        result = await tao_polling_service._list_tao_job_files(
            "ext-job-2", settings=settings
        )
        assert result["success"] is False
        assert result["keys"] == []
        assert "list_files failed" in result["error"]


# ══════════════════════════════════════════════════════════════════════════
# Evaluate prediction materialization
# ══════════════════════════════════════════════════════════════════════════


class TestEvaluatePredictionMaterialization:
    @pytest.mark.parametrize(
        "answer",
        ["", [], 7, None],
        ids=("empty-string", "list", "number", "null"),
    )
    def test_preserves_present_answer_as_schema_invalid_evidence(
        self,
        tmp_path,
        answer,
    ):
        """A present model answer is evidence even when schema-invalid."""
        archive = tmp_path / "evaluate_results.tar.gz"
        body = json.dumps([{"video_id": "images/ex-01.jpg", "answer": answer}]).encode()
        with tarfile.open(archive, "w:gz") as tf:
            info = tarfile.TarInfo("epoch_1/freeform/eval_set/images/ex-01.json")
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))

        cache = tmp_path / "cache"
        result = tao_polling_service._materialize_evaluate_predictions(
            archive,
            cache,
        )

        assert result == {"success": True, "samples": 1, "error": None}
        assert json.loads((cache / "per_sample_predictions").read_text()) == [
            {
                "id": "ex-01",
                "prediction": answer,
                "video_id": "images/ex-01.jpg",
                "datasource": None,
                "correct_answer": None,
                "reasoning": None,
                "full_response": None,
            }
        ]


# ══════════════════════════════════════════════════════════════════════════
# _should_poll cadence
# ══════════════════════════════════════════════════════════════════════════


class TestShouldPoll:
    def test_submitting_is_never_polled(self):
        now = datetime.now(UTC)
        assert not tao_polling_service._should_poll(
            "submitting",
            None,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )

    def test_not_started_is_never_polled(self):
        now = datetime.now(UTC)
        assert not tao_polling_service._should_poll(
            "not_started",
            None,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )

    @pytest.mark.parametrize("status", ["succeeded", "failed", "canceled", "deleted"])
    def test_terminal_statuses_never_polled(self, status):
        now = datetime.now(UTC)
        assert not tao_polling_service._should_poll(
            status,
            None,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )

    @pytest.mark.parametrize("status", ["submitted", "queued"])
    def test_submitted_queued_cadence_30s(self, status):
        now = datetime.now(UTC)
        # Never polled → eligible immediately.
        assert tao_polling_service._should_poll(
            status,
            None,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )
        # Recent poll (10s ago) → skip.
        ten_ago = (now - timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not tao_polling_service._should_poll(
            status,
            ten_ago,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )
        # Old poll (35s ago) → eligible again.
        old = (now - timedelta(seconds=35)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tao_polling_service._should_poll(
            status,
            old,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )

    @pytest.mark.parametrize("status", ["running", "paused"])
    def test_running_paused_cadence_60s(self, status):
        now = datetime.now(UTC)
        # 45s ago — below running threshold → skip.
        recent = (now - timedelta(seconds=45)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert not tao_polling_service._should_poll(
            status,
            recent,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )
        # 70s ago — above threshold → eligible.
        old = (now - timedelta(seconds=70)).strftime("%Y-%m-%dT%H:%M:%SZ")
        assert tao_polling_service._should_poll(
            status,
            old,
            now=now,
            submitted_interval_s=30,
            running_interval_s=60,
        )


# ══════════════════════════════════════════════════════════════════════════
# _poll_single_job: status transitions
# ══════════════════════════════════════════════════════════════════════════


class TestPollSingleJob:
    @pytest.mark.asyncio
    async def test_submitted_to_running_persisted(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        job_id = _add_chain_job(engine, status="submitted")

        mock_poll = AsyncMock(
            return_value={
                "success": True,
                "tao_status_raw": "Running",
                "progress": {"epoch_current": 1, "epoch_total": 3},
                "outputs": None,
                "error": None,
            }
        )
        monkeypatch.setattr(tao_job_service, "poll_tao_job", mock_poll)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, job_id, engine=engine, settings=settings
        )
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=job_id).first()
        assert job.status == "running"
        assert job.progress == {"epoch_current": 1, "epoch_total": 3}
        assert job.started_at is not None
        assert job.last_polled_at is not None

    @pytest.mark.asyncio
    async def test_poll_error_updates_poll_error_ref_only(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        job_id = _add_chain_job(engine, status="running")

        mock_poll = AsyncMock(
            return_value={
                "success": False,
                "tao_status_raw": None,
                "progress": None,
                "outputs": None,
                "error": "transient 503",
            }
        )
        monkeypatch.setattr(tao_job_service, "poll_tao_job", mock_poll)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, job_id, engine=engine, settings=settings
        )
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=job_id).first()
        assert job.status == "running"  # unchanged
        assert job.poll_error_ref == "transient 503"
        assert job.last_polled_at is not None

    @pytest.mark.asyncio
    async def test_emits_tao_job_progress_on_status_change(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        job_id = _add_chain_job(engine, status="submitted")

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
        emitted: list[tuple] = []

        async def fake_emit(project_id, event_type, data):
            emitted.append((project_id, event_type, data))

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", fake_emit)
        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, job_id, engine=engine, settings=settings
        )
        assert any(evt == "tao_job_progress" for _, evt, _ in emitted)


# ══════════════════════════════════════════════════════════════════════════
# Terminal success: artifact fetch + chain advancement
# ══════════════════════════════════════════════════════════════════════════


class TestTerminalSuccess:
    @pytest.mark.asyncio
    async def test_success_fetches_artifacts_emits_completed_and_advances_chain(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        # Two jobs in the same chain — train (running) and evaluate (not_started).
        train_id = _add_chain_job(
            engine,
            status="running",
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

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": {"epoch_current": 3, "epoch_total": 3},
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        artifact_mock = AsyncMock(
            return_value={
                "success": True,
                "artifacts": [
                    {"name": "best_model", "tao_file_path": "/w/best"},
                    {"name": "latest_model", "tao_file_path": "/w/latest"},
                ],
                "error": None,
            }
        )
        logs_mock = AsyncMock(
            return_value={
                "success": True,
                "logs_ref": "https://tao/logs/ext-train",
                "error": None,
            }
        )
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", artifact_mock)
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_logs", logs_mock)

        chain_submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", chain_submit_mock)

        emitted: list[tuple] = []

        async def fake_emit(project_id, event_type, data):
            emitted.append((project_id, event_type, data))

        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", fake_emit)

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, train_id, engine=engine, settings=settings
        )
        # Post-success flow is dispatched as a background
        # task so the polling tick doesn't head-of-line-block on multi-GB
        # downloads. Tests await it explicitly to assert the steady state.
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            train = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
        assert train.status == "succeeded"
        assert train.completed_at is not None
        assert train.outputs["artifacts"][0]["name"] == "best_model"
        assert train.outputs["logs_ref"] == "https://tao/logs/ext-train"

        # Chain advanced: submit_chain_job called for the next sequence.
        assert chain_submit_mock.await_count == 1
        _, kwargs = chain_submit_mock.call_args
        assert chain_submit_mock.await_args.args[1] == eval_id

        # SSE: tao_job_completed and tao_job_progress both emitted.
        events = {evt for _, evt, _ in emitted}
        assert "tao_job_progress" in events
        assert "tao_job_completed" in events


# ══════════════════════════════════════════════════════════════════════════
# Terminal failure: chain halt
# ══════════════════════════════════════════════════════════════════════════


class TestTerminalFailure:
    @pytest.mark.asyncio
    async def test_failure_halts_remaining_not_started_jobs(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        train_id = _add_chain_job(
            engine,
            status="running",
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
        quant_id = _add_chain_job(
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
                    "tao_status_raw": "Failed",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        chain_submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", chain_submit_mock)

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, train_id, engine=engine, settings=settings
        )

        with Session(engine) as s:
            train = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
            ev = s.query(TAOJob).filter_by(tao_job_id=eval_id).first()
            qu = s.query(TAOJob).filter_by(tao_job_id=quant_id).first()
        assert train.status == "failed"
        assert ev.status == "failed"
        assert ev.chain_halted_reason is not None
        assert qu.status == "failed"
        assert qu.chain_halted_reason is not None
        # No next job submitted within the same chain.
        chain_submit_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_chain_isolation_failed_evaluate_does_not_halt_independent_quantize(
        self, tmp_path, monkeypatch
    ):
        """Chain isolation.

        When ``evaluate baseline`` fails (parented on the still-succeeded
        ``train``), independent siblings — ``quantize FP8`` and
        ``quantize W8A16``, both parented on ``train`` — must remain
        ``not_started`` (not halted), and the next eligible quantize
        must be submitted automatically. Their downstream evaluates
        (parented on each quantize) also stay ``not_started``.

        This is the unblock that lets the Blueprint produce the full
        quantization variant set even when cosmos-rl's evaluator hits
        the upstream Qwen3-VL-dense loader gap.
        """
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        # 6-job chain matching the production layout.
        train_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
        )
        eval_baseline_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="running",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id="ext-eval-baseline",
        )
        quant_fp8_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="not_started",
            chain_sequence=3,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        eval_fp8_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="not_started",
            chain_sequence=4,
            action="evaluate",
            parent_tao_job_id=quant_fp8_id,
            tao_external_job_id=None,
        )
        quant_w8a16_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="not_started",
            chain_sequence=5,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        eval_w8a16_id = _add_chain_job(
            engine,
            chain_id="chain-iso",
            status="not_started",
            chain_sequence=6,
            action="evaluate",
            parent_tao_job_id=quant_w8a16_id,
            tao_external_job_id=None,
        )

        # eval baseline transitions to failed.
        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Failed",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        # _fetch_tao_log_text is best-effort; stub it out (not the focus).
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=None),
        )
        chain_submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", chain_submit_mock)

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, eval_baseline_id, engine=engine, settings=settings
        )

        with Session(engine) as s:
            eval_baseline = (
                s.query(TAOJob).filter_by(tao_job_id=eval_baseline_id).first()
            )
            quant_fp8 = s.query(TAOJob).filter_by(tao_job_id=quant_fp8_id).first()
            eval_fp8 = s.query(TAOJob).filter_by(tao_job_id=eval_fp8_id).first()
            quant_w8a16 = s.query(TAOJob).filter_by(tao_job_id=quant_w8a16_id).first()
            eval_w8a16 = s.query(TAOJob).filter_by(tao_job_id=eval_w8a16_id).first()

        # The failed eval is terminal; nothing else changed yet.
        assert eval_baseline.status == "failed"
        # Independent quantize jobs MUST NOT be halted.
        assert quant_fp8.status == "not_started", (
            f"quantize FP8 should not be halted; got {quant_fp8.status} "
            f"(reason={quant_fp8.chain_halted_reason})"
        )
        assert quant_fp8.chain_halted_reason is None
        assert quant_w8a16.status == "not_started"
        assert quant_w8a16.chain_halted_reason is None
        # Their dependent evaluates also stay not_started.
        assert eval_fp8.status == "not_started"
        assert eval_w8a16.status == "not_started"

        # Next eligible job (lowest chain_sequence with parent=succeeded)
        # is the FP8 quantize → submitted.
        assert chain_submit_mock.await_count == 1
        assert chain_submit_mock.await_args.args[1] == quant_fp8_id

    @pytest.mark.asyncio
    async def test_failed_quantize_halts_only_its_evaluate_not_other_quantize(
        self, tmp_path, monkeypatch
    ):
        """Chain isolation: when ``quantize FP8`` fails, only
        its dependent ``eval FP8`` halts. ``quantize W8A16`` (parented on
        the still-succeeded train) and its dependent ``eval W8A16`` stay
        ``not_started``.
        """
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        train_id = _add_chain_job(
            engine,
            chain_id="chain-iso-q",
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
        )
        quant_fp8_id = _add_chain_job(
            engine,
            chain_id="chain-iso-q",
            status="running",
            chain_sequence=3,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id="ext-quant-fp8",
        )
        eval_fp8_id = _add_chain_job(
            engine,
            chain_id="chain-iso-q",
            status="not_started",
            chain_sequence=4,
            action="evaluate",
            parent_tao_job_id=quant_fp8_id,
            tao_external_job_id=None,
        )
        quant_w8a16_id = _add_chain_job(
            engine,
            chain_id="chain-iso-q",
            status="not_started",
            chain_sequence=5,
            action="quantize",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        eval_w8a16_id = _add_chain_job(
            engine,
            chain_id="chain-iso-q",
            status="not_started",
            chain_sequence=6,
            action="evaluate",
            parent_tao_job_id=quant_w8a16_id,
            tao_external_job_id=None,
        )

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Failed",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_log_text",
            AsyncMock(return_value=None),
        )
        chain_submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", chain_submit_mock)

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, quant_fp8_id, engine=engine, settings=settings
        )

        with Session(engine) as s:
            quant_fp8 = s.query(TAOJob).filter_by(tao_job_id=quant_fp8_id).first()
            eval_fp8 = s.query(TAOJob).filter_by(tao_job_id=eval_fp8_id).first()
            quant_w8a16 = s.query(TAOJob).filter_by(tao_job_id=quant_w8a16_id).first()
            eval_w8a16 = s.query(TAOJob).filter_by(tao_job_id=eval_w8a16_id).first()

        assert quant_fp8.status == "failed"
        # Direct dependent: halted.
        assert eval_fp8.status == "failed"
        assert eval_fp8.chain_halted_reason is not None
        # Independent sibling: NOT halted, remains eligible.
        assert quant_w8a16.status == "not_started"
        assert quant_w8a16.chain_halted_reason is None
        assert eval_w8a16.status == "not_started"
        # Next eligible quantize submitted.
        assert chain_submit_mock.await_count == 1
        assert chain_submit_mock.await_args.args[1] == quant_w8a16_id


# ══════════════════════════════════════════════════════════════════════════
# Suite status roll-up
# ══════════════════════════════════════════════════════════════════════════


class TestSuiteRollUp:
    @pytest.mark.asyncio
    async def test_late_success_cannot_reactivate_or_advance_canceled_suite(
        self, tmp_path, monkeypatch
    ):
        """A poll already in flight when suite cancellation wins must stop there."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        train_id = _add_chain_job(
            engine,
            chain_id="chain-canceled",
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
        )
        next_id = _add_chain_job(
            engine,
            chain_id="chain-canceled",
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )
        with Session(engine) as s:
            s.add(
                TrainingSuite(
                    training_suite_id="ts-canceled",
                    project_id=PID,
                    idempotency_key="roll-canceled",
                    guidance_id="g-1",
                    training_preset="standard",
                    export_field_mode="all",
                    include_auto_labeled=False,
                    training_dataset_export_id="de-t",
                    evaluation_dataset_export_id="de-e",
                    selected_student_base_model_config_ids=["mc-1"],
                    quantization_schemes=[],
                    chain_ids_ordered=["chain-canceled"],
                    status="canceled",
                    completed_at=datetime.now(UTC),
                )
            )
            s.commit()

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        await tao_polling_service._poll_single_job(
            PID,
            train_id,
            engine=engine,
            settings=_make_settings(workspace),
        )
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            suite = s.get(TrainingSuite, "ts-canceled")
            next_job = s.get(TAOJob, next_id)
            assert suite is not None
            assert next_job is not None
            assert suite.status == "canceled"
            assert next_job.status == "not_started"
        submit_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_suite_status_becomes_completed_when_all_succeeded(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        # Create a suite with one chain of two jobs.
        train_id = _add_chain_job(
            engine,
            chain_id="chain-X",
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train",
        )
        eval_id = _add_chain_job(
            engine,
            chain_id="chain-X",
            status="running",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id="ext-eval",
        )

        with Session(engine) as s:
            s.add(
                TrainingSuite(
                    training_suite_id="ts-1",
                    project_id=PID,
                    idempotency_key="roll-1",
                    guidance_id="g-1",
                    training_preset="standard",
                    export_field_mode="all",
                    include_auto_labeled=False,
                    training_dataset_export_id="de-t",
                    evaluation_dataset_export_id="de-e",
                    selected_student_base_model_config_ids=["mc-1"],
                    quantization_schemes=[],
                    chain_ids_ordered=["chain-X"],
                    status="running",
                )
            )
            s.commit()

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, eval_id, engine=engine, settings=settings
        )
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            suite = s.query(TrainingSuite).filter_by(training_suite_id="ts-1").first()
        assert suite.status == "completed"
        assert suite.completed_at is not None

    @pytest.mark.asyncio
    async def test_suite_failed_when_any_terminal_job_failed(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        train_id = _add_chain_job(
            engine,
            chain_id="chain-Y",
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-t",
        )
        with Session(engine) as s:
            s.add(
                TrainingSuite(
                    training_suite_id="ts-2",
                    project_id=PID,
                    idempotency_key="roll-2",
                    guidance_id="g-1",
                    training_preset="standard",
                    export_field_mode="all",
                    include_auto_labeled=False,
                    training_dataset_export_id="de-t",
                    evaluation_dataset_export_id="de-e",
                    selected_student_base_model_config_ids=["mc-1"],
                    quantization_schemes=[],
                    chain_ids_ordered=["chain-Y"],
                    status="running",
                )
            )
            s.commit()

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Failed",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, train_id, engine=engine, settings=settings
        )
        with Session(engine) as s:
            suite = s.query(TrainingSuite).filter_by(training_suite_id="ts-2").first()
        assert suite.status == "failed"
        assert suite.completed_at is not None


# ══════════════════════════════════════════════════════════════════════════
# Cross-chain advancement
# ══════════════════════════════════════════════════════════════════════════


class TestCrossChainAdvance:
    @pytest.mark.asyncio
    async def test_next_chain_started_when_current_exhausted(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)

        # Chain A complete (single job — no next sequence).
        a_train = _add_chain_job(
            engine,
            chain_id="chain-A",
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-A",
        )
        # Chain B not yet started.
        b_train = _add_chain_job(
            engine,
            chain_id="chain-B",
            status="not_started",
            chain_sequence=1,
            action="train",
            tao_external_job_id=None,
        )

        with Session(engine) as s:
            s.add(
                TrainingSuite(
                    training_suite_id="ts-cross",
                    project_id=PID,
                    idempotency_key="cross-1",
                    guidance_id="g-1",
                    training_preset="standard",
                    export_field_mode="all",
                    include_auto_labeled=False,
                    training_dataset_export_id="de-t",
                    evaluation_dataset_export_id="de-e",
                    selected_student_base_model_config_ids=["mc-1"],
                    quantization_schemes=[],
                    chain_ids_ordered=["chain-A", "chain-B"],
                    status="running",
                )
            )
            s.commit()

        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", submit_mock)

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, a_train, engine=engine, settings=settings
        )
        await tao_polling_service._await_pending_post_success_tasks()

        # Chain B's first job submitted via cross-chain advancement.
        submit_mock.assert_awaited()
        # At least one of the calls passed b_train.
        called_ids = [args.args[1] for args in submit_mock.await_args_list]
        assert b_train in called_ids


# ══════════════════════════════════════════════════════════════════════════
# tick: integrated path from worker perspective
# ══════════════════════════════════════════════════════════════════════════


class TestTick:
    @pytest.mark.asyncio
    async def test_tick_polls_only_eligible_jobs(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        # One fresh submitted (eligible), one terminal (ignored), one
        # not_started (ignored).
        eligible = _add_chain_job(engine, status="submitted", chain_sequence=1)
        _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=2,
            action="evaluate",
            tao_external_job_id="x",
        )
        _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=3,
            action="quantize",
            tao_external_job_id=None,
        )

        called_ids: list[str] = []

        async def fake_poll(external_id, *, settings):  # noqa: ARG001
            called_ids.append(external_id)
            return {
                "success": True,
                "tao_status_raw": "Running",
                "progress": None,
                "outputs": None,
                "error": None,
            }

        monkeypatch.setattr(tao_job_service, "poll_tao_job", fake_poll)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)

        # Only the eligible submitted job was polled.
        assert called_ids == ["ext-1"]
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=eligible).first()
        assert job.status == "running"


# ══════════════════════════════════════════════════════════════════════════
# TAOJob.outputs_fetch_status lifecycle + restart recovery
# ══════════════════════════════════════════════════════════════════════════
#
# The chain-halting scenario: when the backend crashes
# mid-multi-GB-artifact-download, the next tick observes
# ``status="succeeded"`` and skips ``_handle_succeeded`` entirely
# (``status_changed=False``). The ``outputs_fetch_status`` lifecycle
# + the polling tick recovery scan close the gap.


class TestOutputsFetchLifecycle:
    """``_handle_succeeded`` flips ``outputs_fetch_status`` pending →
    in_progress → completed (or failed) so the recovery scan can
    distinguish "post-success flow finished" from "interrupted, retry me"."""

    @pytest.mark.asyncio
    async def test_happy_path_marks_completed(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        train_id = _add_chain_job(
            engine,
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train-happy",
        )
        # Drive the success transition via the same wire path production uses.
        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )
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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        await tao_polling_service._poll_single_job(
            PID, train_id, engine=engine, settings=settings
        )
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
        assert row.status == "succeeded"
        assert row.outputs_fetch_status == "completed"
        assert row.outputs_fetch_error_ref is None

    @pytest.mark.asyncio
    async def test_artifact_fetch_exception_marks_failed(self, tmp_path, monkeypatch):
        """When ``_handle_succeeded_body`` raises (e.g., S3 GET timeout
        propagates instead of returning ``success=False``), the marker
        flips to ``failed`` and the exception text is captured for the
        operator. The polling loop continues — error is logged, not
        re-raised, so the next per-job poll is unaffected."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        train_id = _add_chain_job(
            engine,
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train-fail",
        )
        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )

        async def _boom(*args, **kwargs):
            raise RuntimeError("simulated S3 outage during artifact GET")

        monkeypatch.setattr(tao_polling_service, "_handle_succeeded_body", _boom)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        # The poll itself MUST NOT raise — the error is contained.
        await tao_polling_service._poll_single_job(
            PID, train_id, engine=engine, settings=settings
        )
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
        assert row.status == "succeeded"  # status flipped before _handle_succeeded
        assert row.outputs_fetch_status == "failed"
        assert row.outputs_fetch_error_ref is not None
        assert "S3 outage" in row.outputs_fetch_error_ref


class TestOutputsFetchRecovery:
    """The recovery scan in ``tick()`` re-fires ``_handle_succeeded`` for
    any TAOJob in ``succeeded + outputs_fetch_status IN (pending,
    in_progress)`` — the fingerprint of a backend crash mid-fetch."""

    @pytest.mark.asyncio
    async def test_resumes_in_progress_artifact_fetch(self, tmp_path, monkeypatch):
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        # Simulate a crash mid-fetch: status is succeeded, but outputs were
        # never persisted (artifact_cache_dir absent) and the lifecycle
        # marker is still in_progress.
        stuck_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-stuck-train",
            outputs_fetch_status="in_progress",
        )

        # Recovery should re-fire _handle_succeeded. We let the real
        # function run with mocked artifact + log fetches.
        artifacts_mock = AsyncMock(
            return_value={
                "success": True,
                "artifacts": [{"name": "best_model", "tao_file_path": "/w/x"}],
                "error": None,
            }
        )
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", artifacts_mock)
        monkeypatch.setattr(
            tao_polling_service,
            "_fetch_tao_logs",
            AsyncMock(return_value={"success": True, "logs_ref": "ref", "error": None}),
        )
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)
        await tao_polling_service._await_pending_post_success_tasks()

        # Recovery refired _handle_succeeded for the stuck job; outputs are now
        # populated and the lifecycle marker is completed.
        assert artifacts_mock.await_count == 1
        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=stuck_id).first()
        assert row.outputs_fetch_status == "completed"
        assert row.outputs is not None
        assert row.outputs.get("artifact_cache_dir") is not None

    @pytest.mark.asyncio
    async def test_resumes_pending_artifact_fetch(self, tmp_path, monkeypatch):
        """Same recovery applies when the marker never advanced past the
        default ``pending`` (e.g., the crash happened before
        ``_handle_succeeded``'s first short-txn write committed)."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        stuck_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="quantize",
            tao_external_job_id="ext-stuck-quant",
            outputs_fetch_status="pending",
        )

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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)
        await tao_polling_service._await_pending_post_success_tasks()

        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=stuck_id).first()
        assert row.outputs_fetch_status == "completed"

    @pytest.mark.asyncio
    async def test_skips_already_completed(self, tmp_path, monkeypatch):
        """Steady state: post-migration succeeded rows have
        ``outputs_fetch_status="completed"`` and MUST NOT be re-fired —
        otherwise every polling tick would re-download multi-GB artifacts."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-done",
            outputs_fetch_status="completed",
        )

        artifacts_mock = AsyncMock()
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", artifacts_mock)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)

        # No artifact re-download.
        assert artifacts_mock.await_count == 0

    @pytest.mark.asyncio
    async def test_skips_failed(self, tmp_path, monkeypatch):
        """``failed`` is terminal for outputs-fetch — the operator must
        manually flip back to ``pending`` to retry. Auto-retry on failed
        would mask real outages (S3 ACL bug, network partition, etc.)."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-failed-fetch",
            outputs_fetch_status="failed",
        )

        artifacts_mock = AsyncMock()
        monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", artifacts_mock)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)

        assert artifacts_mock.await_count == 0

    @pytest.mark.asyncio
    async def test_recovery_advances_chain(self, tmp_path, monkeypatch):
        """After re-firing ``_handle_succeeded`` for a stuck train job,
        the recovery ALSO re-fires chain advancement so the next
        ``not_started`` job (e.g., evaluate/quantize) gets submitted —
        exactly the crash-mid-download failure scenario."""
        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-stuck-advance",
            outputs_fetch_status="in_progress",
        )
        eval_id = _add_chain_job(
            engine,
            status="not_started",
            chain_sequence=2,
            action="evaluate",
            parent_tao_job_id=train_id,
            tao_external_job_id=None,
        )

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
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        chain_submit_mock = AsyncMock(return_value="submitted")
        monkeypatch.setattr(tao_job_service, "submit_chain_job", chain_submit_mock)

        settings = _make_settings(workspace)
        await tao_polling_service.tick(settings)
        await tao_polling_service._await_pending_post_success_tasks()

        # Recovery resumed the train and triggered chain advance for
        # the eval — closing the crash-mid-download gap.
        with Session(engine) as s:
            row = s.query(TAOJob).filter_by(tao_job_id=train_id).first()
        assert row.outputs_fetch_status == "completed"
        # submit_chain_job called for the next chain_sequence.
        assert chain_submit_mock.await_count == 1
        assert chain_submit_mock.await_args.args[1] == eval_id


# ══════════════════════════════════════════════════════════════════════════
# Async post-success dispatch (no head-of-line blocking on the polling
# tick when a multi-GB safetensors download is in flight)
# ══════════════════════════════════════════════════════════════════════════


class TestPostSuccessAsyncDispatch:
    """A synchronous ``await _handle_succeeded`` in the polling tick
    would block every other project's polling for the duration of a
    multi-GB artifact download. ``_post_success_flow`` is therefore
    dispatched as a non-blocking background task via
    ``background_manager.register``, with dedup by ``tao_job_id``."""

    @pytest.mark.asyncio
    async def test_tick_returns_before_artifact_fetch_finishes(
        self, tmp_path, monkeypatch
    ):
        """The polling tick MUST return immediately even
        when ``_handle_succeeded_body`` is slow (e.g., 17 GB safetensors
        download). Dispatch routes through ``background_manager.register``
        so the tick continues without awaiting the multi-GB download."""
        from vlm_feedback_loop.services.background import background_manager

        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        train_id = _add_chain_job(
            engine,
            status="running",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train-async",
        )
        monkeypatch.setattr(
            tao_job_service,
            "poll_tao_job",
            AsyncMock(
                return_value={
                    "success": True,
                    "tao_status_raw": "Done",
                    "progress": None,
                    "outputs": None,
                    "error": None,
                }
            ),
        )

        # _handle_succeeded_body sleeps "forever" — if the tick awaits it,
        # the test deadlocks. The service dispatches it as a background
        # task so the tick returns immediately.
        slow_started = asyncio.Event()
        slow_can_finish = asyncio.Event()

        async def _slow_body(*args, **kwargs):
            slow_started.set()
            await slow_can_finish.wait()

        monkeypatch.setattr(tao_polling_service, "_handle_succeeded_body", _slow_body)
        monkeypatch.setattr(tao_polling_service.sse_manager, "emit", AsyncMock())
        monkeypatch.setattr(
            tao_job_service, "submit_chain_job", AsyncMock(return_value="submitted")
        )

        settings = _make_settings(workspace)
        # ``tick`` MUST NOT block on ``_slow_body``. asyncio.wait_for
        # bounds the test — if the background dispatch regresses, this times out at the inner
        # ``_handle_succeeded_body.wait()``.
        await asyncio.wait_for(
            tao_polling_service._poll_single_job(
                PID, train_id, engine=engine, settings=settings
            ),
            timeout=3.0,
        )

        # Background task was registered for the post-success flow.
        task_id = tao_polling_service._post_success_task_id(train_id)
        assert task_id in background_manager.active_task_ids

        # Wait until the slow body actually started (avoids racing with the
        # event-loop scheduler), then release it and let the test cleanup.
        await slow_started.wait()
        slow_can_finish.set()
        await tao_polling_service._await_pending_post_success_tasks()

    @pytest.mark.asyncio
    async def test_dispatch_is_deduped_by_tao_job_id(self, tmp_path, monkeypatch):
        """The recovery scan and the regular tick can both observe the
        same ``succeeded`` job in a single tick. Dispatch MUST dedup so
        the same multi-GB download isn't kicked off twice."""
        from vlm_feedback_loop.services.background import background_manager

        engine, workspace = _setup(tmp_path)
        _add_minimal_fixtures(engine, workspace / "projects" / PID)
        train_id = _add_chain_job(
            engine,
            status="succeeded",
            chain_sequence=1,
            action="train",
            tao_external_job_id="ext-train-dedup",
            outputs_fetch_status="in_progress",
        )

        # Gate the artifact-fetch body open so the dedup window is held
        # deterministically — without the gate, dedup would only hold
        # because the real body (HTTP fetch + retries) happens to be slow.
        body_started = asyncio.Event()
        body_can_finish = asyncio.Event()

        async def _gated_body(*args, **kwargs):
            body_started.set()
            await body_can_finish.wait()

        monkeypatch.setattr(tao_polling_service, "_handle_succeeded_body", _gated_body)

        # First call to dispatch → registers a background task.
        first = tao_polling_service._dispatch_post_success_flow(
            PID,
            train_id,
            external_id="ext-train-dedup",
            action="train",
            chain_id="chain-a",
            chain_sequence=1,
            engine=engine,
            settings=_make_settings(workspace),
            origin="tick",
        )
        assert first is True
        # The first flow is definitely in flight...
        await asyncio.wait_for(body_started.wait(), timeout=3.0)

        # ...so the second call (e.g., from the recovery scan in the same
        # tick) is skipped by dedup while the task is still in_progress.
        second = tao_polling_service._dispatch_post_success_flow(
            PID,
            train_id,
            external_id="ext-train-dedup",
            action="train",
            chain_id="chain-a",
            chain_sequence=1,
            engine=engine,
            settings=_make_settings(workspace),
            origin="recovery",
        )
        assert second is False

        # Release the body and drain the in-flight task before cleanup.
        body_can_finish.set()
        await tao_polling_service._await_pending_post_success_tasks()
        # After the task drains, dispatch is allowed again (e.g., for a
        # different recovery cycle, though in practice the marker would
        # be ``completed`` and the recovery scan would skip).
        task_id = tao_polling_service._post_success_task_id(train_id)
        assert task_id not in background_manager.active_task_ids


# ══════════════════════════════════════════════════════════════════════════
# TAO ``Done``-before-upload race retry
# ══════════════════════════════════════════════════════════════════════════


class TestEmptyListFilesRetry:
    """TAO FTMS 6.26.3 can mark a cosmos-rl job ``Done`` before
    SeaweedFS finalizes the upload of that job's safetensors. The first
    ``:list_files`` after status-flip can return an empty listing even
    though the upload finalizes seconds later.

    ``_fetch_tao_artifacts`` MUST retry the listing with backoff when
    the workspace listing is genuinely empty AND the caller requested
    real bytes (``local_cache_dir`` is set). Metadata-only callers
    (``local_cache_dir is None``) MUST NOT retry — they explicitly want
    a snapshot.

    The retry trigger is restricted to GENUINELY-EMPTY
    listings only. A non-empty listing without checkpoint shards
    (e.g., ``status.json``, ``microservices_log.txt`` only) means
    cosmos-rl finished and emitted only completion metadata — terminal,
    fail fast. Per-object SeaweedFS finalization means once any key
    appears, retrying will not conjure shards that were never produced.
    See ``test_no_checkpoint_dir_in_listing_returns_failure`` in
    ``test_tao_artifact_download.py`` for the fail-fast contract.
    """

    @pytest.mark.asyncio
    async def test_quantize_retries_until_artifacts_appear(self, tmp_path, monkeypatch):
        """A quantize fetch where the workspace listing is empty on the
        first two attempts but populated on the third MUST succeed —
        the retry-with-backoff masks the TAO-Done race window where
        SeaweedFS has not yet finalized any object for the job.
        """
        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")
        # Workspace bucket lives in deployment.db, not Settings. Seed the
        # singleton so _fetch_tao_artifacts's bucket read finds it.
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
        from vlm_feedback_loop.db.engine import init_deployment_db

        engine = init_deployment_db(tmp_path)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            cfg.tao_workspace_bucket = "b"
            session.commit()

        attempts = {"n": 0}

        async def fake_list(_external_id, *, settings):
            attempts["n"] += 1
            if attempts["n"] < 3:
                # Race window: SeaweedFS has not yet finalized any
                # object for the job — listing is genuinely empty.
                return {
                    "success": True,
                    "keys": [],
                    "error": None,
                }
            return {
                "success": True,
                "keys": [
                    "results/job-1/safetensors/epoch_1/config.json",
                    "results/job-1/safetensors/epoch_1/00000.safetensors",
                    "results/job-1/safetensors/epoch_1/tokenizer.json",
                ],
                "error": None,
            }

        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        async def fake_download(*_args, local_path=None, **_kwargs):
            return {
                "success": True,
                "bytes_written": 1,
                "local_path": str(local_path) if local_path else "",
                "error": None,
            }

        # Stub S3 client builder so we don't touch live boto3.
        monkeypatch.setattr(tao_polling_service, "_list_tao_job_files", fake_list)
        monkeypatch.setattr(tao_polling_service.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(
            tao_polling_service, "_build_workspace_s3_client", lambda _s: object()
        )
        monkeypatch.setattr(
            tao_polling_service, "_download_workspace_s3_object", fake_download
        )

        result = await tao_polling_service._fetch_tao_artifacts(
            "ext-quantize-1",
            settings=settings,
            local_cache_dir=tmp_path / "cache",
            action="quantize",
        )

        assert result["success"] is True, result
        assert attempts["n"] == 3, f"expected 3 attempts, got {attempts['n']}"
        # Two sleeps between the three listings — the first two entries
        # of the 8-entry backoff schedule [10, 20, 40, 60, 90, 150, 240,
        # 360]; we still hit only the first two because the third
        # attempt finds artifacts.
        assert sleeps == [10, 20], sleeps

    @pytest.mark.asyncio
    async def test_metadata_only_does_not_retry_on_empty(self, tmp_path, monkeypatch):
        """When ``local_cache_dir is None``, the caller is enumerating
        the workspace state and an empty listing is a legitimate answer
        — the empty-listing retry MUST NOT fire."""
        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")

        attempts = {"n": 0}

        async def fake_list(_external_id, *, settings):
            attempts["n"] += 1
            return {"success": True, "keys": [], "error": None}

        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(tao_polling_service, "_list_tao_job_files", fake_list)
        monkeypatch.setattr(tao_polling_service.asyncio, "sleep", fake_sleep)

        result = await tao_polling_service._fetch_tao_artifacts(
            "ext-meta-only-1",
            settings=settings,
            local_cache_dir=None,
            action="quantize",
        )

        assert result["success"] is True
        assert result["artifacts"] == []
        assert attempts["n"] == 1, "metadata-only must not retry"
        assert sleeps == []

    @pytest.mark.asyncio
    async def test_exhausts_retries_then_returns_error(self, tmp_path, monkeypatch):
        """If all retries see empty artifact selection, the existing
        ``no merged-HF checkpoint slot`` error MUST surface so callers
        can fall back to the packaging=failed branch.
        """
        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")

        attempts = {"n": 0}

        async def fake_list(_external_id, *, settings):
            attempts["n"] += 1
            # Genuinely-empty listing on every attempt — the race
            # signature that keeps the retry alive.
            return {
                "success": True,
                "keys": [],
                "error": None,
            }

        async def fake_sleep(_seconds):
            return None

        monkeypatch.setattr(tao_polling_service, "_list_tao_job_files", fake_list)
        monkeypatch.setattr(tao_polling_service.asyncio, "sleep", fake_sleep)

        result = await tao_polling_service._fetch_tao_artifacts(
            "ext-stuck-1",
            settings=settings,
            local_cache_dir=tmp_path / "cache",
            action="quantize",
        )

        assert result["success"] is False
        assert "no merged-HF checkpoint slot" in (result["error"] or "")
        # 1 initial + 8 retries = 9 attempts total under the 8-entry
        # backoff schedule [10, 20, 40, 60, 90, 150, 240, 360].
        assert attempts["n"] == 9, f"expected 9 attempts, got {attempts['n']}"

    @pytest.mark.asyncio
    async def test_non_empty_non_checkpoint_listing_fails_fast(
        self, tmp_path, monkeypatch
    ):
        """When the workspace listing
        is non-empty but contains only completion metadata (no
        checkpoint shards), ``_fetch_tao_artifacts`` MUST fail fast on
        the first attempt with no retry. Per-object SeaweedFS
        finalization means once any key appears, the upload set is
        what cosmos-rl produced — retrying will not conjure shards
        that were never written.
        """
        settings = make_tao_settings(tmp_path, TAO_API_KEY="jwt")

        attempts = {"n": 0}

        async def fake_list(_external_id, *, settings):
            attempts["n"] += 1
            return {
                "success": True,
                "keys": [
                    "results/job-1/status.json",
                    "results/job-1/microservices_log.txt",
                ],
                "error": None,
            }

        sleeps: list[float] = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        monkeypatch.setattr(tao_polling_service, "_list_tao_job_files", fake_list)
        monkeypatch.setattr(tao_polling_service.asyncio, "sleep", fake_sleep)

        result = await tao_polling_service._fetch_tao_artifacts(
            "ext-non-empty-no-ckpt-1",
            settings=settings,
            local_cache_dir=tmp_path / "cache",
            action="quantize",
        )

        assert result["success"] is False
        assert "no merged-HF checkpoint slot" in (result["error"] or "")
        assert attempts["n"] == 1, (
            f"non-empty non-checkpoint listing must fail fast on "
            f"the first attempt; got {attempts['n']} attempts"
        )
        assert sleeps == [], (
            f"no retry sleep should fire on non-empty non-checkpoint "
            f"listing; got sleeps={sleeps}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Long-lived worker: settings freshness
# ═══════════════════════════════════════════════════════════════════════════


class TestWorkerSettingsFreshness:
    @pytest.mark.asyncio
    async def test_worker_ticks_with_freshly_resolved_settings(
        self, tmp_path, monkeypatch
    ):
        """Each tick must use the live Settings singleton, not the instance
        captured at worker start — otherwise a runtime settings reload
        (POST /v1/secrets:set persist=true, e.g. TAO configured after
        startup) never reaches the poller and every poll fails preflight
        silently until the backend restarts."""
        startup_settings = _make_settings(tmp_path)
        reloaded_settings = _make_settings(tmp_path)
        assert startup_settings is not reloaded_settings

        seen: list[object] = []

        async def fake_tick(settings) -> None:
            seen.append(settings)

        monkeypatch.setattr(tao_polling_service, "tick", fake_tick)
        monkeypatch.setattr(
            "vlm_feedback_loop.config.get_settings", lambda: reloaded_settings
        )
        shutdown_answers = iter([False, True])
        monkeypatch.setattr(
            tao_polling_service.background_manager,
            "is_shutting_down",
            lambda: next(shutdown_answers, True),
        )

        await tao_polling_service._tao_polling_worker(startup_settings)

        assert seen == [reloaded_settings], (
            "worker must resolve Settings per tick (got the startup "
            "instance — runtime reloads would never reach the poller)"
        )
