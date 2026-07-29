# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration test for checkpoint packaging + TAO re-scoring
with mocked TAO.

Exercises the full pipeline:

    train → succeeded
          → checkpoint packaging
          → StudentModel registered with quality_status="pending"
    evaluate → succeeded
          → TAO re-scoring
          → new RunRecord (evaluation_source="tao")
          → StudentModel.quality_status flipped to "validated"

And the empty-predictions branch: evaluate succeeded with empty predictions →
StudentModel.quality_status="failed".

Runs without a live TAO endpoint — ``tao_polling_service._fetch_tao_artifacts``
is mocked, per-sample predictions are written directly into the
artifact cache, and the Test Pool evaluation archive is written as a
real .tar.gz so the ground-truth loader path is exercised for real.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from conftest import (
    add_endpoint_row,
    add_guidance_row,
    add_model_config_row,
    add_project_row,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.pool import Pool
from vlm_feedback_loop.db.models.run import RunRecord
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import (
    student_model_service,
    tao_polling_service,
)
from vlm_feedback_loop.services.hashing import sha256_file
from vlm_feedback_loop.services.project_service import (
    clear_engine_cache,
)

# In-process e2e over the real proposal/save/eval pipeline: fast standalone,
# but the bare 30s pytest-timeout ceiling is too tight under full-suite
# xdist with coverage instrumentation — and --timeout-method=thread
# hard-exits the whole worker on overrun, hanging the xdist controller.
# Restore the generous ceiling this file had from the integration conftest.
pytestmark = pytest.mark.timeout(120)

PID = "proj-integration"
GID = "g-integration"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    clear_engine_cache()


def _settings(workspace: Path):
    from support import build_test_settings

    return build_test_settings(workspace)


def _make_archive(path: Path, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(path), "w:gz") as tf:
        body = json.dumps(samples).encode("utf-8")
        info = tarfile.TarInfo("annotations.json")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))


def _write_preds(cache_dir: Path, preds: dict[str, dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    arr = [{"id": k, "prediction": json.dumps(v)} for k, v in preds.items()]
    (cache_dir / "per_sample_predictions").write_text(json.dumps(arr))


def _hf_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"x")
    (path / "tokenizer.json").write_text("{}")


def _seed_full_suite(tmp_path: Path):
    """Seed a minimal Cosmos Reason2 8B suite with a real eval archive."""
    engine, pdir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True, subdirs=()
    )

    schema = {
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
                "allowed_values": ["crush", "dent", "scratch"],
                "display_order": 1,
            },
            {
                "field_id": "sev",
                "field_name": "severity",
                "type": "integer",
                "role": "core",
                "minimum": 0,
                "maximum": 3,
                "display_order": 2,
            },
        ]
    }

    eval_samples = [
        {
            "id": "ex-01",
            "images": ["images/ex-01.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\nLabel."},
                {
                    "from": "gpt",
                    "value": json.dumps({"damage_type": "crush", "severity": 2}),
                },
            ],
        },
        {
            "id": "ex-02",
            "images": ["images/ex-02.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\nLabel."},
                {
                    "from": "gpt",
                    "value": json.dumps({"damage_type": "dent", "severity": 1}),
                },
            ],
        },
        {
            "id": "ex-03",
            "images": ["images/ex-03.jpg"],
            "conversations": [
                {"from": "human", "value": "<image>\nLabel."},
                {
                    "from": "gpt",
                    "value": json.dumps({"damage_type": "scratch", "severity": 0}),
                },
            ],
        },
    ]
    eval_archive = pdir / "exports" / "de-eval.tar.gz"
    _make_archive(eval_archive, eval_samples)

    train_id = generate_uuid4()
    eval_id = generate_uuid4()
    chain_id = generate_uuid4()
    suite_id = generate_uuid4()

    train_cache = pdir / "artifacts" / "tao_jobs" / train_id
    _hf_dir(train_cache / "best_model")

    eval_cache = pdir / "artifacts" / "tao_jobs" / eval_id

    with Session(engine) as s:
        add_project_row(s, PID, str(pdir), name="Integration")
        add_endpoint_row(
            s, PID, "ep-1", display_name="hosted", base_url="https://test/v1"
        )
        add_model_config_row(
            s,
            PID,
            "mc-cosmos-8b",
            "ep-1",
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["student_base"]),
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_size",
            visual_budget_support="supported",
        )
        add_guidance_row(s, PID, GID, schema, description="Damage classification")
        s.add(
            Pool(
                pool_id="pool-v1",
                project_id=PID,
                pool_type="test_pool",
                pool_version=1,
                member_example_keys=["ex-01", "ex-02", "ex-03"],
                member_count=3,
                guidance_id=GID,
            )
        )
        s.add(
            DatasetExport(
                dataset_export_id="de-train",
                project_id=PID,
                dataset_intent="training",
                export_field_mode="all",
                guidance_id=GID,
                label_tier_filter="verified_only",
                selection_definition_snapshot={},
                artifact_refs={"archive_path": "/tmp/de-train.tar.gz"},
                manifest_ref="/tmp/m",
                example_count=5,
            )
        )
        s.add(
            DatasetExport(
                dataset_export_id="de-eval",
                project_id=PID,
                dataset_intent="evaluation",
                export_field_mode="all",
                guidance_id=GID,
                label_tier_filter="verified_only",
                selection_definition_snapshot={},
                artifact_refs={
                    "archive_path": str(eval_archive),
                    "checksum_sha256": sha256_file(eval_archive),
                },
                manifest_ref="/tmp/m",
                example_count=3,
            )
        )
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key="idem-integration",
                guidance_id=GID,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-eval",
                selected_student_base_model_config_ids=["mc-cosmos-8b"],
                quantization_schemes=[],
                chain_ids_ordered=[chain_id],
                status="running",
            )
        )
        s.add(
            TAOJob(
                tao_job_id=train_id,
                project_id=PID,
                student_base_model_config_id="mc-cosmos-8b",
                dataset_export_ids=["de-train"],
                action="train",
                status="running",  # poller flips to succeeded
                training_backend="cosmos_rl_tao_vlm",
                training_policy_type="sft",
                job_config={
                    "training_preset": "standard",
                    "lora_config": {"enable_lora": True},
                    "resolved_training_fields": {
                        "policy": {"model_name_or_path": "nvidia/cosmos-reason-2-8b"}
                    },
                },
                tao_create_job_request={
                    "kind": "experiment",
                    "action": "train",
                    "specs": {},
                },
                outputs=None,
                tao_external_job_id="ext-tr",
                chain_id=chain_id,
                chain_sequence=1,
            )
        )
        s.add(
            TAOJob(
                tao_job_id=eval_id,
                project_id=PID,
                student_base_model_config_id="mc-cosmos-8b",
                dataset_export_ids=["de-eval"],
                action="evaluate",
                status="running",
                training_backend="cosmos_rl_tao_vlm",
                job_config={"training_preset": "standard"},
                tao_create_job_request={
                    "kind": "experiment",
                    "action": "evaluate",
                    "specs": {},
                },
                outputs=None,
                tao_external_job_id="ext-ev",
                parent_tao_job_id=train_id,
                chain_id=chain_id,
                chain_sequence=2,
            )
        )
        s.commit()

    return workspace, engine, pdir, train_id, eval_id, train_cache, eval_cache


# ══════════════════════════════════════════════════════════════════════════
# End-to-end happy path
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_end_to_end_train_package_evaluate_rescore(tmp_path, monkeypatch):
    (
        workspace,
        engine,
        pdir,
        train_id,
        eval_id,
        train_cache,
        eval_cache,
    ) = _seed_full_suite(tmp_path)

    # ── Mock TAO HTTP layer ───────────────────────────────────────────
    # For train succeeded: _fetch_tao_artifacts populates the artifact
    # cache. We simulate that by (a) pre-staging `best_model` under the
    # cache dir before the call, (b) returning a metadata-only response.
    async def train_artifact_mock(external_id, *, settings, local_cache_dir=None):
        # local_cache_dir is provided by _handle_succeeded. Don't download
        # anything new — we've pre-staged the HF checkpoint already.
        return {
            "success": True,
            "artifacts": [
                {
                    "name": "best_model",
                    "tao_file_path": "/workspace/best",
                    "local_path": str(train_cache / "best_model"),
                }
            ],
            "error": None,
        }

    async def eval_artifact_mock(external_id, *, settings, local_cache_dir=None):
        # For evaluate succeeded, stage per_sample_predictions in the cache.
        if local_cache_dir is not None:
            _write_preds(
                Path(local_cache_dir),
                {
                    "ex-01": {"damage_type": "crush", "severity": 2},
                    "ex-02": {"damage_type": "dent", "severity": 1},
                    "ex-03": {"damage_type": "scratch", "severity": 0},
                },
            )
        return {
            "success": True,
            "artifacts": [
                {
                    "name": "per_sample_predictions",
                    "tao_file_path": "/workspace/preds",
                    "local_path": str(
                        (local_cache_dir or eval_cache) / "per_sample_predictions"
                    ),
                }
            ],
            "error": None,
        }

    # Route based on external_id so train vs evaluate use the correct mock.
    async def dispatch(external_id, *, settings, local_cache_dir=None, action=None):
        if external_id == "ext-tr":
            return await train_artifact_mock(
                external_id,
                settings=settings,
                local_cache_dir=local_cache_dir,
            )
        return await eval_artifact_mock(
            external_id,
            settings=settings,
            local_cache_dir=local_cache_dir,
        )

    monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", dispatch)
    monkeypatch.setattr(
        tao_polling_service,
        "_fetch_tao_logs",
        AsyncMock(return_value={"success": True, "logs_ref": None, "error": None}),
    )

    settings = _settings(workspace)

    # ── Phase 1: simulate train.succeeded ────────────────────────────
    with Session(engine) as s:
        s.query(TAOJob).filter_by(tao_job_id=train_id).update({"status": "succeeded"})
        s.commit()

    await tao_polling_service._handle_succeeded(
        PID,
        train_id,
        external_id="ext-tr",
        action="train",
        engine=engine,
        settings=settings,
    )

    # Assert StudentModel was registered with quality_status=pending
    with Session(engine) as s:
        students = s.query(StudentModel).all()
        assert len(students) == 1
        student = students[0]
        assert student.tao_job_id == train_id
        assert student.checkpoint_packaging_status == "validated"
        assert student.quality_status == "pending"
        assert student.serving_status == "not_attempted"
        assert student.nim_checkpoint_ref is not None
        assert student.quantize_tao_job_id is None
        assert student.guidance_id == GID

    # ── Phase 2: simulate evaluate.succeeded ──────────────────────────
    with Session(engine) as s:
        s.query(TAOJob).filter_by(tao_job_id=eval_id).update({"status": "succeeded"})
        s.commit()

    await tao_polling_service._handle_succeeded(
        PID,
        eval_id,
        external_id="ext-ev",
        action="evaluate",
        engine=engine,
        settings=settings,
    )

    # Assert a new RunRecord exists + StudentModel flipped to validated
    with Session(engine) as s:
        runs = s.query(RunRecord).filter_by(run_type="evaluation_run").all()
        assert len(runs) == 1
        run = runs[0]
        assert run.evaluation_source == "tao"
        assert run.tao_job_id == eval_id
        assert run.status == "completed"
        assert isinstance(run.rescored_metrics, dict)
        # All 3 predictions matched → exact_match_rate = 1.0
        assert run.rescored_metrics["overall"]["exact_match_rate"] == 1.0
        assert run.rescored_metrics["overall"]["example_count"] == 3
        # Returning/New N/A
        assert run.returning_example_keys is None
        # Coverage gaps present (severity=3 value not in pool)
        assert run.coverage_gaps is not None

        student = s.query(StudentModel).one()
        assert student.quality_status == "validated"
        assert student.quality_evaluation_run_id == run.run_id


# ══════════════════════════════════════════════════════════════════════════
# C2 branch: evaluate.succeeded with empty predictions
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_c2_evaluate_succeeded_with_empty_predictions(tmp_path, monkeypatch):
    (
        workspace,
        engine,
        pdir,
        train_id,
        eval_id,
        train_cache,
        eval_cache,
    ) = _seed_full_suite(tmp_path)

    # Train mock stages the HF checkpoint (same as happy path)
    async def train_artifact_mock(external_id, *, settings, local_cache_dir=None):
        return {
            "success": True,
            "artifacts": [
                {
                    "name": "best_model",
                    "tao_file_path": "/workspace/best",
                    "local_path": str(train_cache / "best_model"),
                }
            ],
            "error": None,
        }

    # Evaluate mock writes NO per_sample_predictions file → triggers C2.
    async def eval_artifact_mock(external_id, *, settings, local_cache_dir=None):
        # Ensure the cache dir exists but is empty.
        if local_cache_dir is not None:
            Path(local_cache_dir).mkdir(parents=True, exist_ok=True)
        return {"success": True, "artifacts": [], "error": None}

    async def dispatch(external_id, *, settings, local_cache_dir=None, action=None):
        if external_id == "ext-tr":
            return await train_artifact_mock(
                external_id,
                settings=settings,
                local_cache_dir=local_cache_dir,
            )
        return await eval_artifact_mock(
            external_id,
            settings=settings,
            local_cache_dir=local_cache_dir,
        )

    monkeypatch.setattr(tao_polling_service, "_fetch_tao_artifacts", dispatch)
    monkeypatch.setattr(
        tao_polling_service,
        "_fetch_tao_logs",
        AsyncMock(return_value={"success": True, "logs_ref": None, "error": None}),
    )

    settings = _settings(workspace)

    # Phase 1: train succeeds → StudentModel registered
    with Session(engine) as s:
        s.query(TAOJob).filter_by(tao_job_id=train_id).update({"status": "succeeded"})
        s.commit()
    await tao_polling_service._handle_succeeded(
        PID,
        train_id,
        external_id="ext-tr",
        action="train",
        engine=engine,
        settings=settings,
    )

    # Phase 2: evaluate succeeds but with no predictions
    with Session(engine) as s:
        s.query(TAOJob).filter_by(tao_job_id=eval_id).update({"status": "succeeded"})
        s.commit()
    await tao_polling_service._handle_succeeded(
        PID,
        eval_id,
        external_id="ext-ev",
        action="evaluate",
        engine=engine,
        settings=settings,
    )

    # Assert C2 path fired
    with Session(engine) as s:
        # No RunRecord was created (rescore returned None)
        assert s.query(RunRecord).filter_by(run_type="evaluation_run").count() == 0
        student = s.query(StudentModel).one()
        assert student.quality_status == "failed"
        assert (
            student.nim_preflight_details.get("quality_failure_reason")
            == "no_parseable_predictions"
        )


@pytest.mark.asyncio
async def test_malformed_materialized_sample_cannot_validate_subset(
    tmp_path,
    monkeypatch,
):
    """A skipped cosmos-rl sample cannot shrink the quality denominator."""
    (
        workspace,
        engine,
        _pdir,
        train_id,
        eval_id,
        train_cache,
        eval_cache,
    ) = _seed_full_suite(tmp_path)

    # The real materializer accepts the archive but can only translate two
    # of its three per-sample files.
    eval_cache.mkdir(parents=True, exist_ok=True)
    tarball = eval_cache / "evaluate_results.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        for example_key, label in (
            ("ex-01", {"damage_type": "crush", "severity": 2}),
            ("ex-02", {"damage_type": "dent", "severity": 1}),
        ):
            body = json.dumps(
                [
                    {
                        "video_id": f"images/{example_key}.jpg",
                        "answer": json.dumps(label),
                    }
                ]
            ).encode()
            info = tarfile.TarInfo(
                f"epoch_1/freeform/eval_set/images/{example_key}.json"
            )
            info.size = len(body)
            tf.addfile(info, io.BytesIO(body))

        malformed = b"{not valid json"
        info = tarfile.TarInfo("epoch_1/freeform/eval_set/images/ex-03.json")
        info.size = len(malformed)
        tf.addfile(info, io.BytesIO(malformed))

    materialized = tao_polling_service._materialize_evaluate_predictions(
        tarball,
        eval_cache,
    )
    assert materialized == {"success": True, "samples": 2, "error": None}

    with Session(engine) as session:
        train_job = session.get(TAOJob, train_id)
        eval_job = session.get(TAOJob, eval_id)
        assert train_job is not None
        assert eval_job is not None
        train_job.status = "succeeded"
        train_job.outputs = {"artifact_cache_dir": str(train_cache)}
        eval_job.status = "succeeded"
        session.commit()

    settings = _settings(workspace)
    student_id = await student_model_service.register_from_tao_terminal(
        PID,
        train_id,
        settings=settings,
    )
    assert student_id is not None

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

    await tao_polling_service._handle_succeeded(
        PID,
        eval_id,
        external_id="ext-ev",
        action="evaluate",
        engine=engine,
        settings=settings,
    )

    with Session(engine) as session:
        assert (
            session.query(RunRecord)
            .filter_by(tao_job_id=eval_id, evaluation_source="tao")
            .count()
            == 0
        )
        student = session.get(StudentModel, student_id)
        assert student is not None
        assert student.quality_status == "failed"
        assert student.quality_evaluation_run_id is None


@pytest.mark.asyncio
async def test_corrupt_frozen_archive_fails_student_quality(tmp_path, monkeypatch):
    """Unreadable frozen evidence terminates quality instead of orphaning it."""
    (
        workspace,
        engine,
        _pdir,
        train_id,
        eval_id,
        train_cache,
        eval_cache,
    ) = _seed_full_suite(tmp_path)
    _write_preds(
        eval_cache,
        {
            "ex-01": {"damage_type": "crush", "severity": 2},
            "ex-02": {"damage_type": "dent", "severity": 1},
            "ex-03": {"damage_type": "scratch", "severity": 0},
        },
    )

    with Session(engine) as session:
        train_job = session.get(TAOJob, train_id)
        eval_job = session.get(TAOJob, eval_id)
        export = session.get(DatasetExport, "de-eval")
        assert train_job is not None
        assert eval_job is not None
        assert export is not None
        train_job.status = "succeeded"
        train_job.outputs = {"artifact_cache_dir": str(train_cache)}
        eval_job.status = "succeeded"
        archive = Path(export.artifact_refs["archive_path"])
        session.commit()

    settings = _settings(workspace)
    student_id = await student_model_service.register_from_tao_terminal(
        PID,
        train_id,
        settings=settings,
    )
    assert student_id is not None

    archive.write_bytes(archive.read_bytes()[:50])
    with Session(engine) as session:
        export = session.get(DatasetExport, "de-eval")
        assert export is not None
        refs = dict(export.artifact_refs)
        refs["checksum_sha256"] = sha256_file(archive)
        export.artifact_refs = refs
        session.commit()

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

    await tao_polling_service._handle_succeeded(
        PID,
        eval_id,
        external_id="ext-ev",
        action="evaluate",
        engine=engine,
        settings=settings,
    )

    with Session(engine) as session:
        assert (
            session.query(RunRecord)
            .filter_by(tao_job_id=eval_id, evaluation_source="tao")
            .count()
            == 0
        )
        eval_job = session.get(TAOJob, eval_id)
        student = session.get(StudentModel, student_id)
        assert eval_job is not None
        assert eval_job.outputs_fetch_status == "completed"
        assert student is not None
        assert student.quality_status == "failed"
        assert student.quality_evaluation_run_id is None
