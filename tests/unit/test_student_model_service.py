# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for student_model_service.

Covers:
* Checkpoint packaging — three LoRA paths + quantize variant.
* StudentModel registration + idempotent re-packaging.
* base_model_path missing → packaging fails with a clear error.
* find_student_for_evaluate_job chain-walking helper.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from conftest import add_tao_job_row, make_settings, seed_tao_chain_project
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.db.models.training_suite import TrainingSuite
from vlm_feedback_loop.services import student_model_service
from vlm_feedback_loop.services.project_service import (
    set_project_engine,
)

PID = "proj-student-packaging"


# ── Settings + DB fixtures ─────────────────────────────────────────────────


def _setup_project(tmp_path: Path):
    workspace = tmp_path / "workspace"
    pdir = workspace / "projects" / PID
    pdir.mkdir(parents=True, exist_ok=True)
    engine = open_project_db(pdir)
    set_project_engine(PID, engine)
    with Session(engine) as s:
        guidance_id = seed_tao_chain_project(s, PID, str(pdir))
        s.commit()
    return engine, workspace, pdir, guidance_id


def _seed_chain(
    engine,
    *,
    chain_id: str,
    guidance_id: str,
    include_quantize: bool = False,
    train_outputs: dict | None = None,
    train_job_config_extra: dict | None = None,
    train_status: str = "succeeded",
    quant_outputs: dict | None = None,
    quant_status: str = "not_started",
):
    """Seed a suite + chain rows. Returns (train_id, eval_id, quant_id, quant_eval_id)."""
    train_id = generate_uuid4()
    eval_id = generate_uuid4()
    quant_id = generate_uuid4() if include_quantize else None
    quant_eval_id = generate_uuid4() if include_quantize else None

    # TrainingSuite (for guidance_id lookup)
    suite_id = generate_uuid4()
    with Session(engine) as s:
        s.add(
            TrainingSuite(
                training_suite_id=suite_id,
                project_id=PID,
                idempotency_key=f"idem-{suite_id}",
                guidance_id=guidance_id,
                training_preset="standard",
                export_field_mode="all",
                include_auto_labeled=False,
                training_dataset_export_id="de-train",
                evaluation_dataset_export_id="de-eval",
                selected_student_base_model_config_ids=["mc-1"],
                quantization_schemes=["FP8_DYNAMIC"] if include_quantize else [],
                chain_ids_ordered=[chain_id],
                status="running",
            )
        )
        train_job_config = {
            "training_preset": "standard",
            "lora_config": {
                "enable_lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"],
            },
            "resolved_training_fields": {
                "policy": {"model_name_or_path": "nvidia/cosmos-reason-2-8b"},
            },
        }
        if train_job_config_extra:
            train_job_config.update(train_job_config_extra)

        add_tao_job_row(
            s,
            PID,
            train_id,
            action="train",
            status=train_status,
            dataset_export_ids=["de-train"],
            job_config=train_job_config,
            outputs=train_outputs,
            tao_external_job_id="ext-train",
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
        if include_quantize:
            add_tao_job_row(
                s,
                PID,
                quant_id,
                action="quantize",
                status=quant_status,
                dataset_export_ids=["de-train"],
                job_config={
                    "training_preset": "standard",
                    "quantization_method": "FP8_DYNAMIC",
                    "lora_config": train_job_config["lora_config"],
                },
                outputs=quant_outputs,
                tao_external_job_id="ext-quant",
                parent_tao_job_id=train_id,
                chain_id=chain_id,
                chain_sequence=3,
            )
            add_tao_job_row(
                s,
                PID,
                quant_eval_id,
                action="evaluate",
                dataset_export_ids=["de-eval"],
                parent_tao_job_id=quant_id,
                chain_id=chain_id,
                chain_sequence=4,
            )
        s.commit()

    return train_id, eval_id, quant_id, quant_eval_id


# ── On-disk checkpoint fixtures ────────────────────────────────────────────


def _make_hf_checkpoint_dir(dir_path: Path) -> None:
    """Produce a realistic NIM-loadable HF checkpoint layout."""
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "config.json").write_text('{"model_type": "cosmos_reason2"}')
    (dir_path / "model.safetensors").write_bytes(b"shard-bytes")
    (dir_path / "tokenizer.json").write_text("{}")
    (dir_path / "tokenizer_config.json").write_text("{}")


def _make_adapter_only_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (dir_path / "adapter_model.safetensors").write_bytes(b"adapter-bytes")


def _make_unrecognized_dir(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "random.log").write_text("nothing useful")


# ══════════════════════════════════════════════════════════════════════════
# Packaging behavior (3 LoRA layouts + quantize)
# ══════════════════════════════════════════════════════════════════════════


class TestResolveMergePython:
    """The merge subprocess must not default to the backend venv
    when a capable interpreter is configured or provisioned — the backend
    deliberately has no transformers/peft, so the shipped default could
    never work without this resolution."""

    def test_setting_wins(self, tmp_path):
        from types import SimpleNamespace

        from vlm_feedback_loop.services.student_model_service import (
            _resolve_merge_python,
        )

        settings = SimpleNamespace(
            MERGE_LORA_PYTHON="/opt/ml/bin/python",
            WORKSPACE_ROOT=str(tmp_path),
        )
        assert _resolve_merge_python(settings) == "/opt/ml/bin/python"

    def test_provisioned_venv_used_when_present(self, tmp_path, monkeypatch):
        import sys as _sys
        from types import SimpleNamespace

        from vlm_feedback_loop.services.student_model_service import (
            _resolve_merge_python,
        )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.student_model_service.Path.home",
            lambda: tmp_path / "fake-home",
        )
        settings = SimpleNamespace(MERGE_LORA_PYTHON=None, WORKSPACE_ROOT=str(tmp_path))
        # No venv → backend interpreter (needs manual dep install).
        assert _resolve_merge_python(settings) == _sys.executable
        venv_python = tmp_path / "merge-lora-venv" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.touch()
        assert _resolve_merge_python(settings) == str(venv_python)

    @pytest.mark.asyncio
    async def test_merge_readiness_fails_for_missing_configured_interpreter(
        self, tmp_path
    ):
        from types import SimpleNamespace

        from vlm_feedback_loop.services.student_model_service import (
            check_lora_merge_readiness,
        )

        missing = tmp_path / "missing-python"
        settings = SimpleNamespace(
            MERGE_LORA_PYTHON=str(missing),
            WORKSPACE_ROOT=str(tmp_path),
        )
        ready, message = await check_lora_merge_readiness(settings)
        assert ready is False
        assert str(missing) in message

    @pytest.mark.asyncio
    async def test_merge_readiness_timeout_uses_shared_process_cleanup(
        self, tmp_path, monkeypatch
    ):
        from unittest.mock import AsyncMock

        python = tmp_path / "python"
        python.touch()
        settings = make_settings(tmp_path, MERGE_LORA_PYTHON=str(python))

        class FakeProcess:
            returncode = None

        async def fake_exec(*_args, **_kwargs):
            return FakeProcess()

        communication = AsyncMock(side_effect=TimeoutError)
        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(
            student_model_service,
            "communicate_with_timeout",
            communication,
        )

        ready, message = await student_model_service.check_lora_merge_readiness(
            settings
        )

        assert ready is False
        assert "probe failed" in message
        communication.assert_awaited_once()


class TestPackaging:
    @pytest.mark.asyncio
    async def test_already_merged_hf_checkpoint_passes_without_merge(self, tmp_path):
        """Already-merged HF checkpoint → validated."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-001"
        _make_hf_checkpoint_dir(cache / "best_model")

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-A",
            guidance_id=guidance_id,
            train_outputs={
                "artifact_cache_dir": str(cache),
                "artifacts": [
                    {"name": "best_model", "local_path": str(cache / "best_model")}
                ],
            },
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert sid is not None

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "validated"
            assert student.nim_checkpoint_ref == str(cache / "best_model")
            assert student.quality_status == "pending"
            assert student.serving_status == "not_attempted"
            assert student.quantization_method is None
            assert student.quantize_tao_job_id is None
            assert student.guidance_id == guidance_id
            assert (
                student.training_suite_id
                == (s.query(TrainingSuite.training_suite_id).one()[0])
            )
            assert student.training_preset == "standard"
            assert student.lora_config["enable_lora"] is True

    @pytest.mark.asyncio
    async def test_adapter_only_with_base_triggers_merge(self, tmp_path, monkeypatch):
        """Adapter-only + base → merged + validated."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-002"
        _make_adapter_only_dir(cache / "best_model")

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-B",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )

        # Mock the merge subprocess — materialize a fake HF checkpoint at the
        # expected merged dir so the post-merge validation succeeds.
        async def fake_run_merge(
            adapter_dir, base, out_dir, *, settings=None, timeout_s=3600.0
        ):
            assert Path(adapter_dir).is_dir()
            assert base == "nvidia/cosmos-reason-2-8b"
            _make_hf_checkpoint_dir(Path(out_dir))
            return {"ok": True, "stdout": "{}", "stderr": "", "returncode": 0}

        monkeypatch.setattr(
            student_model_service,
            "_run_merge_lora_subprocess",
            fake_run_merge,
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert sid is not None

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "validated"
            assert student.nim_checkpoint_ref == str(cache / "merged")
            # Verify the merged dir actually exists on disk
            assert (cache / "merged" / "config.json").is_file()

    @pytest.mark.asyncio
    async def test_adapter_only_without_base_fails_cleanly(self, tmp_path):
        """Missing base_model_path → failed with a clear error."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-003"
        _make_adapter_only_dir(cache / "best_model")

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-C",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
            # Blow away resolved_training_fields.policy.model_name_or_path
            train_job_config_extra={"resolved_training_fields": {}},
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert sid is not None

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "failed"
            assert student.nim_checkpoint_ref is None
            assert (
                student.quality_status == "pending"
            )  # stays pending; failure is on packaging

    @pytest.mark.asyncio
    async def test_unrecognized_layout_fails(self, tmp_path):
        """Neither HF shards nor adapter → failed with unrecognized_checkpoint_layout."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-004"
        _make_unrecognized_dir(cache / "best_model")

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-D",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "failed"

    @pytest.mark.asyncio
    async def test_missing_artifact_cache_dir_fails(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-E",
            guidance_id=guidance_id,
            train_outputs={},
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "failed"
            # Packaging never produced a loadable checkpoint — the error
            # itself is only logged (there is no packaging-error column).
            assert student.nim_checkpoint_ref is None

    @pytest.mark.asyncio
    async def test_merge_subprocess_failure_yields_failed_packaging(
        self, tmp_path, monkeypatch
    ):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-005"
        _make_adapter_only_dir(cache / "best_model")
        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-F",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )

        async def failing_merge(
            adapter_dir, base, out_dir, *, settings=None, timeout_s=3600.0
        ):
            return {
                "ok": False,
                "stdout": "",
                "stderr": "merge_lora: CUDA out of memory",
                "returncode": 2,
            }

        monkeypatch.setattr(
            student_model_service, "_run_merge_lora_subprocess", failing_merge
        )

        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "failed"
            assert student.nim_checkpoint_ref is None

    @pytest.mark.asyncio
    async def test_quantize_variant_requires_hf_shape_no_merge_attempted(
        self, tmp_path, monkeypatch
    ):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        # Train chain must exist for lineage lookup.
        train_cache = pdir / "artifacts" / "tao_jobs" / "tao-train-Q"
        _make_hf_checkpoint_dir(train_cache / "best_model")
        quant_cache = pdir / "artifacts" / "tao_jobs" / "tao-quant-Q"
        _make_hf_checkpoint_dir(quant_cache / "quantized_model")

        train_id, _, quant_id, _ = _seed_chain(
            engine,
            chain_id="chain-Q",
            guidance_id=guidance_id,
            include_quantize=True,
            train_outputs={"artifact_cache_dir": str(train_cache)},
            quant_outputs={"artifact_cache_dir": str(quant_cache)},
            quant_status="succeeded",
        )

        # Guard: merge MUST NOT be invoked for a quantize variant.
        merge_called = {"count": 0}

        async def guard_merge(*args, **kwargs):
            merge_called["count"] += 1
            return {
                "ok": False,
                "stdout": "",
                "stderr": "should not run",
                "returncode": 99,
            }

        monkeypatch.setattr(
            student_model_service, "_run_merge_lora_subprocess", guard_merge
        )

        settings = make_settings(workspace)
        # Quantize packaging succeeds because the TAO output already has HF shape.
        sid = await student_model_service.register_from_tao_terminal(
            PID, quant_id, settings=settings
        )
        assert sid is not None
        assert merge_called["count"] == 0

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "validated"
            assert student.nim_checkpoint_ref == str(quant_cache / "quantized_model")
            # Quantized variant carries method + quantize_tao_job_id
            assert student.quantization_method == "FP8_DYNAMIC"
            assert student.quantize_tao_job_id == quant_id
            # But tao_job_id remains the train job for lineage anchoring
            assert student.tao_job_id == train_id

    @pytest.mark.asyncio
    async def test_quantize_variant_without_hf_shape_fails(self, tmp_path):
        """Quantize output missing HF shape: fail clearly (no merge retry)."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        quant_cache = pdir / "artifacts" / "tao_jobs" / "tao-quant-BAD"
        _make_adapter_only_dir(
            quant_cache / "quantized_model"
        )  # wrong shape for quantize

        train_id, _, quant_id, _ = _seed_chain(
            engine,
            chain_id="chain-QBad",
            guidance_id=guidance_id,
            include_quantize=True,
            train_outputs={"artifact_cache_dir": str(pdir / "t")},
            quant_outputs={"artifact_cache_dir": str(quant_cache)},
            quant_status="succeeded",
        )
        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, quant_id, settings=settings
        )
        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.checkpoint_packaging_status == "failed"

    @pytest.mark.asyncio
    async def test_repackage_refreshes_quantize_root_without_retraining(
        self, tmp_path, monkeypatch
    ):
        """Repackage repairs a mixed-listing quantize fetch in place."""
        from unittest.mock import AsyncMock

        from vlm_feedback_loop.services import tao_polling_service

        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        quant_cache = pdir / "artifacts" / "tao_jobs" / "tao-quant-refresh"
        _make_adapter_only_dir(quant_cache)
        _, _, quant_id, _ = _seed_chain(
            engine,
            chain_id="chain-QRefresh",
            guidance_id=guidance_id,
            include_quantize=True,
            train_outputs={"artifact_cache_dir": str(pdir / "train")},
            quant_outputs={"artifact_cache_dir": str(quant_cache)},
            quant_status="succeeded",
        )
        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, quant_id, settings=settings
        )

        async def refresh_quantize(*args, **kwargs):
            assert args == ("ext-quant",)
            assert kwargs["local_cache_dir"] == quant_cache
            _make_hf_checkpoint_dir(quant_cache)
            return {"success": True, "artifacts": [], "error": None}

        refresh = AsyncMock(side_effect=refresh_quantize)
        monkeypatch.setattr(
            tao_polling_service, "refresh_quantized_checkpoint_artifacts", refresh
        )

        result = await student_model_service.repackage_student_model(
            project_id=PID, student_model_id=sid, settings=settings
        )

        assert result == {
            "error": None,
            "student_model_id": sid,
            "checkpoint_packaging_status": "validated",
        }
        refresh.assert_awaited_once()
        with Session(engine) as s:
            student = s.get(StudentModel, sid)
            assert student.nim_checkpoint_ref == str(quant_cache)

    @pytest.mark.asyncio
    async def test_evaluate_action_is_a_noop(self, tmp_path):
        """Only train + quantize trigger registration."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        train_id, eval_id, *_ = _seed_chain(
            engine,
            chain_id="chain-X",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(pdir / "t")},
        )
        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, eval_id, settings=settings
        )
        assert sid is None
        with Session(engine) as s:
            assert s.query(StudentModel).count() == 0


# ══════════════════════════════════════════════════════════════════════════
# Idempotency
# ══════════════════════════════════════════════════════════════════════════


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_second_register_updates_in_place(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-Idem"
        _make_hf_checkpoint_dir(cache / "best_model")

        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-Idem",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )

        settings = make_settings(workspace)
        sid1 = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        sid2 = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        assert sid1 == sid2

        with Session(engine) as s:
            assert s.query(StudentModel).count() == 1


# ══════════════════════════════════════════════════════════════════════════
# find_student_for_evaluate_job
# ══════════════════════════════════════════════════════════════════════════


class TestFindStudentForEvaluate:
    @pytest.mark.asyncio
    async def test_baseline_evaluate_resolves_to_baseline_student(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-E1"
        _make_hf_checkpoint_dir(cache / "best_model")

        train_id, eval_id, *_ = _seed_chain(
            engine,
            chain_id="chain-E1",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )
        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        with Session(engine) as s:
            eval_job = s.query(TAOJob).filter_by(tao_job_id=eval_id).one()
            student = student_model_service.find_student_for_evaluate_job(
                s, project_id=PID, evaluate_job=eval_job
            )
            assert student is not None
            assert student.student_model_id == sid
            assert student.quantize_tao_job_id is None

    @pytest.mark.asyncio
    async def test_quantized_evaluate_resolves_to_quantized_student(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        train_cache = pdir / "artifacts" / "tao_jobs" / "tao-train-E2"
        quant_cache = pdir / "artifacts" / "tao_jobs" / "tao-quant-E2"
        _make_hf_checkpoint_dir(train_cache / "best_model")
        _make_hf_checkpoint_dir(quant_cache / "quantized_model")

        train_id, _, quant_id, quant_eval_id = _seed_chain(
            engine,
            chain_id="chain-E2",
            guidance_id=guidance_id,
            include_quantize=True,
            train_outputs={"artifact_cache_dir": str(train_cache)},
            quant_outputs={"artifact_cache_dir": str(quant_cache)},
            quant_status="succeeded",
        )
        settings = make_settings(workspace)
        baseline_sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )
        quantized_sid = await student_model_service.register_from_tao_terminal(
            PID, quant_id, settings=settings
        )
        assert baseline_sid != quantized_sid

        with Session(engine) as s:
            quant_eval_job = s.query(TAOJob).filter_by(tao_job_id=quant_eval_id).one()
            student = student_model_service.find_student_for_evaluate_job(
                s, project_id=PID, evaluate_job=quant_eval_job
            )
            assert student is not None
            assert student.student_model_id == quantized_sid
            assert student.quantize_tao_job_id == quant_id

    @pytest.mark.asyncio
    async def test_returns_none_for_non_evaluate_job(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-NE",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(pdir / "t")},
        )
        with Session(engine) as s:
            train_job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            assert (
                student_model_service.find_student_for_evaluate_job(
                    s, project_id=PID, evaluate_job=train_job
                )
                is None
            )


# ══════════════════════════════════════════════════════════════════════════
# mark_student_quality_failed
# ══════════════════════════════════════════════════════════════════════════


class TestMarkQualityFailed:
    @pytest.mark.asyncio
    async def test_flips_quality_status_and_captures_reason(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-train-QF"
        _make_hf_checkpoint_dir(cache / "best_model")
        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-QF",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )
        settings = make_settings(workspace)
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=settings
        )

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.quality_status == "pending"
            student_model_service.mark_student_quality_failed(
                student=student,
                reason="no_parseable_predictions",
            )
            s.commit()

        with Session(engine) as s:
            student = s.query(StudentModel).filter_by(student_model_id=sid).one()
            assert student.quality_status == "failed"
            assert (
                student.nim_preflight_details.get("quality_failure_reason")
                == "no_parseable_predictions"
            )


# ══════════════════════════════════════════════════════════════════════════
# list / get helpers
# ══════════════════════════════════════════════════════════════════════════


class TestListGet:
    @pytest.mark.asyncio
    async def test_list_returns_newest_first(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)

        # Register two students with distinct created_at timestamps.
        for i in range(2):
            cache = pdir / "artifacts" / "tao_jobs" / f"tao-list-{i}"
            _make_hf_checkpoint_dir(cache / "best_model")
            train_id, *_ = _seed_chain(
                engine,
                chain_id=f"chain-list-{i}",
                guidance_id=guidance_id,
                train_outputs={"artifact_cache_dir": str(cache)},
            )
            await student_model_service.register_from_tao_terminal(
                PID,
                train_id,
                settings=make_settings(workspace),
            )
            # Force distinct created_at values by explicitly overwriting
            with Session(engine) as s:
                rows = (
                    s.query(StudentModel).filter(StudentModel.project_id == PID).all()
                )
                for ix, row in enumerate(rows):
                    row.created_at = f"2026-04-{15 + ix:02d}T00:00:00Z"
                s.commit()

        items, next_cursor = student_model_service.list_student_models(
            project_id=PID,
            workspace_root=str(workspace),
        )
        assert len(items) == 2
        # Descending by created_at
        assert items[0].created_at >= items[1].created_at
        assert next_cursor is None

    @pytest.mark.asyncio
    async def test_keyset_pagination_no_drops_or_dupes_same_timestamp(self, tmp_path):
        """Paging across a same-second boundary must not drop or duplicate rows.

        Regression: the old cursor filtered a random UUID against a created_at
        ordering (incoherent) and a bare timestamp cursor would skip every row
        sharing the boundary second. Insert five rows — two sharing one
        timestamp straddling a page boundary — and page through with limit=2,
        asserting each id is seen exactly once.
        """
        engine, workspace, _pdir, guidance_id = _setup_project(tmp_path)
        # Two rows share 10:00:00; the rest are distinct.
        stamps = [
            "2026-05-01T10:00:00Z",
            "2026-05-01T09:00:00Z",
            "2026-05-01T08:00:00Z",
            "2026-05-01T08:00:00Z",  # same-second collision at a page boundary
            "2026-05-01T07:00:00Z",
        ]
        expected_ids = set()
        with Session(engine) as s:
            for i, ts in enumerate(stamps):
                sid = f"sm-{i}"
                expected_ids.add(sid)
                s.add(
                    StudentModel(
                        student_model_id=sid,
                        project_id=PID,
                        student_base_model_config_id="mc-x",
                        tao_job_id=f"tao-{i}",
                        guidance_id=guidance_id,
                        dataset_export_ids=[],
                        training_preset="standard",
                        lora_config={},
                        created_at=ts,
                    )
                )
            s.commit()

        seen: list[str] = []
        cursor = None
        for _ in range(10):  # safety bound
            items, cursor = student_model_service.list_student_models(
                project_id=PID, workspace_root=str(workspace), limit=2, cursor=cursor
            )
            seen.extend(m.student_model_id for m in items)
            if cursor is None:
                break

        assert len(seen) == len(expected_ids), f"drops/dupes: {seen}"
        assert set(seen) == expected_ids
        assert len(set(seen)) == len(seen)  # no duplicates

    @pytest.mark.asyncio
    async def test_get_returns_full_record(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "tao-get"
        _make_hf_checkpoint_dir(cache / "best_model")
        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-get",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )
        sid = await student_model_service.register_from_tao_terminal(
            PID, train_id, settings=make_settings(workspace)
        )
        row = student_model_service.get_student_model(
            project_id=PID,
            student_model_id=sid,
            workspace_root=str(workspace),
        )
        assert row is not None
        assert row.student_model_id == sid
        assert row.checkpoint_packaging_status == "validated"

    @pytest.mark.asyncio
    async def test_get_cross_project_is_404(self, tmp_path):
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        row = student_model_service.get_student_model(
            project_id=PID,
            student_model_id="nonexistent-id",
            workspace_root=str(workspace),
        )
        assert row is None


# ══════════════════════════════════════════════════════════════════════════
# Direct packaging API (lower-level)
# ══════════════════════════════════════════════════════════════════════════


class TestPackageCheckpointDirect:
    @pytest.mark.asyncio
    async def test_returns_packagingresult_dataclass(self, tmp_path):
        """Sanity — the function returns the documented dataclass shape."""
        engine, workspace, pdir, guidance_id = _setup_project(tmp_path)
        cache = pdir / "artifacts" / "tao_jobs" / "direct"
        _make_hf_checkpoint_dir(cache / "best_model")
        train_id, *_ = _seed_chain(
            engine,
            chain_id="chain-direct",
            guidance_id=guidance_id,
            train_outputs={"artifact_cache_dir": str(cache)},
        )
        with Session(engine) as s:
            job = s.query(TAOJob).filter_by(tao_job_id=train_id).one()
            s.expunge(job)

        result = await student_model_service._package_checkpoint(
            job, settings=make_settings(workspace)
        )
        assert isinstance(result, student_model_service.PackagingResult)
        assert result.status == "validated"
        assert result.merged is False
        assert result.error is None


class TestMergeSubprocessEnv:
    @pytest.mark.asyncio
    async def test_merge_cancellation_uses_shared_cleanup_and_propagates(
        self, tmp_path, monkeypatch
    ):
        import asyncio
        from unittest.mock import AsyncMock

        class FakeProcess:
            returncode = None

        async def fake_exec(*_args, **_kwargs):
            return FakeProcess()

        communication = AsyncMock(side_effect=asyncio.CancelledError)
        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        monkeypatch.setattr(
            student_model_service,
            "communicate_with_timeout",
            communication,
        )

        with pytest.raises(asyncio.CancelledError):
            await student_model_service._run_merge_lora_subprocess(
                adapter_dir=tmp_path / "adapter",
                base_model_path="nvidia/Cosmos3-Nano-Reasoner",
                out_dir=tmp_path / "merged",
                settings=make_settings(tmp_path),
            )

        communication.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_merge_spawn_forwards_hf_token(self, tmp_path, monkeypatch):
        """The LoRA merge pulls a gated/auth'd HF base model; the spawned
        subprocess must carry settings.HF_TOKEN (both env names
        huggingface_hub honors) — the backend process env does not
        necessarily have it (observed live: first adapter merge ran
        unauthenticated and failed 'not a valid model identifier')."""
        import asyncio as aio

        captured: dict = {}
        opaque_token = "opaque-merge-credential-9274"

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (
                    f"stdout echoed {opaque_token}".encode(),
                    f"stderr echoed {opaque_token}".encode(),
                )

        async def fake_exec(*cmd, **kwargs):
            captured["env"] = kwargs.get("env")
            return _FakeProc()

        monkeypatch.setattr(aio, "create_subprocess_exec", fake_exec)
        settings = make_settings(tmp_path, HF_TOKEN=opaque_token)
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        result = await student_model_service._run_merge_lora_subprocess(
            adapter_dir=str(adapter),
            base_model_path="nvidia/Cosmos3-Nano-Reasoner",
            out_dir=str(tmp_path / "merged"),
            settings=settings,
        )
        env = captured["env"]
        assert env["HF_TOKEN"] == opaque_token
        assert env["HUGGING_FACE_HUB_TOKEN"] == opaque_token
        assert opaque_token not in result["stdout"]
        assert opaque_token not in result["stderr"]
        assert result["stdout"] == "stdout echoed [REDACTED]"
        assert result["stderr"] == "stderr echoed [REDACTED]"
