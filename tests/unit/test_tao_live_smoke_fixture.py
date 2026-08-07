# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the disposable TAO wiring-smoke fixture."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy.orm import Session

from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.label import Label
from vlm_feedback_loop.db.models.operation import OperationRecord
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.services.project_service import get_project_engine

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

tao_live_smoke = importlib.import_module("tao_live_smoke")


def test_wiring_fixture_uses_real_schema_and_honest_import_lineage(
    tmp_workspace: Path,
) -> None:
    """Fixture setup uses production services and never claims a Teacher call."""
    settings = Settings(WORKSPACE_ROOT=str(tmp_workspace))

    assembly = tao_live_smoke._build_smoke_project(settings, "fixture-experiment")

    engine = get_project_engine(assembly["project_id"], settings.WORKSPACE_ROOT)
    assert engine is not None
    with Session(engine) as session:
        project = session.get(Project, assembly["project_id"])
        guidance = session.query(Guidance).one()
        examples = session.query(Example).order_by(Example.example_key).all()
        labels = session.query(Label).order_by(Label.example_key).all()
        operations = (
            session.query(OperationRecord).order_by(OperationRecord.example_key).all()
        )

    # SchemaCore, not a hard-coded placeholder, produced the training schema.
    assert guidance.schema["derived_json_schema"]["type"] == "object"
    assert guidance.schema["schema_hash"] != "live-smoke"
    assert project is not None
    assert project.scaleup_min_test_pool_size == 1

    assert len(examples) == 3
    assert all(example.state == "Verified" for example in examples)
    assert all(
        Path(example.storage_ref).is_relative_to(
            Path(assembly["project_dir"]) / "artifacts"
        )
        for example in examples
    )

    assert len(labels) == 3
    assert all(label.verified_outcome == "Edit" for label in labels)
    assert all(label.rationale_source == "sme_edited" for label in labels)
    assert sum(label.pool_assignment == "test_pool" for label in labels) == 1

    assert len(operations) == 3
    assert all(
        operation.invocation_status == "schema_invalid" for operation in operations
    )
    assert all(operation.schema_valid_core is False for operation in operations)
    assert all(operation.model_config_id is None for operation in operations)


def test_verifier_accepts_spec_compliant_lora_baseline_nim_lineage() -> None:
    """The live smoke recognizes the merged-LoRA Student-NIM quality path."""
    student = SimpleNamespace(student_model_id="student-1", tao_job_id="train-1")
    train_job = SimpleNamespace(
        tao_job_id="train-1",
        action="train",
        status="succeeded",
        tao_external_job_id="external-train-1",
    )
    run = SimpleNamespace(
        run_id="run-1",
        evaluation_source="nim",
        student_model_config_id="student-1",
        metrics={"overall": {"exact_match_rate": 1.0}},
        rescored_metrics=None,
        tao_native_metrics=None,
        tao_job_id=None,
    )
    evaluate_job = SimpleNamespace(
        tao_job_id="eval-1",
        action="evaluate",
        status="succeeded",
        training_backend="student_nim_local",
        parent_tao_job_id="train-1",
        outputs={
            "evaluation_source": "student_nim_local",
            "evaluation_run_id": "run-1",
            "student_model_id": "student-1",
        },
    )

    failures = tao_live_smoke._evaluation_lineage_failures(
        run=run,
        student=student,
        evaluate_job=evaluate_job,
        train_job=train_job,
    )

    assert failures == []


def test_verifier_rejects_lora_baseline_without_run_linkage() -> None:
    """A successful-looking local evaluate row must link the exact quality run."""
    student = SimpleNamespace(student_model_id="student-1", tao_job_id="train-1")
    train_job = SimpleNamespace(
        tao_job_id="train-1",
        action="train",
        status="succeeded",
        tao_external_job_id="external-train-1",
    )
    run = SimpleNamespace(
        run_id="run-1",
        evaluation_source="nim",
        student_model_config_id="student-1",
        metrics={"overall": {"exact_match_rate": 1.0}},
        rescored_metrics=None,
        tao_native_metrics=None,
        tao_job_id=None,
    )
    evaluate_job = SimpleNamespace(
        tao_job_id="eval-1",
        action="evaluate",
        status="succeeded",
        training_backend="student_nim_local",
        parent_tao_job_id="train-1",
        outputs={
            "evaluation_source": "student_nim_local",
            "evaluation_run_id": "different-run",
            "student_model_id": "student-1",
        },
    )

    failures = tao_live_smoke._evaluation_lineage_failures(
        run=run,
        student=student,
        evaluate_job=evaluate_job,
        train_job=train_job,
    )

    assert "LoRA baseline evaluate output does not link quality RunRecord" in failures
