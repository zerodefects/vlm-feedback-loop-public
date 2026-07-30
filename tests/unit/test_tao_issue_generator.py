# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tao_issue Action Request generator.

Unlike earlier generators, this one reads project DB state.  Tests seed a
TAOJob + ModelConfig via SQLAlchemy and patch the generator's
``get_settings`` import so it points at the temp workspace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

import vlm_feedback_loop.services.tao_issue_generator  # noqa: F401 — ensures registered
from conftest import (
    add_endpoint_row,
    add_model_config_row,
    add_project_row,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.db.base import utc_now
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services.action_requests import generate_action_request

PID = "test-proj"
MCID = "mc-cosmos-8b"
TID = "tao-abc"
EID = "ep-1"


def _setup_project_db(tmp_path: Path):
    engine, _, workspace = open_project_workspace(tmp_path, PID, register_engine=True)
    return engine, workspace


def _seed_project(session, project_dir):
    add_project_row(session, PID, str(project_dir))
    add_endpoint_row(session, PID, EID)
    add_model_config_row(
        session,
        PID,
        MCID,
        EID,
        model_name="nvidia/cosmos-reason2-8b",
        eligible_roles=json.dumps(["teacher", "student_base"]),
        thinking_toggle_mode="qwen_enable_thinking",
        thinking_toggle_support="supported",
        visual_budget_mode="mm_processor_size",
        visual_budget_support="supported",
    )


def _seed_tao_job(
    session,
    *,
    tao_job_id=TID,
    status="failed",
    error_ref="CUDA out of memory on GPU 3",
    tao_external_job_id="ext-xyz-123",
    action="train",
):
    session.add(
        TAOJob(
            tao_job_id=tao_job_id,
            project_id=PID,
            student_base_model_config_id=MCID,
            dataset_export_ids=["de-a", "de-b"],
            action=action,
            status=status,
            training_backend="cosmos_rl_tao_vlm",
            training_policy_type="sft" if action == "train" else None,
            job_config={
                "training_preset": "standard",
                "training_backend": "cosmos_rl_tao_vlm",
                "hyperparameters": {"train": {"epoch": 3}},
                "tao_release_version": "6.26.3",
                "cosmos_rl_container_tag": "6.26.3-cosmos-rl",
            },
            tao_create_job_request={
                "kind": "experiment",
                "action": action,
                "specs": {"train": {"epoch": 3}},
            },
            tao_external_job_id=tao_external_job_id,
            error_ref=error_ref,
            created_at=utc_now(),
        )
    )


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    engine, workspace = _setup_project_db(tmp_path)
    settings = make_tao_settings(workspace)
    with Session(engine) as s:
        _seed_project(s, workspace / "projects" / PID)
        _seed_tao_job(s)
        s.commit()
    # Patch the generator's get_settings to return our test settings.
    monkeypatch.setattr(
        "vlm_feedback_loop.services.tao_issue_generator.get_settings",
        lambda: settings,
    )
    yield engine, settings


# ═══════════════════════════════════════════════════════════════════════════
# Happy path: content assertions
# ═══════════════════════════════════════════════════════════════════════════


class TestRenderedContent:
    def test_rendered_text_contains_required_fields(self, seeded):
        result = generate_action_request(
            "tao_issue",
            "Test Project",
            PID,
            {"tao_job_id": TID},
        )
        text = result["rendered_text"]
        # Required content:
        assert "TAO Endpoint: http://tao.test/api/v2" in text
        assert "TAO Organization: example-org" in text
        assert f"Job ID: {TID}" in text
        assert "TAO External Job ID: ext-xyz-123" in text
        assert "Action: train" in text
        assert "Base Model: nvidia/cosmos-reason2-8b" in text
        assert "Training Preset: standard" in text
        assert "CUDA out of memory" in text
        # Diagnostic hint includes the logs endpoint.
        assert (
            "GET http://tao.test/api/v2/orgs/example-org/jobs/ext-xyz-123:logs" in text
        )
        # Job config summary
        assert "Epochs: 3" in text
        assert "TAO release: 6.26.3" in text
        assert "Cosmos-RL tag: 6.26.3-cosmos-rl" in text
        assert "Dataset Exports: de-a, de-b" in text
        # Standard diagnostic suggestions
        assert "R580+" in text  # driver compatibility hint

    def test_generic_status_is_replaced_from_captured_worker_logs(
        self, tmp_path, monkeypatch
    ):
        engine, workspace = _setup_project_db(tmp_path)
        settings = make_tao_settings(workspace)
        with Session(engine) as s:
            _seed_project(s, workspace / "projects" / PID)
            _seed_tao_job(
                s,
                action="quantize",
                error_ref="quantize action failed for cosmos-rl",
            )
            job = s.query(TAOJob).filter_by(tao_job_id=TID).one()
            job.outputs = {
                "tao_logs_text": (
                    "Quantization failed: offset overflow while concatenating "
                    "arrays, consider casting to large_list first.\n"
                    "quantize action failed for cosmos-rl"
                )
            }
            s.commit()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_issue_generator.get_settings",
            lambda: settings,
        )

        result = generate_action_request(
            "tao_issue", "Test Project", PID, {"tao_job_id": TID}
        )
        text = result["rendered_text"]
        assert f"Job ID: {TID}" in text
        assert "Action: quantize" in text
        assert "offset overflow while concatenating arrays" in text
        assert "num_calibration_samples" in text


# ═══════════════════════════════════════════════════════════════════════════
# Framework redact on secrets
# ═══════════════════════════════════════════════════════════════════════════


class TestNoSecrets:
    def test_bearer_and_nvapi_tokens_redacted(self, tmp_path, monkeypatch):
        engine, workspace = _setup_project_db(tmp_path)
        settings = make_tao_settings(workspace)
        with Session(engine) as s:
            _seed_project(s, workspace / "projects" / PID)
            _seed_tao_job(
                s,
                error_ref=(
                    "TAO rejected request: Bearer nvapi-LEAKEDTOKEN12345 "
                    "failed for workspace_abc"
                ),
            )
            s.commit()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_issue_generator.get_settings",
            lambda: settings,
        )

        result = generate_action_request(
            "tao_issue", "Secrets Test", PID, {"tao_job_id": TID}
        )
        text = result["rendered_text"]
        # Framework redact replaces nvapi-... and "Bearer ..."
        assert "nvapi-LEAKEDTOKEN12345" not in text
        assert "Bearer nvapi-LEAKEDTOKEN12345" not in text
        assert "[REDACTED]" in text


# ═══════════════════════════════════════════════════════════════════════════
# Missing / malformed context
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingContext:
    def test_missing_tao_job_id_renders_placeholder(self, seeded):
        result = generate_action_request("tao_issue", "Test Project", PID, {})
        text = result["rendered_text"]
        # Should not raise; should say "(job id not provided)".
        assert "(job id not provided)" in text
        # Should still include the TAO endpoint/org.
        assert "http://tao.test/api/v2" in text
        assert "example-org" in text

    def test_nonexistent_tao_job_id_renders_not_found(self, seeded):
        result = generate_action_request(
            "tao_issue",
            "Test Project",
            PID,
            {"tao_job_id": "does-not-exist"},
        )
        text = result["rendered_text"]
        # Job record not resolved, status reflects that.
        assert "(TAO job record not found)" in text
        # No external id → suggestion mentions that.
        assert "no external job id available" in text.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Generator registration
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistered:
    def test_generate_action_request_returns_tao_issue_shape(self, seeded):
        result = generate_action_request(
            "tao_issue", "Test Project", PID, {"tao_job_id": TID}
        )
        assert result["request_type"] == "tao_issue"
        assert result["project_name"] == "Test Project"
        assert "generated_at" in result
        assert "rendered_text" in result
        assert "technical_requirements" in result
        assert "current_environment" in result


# ═══════════════════════════════════════════════════════════════════════════
# Job without an external_id (e.g., submission_interrupted recovery)
# ═══════════════════════════════════════════════════════════════════════════


class TestNoExternalId:
    def test_handles_missing_external_id(self, tmp_path, monkeypatch):
        engine, workspace = _setup_project_db(tmp_path)
        settings = make_tao_settings(workspace)
        with Session(engine) as s:
            _seed_project(s, workspace / "projects" / PID)
            _seed_tao_job(
                s,
                status="failed",
                tao_external_job_id=None,
                error_ref="submission_interrupted",
            )
            s.commit()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_issue_generator.get_settings",
            lambda: settings,
        )

        result = generate_action_request("tao_issue", "Test", PID, {"tao_job_id": TID})
        text = result["rendered_text"]
        assert "TAO External Job ID: (not submitted)" in text
        # diagnostic endpoint falls back to the no-external-id variant
        assert "no external job id available" in text.lower()
        tech = result["technical_requirements"]
        assert tech["diagnostic_endpoint"] is None
