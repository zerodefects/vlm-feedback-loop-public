# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for first-use TAO Student-base provisioning."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.services.project_service import get_project_engine
from vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service import (
    execute_provisioning_run,
    recover_interrupted_provisioning_runs,
)
from vlm_feedback_loop.services.tao_base_experiment_provisioning_service import (
    ProvisioningResult,
)
from vlm_feedback_loop.services.tao_bootstrap_service import (
    patch_model_configs_across_projects,
)

_RUN_SVC = "vlm_feedback_loop.services.tao_base_experiment_provisioning_run_service"


def _seed_project(client) -> str:
    response = client.post(
        "/v1/projects",
        json={"name": "first-use-provisioning", "description": ""},
    )
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


def _first_student_base(client, project_id: str) -> dict:
    response = client.get(
        f"/v1/projects/{project_id}/model_configs",
        params={"eligible_role": "student_base"},
    )
    assert response.status_code == 200, response.text
    return response.json()["items"][0]


def _settings(client):
    return client.app.dependency_overrides[get_current_settings]()


def _set_model_state(
    client,
    project_id: str,
    model_config_id: str,
    *,
    experiment_id: str | None,
    pull_status: str | None,
) -> None:
    settings = _settings(client)
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None
    with Session(engine) as session:
        row = session.get(ModelConfig, model_config_id)
        assert row is not None
        row.tao_base_experiment_id = experiment_id
        row.tao_base_experiment_pull_status = pull_status
        session.commit()


def _model_state(
    client, project_id: str, model_config_id: str
) -> tuple[str | None, str | None]:
    settings = _settings(client)
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None
    with Session(engine) as session:
        row = session.get(ModelConfig, model_config_id)
        assert row is not None
        return row.tao_base_experiment_id, row.tao_base_experiment_pull_status


def _capture_background(monkeypatch):
    captured: dict[str, object] = {}

    def capture(task_id, worker):
        captured["task_id"] = task_id
        captured["worker"] = worker
        worker.close()
        return None

    monkeypatch.setattr(f"{_RUN_SVC}.background_manager.register", capture)
    return captured


def test_ready_selection_returns_immediate_success_without_background_work(
    test_app_client, monkeypatch
):
    """Already-provisioned bases preserve the fast existing training path."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    _set_model_state(
        test_app_client,
        project_id,
        base["model_config_id"],
        experiment_id="exp-ready",
        pull_status="pull_complete",
    )
    captured = _capture_background(monkeypatch)

    response = test_app_client.post(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
        json={"student_base_model_config_ids": [base["model_config_id"]]},
    )

    assert response.status_code == 202, response.text
    assert response.json()["status"] == "succeeded"
    assert response.json()["requested_model_config_ids"] == []
    assert captured == {}


def test_missing_selection_is_queued_and_visible_through_get(
    test_app_client, monkeypatch
):
    """The first click returns promptly and exposes a durable polling record."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    captured = _capture_background(monkeypatch)

    with patch(f"{_RUN_SVC}._provisioning_prerequisite_error", return_value=None):
        response = test_app_client.post(
            f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
            json={"student_base_model_config_ids": [base["model_config_id"]]},
        )

    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "queued"
    assert body["requested_model_config_ids"] == [base["model_config_id"]]
    assert captured["task_id"] == (f"tao-base-provision-{body['provisioning_run_id']}")
    assert _model_state(test_app_client, project_id, base["model_config_id"]) == (
        None,
        "pulling",
    )

    polled = test_app_client.get(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning/"
        f"{body['provisioning_run_id']}"
    )
    assert polled.status_code == 200
    assert polled.json()["status"] == "queued"


@pytest.mark.asyncio
async def test_worker_success_patches_ready_state_before_terminal_success(
    test_app_client, monkeypatch
):
    """A successful background run makes suite creation safe before UI continuation."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    captured = _capture_background(monkeypatch)
    settings = _settings(test_app_client)

    with patch(f"{_RUN_SVC}._provisioning_prerequisite_error", return_value=None):
        response = test_app_client.post(
            f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
            json={"student_base_model_config_ids": [base["model_config_id"]]},
        )
    run_id = response.json()["provisioning_run_id"]
    assert captured

    async def fake_provisioner(settings_arg, **_kwargs):
        patch_model_configs_across_projects(
            Path(settings_arg.WORKSPACE_ROOT),
            base_experiment_map={base["model_name"]: "exp-new"},
        )
        return ProvisioningResult(registered=[base["model_config_id"]])

    await execute_provisioning_run(
        run_id,
        settings,
        _provisioner=fake_provisioner,
    )

    polled = test_app_client.get(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning/{run_id}"
    )
    assert polled.status_code == 200
    assert polled.json()["status"] == "succeeded"
    assert _model_state(test_app_client, project_id, base["model_config_id"]) == (
        "exp-new",
        "pull_complete",
    )


@pytest.mark.asyncio
async def test_deployment_level_failure_clears_pulling_state(
    test_app_client, monkeypatch
):
    """A generic workspace failure leaves every selected base retriable."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    _capture_background(monkeypatch)
    settings = _settings(test_app_client)

    with patch(f"{_RUN_SVC}._provisioning_prerequisite_error", return_value=None):
        response = test_app_client.post(
            f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
            json={"student_base_model_config_ids": [base["model_config_id"]]},
        )
    run_id = response.json()["provisioning_run_id"]

    async def fake_provisioner(_settings, **_kwargs):
        return ProvisioningResult(
            failed=[("workspace", "S3 endpoint became unreachable")]
        )

    await execute_provisioning_run(
        run_id,
        settings,
        _provisioner=fake_provisioner,
    )

    polled = test_app_client.get(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning/{run_id}"
    )
    assert polled.json()["status"] == "failed"
    assert _model_state(test_app_client, project_id, base["model_config_id"]) == (
        None,
        "failed",
    )


def test_startup_recovery_fails_interrupted_run_and_allows_retry(
    test_app_client, monkeypatch
):
    """A restart cannot leave Student Training polling a permanent active state."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    _capture_background(monkeypatch)
    settings = _settings(test_app_client)

    with patch(f"{_RUN_SVC}._provisioning_prerequisite_error", return_value=None):
        response = test_app_client.post(
            f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
            json={"student_base_model_config_ids": [base["model_config_id"]]},
        )
    run_id = response.json()["provisioning_run_id"]

    assert recover_interrupted_provisioning_runs(settings) == 1
    polled = test_app_client.get(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning/{run_id}"
    )
    assert polled.json()["status"] == "failed"
    assert "restart interrupted" in polled.json()["error_ref"]
    assert _model_state(test_app_client, project_id, base["model_config_id"]) == (
        None,
        "failed",
    )


def test_missing_hf_token_is_an_actionable_start_error(test_app_client):
    """The click fails before queuing work when gated-Hub auth is absent."""
    project_id = _seed_project(test_app_client)
    base = _first_student_base(test_app_client, project_id)
    settings = _settings(test_app_client)
    settings.HF_TOKEN = None
    settings.TAO_WORKSPACE_S3_ACCESS_KEY = "access"
    settings.TAO_WORKSPACE_S3_SECRET_KEY = "secret"
    engine = init_deployment_db(settings.WORKSPACE_ROOT)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).one()
        cfg.bootstrap_status = "bootstrapped"
        cfg.tao_workspace_id = "workspace-id"
        cfg.tao_workspace_bucket = "bucket"
        cfg.tao_workspace_s3_endpoint_url_external = "http://s3.example"
        session.commit()

    response = test_app_client.post(
        f"/v1/projects/{project_id}/tao_base_experiment_provisioning",
        json={"student_base_model_config_ids": [base["model_config_id"]]},
    )

    assert response.status_code == 400
    assert "HF_TOKEN is required" in response.text
