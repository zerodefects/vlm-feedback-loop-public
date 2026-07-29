# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router tests for POST /student_models/{id}:deploy_nim.

Covers:
  - 202 happy path, 404 student_not_found, 400 checkpoint_not_validated,
    400 invalid_external_url, 409 deploy_in_progress.
  - External-mode dispatch skips local container creation and persists
    ``nim_deployment_mode="external"`` with the supplied URL.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from sqlalchemy.orm import Session

from conftest import create_project_via_api
from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.db.models.student_model import StudentModel


def _create_project(client) -> str:
    return create_project_via_api(
        client, name="step-12-1-deploy-nim", description="deploy_nim test"
    )["project_id"]


def _settings_for(client):
    from vlm_feedback_loop.main import app
    from vlm_feedback_loop.routers.projects import get_current_settings

    return app.dependency_overrides[get_current_settings]()


def _project_engine(client, project_id: str):
    from vlm_feedback_loop.services.project_service import get_project_engine

    settings = _settings_for(client)
    return get_project_engine(project_id, settings.WORKSPACE_ROOT)


def _insert_student(
    client,
    project_id: str,
    *,
    checkpoint_packaging_status: str = "validated",
    nim_checkpoint_ref: str = "/tmp/ckpt",
    quality_status: str = "validated",
) -> str:
    sid = generate_uuid4()
    engine = _project_engine(client, project_id)
    assert engine is not None
    with Session(engine) as session:
        session.add(
            StudentModel(
                student_model_id=sid,
                project_id=project_id,
                student_base_model_config_id="mc-base",
                tao_job_id="tao-1",
                guidance_id="g-1",
                dataset_export_ids=["de-1"],
                training_preset="standard",
                lora_config={"enable_lora": True},
                created_at="2026-04-29T00:00:00Z",
                checkpoint_packaging_status=checkpoint_packaging_status,
                nim_checkpoint_ref=nim_checkpoint_ref,
                quality_status=quality_status,
                serving_status="not_attempted",
            )
        )
        session.commit()
    return sid


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — patch the lifecycle to no-op so the router test stays unit-scoped.
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _stub_lifecycle(monkeypatch):
    """Replace the lifecycle coroutine with a no-op so tests don't try to
    actually call docker. Yields the captured kwargs from the latest call.
    """
    captured: dict[str, Any] = {}

    async def _fake_lifecycle(**kwargs: Any) -> None:
        captured.clear()
        captured.update(kwargs)
        # Sleep just enough so the task is briefly visible to active_task_ids
        # for the deploy_in_progress test, then return.
        await asyncio.sleep(0)

    from vlm_feedback_loop.services import student_nim_lifecycle

    monkeypatch.setattr(
        student_nim_lifecycle,
        "run_student_deployment_lifecycle",
        _fake_lifecycle,
    )
    yield captured


# ═══════════════════════════════════════════════════════════════════════════
# 202 happy path
# ═══════════════════════════════════════════════════════════════════════════


class TestHappyPath:
    def test_local_mode_returns_202_and_dispatches(
        self, test_app_client, _stub_lifecycle
    ):
        pid = _create_project(test_app_client)
        sid = _insert_student(test_app_client, pid)

        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={},
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["student_model_id"] == sid
        assert body["nim_deployment_mode"] == "local"
        assert body["serving_status"] == "pending"
        assert body["task_id"].startswith(f"student-nim-{pid}-{sid[:8]}-")
        assert body["created_at"]

        # Persisted state mirrors the dispatch.
        engine = _project_engine(test_app_client, pid)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.serving_status == "pending"
            assert student.nim_deployment_mode == "local"
            assert student.nim_endpoint_url is None

    def test_dispatched_lifecycle_called_with_request_kwargs(
        self, test_app_client, _stub_lifecycle
    ):
        pid = _create_project(test_app_client)
        sid = _insert_student(test_app_client, pid)

        test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={
                "nim_container_image": "nvcr.io/nim/test:1.0",
                "gpu_assignment": "device=2",
                "nim_release_version": "1.6.0",
            },
        )

        # Wait for the brief await asyncio.sleep(0) inside the stub.
        # In practice the TestClient finishes the dispatch synchronously.
        async def _wait():
            for _ in range(50):
                if _stub_lifecycle:
                    return
                await asyncio.sleep(0.01)

        asyncio.run(_wait())

        assert _stub_lifecycle.get("project_id") == pid
        assert _stub_lifecycle.get("student_model_id") == sid
        assert _stub_lifecycle.get("mode") == "local"
        assert _stub_lifecycle.get("nim_container_image") == "nvcr.io/nim/test:1.0"
        assert _stub_lifecycle.get("gpu_assignment") == "device=2"
        assert _stub_lifecycle.get("nim_release_version") == "1.6.0"


# ═══════════════════════════════════════════════════════════════════════════
# External mode skips local container, registers the URL
# ═══════════════════════════════════════════════════════════════════════════


class TestExternalMode:
    def test_external_url_persists_mode_and_url(self, test_app_client, _stub_lifecycle):
        pid = _create_project(test_app_client)
        sid = _insert_student(test_app_client, pid)

        external = "http://student.example.local:8000/v1"
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={
                "nim_endpoint_url": external,
                "auth_mode": "bearer",
            },
        )
        assert resp.status_code == 202, resp.text
        body = resp.json()
        assert body["nim_deployment_mode"] == "external"

        engine = _project_engine(test_app_client, pid)
        with Session(engine) as session:
            student = session.get(StudentModel, sid)
            assert student.nim_deployment_mode == "external"
            assert student.nim_endpoint_url == external


# ═══════════════════════════════════════════════════════════════════════════
# Validation errors
# ═══════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_404_student_not_found(self, test_app_client, _stub_lifecycle):
        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{generate_uuid4()}:deploy_nim",
            json={},
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Student model not found"

    def test_400_checkpoint_not_validated(self, test_app_client, _stub_lifecycle):
        pid = _create_project(test_app_client)
        sid = _insert_student(
            test_app_client,
            pid,
            checkpoint_packaging_status="pending",
        )
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={},
        )
        assert resp.status_code == 400
        assert "checkpoint" in resp.json()["detail"].lower()

    def test_400_checkpoint_failed(self, test_app_client, _stub_lifecycle):
        pid = _create_project(test_app_client)
        sid = _insert_student(
            test_app_client,
            pid,
            checkpoint_packaging_status="failed",
        )
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={},
        )
        assert resp.status_code == 400
        assert "checkpoint" in resp.json()["detail"].lower()

    def test_400_invalid_external_url(self, test_app_client, _stub_lifecycle):
        pid = _create_project(test_app_client)
        sid = _insert_student(test_app_client, pid)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={"nim_endpoint_url": "ftp://nope"},
        )
        assert resp.status_code == 400
        assert "nim_endpoint_url" in resp.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════
# 409 deploy_in_progress (single-active-per-project lock)
# ═══════════════════════════════════════════════════════════════════════════


class TestConcurrencyLock:
    """The single-active-per-project invariant is enforced at the service
    layer by ``_has_in_flight_deploy``. The TestClient's per-request
    event loop tears down between calls, so we exercise the lock at the
    service level instead — same code path the router would hit.
    """

    def test_409_when_in_flight_task_present(self, test_app_client, monkeypatch):
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services import student_model_service

        settings = test_app_client.app.dependency_overrides[get_current_settings]()

        pid = _create_project(test_app_client)
        # First Student exists for completeness even though we only POST
        # against sid_b — the lock applies project-wide.
        _insert_student(test_app_client, pid)
        sid_b = _insert_student(test_app_client, pid)

        # Force `_has_in_flight_deploy` to return True without actually
        # running a background task. Same response a router would see.
        monkeypatch.setattr(
            student_model_service, "_has_in_flight_deploy", lambda _pid: True
        )

        async def _go():
            return await student_model_service.deploy_nim(
                project_id=pid,
                student_model_id=sid_b,
                nim_endpoint_url=None,
                nim_container_image=None,
                nim_release_version=None,
                gpu_assignment=None,
                auth_mode="none",
                settings=settings,
            )

        result = asyncio.run(_go())
        assert result.get("error") == "deploy_in_progress"

    def test_409_at_router_level(self, test_app_client, monkeypatch):
        """Same scenario but exercised through the actual HTTP route so the
        409 status code (not the underlying error string) is verified.
        """
        from vlm_feedback_loop.services import student_model_service

        pid = _create_project(test_app_client)
        sid = _insert_student(test_app_client, pid)
        monkeypatch.setattr(
            student_model_service, "_has_in_flight_deploy", lambda _pid: True
        )

        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{sid}:deploy_nim",
            json={},
        )
        assert resp.status_code == 409
        assert "in progress" in resp.json()["detail"].lower()
