# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the test-emit SSE endpoint.

The testing router is deliberately NOT mounted in the production app (it is
gated on ``VLM_ENABLE_TESTING_ROUTES`` — see main.py), so these tests mount it
on a self-contained app alongside the projects router rather than relying on
the shared ``test_app_client``.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_settings


@pytest.fixture()
def testing_client(tmp_path):
    from fastapi import APIRouter, FastAPI
    from starlette.testclient import TestClient

    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.routers.projects import router as projects_router
    from vlm_feedback_loop.routers.testing import router as testing_router

    workspace = tmp_path / "workspace"
    settings = make_settings(workspace)

    app = FastAPI()
    api = APIRouter(prefix="/v1")
    api.include_router(projects_router)
    api.include_router(testing_router)
    app.include_router(api)
    app.dependency_overrides[get_current_settings] = lambda: settings

    client = TestClient(app, raise_server_exceptions=False)
    yield client

    app.dependency_overrides.clear()


class TestEmitTestEvent:
    """POST /v1/testing/projects/{project_id}/events:emit"""

    def test_emit_returns_ok(self, testing_client):
        """Emit succeeds for an existing project."""
        # Create a project first
        create_resp = testing_client.post("/v1/projects", json={"name": "Emit Test"})
        assert create_resp.status_code == 201
        project_id = create_resp.json()["project_id"]

        resp = testing_client.post(
            f"/v1/testing/projects/{project_id}/events:emit",
            json={
                "event_type": "evaluation_progress",
                "data": {"run_id": "r1", "processed": 5, "total": 10},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_emit_404_for_missing_project(self, testing_client):
        """404 when project does not exist."""
        resp = testing_client.post(
            "/v1/testing/projects/00000000-0000-4000-8000-000000000000/events:emit",
            json={
                "event_type": "test_event",
                "data": {"run_id": "r1"},
            },
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"

    def test_emit_delivers_to_sse_subscriber(self, testing_client):
        """Emitted event is delivered to an SSE subscriber queue."""
        from vlm_feedback_loop.services.sse import sse_manager

        # Create a project
        create_resp = testing_client.post(
            "/v1/projects", json={"name": "SSE Delivery Test"}
        )
        project_id = create_resp.json()["project_id"]

        # Subscribe to the project's SSE stream
        queue = sse_manager.subscribe(project_id)

        # Emit a test event
        resp = testing_client.post(
            f"/v1/testing/projects/{project_id}/events:emit",
            json={
                "event_type": "evaluation_completed",
                "data": {
                    "run_id": "r1",
                    "status": "completed",
                    "summary": {"accuracy": 0.85},
                },
            },
        )
        assert resp.status_code == 200

        # Verify the event arrived on the queue
        assert not queue.empty()
        msg = queue.get_nowait()
        assert msg.startswith("event: evaluation_completed\n")
        data = json.loads(msg.split("data: ")[1].strip())
        assert data["run_id"] == "r1"
        assert data["status"] == "completed"
        assert "timestamp" in data

        # Cleanup
        sse_manager.unsubscribe(project_id, queue)

    def test_emit_rejects_extra_fields(self, testing_client):
        """Request body with extra fields is rejected (ConfigDict extra=forbid)."""
        create_resp = testing_client.post(
            "/v1/projects", json={"name": "Extra Fields Test"}
        )
        project_id = create_resp.json()["project_id"]

        resp = testing_client.post(
            f"/v1/testing/projects/{project_id}/events:emit",
            json={
                "event_type": "test_event",
                "data": {"run_id": "r1"},
                "unexpected_field": True,
            },
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("unexpected_field" in err.get("loc", []) for err in errors), errors
