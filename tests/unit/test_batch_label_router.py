# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the batch_label_runs router.

Router-level HTTP tests using TestClient, mocking the service layer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from vlm_feedback_loop.db.base import utc_now

# Service-level patch targets
_SVC = "vlm_feedback_loop.services.batch_label_service"


# ── Helpers ─────────────────────────────────────────────────────────────────

PID = "proj-001"
RID = "run-001"

_SAMPLE_CREATE_RESPONSE = {
    "run_id": RID,
    "run_type": "batch_label_run",
    "status": "queued",
    "guidance_id": "g-1",
    "model_config_id": "mc-1",
    "generation_preset_key": "precise",
    "thinking_mode_effective": "on",
    "visual_budget_preset_key": "balanced",
    "structured_generation_mode_effective": "auto",
    "examples_total": 10,
    "created_at": utc_now(),
}

_SAMPLE_RUN_DETAIL = {
    "run_id": RID,
    "run_type": "batch_label_run",
    "status": "completed",
    "status_reason": None,
    "paused_reason": None,
    "guidance_id": "g-1",
    "model_config_id": "mc-1",
    "generation_preset_key": "precise",
    "thinking_mode_effective": "on",
    "visual_budget_preset_key": "balanced",
    "structured_generation_mode_effective": "auto",
    "progress": None,
    "examples_succeeded": 8,
    "examples_schema_invalid": 1,
    "examples_timeout": 1,
    "examples_endpoint_error": 0,
    "examples_total": 10,
    "created_at": utc_now(),
    "started_at": utc_now(),
    "completed_at": utc_now(),
    "cancel_requested_at": None,
    "recovered_from_restart": False,
}


# ══════════════════════════════════════════════════════════════════════════════
# POST /batch_label_runs
# ══════════════════════════════════════════════════════════════════════════════


class TestPostCreate:
    def test_create_returns_201(self, test_app_client):
        with patch(
            f"{_SVC}.start_batch_label_run", new_callable=AsyncMock
        ) as mock_start:
            mock_start.return_value = _SAMPLE_CREATE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs",
                json={},
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["run_id"] == RID
        assert body["run_type"] == "batch_label_run"
        assert body["status"] == "queued"

    def test_create_forwards_icl_mode(self, test_app_client):
        """The request's icl_mode reaches the service; an invalid
        value is rejected at the schema edge (422), never reaching it."""
        with patch(
            f"{_SVC}.start_batch_label_run", new_callable=AsyncMock
        ) as mock_start:
            mock_start.return_value = _SAMPLE_CREATE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs",
                json={"icl_mode": "disabled"},
            )
        assert resp.status_code == 201
        assert mock_start.call_args.kwargs["icl_mode"] == "disabled"

        resp = test_app_client.post(
            f"/v1/projects/{PID}/batch_label_runs",
            json={"icl_mode": "off"},
        )
        assert resp.status_code == 422

    def test_create_gate_not_ready_returns_409(self, test_app_client):
        with patch(
            f"{_SVC}.start_batch_label_run", new_callable=AsyncMock
        ) as mock_start:
            mock_start.return_value = "conflict: Scale-Up Readiness Gate not ready"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs",
                json={},
            )
        assert resp.status_code == 409
        assert "Gate not ready" in resp.json()["detail"]

    def test_create_no_guidance_returns_400(self, test_app_client):
        with patch(
            f"{_SVC}.start_batch_label_run", new_callable=AsyncMock
        ) as mock_start:
            mock_start.return_value = "No active Guidance configured"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs",
                json={},
            )
        assert resp.status_code == 400
        assert "Guidance" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# GET /batch_label_runs/{run_id}
# ══════════════════════════════════════════════════════════════════════════════


class TestGetStatus:
    def test_get_returns_200(self, test_app_client):
        with patch(f"{_SVC}.get_batch_label_run") as mock_get:
            mock_get.return_value = _SAMPLE_RUN_DETAIL
            resp = test_app_client.get(f"/v1/projects/{PID}/batch_label_runs/{RID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["examples_succeeded"] == 8

    def test_get_nonexistent_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_batch_label_run") as mock_get:
            mock_get.return_value = "not found: Batch label run no-such"
            resp = test_app_client.get(f"/v1/projects/{PID}/batch_label_runs/no-such")
        assert resp.status_code == 404
        assert "Batch label run" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# GET /batch_label_runs
# ══════════════════════════════════════════════════════════════════════════════


class TestGetList:
    def test_list_returns_200(self, test_app_client):
        with patch(f"{_SVC}.list_batch_label_runs") as mock_list:
            mock_list.return_value = ([_SAMPLE_RUN_DETAIL], None)
            resp = test_app_client.get(f"/v1/projects/{PID}/batch_label_runs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["next_cursor"] is None


# ══════════════════════════════════════════════════════════════════════════════
# POST /batch_label_runs/{run_id}:resume
# ══════════════════════════════════════════════════════════════════════════════


class TestResume:
    def test_resume_returns_200(self, test_app_client):
        with patch(
            f"{_SVC}.resume_batch_label_run", new_callable=AsyncMock
        ) as mock_resume:
            mock_resume.return_value = {"run_id": RID, "status": "queued"}
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs/{RID}:resume",
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"

    def test_resume_non_paused_returns_409(self, test_app_client):
        with patch(
            f"{_SVC}.resume_batch_label_run", new_callable=AsyncMock
        ) as mock_resume:
            mock_resume.return_value = (
                "conflict: Run run-001 is not paused (status=completed)"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs/{RID}:resume",
            )
        assert resp.status_code == 409
        assert "not paused" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# POST /batch_label_runs/{run_id}:cancel
# ══════════════════════════════════════════════════════════════════════════════


class TestCancel:
    def test_cancel_returns_200(self, test_app_client):
        with patch(
            f"{_SVC}.cancel_batch_label_run", new_callable=AsyncMock
        ) as mock_cancel:
            mock_cancel.return_value = {
                "run_id": RID,
                "status": "canceling",
                "cancel_requested_at": utc_now(),
            }
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs/{RID}:cancel",
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "canceling"

    def test_cancel_terminal_returns_409(self, test_app_client):
        with patch(
            f"{_SVC}.cancel_batch_label_run", new_callable=AsyncMock
        ) as mock_cancel:
            mock_cancel.return_value = (
                "conflict: Run run-001 already in terminal state (completed)"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/batch_label_runs/{RID}:cancel",
            )
        assert resp.status_code == 409
        assert "terminal" in resp.json()["detail"]
