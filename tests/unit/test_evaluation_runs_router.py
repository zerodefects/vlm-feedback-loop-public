# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP-level tests for the evaluation_runs router.

Service behavior is covered by ``test_evaluation_service``; this file
exercises the per-endpoint error mapping (404 / 409 / 400) through the
shared classifier (services/errors.py::map_service_error, pinned directly
in test_error_mapping.py) — router branch logic that service-level
coverage does not reach.

Mocks the service layer at the import boundary; asserts response code
plus a substring of ``detail`` so failures point at the right branch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

_SVC = "vlm_feedback_loop.services.evaluation_service"

PID = "proj-eval-router"
RID = "run-eval-001"


# ══════════════════════════════════════════════════════════════════════════════
# POST /evaluation_runs (create) — error-mapping integration through the router
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateEvaluationRunErrors:
    """Exercise the router's error path when ``start_evaluation_run``
    returns a string."""

    def test_empty_pool_returns_400(self, test_app_client):
        with patch(f"{_SVC}.start_evaluation_run", new_callable=AsyncMock) as mock:
            mock.return_value = "Empty pool — cannot run evaluation"
            resp = test_app_client.post(f"/v1/projects/{PID}/evaluation_runs", json={})
        assert resp.status_code == 400
        assert "Empty pool" in resp.json()["detail"]

    def test_no_teacher_returns_400(self, test_app_client):
        with patch(f"{_SVC}.start_evaluation_run", new_callable=AsyncMock) as mock:
            mock.return_value = "no teacher model configured"
            resp = test_app_client.post(f"/v1/projects/{PID}/evaluation_runs", json={})
        assert resp.status_code == 400
        assert "teacher" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# GET /evaluation_runs/{run_id} — get
# ══════════════════════════════════════════════════════════════════════════════


class TestGetEvaluationRunErrors:
    def test_run_not_found_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_evaluation_run") as mock:
            mock.return_value = f"Evaluation run not found: {RID}"
            resp = test_app_client.get(f"/v1/projects/{PID}/evaluation_runs/{RID}")
        assert resp.status_code == 404
        assert "Evaluation run not found" in resp.json()["detail"]


class TestGetEvaluationRunProvenance:
    """The response schema must pass ``student_model_config_id`` through:
    set on Student quality/serving runs, null on Teacher runs. The Compare
    screen uses it to tell the Teacher baseline apart from Student runs;
    a schema that drops the field silently breaks that selection."""

    def _run_dict(self, **overrides):
        base = {
            "run_id": RID,
            "run_type": "evaluation_run",
            "status": "completed",
            "created_at": "2026-07-16T00:00:00Z",
        }
        base.update(overrides)
        return base

    def test_student_run_serializes_student_model_config_id(self, test_app_client):
        with patch(f"{_SVC}.get_evaluation_run") as mock:
            mock.return_value = self._run_dict(student_model_config_id="mc-student-1")
            resp = test_app_client.get(f"/v1/projects/{PID}/evaluation_runs/{RID}")
        assert resp.status_code == 200
        assert resp.json()["student_model_config_id"] == "mc-student-1"

    def test_teacher_run_serializes_null(self, test_app_client):
        with patch(f"{_SVC}.get_evaluation_run") as mock:
            mock.return_value = self._run_dict(student_model_config_id=None)
            resp = test_app_client.get(f"/v1/projects/{PID}/evaluation_runs/{RID}")
        assert resp.status_code == 200
        assert resp.json()["student_model_config_id"] is None


# ══════════════════════════════════════════════════════════════════════════════
# GET /evaluation_runs (list) — basis query param
# ══════════════════════════════════════════════════════════════════════════════


class TestListEvaluationRunsBasis:
    """``basis`` must reach the service; a silently dropped param would
    put Student benchmarks back on the evaluation strip. Filter semantics
    are pinned at the service level."""

    def test_basis_forwarded_to_service(self, test_app_client):
        with patch(f"{_SVC}.list_evaluation_runs") as mock:
            mock.return_value = ([], None)
            resp = test_app_client.get(f"/v1/projects/{PID}/evaluation_runs?basis=gate")
        assert resp.status_code == 200
        assert mock.call_args.kwargs["basis"] == "gate"

    def test_invalid_basis_rejected_422(self, test_app_client):
        resp = test_app_client.get(f"/v1/projects/{PID}/evaluation_runs?basis=bogus")
        assert resp.status_code == 422
        assert "basis" in str(resp.json()["detail"])


# ══════════════════════════════════════════════════════════════════════════════
# POST /evaluation_runs/{run_id}:cancel
# ══════════════════════════════════════════════════════════════════════════════


class TestCancelEvaluationRun:
    def test_terminal_state_returns_409(self, test_app_client):
        with patch(f"{_SVC}.cancel_evaluation_run", new_callable=AsyncMock) as mock:
            mock.return_value = "Cannot cancel a run in terminal status: completed"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/evaluation_runs/{RID}:cancel"
            )
        assert resp.status_code == 409
        assert "terminal" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# POST /evaluation_trigger_status:dismiss
# ══════════════════════════════════════════════════════════════════════════════


class TestDismissTrigger:
    def test_unknown_trigger_type_returns_422(self, test_app_client):
        # The request schema's Literal rejects unknown trigger types with a
        # 422 before the service is ever called.
        with patch(f"{_SVC}.dismiss_trigger") as mock:
            resp = test_app_client.post(
                f"/v1/projects/{PID}/evaluation_trigger_status:dismiss",
                json={"trigger_type": "foo"},
            )
        assert resp.status_code == 422
        assert "trigger_type" in resp.text
        mock.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Project-not-found wiring — one thin check per handler
# ══════════════════════════════════════════════════════════════════════════════


class TestProjectNotFoundWiring:
    """Pins router→mapper wiring only: each handler forwards the
    service's "Project not found" string through map_service_error. The
    string→404 mapping rule itself is pinned in test_error_mapping.py."""

    @pytest.mark.parametrize(
        ("service_fn", "is_async", "method", "path"),
        [
            (
                "start_evaluation_run",
                True,
                "post",
                f"/v1/projects/{PID}/evaluation_runs",
            ),
            ("compute_scaleup_gate", False, "get", f"/v1/projects/{PID}/scaleup_gate"),
        ],
    )
    def test_project_not_found_maps_404(
        self, test_app_client, service_fn, is_async, method, path
    ):
        kwargs = {"new_callable": AsyncMock} if is_async else {}
        with patch(f"{_SVC}.{service_fn}", **kwargs) as mock:
            mock.return_value = f"Project not found: {PID}"
            if method == "post":
                resp = test_app_client.post(path, json={})
            else:
                resp = test_app_client.get(path)
        assert resp.status_code == 404
        assert "Project not found" in resp.json()["detail"]
