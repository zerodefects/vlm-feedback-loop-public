# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tao_jobs router.

Router-level HTTP tests using ``test_app_client``; the service layer is
mocked so the tests cover URL shape, status codes, error mapping, and
request/response schema validation only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vlm_feedback_loop.db.base import utc_now

_SVC = "vlm_feedback_loop.services.tao_job_service"

PID = "proj-001"
TID = "tao-001"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_request_body() -> dict:
    """Minimum-valid POST body for the TAO job endpoint."""
    return {
        "student_base_model_config_id": "mc-cosmos-8b",
        "dataset_export_ids": ["de-a"],
        "job_config": {
            "training_preset": "standard",
            "lora_config": {
                "enable_lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q_proj", "v_proj"],
            },
            "hyperparameters": {"train": {"epoch": 3}},
            "tao_release_version": "6.26.3",
            "cosmos_rl_container_tag": "6.26.3-cosmos-rl",
        },
        "tao_create_job_request": {
            "kind": "experiment",
            "action": "train",
            "specs": {"train": {"epoch": 3}},
        },
    }


def _sample_record() -> dict:
    return {
        "tao_job_id": TID,
        "project_id": PID,
        "status": "submitted",
        "tao_status_raw": None,
        "action": "train",
        "training_backend": "cosmos_rl_tao_vlm",
        "training_policy_type": "sft",
        "student_base_model_config_id": "mc-cosmos-8b",
        "dataset_export_ids": ["de-a"],
        "job_config": {"training_preset": "standard"},
        "tao_create_job_request": {
            "kind": "experiment",
            "action": "train",
            "specs": {},
        },
        "tao_external_job_id": "ext-42",
        "progress": None,
        "outputs": None,
        "parent_tao_job_id": None,
        "chain_id": None,
        "chain_sequence": None,
        "chain_halted_reason": None,
        "preflight_result": None,
        "error_ref": None,
        "poll_error_ref": None,
        "created_at": utc_now(),
        "started_at": utc_now(),
        "completed_at": None,
        "last_polled_at": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST /tao_jobs
# ═══════════════════════════════════════════════════════════════════════════


class TestPostCreate:
    def test_returns_201_with_full_record(self, test_app_client):
        with patch(f"{_SVC}.create_tao_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = _sample_record()
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs", json=_create_request_body()
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["tao_job_id"] == TID
        assert body["status"] == "submitted"
        assert body["tao_external_job_id"] == "ext-42"
        assert body["training_backend"] == "cosmos_rl_tao_vlm"
        assert body["training_policy_type"] == "sft"
        assert body["action"] == "train"

    def test_mixed_export_field_mode_returns_400(self, test_app_client):
        with patch(f"{_SVC}.create_tao_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = (
                "validation: MIXED_EXPORT_FIELD_MODE: exports differ (all vs core_only)"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs", json=_create_request_body()
            )
        assert resp.status_code == 400
        assert "MIXED_EXPORT_FIELD_MODE" in resp.text

    def test_student_base_role_missing_returns_400(self, test_app_client):
        with patch(f"{_SVC}.create_tao_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = (
                "validation: ModelConfig mc-x does not have student_base role"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs", json=_create_request_body()
            )
        assert resp.status_code == 400
        assert "student_base" in resp.text

    def test_dataset_export_not_found_returns_404(self, test_app_client):
        with patch(f"{_SVC}.create_tao_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "not found: dataset_export(s) not found: de-x"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs", json=_create_request_body()
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found: dataset_export(s) not found: de-x"

    def test_cross_project_dataset_export_returns_404(self, test_app_client):
        with patch(f"{_SVC}.create_tao_job", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = (
                "not found: dataset_export(s) belong to a different project: de-x"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs", json=_create_request_body()
            )
        assert resp.status_code == 404
        assert (
            resp.json()["detail"]
            == "not found: dataset_export(s) belong to a different project: de-x"
        )

    def test_malformed_request_returns_422(self, test_app_client):
        # Missing required fields → pydantic rejects before service is called.
        resp = test_app_client.post(
            f"/v1/projects/{PID}/tao_jobs", json={"action": "train"}
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any(
            "student_base_model_config_id" in err.get("loc", []) for err in errors
        ), errors


# ═══════════════════════════════════════════════════════════════════════════
# GET /tao_jobs/{id}
# ═══════════════════════════════════════════════════════════════════════════


class TestGet:
    def test_returns_200_with_full_record(self, test_app_client):
        with patch(f"{_SVC}.get_tao_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _sample_record()
            resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs/{TID}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tao_job_id"] == TID

    def test_not_found_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_tao_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = "not found: TAOJob tao-missing"
            resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs/tao-missing")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found: TAOJob tao-missing"

    def test_refresh_true_invokes_service_with_refresh(self, test_app_client):
        with patch(f"{_SVC}.get_tao_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _sample_record()
            resp = test_app_client.get(
                f"/v1/projects/{PID}/tao_jobs/{TID}?refresh=true"
            )
        assert resp.status_code == 200
        # Service was called with refresh=True
        _, kwargs = mock_get.call_args
        assert kwargs.get("refresh") is True

    def test_refresh_false_by_default(self, test_app_client):
        with patch(f"{_SVC}.get_tao_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = _sample_record()
            resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs/{TID}")
        assert resp.status_code == 200
        _, kwargs = mock_get.call_args
        assert kwargs.get("refresh") is False


# ═══════════════════════════════════════════════════════════════════════════
# GET /tao_jobs
# ═══════════════════════════════════════════════════════════════════════════


class TestList:
    def test_returns_200(self, test_app_client):
        with patch(f"{_SVC}.list_tao_jobs") as mock_list:
            mock_list.return_value = ([_sample_record()], None)
            resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["next_cursor"] is None

    def test_status_filter_forwarded(self, test_app_client):
        with patch(f"{_SVC}.list_tao_jobs") as mock_list:
            mock_list.return_value = ([], None)
            resp = test_app_client.get(
                f"/v1/projects/{PID}/tao_jobs?status=queued&limit=5"
            )
        assert resp.status_code == 200
        _, kwargs = mock_list.call_args
        assert kwargs.get("status_filter") == "queued"
        assert kwargs.get("limit") == 5

    def test_invalid_cursor_returns_400(self, test_app_client):
        with patch(f"{_SVC}.list_tao_jobs") as mock_list:
            mock_list.return_value = "validation: invalid cursor"
            resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs?cursor=bogus")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "validation: invalid cursor"

    def test_limit_bounds_enforced_at_query_level(self, test_app_client):
        # limit=0 should be rejected by Query(..., ge=1, le=100)
        resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs?limit=0")
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("limit" in err.get("loc", []) for err in errors), errors

        resp = test_app_client.get(f"/v1/projects/{PID}/tao_jobs?limit=500")
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("limit" in err.get("loc", []) for err in errors), errors


# ═══════════════════════════════════════════════════════════════════════════
# POST /tao_jobs/{id}:cancel
# ═══════════════════════════════════════════════════════════════════════════


class TestCancel:
    def test_returns_200_with_canceled_record(self, test_app_client):
        canceled = {**_sample_record(), "status": "canceled"}
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = canceled
            resp = test_app_client.post(f"/v1/projects/{PID}/tao_jobs/{TID}:cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "canceled"
        assert body["tao_job_id"] == TID

    def test_not_found_returns_404(self, test_app_client):
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = "not found: TAOJob tao-missing"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/tao_jobs/tao-missing:cancel"
            )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found: TAOJob tao-missing"

    def test_terminal_status_returns_409(self, test_app_client):
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = (
                "conflict: cannot cancel TAOJob in terminal status 'succeeded'"
            )
            resp = test_app_client.post(f"/v1/projects/{PID}/tao_jobs/{TID}:cancel")
        assert resp.status_code == 409
        assert (
            resp.json()["detail"]
            == "conflict: cannot cancel TAOJob in terminal status 'succeeded'"
        )

    def test_tao_failure_returns_502(self, test_app_client):
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = (
                "tao_error: TAO cancel failed: 503 service unavailable"
            )
            resp = test_app_client.post(f"/v1/projects/{PID}/tao_jobs/{TID}:cancel")
        assert resp.status_code == 502
        assert (
            resp.json()["detail"]
            == "tao_error: TAO cancel failed: 503 service unavailable"
        )

    def test_tao_unreachable_returns_503(self, test_app_client):
        """Documented error contract (docs/API.md): upstream connect
        failures are 503, distinct from provider refusals (502)."""
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = "tao_unreachable: TAO cancel failed: transport"
            resp = test_app_client.post(f"/v1/projects/{PID}/tao_jobs/{TID}:cancel")
        assert resp.status_code == 503
        assert resp.json()["detail"] == "tao_unreachable: TAO cancel failed: transport"

    def test_tao_timeout_returns_504(self, test_app_client):
        """Documented error contract (docs/API.md): upstream timeouts are 504."""
        with patch(f"{_SVC}.cancel_tao_job", new_callable=AsyncMock) as mock_cancel:
            mock_cancel.return_value = "tao_timeout: TAO cancel failed: timed out"
            resp = test_app_client.post(f"/v1/projects/{PID}/tao_jobs/{TID}:cancel")
        assert resp.status_code == 504
        assert resp.json()["detail"] == "tao_timeout: TAO cancel failed: timed out"


# ═══════════════════════════════════════════════════════════════════════════
# Project-not-found wiring — one thin check per handler
# ═══════════════════════════════════════════════════════════════════════════


class TestProjectNotFoundWiring:
    """Pins router→mapper wiring only: each handler forwards the
    service's "not found: Project ..." string through map_service_error.
    The string→404 mapping rule itself is pinned in test_error_mapping.py."""

    @pytest.mark.parametrize(
        ("service_fn", "is_async", "method", "path", "body"),
        [
            (
                "create_tao_job",
                True,
                "post",
                f"/v1/projects/{PID}/tao_jobs",
                _create_request_body(),
            ),
            ("list_tao_jobs", False, "get", f"/v1/projects/{PID}/tao_jobs", None),
            (
                "cancel_tao_job",
                True,
                "post",
                f"/v1/projects/{PID}/tao_jobs/{TID}:cancel",
                None,
            ),
        ],
    )
    def test_project_not_found_maps_404(
        self, test_app_client, service_fn, is_async, method, path, body
    ):
        kwargs = {"new_callable": AsyncMock} if is_async else {}
        with patch(f"{_SVC}.{service_fn}", **kwargs) as mock:
            mock.return_value = "not found: Project proj-x"
            if method == "post":
                resp = test_app_client.post(path, json=body)
            else:
                resp = test_app_client.get(path)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found: Project proj-x"
