# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training_suites router.

Router-level HTTP tests using ``test_app_client``; the service layer is
mocked so the tests cover URL shape, status codes, error mapping, and
request/response schema validation only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vlm_feedback_loop.db.base import utc_now

_SVC = "vlm_feedback_loop.services.training_suite_service"

PID = "proj-001"
TSID = "ts-001"


def _create_request_body() -> dict:
    return {
        "student_base_model_config_ids": ["mc-8b"],
        "training_preset": "standard",
        "include_auto_labeled": True,
        "export_field_mode": "all",
        "quantization_schemes": ["FP8_DYNAMIC", "W4A16"],
        "idempotency_key": "req-123",
    }


def _sample_response() -> dict:
    return {
        "training_suite_id": TSID,
        "project_id": PID,
        "idempotency_key": "req-123",
        "guidance_id": "g-1",
        "training_preset": "standard",
        "export_field_mode": "all",
        "include_auto_labeled": True,
        "quantization_schemes": ["FP8_DYNAMIC", "W4A16"],
        "training_dataset_export_id": "de-train",
        "evaluation_dataset_export_id": "de-eval",
        "selected_student_base_model_config_ids": ["mc-8b"],
        "chain_ids_ordered": ["chain-1"],
        "chains": [
            {
                "chain_id": "chain-1",
                "student_base_model_config_id": "mc-8b",
                "base_model_name": "nvidia/cosmos-reason2-8b",
                "jobs": [
                    {
                        "tao_job_id": "t1",
                        "action": "train",
                        "chain_sequence": 1,
                        "status": "submitted",
                        "tao_external_job_id": "ext-1",
                        "chain_halted_reason": None,
                    },
                    {
                        "tao_job_id": "t2",
                        "action": "evaluate",
                        "chain_sequence": 2,
                        "status": "not_started",
                        "tao_external_job_id": None,
                        "chain_halted_reason": None,
                    },
                ],
            }
        ],
        "status": "running",
        "created_at": utc_now(),
        "started_at": utc_now(),
        "completed_at": None,
    }


def _sample_cancel_response() -> dict:
    suite = {**_sample_response(), "status": "canceled", "completed_at": utc_now()}
    return {
        "training_suite": suite,
        "jobs_canceled": 1,
        "jobs_already_terminal": 1,
        "setup_tasks_canceled": 0,
        "remote_cancel_failures": [],
    }


# ═══════════════════════════════════════════════════════════════════════════
# POST
# ═══════════════════════════════════════════════════════════════════════════


class TestPost:
    def test_returns_201_on_new_create(self, test_app_client):
        with patch(
            f"{_SVC}.launch_training_suite", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _sample_response()
            resp = test_app_client.post(
                f"/v1/projects/{PID}/training_suites", json=_create_request_body()
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["training_suite_id"] == TSID
        assert len(body["chains"]) == 1
        assert len(body["chains"][0]["jobs"]) == 2

    def test_omitted_quantization_uses_one_fp8_validation_variant(
        self, test_app_client
    ):
        req = _create_request_body()
        del req["quantization_schemes"]
        with patch(
            f"{_SVC}.launch_training_suite", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = {
                **_sample_response(),
                "quantization_schemes": ["FP8_DYNAMIC"],
            }
            resp = test_app_client.post(f"/v1/projects/{PID}/training_suites", json=req)

        assert resp.status_code == 201, resp.text
        assert resp.json()["quantization_schemes"] == ["FP8_DYNAMIC"]
        assert mock_create.await_args.kwargs["quantization_schemes"] == ["FP8_DYNAMIC"]

    def test_invalid_preset_returns_400(self, test_app_client):
        # Schema validator rejects the invalid literal before the service is called.
        req = _create_request_body()
        req["training_preset"] = "ludicrous"
        resp = test_app_client.post(f"/v1/projects/{PID}/training_suites", json=req)
        assert resp.status_code == 422  # pydantic enum mismatch
        errors = resp.json()["detail"]
        assert any("training_preset" in err.get("loc", []) for err in errors), errors

    def test_service_validation_error_returns_400(self, test_app_client):
        with patch(
            f"{_SVC}.launch_training_suite", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = (
                "validation: ModelConfig mc-8b does not have student_base role"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/training_suites", json=_create_request_body()
            )
        assert resp.status_code == 400
        assert "student_base" in resp.text

    def test_missing_required_field_returns_422(self, test_app_client):
        req = _create_request_body()
        del req["training_preset"]
        resp = test_app_client.post(f"/v1/projects/{PID}/training_suites", json=req)
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("training_preset" in err.get("loc", []) for err in errors), errors

    def test_empty_models_list_returns_422(self, test_app_client):
        req = _create_request_body()
        req["student_base_model_config_ids"] = []
        resp = test_app_client.post(f"/v1/projects/{PID}/training_suites", json=req)
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any(
            "student_base_model_config_ids" in err.get("loc", []) for err in errors
        ), errors

    def test_idempotency_replay_still_201(self, test_app_client):
        """Even on replay, the resource exists after the call → 201."""
        with patch(
            f"{_SVC}.launch_training_suite", new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = _sample_response()
            resp1 = test_app_client.post(
                f"/v1/projects/{PID}/training_suites", json=_create_request_body()
            )
            resp2 = test_app_client.post(
                f"/v1/projects/{PID}/training_suites", json=_create_request_body()
            )
        assert resp1.status_code == 201
        assert resp2.status_code == 201
        assert resp1.json()["training_suite_id"] == resp2.json()["training_suite_id"]

    def test_unknown_quantization_scheme_returns_422(self, test_app_client):
        # Pydantic Literal rejects the invalid scheme at request validation.
        req = _create_request_body()
        req["quantization_schemes"] = ["INT128_HYPE"]
        resp = test_app_client.post(f"/v1/projects/{PID}/training_suites", json=req)
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("quantization_schemes" in err.get("loc", []) for err in errors), (
            errors
        )


class TestPresetResolution:
    def test_returns_server_resolved_patches_without_setup_checks(
        self, test_app_client
    ):
        with patch(f"{_SVC}.resolve_training_presets_for_models") as resolve:
            resolve.return_value = {
                "resolved_presets": {
                    "mc-8b": {
                        "standard": {
                            "train": {
                                "epoch": 3,
                                "resume": False,
                                "ckpt": {
                                    "enable_checkpoint": True,
                                    "save_freq_in_epoch": 1,
                                    "max_keep": 1,
                                    "export_safetensors": True,
                                },
                            }
                        }
                    }
                }
            }
            response = test_app_client.post(
                f"/v1/projects/{PID}/training_presets:resolve",
                json={"student_base_model_config_ids": ["mc-8b"]},
            )

        assert response.status_code == 200, response.text
        assert (
            response.json()["resolved_presets"]["mc-8b"]["standard"]["train"]["epoch"]
            == 3
        )
        resolve.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# POST /{id}:cancel
# ═══════════════════════════════════════════════════════════════════════════


class TestCancel:
    def test_returns_best_effort_cancel_summary(self, test_app_client):
        with patch(
            f"{_SVC}.cancel_training_suite", new_callable=AsyncMock
        ) as mock_cancel:
            mock_cancel.return_value = _sample_cancel_response()
            response = test_app_client.post(
                f"/v1/projects/{PID}/training_suites/{TSID}:cancel"
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["training_suite"]["status"] == "canceled"
        assert body["jobs_canceled"] == 1
        mock_cancel.assert_awaited_once()

    def test_terminal_suite_returns_conflict(self, test_app_client):
        with patch(
            f"{_SVC}.cancel_training_suite", new_callable=AsyncMock
        ) as mock_cancel:
            mock_cancel.return_value = (
                "conflict: cannot cancel TrainingSuite in terminal status 'completed'"
            )
            response = test_app_client.post(
                f"/v1/projects/{PID}/training_suites/{TSID}:cancel"
            )

        assert response.status_code == 409
        assert "completed" in response.json()["detail"]


# ═══════════════════════════════════════════════════════════════════════════
# GET /{id}
# ═══════════════════════════════════════════════════════════════════════════


class TestGet:
    def test_returns_200(self, test_app_client):
        with patch(f"{_SVC}.get_training_suite") as mock_get:
            mock_get.return_value = _sample_response()
            resp = test_app_client.get(f"/v1/projects/{PID}/training_suites/{TSID}")
        assert resp.status_code == 200
        assert resp.json()["training_suite_id"] == TSID

    def test_not_found_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_training_suite") as mock_get:
            mock_get.return_value = "not found: TrainingSuite ts-missing"
            resp = test_app_client.get(f"/v1/projects/{PID}/training_suites/ts-missing")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "not found: TrainingSuite ts-missing"


# ═══════════════════════════════════════════════════════════════════════════
# GET list
# ═══════════════════════════════════════════════════════════════════════════


class TestList:
    def test_returns_200_with_items(self, test_app_client):
        with patch(f"{_SVC}.list_training_suites") as mock_list:
            mock_list.return_value = ([_sample_response()], None)
            resp = test_app_client.get(f"/v1/projects/{PID}/training_suites")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["next_cursor"] is None

    def test_limit_bounds_enforced(self, test_app_client):
        resp = test_app_client.get(f"/v1/projects/{PID}/training_suites?limit=0")
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("limit" in err.get("loc", []) for err in errors), errors
        resp = test_app_client.get(f"/v1/projects/{PID}/training_suites?limit=500")
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("limit" in err.get("loc", []) for err in errors), errors

    def test_invalid_cursor_returns_400(self, test_app_client):
        with patch(f"{_SVC}.list_training_suites") as mock_list:
            mock_list.return_value = "validation: invalid cursor"
            resp = test_app_client.get(
                f"/v1/projects/{PID}/training_suites?cursor=bogus"
            )
        assert resp.status_code == 400
        assert resp.json()["detail"] == "validation: invalid cursor"


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
                "launch_training_suite",
                True,
                "post",
                f"/v1/projects/{PID}/training_suites",
                _create_request_body(),
            ),
            (
                "list_training_suites",
                False,
                "get",
                f"/v1/projects/{PID}/training_suites",
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
