# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NIM router endpoints (routers/nim.py).

Covers: environment assessment, transient connection test, NimEndpoint CRUD,
EmbeddingDeploymentConfig update, and nim_setup Action Request generator.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient

from conftest import create_project_via_api
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    EMBEDDING_MODEL_ID,
    EMBEDDING_NIM_GPU_MIN_GB,
    EMBEDDING_NIM_IMAGE,
    STEP_3_7_FLASH,
)
from vlm_feedback_loop.services import environment as environment_service
from vlm_feedback_loop.services.environment import GpuInfo
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG

# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_project(client: TestClient) -> str:
    """Create a project and return its project_id."""
    return create_project_via_api(client, name="Test Project")["project_id"]


# ── GET /v1/environment ─────────────────────────────────────────────────────


class TestEnvironmentEndpoint:
    @pytest.fixture(autouse=True)
    def _reset_machine_assessment_cache(self):
        environment_service.invalidate_machine_assessment_cache()
        yield
        environment_service.invalidate_machine_assessment_cache()

    def test_caches_hardware_until_an_explicit_refresh(self, test_app_client):
        """Ordinary reads reuse host probes; explicit refresh replaces the snapshot."""
        docker_probe = AsyncMock(return_value=(True, None))
        toolkit_probe = AsyncMock(return_value=(True, None))
        gpu_probe = AsyncMock(
            side_effect=[
                [GpuInfo(name="GPU before change", memory_total_mb=40960)],
                [GpuInfo(name="GPU after change", memory_total_mb=97887)],
            ]
        )
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                docker_probe,
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                toolkit_probe,
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                gpu_probe,
            ),
        ):
            before = test_app_client.get("/v1/environment")
            cached = test_app_client.get("/v1/environment")
            refreshed = test_app_client.get("/v1/environment?refresh_hardware=true")

        assert before.status_code == 200
        assert cached.status_code == 200
        assert refreshed.status_code == 200
        assert before.json()["gpus"][0]["name"] == "GPU before change"
        assert cached.json()["gpus"][0]["name"] == "GPU before change"
        assert refreshed.json()["gpus"][0]["name"] == "GPU after change"
        assert docker_probe.await_count == 2
        assert toolkit_probe.await_count == 2
        assert gpu_probe.await_count == 2

    def test_returns_all_required_fields(self, test_app_client):
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                new_callable=AsyncMock,
                return_value=(False, "not available"),
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                new_callable=AsyncMock,
                return_value=(False, "not available"),
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = test_app_client.get("/v1/environment")

        assert resp.status_code == 200
        data = resp.json()

        # All required fields present
        required_fields = [
            "hosted_nim_available",
            "local_deploy_available",
            "docker_available",
            "nvidia_toolkit_available",
            "nvidia_api_key_configured",
            "ngc_api_key_configured",
            "gpus",
            "local_deployable_models",
            "embedding_deployment",
            "missing_prerequisites",
            "recommended_teacher_mode",
            "recommended_embedding_mode",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_no_secrets_in_response(self, test_app_client):
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = test_app_client.get("/v1/environment")

        data = resp.json()
        flat = str(data)
        # Default test settings have no API keys
        assert "nvapi-" not in flat
        assert "ngc-" not in flat
        # Boolean flags only
        assert data["nvidia_api_key_configured"] is False
        assert data["ngc_api_key_configured"] is False

    def test_deployment_scoped_no_project_id(self, test_app_client):
        """Endpoint works without project context."""
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = test_app_client.get("/v1/environment")
        assert resp.status_code == 200
        # A full environment report came back, not just a routed response.
        data = resp.json()
        assert "gpus" in data, data
        assert "recommended_teacher_mode" in data, data

    def test_embedding_deployment_from_config(self, test_app_client):
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = test_app_client.get("/v1/environment")

        emb = resp.json()["embedding_deployment"]
        assert emb["model_name"] == EMBEDDING_MODEL_ID
        assert emb["nim_container_image"] == EMBEDDING_NIM_IMAGE
        assert emb["gpu_memory_minimum_gb"] == EMBEDDING_NIM_GPU_MIN_GB

    def test_local_deployable_models_structure(self, test_app_client):
        with (
            patch(
                "vlm_feedback_loop.services.environment.check_docker_available",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.check_nvidia_toolkit",
                new_callable=AsyncMock,
                return_value=(False, None),
            ),
            patch(
                "vlm_feedback_loop.services.environment.probe_gpu_inventory",
                new_callable=AsyncMock,
                return_value=[GpuInfo(name="A100", memory_total_mb=81920)],
            ),
        ):
            resp = test_app_client.get("/v1/environment")

        models = resp.json()["local_deployable_models"]
        # Only models with local_deploy_metadata
        names = {m["model_name"] for m in models}
        assert COSMOS_REASON2_8B in names
        assert COSMOS_REASON2_2B in names
        for m in models:
            assert "fits" in m
            assert "nim_container_image" in m
            assert "gpu_memory_minimum_gb" in m


# ── POST /v1/nim/test_connection ────────────────────────────────────────────


class TestConnectionTest:
    def test_models_probe_success(self, test_app_client):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "model-a"}, {"id": "model-b"}]},
                error_class=None,
                attempts=1,
            )

            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "auth_mode": "bearer",
                    "credential_transient": "nvapi-test",
                    "probe_kind": "models",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["models"] == ["model-a", "model-b"]
        assert data["error"] is None

    def test_models_probe_auth_failure(self, test_app_client):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=401,
                body={"error": "unauthorized"},
                error_class="endpoint_error",
                error_detail="HTTP 401",
                attempts=1,
            )

            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "auth_mode": "bearer",
                    "credential_transient": "bad-key",
                    "probe_kind": "models",
                },
            )

        data = resp.json()
        assert data["success"] is False
        assert data["error"] is not None
        # Error should be user-facing, not stack trace
        assert "Traceback" not in (data["error"] or "")

    def test_embeddings_probe(self, test_app_client):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=200,
                body={
                    "data": [{"index": 0, "embedding": [0.1] * 2048}],
                    "model": EMBEDDING_MODEL_ID,
                },
                error_class=None,
                attempts=1,
            )

            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "http://embedding.internal:8000/v1",
                    "auth_mode": "none",
                    "probe_kind": "embeddings",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True
        request = mock_req.await_args
        assert request.args[1] == "http://embedding.internal:8000/v1/embeddings"
        assert request.kwargs["json_body"]["model"] == EMBEDDING_MODEL_ID
        assert request.kwargs["json_body"]["input_type"] == "passage"

    def test_embeddings_probe_rejects_wrong_dimension(self, test_app_client):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"index": 0, "embedding": [0.1]}]},
                error_class=None,
                attempts=1,
            )
            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "http://teacher-only.internal:8000/v1",
                    "auth_mode": "none",
                    "probe_kind": "embeddings",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert "2048-dimensional" in resp.json()["error"]

    def test_missing_credential_returns_error(self, test_app_client):
        resp = test_app_client.post(
            "/v1/nim/test_connection",
            json={
                "base_url": "https://integrate.api.nvidia.com/v1",
                "auth_mode": "bearer",
                # No credential_transient
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "credential" in data["error"].lower()

    def test_no_auth_mode_succeeds(self, test_app_client):
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "local-model"}]},
                error_class=None,
                attempts=1,
            )

            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "http://localhost:8000/v1",
                    "auth_mode": "none",
                    "probe_kind": "models",
                },
            )

        assert resp.json()["success"] is True


# ── NimEndpoint CRUD ────────────────────────────────────────────────────────


class TestNimEndpointCRUD:
    @pytest.fixture(autouse=True)
    def _mock_probe(self):
        """Mock the NIM client for all CRUD tests to avoid real network calls."""
        with patch(
            "vlm_feedback_loop.services.nim_client.resilient_request",
            new_callable=AsyncMock,
        ) as mock_req:
            from vlm_feedback_loop.services.http_client import HttpResult

            mock_req.return_value = HttpResult(
                status_code=200,
                body={"data": [{"id": "model-a"}]},
                error_class=None,
                attempts=1,
            )
            yield mock_req

    def test_create_endpoint(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "My Self-Hosted NIM",
                "endpoint_mode": "self_hosted",
                "base_url": "http://10.0.1.50:8000/v1",
                "auth_mode": "none",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["display_name"] == "My Self-Hosted NIM"
        assert data["endpoint_mode"] == "self_hosted"
        assert data["base_url"] == "http://10.0.1.50:8000/v1"
        assert data["endpoint_id"]  # UUID4
        assert data["project_id"] == project_id
        assert data["source_kind"] == "user_configured"
        assert data["usage_policy"] == "operator_managed"
        # Auto-probe ran
        assert data["last_probe_at"] is not None
        assert data["last_probe_status"] == "healthy"

    def test_list_endpoints(self, test_app_client):
        project_id = _create_project(test_app_client)
        # Project creation seeds one hosted endpoint
        resp = test_app_client.get(f"/v1/projects/{project_id}/nim_endpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) >= 1
        # The seeded endpoint
        seeded = data["items"][0]
        assert seeded["source_kind"] == "seeded_hosted"
        assert seeded["usage_policy"] == "evaluation_only"

    def test_catalog_host_is_evaluation_only_even_when_user_configured(
        self, test_app_client
    ):
        """Usage policy follows the service host, not endpoint provenance."""
        project_id = _create_project(test_app_client)
        response = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "Catalog override",
                "endpoint_mode": "hosted",
                "base_url": "https://INTEGRATE.API.NVIDIA.COM./v1",
                "auth_mode": "bearer",
            },
        )

        assert response.status_code == 201
        assert response.json()["source_kind"] == "user_configured"
        assert response.json()["usage_policy"] == "evaluation_only"

    def test_get_endpoint_by_id(self, test_app_client):
        project_id = _create_project(test_app_client)
        # Create one
        create_resp = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "Test EP",
                "endpoint_mode": "self_hosted",
                "base_url": "http://host:8000/v1",
            },
        )
        ep_id = create_resp.json()["endpoint_id"]

        # Get by ID
        resp = test_app_client.get(f"/v1/projects/{project_id}/nim_endpoints/{ep_id}")
        assert resp.status_code == 200
        assert resp.json()["endpoint_id"] == ep_id

    def test_get_nonexistent_returns_404(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{project_id}/nim_endpoints/nonexistent-id"
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Endpoint not found"

    def test_update_endpoint(self, test_app_client):
        project_id = _create_project(test_app_client)
        create_resp = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "Old Name",
                "endpoint_mode": "self_hosted",
                "base_url": "http://old:8000/v1",
            },
        )
        ep_id = create_resp.json()["endpoint_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/nim_endpoints/{ep_id}",
            json={"display_name": "New Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "New Name"

    def test_update_base_url_triggers_reprobe(self, test_app_client):
        project_id = _create_project(test_app_client)
        create_resp = test_app_client.post(
            f"/v1/projects/{project_id}/nim_endpoints",
            json={
                "display_name": "EP",
                "endpoint_mode": "self_hosted",
                "base_url": "http://old:8000/v1",
            },
        )
        ep_id = create_resp.json()["endpoint_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{project_id}/nim_endpoints/{ep_id}",
            json={"base_url": "http://new:8000/v1"},
        )
        assert resp.status_code == 200
        assert resp.json()["base_url"] == "http://new:8000/v1"
        # Probe ran again on the base_url change
        assert resp.json()["last_probe_at"] is not None

    def test_no_api_keys_in_response(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{project_id}/nim_endpoints")
        flat = str(resp.json())
        assert "nvapi-" not in flat
        assert "Bearer" not in flat

    def test_nonexistent_project_returns_404(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/nonexistent/nim_endpoints",
            json={
                "display_name": "EP",
                "endpoint_mode": "self_hosted",
                "base_url": "http://host:8000/v1",
            },
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


class TestConfigureSelfHostedTeacher:
    @pytest.fixture(autouse=True)
    def _mock_capability_probes(self):
        with (
            patch(
                "vlm_feedback_loop.services.model_config_service."
                "probe_structured_generation",
                new=AsyncMock(return_value="supported"),
            ),
            patch(
                "vlm_feedback_loop.services.model_config_service.probe_thinking_toggle",
                new=AsyncMock(return_value="unsupported"),
            ),
            patch(
                "vlm_feedback_loop.services.model_config_service.probe_visual_budget",
                new=AsyncMock(return_value="unsupported"),
            ),
            patch(
                "vlm_feedback_loop.services.model_config_service."
                "_probe_image_cap_support",
                new=AsyncMock(return_value="supported"),
            ),
        ):
            yield

    def _selected_teacher(self, client: TestClient, project_id: str) -> dict:
        project = client.get(f"/v1/projects/{project_id}").json()
        teachers = client.get(
            f"/v1/projects/{project_id}/model_configs?eligible_role=teacher"
        ).json()["items"]
        return next(
            item
            for item in teachers
            if item["model_config_id"] == project["teacher_model_config_id"]
        )

    def test_verified_endpoint_is_bound_selected_and_capability_probed(
        self, test_app_client
    ):
        from vlm_feedback_loop.services.nim_client import NimListModelsResult

        project_id = _create_project(test_app_client)
        teacher = self._selected_teacher(test_app_client, project_id)
        with patch(
            "vlm_feedback_loop.services.nim_endpoint_service.nim_client.list_models",
            new=AsyncMock(
                return_value=NimListModelsResult(
                    success=True,
                    models=[teacher["model_name"]],
                    status_code=200,
                )
            ),
        ) as list_models:
            response = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json={
                    "base_url": "http://nim.internal:8000/v1/",
                    "model_config_id": teacher["model_config_id"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["endpoint"]["endpoint_mode"] == "self_hosted"
        assert data["endpoint"]["base_url"] == "http://nim.internal:8000/v1"
        assert data["endpoint"]["auth_mode"] == "none"
        assert data["endpoint"]["last_probe_status"] == "healthy"
        assert data["model_config_id"] == teacher["model_config_id"]
        assert data["model_name"] == teacher["model_name"]
        assert data["structured_generation_support"] == "supported"
        assert data["thinking_toggle_support"] == "unsupported"
        assert data["visual_budget_support"] == "unsupported"
        list_models.assert_awaited_once_with(
            base_url="http://nim.internal:8000/v1",
            auth_headers={},
            deadline_s=180.0,
            max_retries=1,
        )

        selected = self._selected_teacher(test_app_client, project_id)
        assert selected["model_config_id"] == teacher["model_config_id"]
        assert selected["endpoint_id"] == data["endpoint"]["endpoint_id"]
        assert selected["availability"] == {"available": True, "reason": None}

    def test_repeat_apply_reuses_endpoint_instead_of_duplicating(self, test_app_client):
        from vlm_feedback_loop.services.nim_client import NimListModelsResult

        project_id = _create_project(test_app_client)
        teacher = self._selected_teacher(test_app_client, project_id)
        probe = AsyncMock(
            return_value=NimListModelsResult(
                success=True,
                models=[teacher["model_name"]],
                status_code=200,
            )
        )
        payload = {
            "base_url": "http://nim.internal:8000/v1",
            "model_config_id": teacher["model_config_id"],
        }
        with patch(
            "vlm_feedback_loop.services.nim_endpoint_service.nim_client.list_models",
            probe,
        ):
            first = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json=payload,
            )
            second = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json=payload,
            )

        assert first.status_code == second.status_code == 200
        assert (
            first.json()["endpoint"]["endpoint_id"]
            == second.json()["endpoint"]["endpoint_id"]
        )
        endpoints = test_app_client.get(
            f"/v1/projects/{project_id}/nim_endpoints"
        ).json()["items"]
        matching = [
            endpoint
            for endpoint in endpoints
            if endpoint["endpoint_mode"] == "self_hosted"
            and endpoint["base_url"] == payload["base_url"]
        ]
        assert len(matching) == 1

    def test_apply_reuses_and_canonicalizes_an_existing_self_hosted_endpoint(
        self, test_app_client
    ):
        """Save binds one existing URL and enforces the credential-free contract."""
        from vlm_feedback_loop.services.nim_client import NimListModelsResult

        project_id = _create_project(test_app_client)
        teacher = self._selected_teacher(test_app_client, project_id)
        create_probe = AsyncMock(
            return_value=NimListModelsResult(
                success=True,
                models=[teacher["model_name"]],
                status_code=200,
            )
        )
        with patch(
            "vlm_feedback_loop.services.nim_endpoint_service.nim_client.list_models",
            create_probe,
        ):
            existing = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints",
                json={
                    "display_name": "Old custom name",
                    "endpoint_mode": "self_hosted",
                    "base_url": "http://nim.internal:8000/v1",
                    "auth_mode": "bearer",
                },
            )
            configured = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json={
                    "base_url": "http://nim.internal:8000/v1",
                    "model_config_id": teacher["model_config_id"],
                },
            )

        assert existing.status_code == 201
        assert configured.status_code == 200
        endpoint = configured.json()["endpoint"]
        assert endpoint["endpoint_id"] == existing.json()["endpoint_id"]
        assert endpoint["auth_mode"] == "none"
        assert endpoint["display_name"] == (
            "Self-hosted Teacher (http://nim.internal:8000/v1)"
        )

    def test_model_mismatch_leaves_project_binding_unchanged(self, test_app_client):
        from vlm_feedback_loop.services.nim_client import NimListModelsResult

        project_id = _create_project(test_app_client)
        teacher = self._selected_teacher(test_app_client, project_id)
        original_endpoint_id = teacher["endpoint_id"]
        with patch(
            "vlm_feedback_loop.services.nim_endpoint_service.nim_client.list_models",
            new=AsyncMock(
                return_value=NimListModelsResult(
                    success=True,
                    models=["a-different-model"],
                    status_code=200,
                )
            ),
        ):
            response = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json={
                    "base_url": "http://wrong-model:8000/v1",
                    "model_config_id": teacher["model_config_id"],
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "model_not_served"
        selected = self._selected_teacher(test_app_client, project_id)
        assert selected["endpoint_id"] == original_endpoint_id
        endpoints = test_app_client.get(
            f"/v1/projects/{project_id}/nim_endpoints"
        ).json()["items"]
        assert not any(
            endpoint["base_url"] == "http://wrong-model:8000/v1"
            for endpoint in endpoints
        )

    def test_embedded_credentials_are_rejected_before_network_or_persistence(
        self, test_app_client
    ):
        project_id = _create_project(test_app_client)
        teacher = self._selected_teacher(test_app_client, project_id)
        with patch(
            "vlm_feedback_loop.services.nim_endpoint_service.nim_client.list_models",
            new=AsyncMock(),
        ) as probe:
            response = test_app_client.post(
                f"/v1/projects/{project_id}/nim_endpoints:configure_self_hosted_teacher",
                json={
                    "base_url": "http://operator:secret@nim.internal:8000/v1",
                    "model_config_id": teacher["model_config_id"],
                },
            )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "embedded_credentials_forbidden"
        probe.assert_not_awaited()


# ── PATCH /v1/embedding_deployment_config ───────────────────────────────────


class TestEmbeddingDeploymentConfig:
    def test_update_provider(self, test_app_client):
        # Create a project first to ensure deployment.db is initialized
        _create_project(test_app_client)

        resp = test_app_client.patch(
            "/v1/embedding_deployment_config",
            json={"provider": "self_hosted_nvclip"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["provider"] == "self_hosted_nvclip"
        assert data["model_name"] == EMBEDDING_MODEL_ID

    def test_update_endpoint_url(self, test_app_client):
        _create_project(test_app_client)

        resp = test_app_client.patch(
            "/v1/embedding_deployment_config",
            json={"endpoint_url": "http://local-nvclip:8001/v1"},
        )
        assert resp.status_code == 200
        assert resp.json()["endpoint_url"] == "http://local-nvclip:8001/v1"

    def test_update_gpu_assignment(self, test_app_client):
        _create_project(test_app_client)

        resp = test_app_client.patch(
            "/v1/embedding_deployment_config",
            json={"gpu_assignment": "device=1"},
        )
        assert resp.status_code == 200
        assert resp.json()["gpu_assignment"] == "device=1"

    def test_partial_update_preserves_other_fields(self, test_app_client):
        _create_project(test_app_client)

        resp = test_app_client.patch(
            "/v1/embedding_deployment_config",
            json={"provider": "local_nvclip"},
        )
        data = resp.json()
        # Other fields unchanged
        assert data["embedding_dim"] == 2048
        assert data["nim_container_image"] == EMBEDDING_NIM_IMAGE
        assert data["preferred_host_port"] == 8001


class TestConfigureSelfHostedEmbedding:
    def test_verified_vector_is_persisted_and_projects_are_reswept(
        self, test_app_client
    ):
        from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult

        _create_project(test_app_client)
        probe = AsyncMock(
            return_value=NimEmbeddingsResult(
                success=True,
                embeddings=[[0.25] * 2048],
                model=EMBEDDING_MODEL_ID,
                status_code=200,
            )
        )
        resweep = AsyncMock()
        with (
            patch(
                "vlm_feedback_loop.services.local_nim_service.create_embeddings",
                probe,
            ),
            patch(
                "vlm_feedback_loop.services.local_nim_service.resweep_embedding_tasks",
                resweep,
            ),
        ):
            response = test_app_client.post(
                "/v1/embedding_deployment_config:configure_self_hosted",
                json={"base_url": "http://embedding.internal:8000/v1/"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["provider"] == "self_hosted_nvclip"
        assert data["endpoint_url"] == "http://embedding.internal:8000/v1"
        assert data["gpu_assignment"] is None
        probe.assert_awaited_once()
        assert probe.await_args.kwargs["model"] == EMBEDDING_MODEL_ID
        assert probe.await_args.kwargs["input_type"] == "passage"
        resweep.assert_awaited_once()

    def test_failed_probe_does_not_mutate_configuration(self, test_app_client):
        from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult

        _create_project(test_app_client)
        with patch(
            "vlm_feedback_loop.services.local_nim_service.create_embeddings",
            new=AsyncMock(
                return_value=NimEmbeddingsResult(
                    success=False,
                    error="Endpoint error: HTTP 404",
                    status_code=404,
                )
            ),
        ):
            response = test_app_client.post(
                "/v1/embedding_deployment_config:configure_self_hosted",
                json={"base_url": "http://teacher-only.internal:8000/v1"},
            )

        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "embedding_probe_failed"
        assert response.json()["detail"]["message"] == (
            "This endpoint does not expose the required /embeddings operation "
            "for the configured NeMo Retriever model."
        )
        config = test_app_client.patch(
            "/v1/embedding_deployment_config", json={}
        ).json()
        assert config["provider"] == "none"

    def test_different_url_cannot_orphan_active_managed_embedding(
        self, test_app_client
    ):
        from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult

        project_id = _create_project(test_app_client)
        with (
            patch(
                "vlm_feedback_loop.services.local_nim_service.create_embeddings",
                new=AsyncMock(
                    return_value=NimEmbeddingsResult(
                        success=True,
                        embeddings=[[0.25] * 2048],
                        status_code=200,
                    )
                ),
            ),
            patch(
                "vlm_feedback_loop.services.local_nim_service."
                "_scan_active_deployment_placements",
                return_value=[("0", "embedding", project_id, "missing", 8001)],
            ),
        ):
            response = test_app_client.post(
                "/v1/embedding_deployment_config:configure_self_hosted",
                json={"base_url": "http://different.internal:8000/v1"},
            )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "local_embedding_active"
        config = test_app_client.patch(
            "/v1/embedding_deployment_config", json={}
        ).json()
        assert config["provider"] == "none"
        assert config["endpoint_url"] is None

    def test_exact_active_managed_url_keeps_gpu_assignment(self, test_app_client):
        from pathlib import Path

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.local_nim_deployment import (
            LocalNimDeployment,
        )
        from vlm_feedback_loop.services.nim_client import NimEmbeddingsResult
        from vlm_feedback_loop.services.project_service import get_project_engine

        project_id = _create_project(test_app_client)
        deployment_id = "active-embedding"
        project_dir = Path(
            test_app_client.get(f"/v1/projects/{project_id}").json()["project_dir"]
        )
        engine = get_project_engine(project_id, str(project_dir.parents[1]))
        assert engine is not None
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=deployment_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://managed.internal:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()
        test_app_client.patch(
            "/v1/embedding_deployment_config",
            json={
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://managed.internal:8000/v1",
                "gpu_assignment": "device=0",
            },
        )
        with (
            patch(
                "vlm_feedback_loop.services.local_nim_service.create_embeddings",
                new=AsyncMock(
                    return_value=NimEmbeddingsResult(
                        success=True,
                        embeddings=[[0.25] * 2048],
                        status_code=200,
                    )
                ),
            ),
            patch(
                "vlm_feedback_loop.services.local_nim_service."
                "_scan_active_deployment_placements",
                return_value=[("0", "embedding", project_id, deployment_id, 8001)],
            ),
            patch(
                "vlm_feedback_loop.services.local_nim_service.resweep_embedding_tasks",
                new=AsyncMock(),
            ),
        ):
            response = test_app_client.post(
                "/v1/embedding_deployment_config:configure_self_hosted",
                json={"base_url": "http://managed.internal:8000/v1/"},
            )

        assert response.status_code == 200
        assert response.json()["gpu_assignment"] == "device=0"


# ── nim_setup Action Request ────────────────────────────────────────────────


class TestNimSetupActionRequest:
    def test_generates_with_model_names(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_type"] == "nim_setup"
        assert data["project_name"] == "Test Project"

        # rendered_text contains model names
        text = data["rendered_text"]
        assert COSMOS_REASON2_8B in text
        assert COSMOS_REASON2_2B in text
        assert STEP_3_7_FLASH in text

    def test_contains_endpoint_config_fields(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        text = resp.json()["rendered_text"]
        # The verification example reads ``{base_url}/models`` with an
        # explicit note that base_url already includes /v1 — rendering
        # ``{base_url}/v1/models`` would double the ``/v1`` segment when
        # an SME pastes a real base_url like ``http://host:8000/v1``.
        # Either rendering surfaces a ``/models`` suffix; this assertion
        # stays generic so future rewordings (e.g. cleaner placeholder
        # syntax) won't regress.
        assert "/models" in text
        assert "/v1/v1/models" not in text, (
            "doubled /v1/v1/ regressed; see services/nim_setup_generator.py"
        )
        assert "base URL" in text.lower() or "base_url" in text.lower()

    def test_contains_gpu_requirements(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        text = resp.json()["rendered_text"]
        # The 8B memory requirement figure (NIM_GPU_MEMORY_8B_BF16_GB) must
        # render — a bare "GPU" heading always appears, so assert the number.
        assert "at least 56 GB" in text

    def test_technical_requirements_structure(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        tech = resp.json()["technical_requirements"]
        assert "endpoint_config" in tech
        assert "target_models" in tech
        assert "gpu_requirements" in tech
        assert "verification_endpoint" in tech
        assert len(tech["target_models"]) == len(SEEDED_MODEL_CATALOG)

    def test_request_sets_clear_scope_and_secret_handling(self, test_app_client):
        project_id = _create_project(test_app_client)

        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        assert resp.status_code == 200
        data = resp.json()
        text = data["rendered_text"]

        assert "only one is needed" in text
        assert "do not send it back by email" in text
        assert "does not send per-request credentials" in text
        assert "auth_header" not in data["technical_requirements"]["endpoint_config"]


# ── POST /v1/nim/test_ngc_credential ────────────────────────────────────────
# ── POST /v1/nim/test_nvidia_credential ─────────────────────────────────────


class TestKeyCredentialEndpoints:
    """Coverage for the dedicated NGC + NVIDIA credential probe endpoints.

    Both endpoints accept two modes:
      * non-empty ``credential_transient`` → probe THAT key.
      * empty/null → probe the effective key via runtime_secrets.

    Both probes route through the canonical ``resilient_request`` client,
    which instantiates ``httpx.AsyncClient`` at call time — so patching
    the class with a MockTransport-injecting factory intercepts them.
    """

    @staticmethod
    def _patched_client(handler):
        """Return an AsyncClient-compatible class whose instances route
        every request through ``handler`` via MockTransport. The probe
        functions call ``async with httpx.AsyncClient(timeout=...) as
        client:`` with no transport kwarg, so we replace the class
        itself in the router module's import namespace.
        """
        import httpx

        real_async_client = httpx.AsyncClient

        class _Factory:
            def __init__(self, *args, **kwargs):
                # Strip transport=... if any caller ever passes it; we
                # always inject MockTransport.
                kwargs.pop("transport", None)
                self._real = real_async_client(
                    transport=httpx.MockTransport(handler),
                    **kwargs,
                )

            async def __aenter__(self):
                return self._real

            async def __aexit__(self, *exc):
                await self._real.__aexit__(*exc)

        return _Factory

    # ── NGC ────────────────────────────────────────────────────────────

    def test_ngc_explicit_credential_success(self, test_app_client):
        import httpx

        def handler(_request):
            return httpx.Response(200, json={"token": "abc"})

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_ngc_credential",
                json={"credential_transient": "ngck-good"},
            )
        assert resp.status_code == 200
        assert resp.json() == {"success": True, "models": None, "error": None}

    def test_ngc_explicit_credential_403_wrong_scope(self, test_app_client):
        """An ``nvapi-`` build.nvidia.com Personal Key pasted into the
        NGC field: nvcr.io returns 403/DENIED."""
        import httpx

        def handler(_request):
            return httpx.Response(403, json={"errors": [{"code": "DENIED"}]})

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_ngc_credential",
                json={"credential_transient": "nvapi-wrong-scope"},
            )
        body = resp.json()
        assert body["success"] is False
        assert "NGC Catalog" in body["error"]
        assert "Private Registry" in body["error"]

    def test_ngc_401_garbage_or_empty(self, test_app_client):
        import httpx

        def handler(_request):
            return httpx.Response(401)

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_ngc_credential",
                json={"credential_transient": "garbage"},
            )
        body = resp.json()
        assert body["success"] is False
        assert "not recognised" in body["error"].lower()

    def test_ngc_probe_carries_source_header_and_basic_auth(self, test_app_client):
        """The probe must go through the canonical client: Blueprint
        source header present (usage-tracking convention) and registry
        Basic auth built from $oauthtoken:key."""
        import base64

        import httpx

        seen: dict[str, str] = {}

        def handler(request):
            seen.update(request.headers)
            return httpx.Response(200, json={"token": "abc"})

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            test_app_client.post(
                "/v1/nim/test_ngc_credential",
                json={"credential_transient": "ngck-good"},
            )
        assert seen.get("source") == "vlm-feedback-loop"
        expected = base64.b64encode(b"$oauthtoken:ngck-good").decode()
        assert seen.get("authorization") == f"Basic {expected}"

    def test_ngc_effective_mode_no_credential_in_request(self, test_app_client):
        """``credential_transient`` absent ⇒ backend resolves from runtime
        secrets via ``get_effective_secret``. With no NVIDIA env key in
        the test fixture, this returns the 'not configured' friendly
        message before any HTTP call fires."""
        resp = test_app_client.post("/v1/nim/test_ngc_credential", json={})
        body = resp.json()
        assert body["success"] is False
        assert "No NGC API key is configured" in body["error"]

    # ── NVIDIA ─────────────────────────────────────────────────────────

    def test_nvidia_probe_carries_source_and_bearer_headers(self, test_app_client):
        """Canonical-client contract: source header + Bearer auth built by
        nim_client.build_auth_headers, not a hand-rolled header dict."""
        import httpx

        seen: dict[str, str] = {}

        def handler(request):
            seen.update(request.headers)
            return httpx.Response(200, json={"choices": []})

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            test_app_client.post(
                "/v1/nim/test_nvidia_credential",
                json={"credential_transient": "nvapi-good"},
            )
        assert seen.get("source") == "vlm-feedback-loop"
        assert seen.get("authorization") == "Bearer nvapi-good"

    def test_nvidia_explicit_credential_success(self, test_app_client):
        """Probes the actually-gated POST /v1/chat/completions path. A 200
        response means the bearer was accepted."""
        import httpx

        def handler(_request):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "ok"}}]}
            )

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_nvidia_credential",
                json={"credential_transient": "nvapi-good"},
            )
        body = resp.json()
        assert body["success"] is True

    def test_nvidia_400_after_auth_means_valid_key_and_probe_never_queues(
        self, test_app_client
    ):
        """The probe sends an intentionally invalid payload (empty
        ``messages``) so it dies at request validation instead of waiting
        in the model's inference queue: a 400 response therefore means the
        bearer cleared auth and the key is VALID. Pins both halves — the
        classification (400 ⇒ success) and the queue-free payload (a
        refactor restoring real messages would silently reintroduce
        20–52 s probe latency under hosted-queue load)."""
        import json

        import httpx

        bodies: list[dict] = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(
                400,
                json={"error": {"message": "list index out of range", "code": 400}},
            )

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_nvidia_credential",
                json={"credential_transient": "nvapi-good"},
            )
        body = resp.json()
        assert body["success"] is True
        assert bodies and bodies[0]["messages"] == []

    def test_nvidia_explicit_credential_403_rejected(self, test_app_client):
        """build.nvidia.com returns 403 Forbidden / Authorization failed
        for a bad bearer on chat/completions."""
        import httpx

        def handler(_request):
            return httpx.Response(403, json={"status": 403, "title": "Forbidden"})

        with patch("httpx.AsyncClient", self._patched_client(handler)):
            resp = test_app_client.post(
                "/v1/nim/test_nvidia_credential",
                json={"credential_transient": "nvapi-bad"},
            )
        body = resp.json()
        assert body["success"] is False
        assert "rejected by build.nvidia.com" in body["error"]

    def test_nvidia_effective_mode_no_credential_in_request(self, test_app_client):
        resp = test_app_client.post("/v1/nim/test_nvidia_credential", json={})
        body = resp.json()
        assert body["success"] is False
        assert "No NVIDIA API key is configured" in body["error"]
