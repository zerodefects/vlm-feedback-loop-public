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
    MISTRAL_LARGE_3,
)
from vlm_feedback_loop.services.environment import GpuInfo
from vlm_feedback_loop.services.project_service import SEEDED_MODEL_CATALOG

# ── Helpers ─────────────────────────────────────────────────────────────────


def _create_project(client: TestClient) -> str:
    """Create a project and return its project_id."""
    return create_project_via_api(client, name="Test Project")["project_id"]


# ── GET /v1/environment ─────────────────────────────────────────────────────


class TestEnvironmentEndpoint:
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
                body={"data": [{"index": 0, "embedding": [0.1]}], "model": "nvclip"},
                error_class=None,
                attempts=1,
            )

            resp = test_app_client.post(
                "/v1/nim/test_connection",
                json={
                    "base_url": "https://integrate.api.nvidia.com/v1",
                    "auth_mode": "bearer",
                    "credential_transient": "nvapi-test",
                    "probe_kind": "embeddings",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

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
        assert MISTRAL_LARGE_3 in text

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
