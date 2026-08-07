# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for local NIM deployment router endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from conftest import create_project_via_api
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.model_catalog_constants import EMBEDDING_NIM_GPU_MIN_GB
from vlm_feedback_loop.services.local_nim_service import (
    ActiveNimResident,
    GpuExhaustedError,
    PreflightCheckResult,
    PreflightResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _create_project(client) -> str:
    """Create a project and return its project_id."""
    return create_project_via_api(client, name="Test NIM Deploy")["project_id"]


def _mock_deployment():
    """Create a mock LocalNimDeployment object."""
    dep = MagicMock()
    dep.local_nim_deployment_id = generate_uuid4()
    dep.project_id = "test-project"
    dep.model_config_id = "test-mc"
    dep.role = "teacher"
    dep.nim_container_image = "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0"
    dep.container_name = "vlm-teacher-test1234"
    dep.container_id = "abc123"
    dep.host_port = 8000
    dep.endpoint_url = "http://localhost:8000/v1"
    dep.gpu_assignment = "device=0"
    dep.status = "starting"
    dep.status_reason = None
    dep.activate_on_success = False
    dep.deployed_at = None
    dep.stopped_at = None
    dep.created_at = utc_now()
    # Displacement audit fields default to None on a
    # freshly-created mock (no displacement yet).
    dep.displaced_by_deployment_id = None
    dep.displaced_at = None
    return dep


# ── Preflight Tests ───────────────────────────────────────────────────────────


class TestPreflightEndpoint:
    """POST /projects/{pid}/local_nim/preflight."""

    @pytest.fixture(autouse=True)
    def _patch_service(self, monkeypatch):
        """Mock all service calls to avoid real Docker/subprocess operations."""
        self._mock_preflight = AsyncMock(
            return_value=PreflightResult(
                all_passed=True,
                checks=[
                    PreflightCheckResult("docker", True, "Docker available"),
                    PreflightCheckResult("nvidia_toolkit", True, "Toolkit available"),
                    PreflightCheckResult(
                        "gpu_memory", True, "GPU 0: 80 GB, need >56 GB"
                    ),
                    PreflightCheckResult("ngc_api_key", True, "NGC key configured"),
                    PreflightCheckResult(
                        "model_profile", True, "Compatible profile found"
                    ),
                    PreflightCheckResult("image_pullable", True, "Image accessible"),
                ],
                docker_run_command="docker run -d ...",
            )
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.run_preflight_checks",
            self._mock_preflight,
        )
        self._mock_placement = AsyncMock(return_value="device=0")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            self._mock_placement,
        )

    def test_preflight_returns_check_results(self, test_app_client):
        client = test_app_client
        pid = _create_project(client)

        # Get a model_config_id with local_deploy_metadata
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/preflight",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["all_passed"] is True
        assert len(body["checks"]) == 6
        assert body["docker_run_command"] is not None

    def test_preflight_nonexistent_project(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/nonexistent/local_nim/preflight",
            json={"role": "teacher", "model_config_id": "x"},
        )
        assert resp.status_code == 404
        assert "Project not found" in resp.json()["detail"]


# ── Deploy Tests ──────────────────────────────────────────────────────────────


class TestDeployEndpoint:
    """POST /projects/{pid}/local_nim/deploy."""

    @pytest.fixture(autouse=True)
    def _patch_service(self, monkeypatch):
        dep = _mock_deployment()
        self._mock_deploy = AsyncMock(
            return_value={
                "deployment": dep,
                "preflight": PreflightResult(
                    all_passed=True,
                    checks=[PreflightCheckResult("docker", True, "ok")],
                    docker_run_command="docker run -d ...",
                ),
            }
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.deploy_local_nim",
            self._mock_deploy,
        )
        self._mock_placement = AsyncMock(return_value="device=0")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            self._mock_placement,
        )

    def test_matching_running_teacher_is_reused_without_placement_or_deploy(
        self, test_app_client, monkeypatch
    ):
        """A second project attaches to the identical resident immediately."""

        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )
        dep = _mock_deployment()
        dep.status = "running"
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Owner project",
            deployment_id=dep.local_nim_deployment_id,
            model_config_id="owner-mc",
            role="teacher",
            model_name=teacher_mc["model_name"],
            nim_container_image=teacher_mc["local_deploy_metadata"][
                "nim_container_image"
            ],
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8000/v1",
            host_port=8000,
            status="running",
            nim_model_size=None,
            nim_model_profile=None,
            extra_container_env=(),
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service."
            "reuse_compatible_running_teacher",
            MagicMock(return_value=resident),
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["disposition"] == "reused"
        assert body["deployment"] is None
        assert body["resident"]["project_name"] == "Owner project"
        assert body["preflight"]["checks"][0]["check_name"] == "resident_reused"
        assert self._mock_placement.await_count == 0
        assert self._mock_deploy.await_count == 0

    def test_deploy_returns_deployment_record(self, test_app_client):
        client = test_app_client
        pid = _create_project(client)

        # The allocator may move away from the catalog's preferred port.
        # Both response sections must expose the port actually reserved.
        deployment = self._mock_deploy.return_value["deployment"]
        deployment.host_port = 49500
        deployment.endpoint_url = "http://localhost:49500/v1"

        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "deployment" in body
        assert "preflight" in body
        assert body["deployment"]["role"] == "teacher"
        assert body["deployment"]["host_port"] == 49500
        assert body["preflight"]["resolved_port"] == 49500
        assert self._mock_deploy.await_args.kwargs["background"] is True

    def test_teacher_activation_intent_is_durable_deploy_input(self, test_app_client):
        """NIM Configuration may select the Teacher only after verification."""

        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            config
            for config in configs
            if config.get("local_deploy_metadata") and "8b" in config["model_name"]
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
                "activate_on_success": True,
            },
        )

        assert resp.status_code == 201
        assert self._mock_deploy.await_args.kwargs["activate_on_success"] is True

    def test_reused_teacher_is_selected_only_after_exact_reuse(
        self, test_app_client, monkeypatch
    ):
        """A healthy exact resident can be selected without another deploy."""

        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            config
            for config in configs
            if config.get("local_deploy_metadata") and "8b" in config["model_name"]
        )
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Owner project",
            deployment_id="resident-deployment",
            model_config_id="owner-config",
            role="teacher",
            model_name=teacher_mc["model_name"],
            nim_container_image=teacher_mc["local_deploy_metadata"][
                "nim_container_image"
            ],
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8000/v1",
            host_port=8000,
            status="running",
            nim_model_size=None,
            nim_model_profile=None,
            extra_container_env=(),
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service."
            "reuse_compatible_running_teacher",
            MagicMock(return_value=resident),
        )
        activate = MagicMock(return_value=True)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service."
            "activate_teacher_model_config",
            activate,
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
                "activate_on_success": True,
            },
        )

        assert resp.status_code == 201
        assert resp.json()["disposition"] == "reused"
        activate.assert_called_once()
        assert activate.call_args.args[:2] == (
            pid,
            teacher_mc["model_config_id"],
        )
        assert self._mock_deploy.await_count == 0

    def test_embedding_rejects_teacher_activation_intent(self, test_app_client):
        """An embedding deployment cannot mutate the project's Teacher."""

        pid = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={"role": "embedding", "activate_on_success": True},
        )

        assert resp.status_code == 400
        assert "only for teacher" in resp.json()["detail"]

    def test_deploy_teacher_requires_model_config_id(self, test_app_client):
        client = test_app_client
        pid = _create_project(client)

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={"role": "teacher"},
        )
        assert resp.status_code == 400
        # This test exercises the real service, so the detail pins the
        # correct 400 cause, not just any validation failure.
        assert "model_config_id" in resp.json()["detail"]

    def test_deploy_nonexistent_project(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/nonexistent/local_nim/deploy",
            json={"role": "teacher", "model_config_id": "x"},
        )
        assert resp.status_code == 404
        assert "Project not found" in resp.json()["detail"]


# ── replace_resident behavior ──────────────────────────────────


class TestDeployReplaceResident:
    """One-NIM-per-GPU invariant:
    ``POST /local_nim/deploy`` rejects with ``409 gpu_occupied`` when
    the target GPU has an active resident and the caller did not opt
    into replace semantics. With ``replace_resident=true``, the
    request succeeds and ``deploy_local_nim`` is invoked with
    ``replace_resident=True`` so the service stops the resident
    before docker_run."""

    @pytest.fixture(autouse=True)
    def _patch_service(self, monkeypatch):
        dep = _mock_deployment()
        self._mock_deploy = AsyncMock(
            return_value={
                "deployment": dep,
                "preflight": PreflightResult(
                    all_passed=True,
                    checks=[PreflightCheckResult("docker", True, "ok")],
                    docker_run_command="docker run -d ...",
                ),
                "displaced": [],
            }
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.deploy_local_nim",
            self._mock_deploy,
        )
        # Auto-placer returns device=0 by default; tests mark that
        # device occupied via _set_residents() below.
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            AsyncMock(return_value="device=0"),
        )
        # Empty scan by default — tests override per-test.
        self._mock_scan = MagicMock(return_value={})
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.scan_active_residents_by_device",
            self._mock_scan,
        )
        self._mock_resident_list = MagicMock(return_value=[])
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.list_active_nim_residents",
            self._mock_resident_list,
        )

    def _set_residents(self, device_index: str, deployment_ids: list[str]) -> None:
        """Helper: pretend the workspace has active residents on a
        given device. The router uses the keys of this map to gate."""
        self._mock_scan.return_value = {
            device_index: [("test-project", did) for did in deployment_ids]
        }

    def test_returns_409_gpu_occupied_when_replace_resident_false(
        self, test_app_client
    ):
        """Target GPU has an active resident + caller did not opt in →
        ``409 gpu_occupied`` with a structured error code."""
        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        self._set_residents("0", [generate_uuid4()])

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "gpu_occupied"
        assert "replace_resident=true" in detail["message"]
        # The service-level deploy_local_nim must NOT be invoked when
        # the router gate rejects.
        assert self._mock_deploy.await_count == 0

    def test_409_names_the_blueprint_resident_and_allows_replacement(
        self, test_app_client
    ):
        """Conflict detail identifies the managed NIM the user may replace."""

        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Existing project",
            deployment_id="resident-dep",
            model_config_id="resident-mc",
            role="teacher",
            model_name="nvidia/cosmos3-nano-reasoner",
            nim_container_image="nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8001/v1",
            host_port=8001,
            status="running",
            nim_model_size="nano",
            nim_model_profile="nano-profile",
            extra_container_env=(),
        )
        self._set_residents("0", [resident.deployment_id])
        self._mock_resident_list.return_value = [resident]

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "gpu_occupied"
        assert detail["can_replace"] is True
        assert detail["matches_requested_model"] is False
        assert detail["resident"]["project_name"] == "Existing project"
        assert detail["resident"]["model_name"] == "nvidia/cosmos3-nano-reasoner"

    def test_same_model_still_starting_is_not_offered_for_replacement(
        self, test_app_client
    ):
        """A matching cold-start is a wait/retry state, not a replace prompt."""

        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )
        metadata = teacher_mc["local_deploy_metadata"]
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Existing project",
            deployment_id="resident-dep",
            model_config_id="resident-mc",
            role="teacher",
            model_name=teacher_mc["model_name"],
            nim_container_image=metadata["nim_container_image"],
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8001/v1",
            host_port=8001,
            status="starting",
            nim_model_size=metadata.get("nim_model_size"),
            nim_model_profile=metadata.get("nim_model_profile"),
            extra_container_env=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in metadata.get("extra_container_env", {}).items()
                )
            ),
        )
        self._set_residents("0", [resident.deployment_id])
        self._mock_resident_list.return_value = [resident]

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )

        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "resident_starting"
        assert detail["matches_requested_model"] is True
        assert detail["can_replace"] is False

    def test_replace_resident_true_calls_service_with_replace_flag(
        self, test_app_client
    ):
        """``replace_resident=true`` + occupied GPU → 201; the service
        is invoked with ``replace_resident=True`` so it stops the
        resident before docker_run."""
        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        self._set_residents("0", [generate_uuid4()])

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
                "replace_resident": True,
            },
        )
        assert resp.status_code == 201
        # deploy_local_nim must be invoked with replace_resident=True.
        assert self._mock_deploy.await_count == 1
        call_kwargs = self._mock_deploy.await_args.kwargs
        assert call_kwargs["replace_resident"] is True

    def test_replace_fallback_targets_floor_qualified_device(
        self, test_app_client, monkeypatch
    ):
        """``replace_resident=true`` with every candidate GPU taken or
        below the floor routes through ``resolve_replace_target`` — the
        deploy lands on the device it picks, not a hardcoded device=0."""
        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            AsyncMock(side_effect=GpuExhaustedError("All 2 GPU(s) occupied")),
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_replace_target",
            AsyncMock(return_value="device=1"),
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
                "replace_resident": True,
            },
        )
        assert resp.status_code == 201
        assert self._mock_deploy.await_count == 1
        call_kwargs = self._mock_deploy.await_args.kwargs
        assert call_kwargs["gpu_assignment"] == "device=1"
        assert call_kwargs["replace_resident"] is True

    def test_embedding_replace_below_floor_returns_409_gpu_exhausted(
        self, test_app_client, monkeypatch
    ):
        """``replace_resident=true`` cannot bypass the embedding memory
        floor: when no GPU on the host meets it, the deploy is refused
        with ``409 gpu_exhausted`` instead of landing the embedding NIM
        on a device that cannot hold it."""
        client = test_app_client
        pid = _create_project(client)

        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            AsyncMock(
                side_effect=GpuExhaustedError(
                    "No free GPU meets the 24 GB memory floor"
                )
            ),
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_replace_target",
            AsyncMock(
                side_effect=GpuExhaustedError(
                    "No GPU on this host meets the 24 GB memory floor "
                    "for role=embedding (largest is 16 GB)"
                )
            ),
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={"role": "embedding", "replace_resident": True},
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "gpu_exhausted"
        assert "memory floor" in detail["message"]
        assert self._mock_deploy.await_count == 0

    def test_deploy_passes_memory_floor_to_automatic_placement(
        self, test_app_client, monkeypatch
    ):
        """Automatic placement receives each deployable model's memory floor.

        Without this, a multi-GPU host could choose a free undersized GPU and
        fail only later in preflight even though a qualifying device exists.
        """
        client = test_app_client
        pid = _create_project(client)

        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        placement_spy = AsyncMock(return_value="device=0")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            placement_spy,
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={"role": "embedding"},
        )
        assert resp.status_code == 201
        assert (
            placement_spy.await_args.kwargs["min_gpu_memory_gb"]
            == EMBEDDING_NIM_GPU_MIN_GB
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )
        assert resp.status_code == 201
        assert (
            placement_spy.await_args.kwargs["min_gpu_memory_gb"]
            == (teacher_mc["local_deploy_metadata"]["nim_gpu_memory_minimum_gb"])
        )

    def test_returns_409_gpu_exhausted_on_resolve_failure(
        self, test_app_client, monkeypatch
    ):
        """When the auto-placer raises ``GpuExhaustedError`` (all GPUs
        occupied) and the caller did NOT pass ``replace_resident=true``,
        the router returns ``409 gpu_exhausted`` (not a generic 422)."""
        client = test_app_client
        pid = _create_project(client)
        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        teacher_mc = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "8b" in c["model_name"]
        )

        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.resolve_gpu_placement",
            AsyncMock(
                side_effect=GpuExhaustedError("All 1 GPU(s) occupied"),
            ),
        )

        resp = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": teacher_mc["model_config_id"],
            },
        )
        assert resp.status_code == 409
        detail = resp.json()["detail"]
        assert detail["code"] == "gpu_exhausted"


# ── List / Get Tests ──────────────────────────────────────────────────────────


class TestListGetEndpoints:
    """GET /projects/{pid}/local_nim/deployments."""

    @pytest.fixture(autouse=True)
    def _patch_service(self, monkeypatch):
        dep = _mock_deployment()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.list_local_deployments",
            MagicMock(return_value=[dep]),
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.get_local_deployment",
            MagicMock(return_value=dep),
        )

    def test_list_deployments(self, test_app_client):
        client = test_app_client
        pid = _create_project(client)

        resp = client.get(f"/v1/projects/{pid}/local_nim/deployments")
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert len(body["items"]) == 1

    def test_get_deployment(self, test_app_client):
        client = test_app_client
        pid = _create_project(client)

        resp = client.get(f"/v1/projects/{pid}/local_nim/deployments/some-id")
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "teacher"

    def test_get_nonexistent_deployment(self, test_app_client, monkeypatch):
        client = test_app_client
        pid = _create_project(client)

        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.get_local_deployment",
            MagicMock(return_value=None),
        )

        resp = client.get(f"/v1/projects/{pid}/local_nim/deployments/nonexistent")
        assert resp.status_code == 404
        assert "Deployment not found" in resp.json()["detail"]


# ── Stop Tests ────────────────────────────────────────────────────────────────


class TestStopEndpoint:
    """POST /projects/{pid}/local_nim/deployments/{did}:stop."""

    def test_stop_deployment(self, test_app_client, monkeypatch):
        client = test_app_client
        pid = _create_project(client)

        dep = _mock_deployment()
        dep.status = "stopped"
        dep.stopped_at = utc_now()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.stop_local_nim",
            AsyncMock(return_value=dep),
        )

        resp = client.post(f"/v1/projects/{pid}/local_nim/deployments/some-id:stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_stop_nonexistent(self, test_app_client, monkeypatch):
        client = test_app_client
        pid = _create_project(client)

        monkeypatch.setattr(
            "vlm_feedback_loop.services.local_nim_service.stop_local_nim",
            AsyncMock(return_value=None),
        )

        resp = client.post(f"/v1/projects/{pid}/local_nim/deployments/nonexistent:stop")
        assert resp.status_code == 404
        assert "Deployment not found" in resp.json()["detail"]
