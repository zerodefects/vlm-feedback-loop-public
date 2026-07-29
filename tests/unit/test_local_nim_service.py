# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for local NIM deployment service.

Covers preflight checks, GPU placement, container lifecycle,
health polling, port allocation, restart recovery, and Action Request fallback.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

import vlm_feedback_loop.services.local_nim_service as svc
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.deployment_models import EmbeddingDeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.student_model import StudentModel
from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_REASONER_NIM_IMAGE,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_NIM_IMAGE,
    EMBEDDING_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
)
from vlm_feedback_loop.services.environment import GpuInfo

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_settings(tmp_workspace, patch_config_paths, write_config, write_env):
    """Create a Settings instance with NGC_API_KEY and workspace."""
    write_config()
    write_env({"NGC_API_KEY": "test-ngc-key-123", "NVIDIA_API_KEY": "test-nvidia-key"})
    from vlm_feedback_loop.config import get_settings, init_settings

    init_settings()
    return get_settings()


@pytest.fixture()
def project_with_db(mock_settings):
    """Create a project and return (project_id, engine)."""
    from vlm_feedback_loop.services.project_service import create_project

    project = create_project(
        name="Test Project",
        description=None,
        settings=mock_settings,
    )
    project_id = project.project_id
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(project_id, mock_settings.WORKSPACE_ROOT)
    return project_id, engine


def _make_subprocess_mock(
    *,
    docker_ok: bool = True,
    toolkit_ok: bool = True,
    gpu_csv: str = "NVIDIA A100, 81920",
    ngc_key: bool = True,
    registry_login_ok: bool = True,
    profile_ok: bool = True,
    profile_timeout: bool = False,
    manifest_ok: bool = True,
    docker_run_ok: bool = True,
    container_id: str = "abc123def456",
) -> AsyncMock:
    """Build a ``run_subprocess`` mock that handles multiple Docker commands."""

    async def _mock_subprocess(
        *args, timeout_s=10.0, stdin_input=None, secret_env=None
    ):
        cmd = " ".join(str(a) for a in args)

        if args[0] == "docker" and args[1] == "info":
            if docker_ok:
                return (0, "24.0.7", "")
            return (1, "", "Docker not available")

        # docker login nvcr.io (check 5 — registry_auth). The NGC key is
        # handed over stdin, never argv, so it must never appear in cmd.
        if args[0] == "docker" and args[1] == "login":
            if registry_login_ok:
                return (0, "Login Succeeded", "")
            return (1, "", "unauthorized: authentication required")

        # docker pull (check 6 — image_pullable, runs before list-model-profiles)
        if args[0] == "docker" and args[1] == "pull":
            if manifest_ok:
                return (0, "Status: Image is up to date", "")
            return (1, "", "pull failed: manifest not found")

        # list-model-profiles (check 6) — must be checked before generic
        # --runtime=nvidia handling (toolkit check 2)
        if args[0] == "docker" and "list-model-profiles" in cmd:
            if profile_timeout:
                # run_subprocess returns (-1, "", "...timed out...") and kills
                # only the docker client; the container keeps holding the GPU.
                return (-1, "", f"Command timed out after {timeout_s}s")
            if profile_ok:
                return (0, "Profile: default (runnable)\n", "")
            return (1, "", "No runnable profile found")

        # NVIDIA toolkit check (check 2) — docker run --runtime=nvidia ... nvidia-smi
        if args[0] == "docker" and "--runtime=nvidia" in cmd and "nvidia-smi" in cmd:
            if toolkit_ok:
                return (0, "GPU 0: NVIDIA A100", "")
            return (1, "", "Toolkit not available")

        if args[0] == "nvidia-smi":
            if gpu_csv:
                return (0, gpu_csv, "")
            return (1, "", "nvidia-smi failed")

        if args[0] == "docker" and args[1] == "run" and "-d" in args:
            if docker_run_ok:
                return (0, container_id, "")
            return (1, "", "docker run failed")

        if args[0] == "docker" and args[1] == "rm":
            return (0, "", "")

        if args[0] == "docker" and args[1] == "stop":
            return (0, "", "")

        if args[0] == "docker" and args[1] == "inspect":
            return (0, "running", "")

        return (0, "", "")

    return AsyncMock(side_effect=_mock_subprocess)


def _read_embedding_config(workspace_root: str) -> EmbeddingDeploymentConfig:
    """Read the deployment.db EmbeddingDeploymentConfig singleton (detached)."""
    engine = init_deployment_db(workspace_root)
    with Session(engine) as session:
        config = session.execute(select(EmbeddingDeploymentConfig)).scalar_one()
        session.expunge(config)
        return config


def _docker_calls(subprocess_mock: AsyncMock) -> list[tuple[str, ...]]:
    """All docker argv tuples issued through a run_subprocess mock."""
    return [
        tuple(str(a) for a in c.args)
        for c in subprocess_mock.await_args_list
        if c.args and c.args[0] == "docker"
    ]


# ── Preflight Checks ─────────────────────────────────────────────────────────


class TestPreflightChecks:
    """The six deploy preflight checks."""

    @pytest.fixture(autouse=True)
    def _patch_subprocess(self, monkeypatch):
        self._subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", self._subprocess_mock)
        monkeypatch.setattr(
            "vlm_feedback_loop.services.environment.run_subprocess",
            self._subprocess_mock,
        )
        # Also patch the functions imported from environment into local_nim_service
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )

    @pytest.mark.asyncio
    async def test_all_checks_pass(self, mock_settings):
        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is True
        assert len(result.checks) == 7
        check_names = [c.check_name for c in result.checks]
        assert check_names == [
            "docker",
            "nvidia_toolkit",
            "gpu_memory",
            "ngc_api_key",
            "registry_auth",
            "image_pullable",
            "model_profile",
        ]
        assert all(c.passed for c in result.checks)
        assert result.docker_run_command is not None

    @pytest.mark.asyncio
    async def test_docker_unavailable_fails_first_check(
        self, mock_settings, monkeypatch
    ):
        monkeypatch.setattr(
            svc,
            "check_docker_available",
            AsyncMock(return_value=(False, "Docker not found")),
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        assert len(result.checks) == 1  # short-circuit after check 1
        assert result.checks[0].check_name == "docker"
        assert result.checks[0].passed is False

    @pytest.mark.asyncio
    async def test_nvidia_toolkit_fails(self, mock_settings, monkeypatch):
        monkeypatch.setattr(
            svc,
            "check_nvidia_toolkit",
            AsyncMock(return_value=(False, "Toolkit failed")),
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        assert len(result.checks) == 2  # check 1 passes, check 2 fails
        assert result.checks[1].check_name == "nvidia_toolkit"
        assert result.checks[1].passed is False

    @pytest.mark.asyncio
    async def test_gpu_memory_below_minimum_fails(self, mock_settings, monkeypatch):
        """GPU memory fast-fail: diagnostic includes GPU name and requirement."""
        monkeypatch.setattr(
            svc,
            "run_subprocess",
            _make_subprocess_mock(gpu_csv="NVIDIA A10, 24576"),
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        gpu_check = next(c for c in result.checks if c.check_name == "gpu_memory")
        assert gpu_check.passed is False
        assert "24" in gpu_check.diagnostic
        assert "56" in gpu_check.diagnostic
        assert "A10" in gpu_check.diagnostic

    @pytest.mark.asyncio
    async def test_gpu_compute_capability_below_model_floor_fails(
        self, mock_settings, monkeypatch
    ):
        """A large pre-Hopper GPU cannot pass a cc>=9.0 Omni preflight."""
        monkeypatch.setattr(
            svc,
            "run_subprocess",
            _make_subprocess_mock(gpu_csv="NVIDIA A100, 98304, 8.0"),
        )

        result = await svc.run_preflight_checks(
            nim_container_image=(
                "nvcr.io/nim/nvidia/"
                "nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant"
            ),
            gpu_memory_minimum_gb=80,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
            gpu_compute_capability_minimum=9.0,
        )

        assert result.all_passed is False
        gpu_check = next(c for c in result.checks if c.check_name == "gpu_memory")
        assert gpu_check.passed is False
        assert "compute capability 8" in gpu_check.diagnostic
        assert "need >=9" in gpu_check.diagnostic

    @pytest.mark.asyncio
    async def test_ngc_api_key_missing_fails(self, mock_settings, monkeypatch):
        monkeypatch.setattr(mock_settings, "NGC_API_KEY", None)

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        ngc_check = next(c for c in result.checks if c.check_name == "ngc_api_key")
        assert ngc_check.passed is False

    @pytest.mark.asyncio
    async def test_registry_auth_failure_blocks_pull(self, mock_settings, monkeypatch):
        """A failed nvcr.io docker login fails preflight BEFORE the image pull.

        A configured NGC key is not enough — the docker daemon must be
        authenticated to nvcr.io or the private-image pull is denied. When
        ``docker login`` fails, preflight stops at ``registry_auth`` with an
        actionable diagnostic and never attempts ``docker pull``.
        """
        mock = _make_subprocess_mock(registry_login_ok=False)
        monkeypatch.setattr(svc, "run_subprocess", mock)
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        auth_check = next(c for c in result.checks if c.check_name == "registry_auth")
        assert auth_check.passed is False
        # Short-circuits before the pull — no image_pullable check ran.
        assert all(c.check_name != "image_pullable" for c in result.checks)
        docker_argv = _docker_calls(mock)
        assert not any(a[:2] == ("docker", "pull") for a in docker_argv)

    @pytest.mark.asyncio
    async def test_registry_auth_uses_password_stdin_not_argv(
        self, mock_settings, monkeypatch
    ):
        """The NGC key is handed to ``docker login`` via stdin, never argv.

        Passing a secret on the command line would leak it into ``ps`` /
        process listings; ``--password-stdin`` is the safe convention.
        """
        monkeypatch.setattr(mock_settings, "NGC_API_KEY", "nvapi-super-secret-value")
        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is True
        login_calls = [
            c
            for c in self._subprocess_mock.await_args_list
            if c.args and c.args[:2] == ("docker", "login")
        ]
        assert len(login_calls) == 1
        call = login_calls[0]
        # Username is the fixed $oauthtoken convention; password comes via stdin.
        assert "--password-stdin" in call.args
        assert "$oauthtoken" in call.args
        assert call.kwargs.get("stdin_input") == "nvapi-super-secret-value"
        # The secret must NOT appear anywhere in the argv.
        assert all("nvapi-super-secret-value" not in str(a) for a in call.args)

    @pytest.mark.asyncio
    async def test_model_profile_no_runnable(self, mock_settings, monkeypatch):
        mock = _make_subprocess_mock(profile_ok=False)
        monkeypatch.setattr(svc, "run_subprocess", mock)
        # Re-patch the high-level helpers since autouse fixture already set them
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        profile_check = next(
            c for c in result.checks if c.check_name == "model_profile"
        )
        assert profile_check.passed is False

    @pytest.mark.asyncio
    async def test_model_profile_timeout_is_inconclusive_not_failure(
        self, mock_settings, monkeypatch
    ):
        """list-model-profiles on current VLM NIM images does a full engine
        init and can exceed the probe timeout. A timeout must be treated as
        INCONCLUSIVE (deploy proceeds; serve health poll is authoritative), and
        the probe container must be force-removed so it can't orphan and hold
        the GPU (the bug that caused 'Detected 0 compatible profile(s)' on the
        next deploy)."""
        mock = _make_subprocess_mock(profile_timeout=True)
        monkeypatch.setattr(svc, "run_subprocess", mock)
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )

        result = await svc.run_preflight_checks(
            nim_container_image="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.7.0",
            gpu_memory_minimum_gb=36,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )

        # Timeout is inconclusive, not a failure → deploy proceeds.
        profile_check = next(
            c for c in result.checks if c.check_name == "model_profile"
        )
        assert profile_check.passed is True
        assert "inconclusive" in profile_check.diagnostic.lower()
        assert result.all_passed is True

        # The probe container must be force-removed (orphan-leak guard).
        rm_calls = [
            c
            for c in mock.call_args_list
            if c.args[:3] == ("docker", "rm", "-f")
            and "vlm-preflight-profile-0" in c.args
        ]
        assert rm_calls, "expected 'docker rm -f vlm-preflight-profile-0' cleanup"

        # Some current VLM images perform a full model download/init even for this
        # probe. The disposable probe must populate the same persistent cache
        # as the real deploy and use the same writable host identity.
        profile_call = next(
            call
            for call in mock.call_args_list
            if call.args[:2] == ("docker", "run") and "list-model-profiles" in call.args
        )
        assert "-v" in profile_call.args
        cache_mount = profile_call.args[profile_call.args.index("-v") + 1]
        assert cache_mount.endswith(":/opt/nim/.cache")
        assert profile_call.args[profile_call.args.index("-u") + 1] == str(
            svc.os.getuid()
        )

    @pytest.mark.asyncio
    async def test_model_profile_probe_uses_requested_shared_image_identity(
        self, mock_settings
    ):
        """The shared Cosmos image defaults to Nano, so a Super preflight
        must pass the same size/profile/name selectors as the real deploy."""
        profile = "super-profile-id"
        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
            nim_model_size="super",
            nim_model_profile=profile,
            nim_served_model_name=COSMOS3_SUPER_REASONER,
        )

        assert result.all_passed is True
        profile_call = next(
            call
            for call in self._subprocess_mock.await_args_list
            if call.args
            and call.args[:2] == ("docker", "run")
            and "list-model-profiles" in call.args
        )
        assert "NIM_MODEL_SIZE=super" in profile_call.args
        assert f"NIM_MODEL_PROFILE={profile}" in profile_call.args
        assert f"NIM_SERVED_MODEL_NAME={COSMOS3_SUPER_REASONER}" in profile_call.args
        ngc_index = profile_call.args.index("NGC_API_KEY")
        assert profile_call.args[ngc_index - 1] == "-e"
        assert not any(
            str(token).startswith("NGC_API_KEY=") for token in profile_call.args
        )
        assert "test-ngc-key-123" not in " ".join(map(str, profile_call.args))
        assert profile_call.kwargs["secret_env"] == {"NGC_API_KEY": "test-ngc-key-123"}
        assert result.docker_run_command is not None
        assert "NIM_MODEL_SIZE=super" in result.docker_run_command

    @pytest.mark.asyncio
    async def test_image_not_pullable(self, mock_settings, monkeypatch):
        mock = _make_subprocess_mock(manifest_ok=False)
        monkeypatch.setattr(svc, "run_subprocess", mock)
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        img_check = next(c for c in result.checks if c.check_name == "image_pullable")
        assert img_check.passed is False
        # Image pull fails at check 5, so model_profile (check 6) is never reached
        check_names = [c.check_name for c in result.checks]
        assert "model_profile" not in check_names

    @pytest.mark.asyncio
    async def test_docker_run_command_always_populated(
        self, mock_settings, monkeypatch
    ):
        """docker_run_command is built even when preflight fails."""
        monkeypatch.setattr(
            svc,
            "check_docker_available",
            AsyncMock(return_value=(False, "Docker not found")),
        )

        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert result.all_passed is False
        assert result.docker_run_command is not None
        assert "docker run" in result.docker_run_command

    @pytest.mark.asyncio
    async def test_checks_run_in_order(self, mock_settings, monkeypatch):
        """Verify all 7 checks appear in the correct order when all pass."""
        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        names = [c.check_name for c in result.checks]
        expected = [
            "docker",
            "nvidia_toolkit",
            "gpu_memory",
            "ngc_api_key",
            "registry_auth",
            "image_pullable",
            "model_profile",
        ]
        assert names == expected

    @pytest.mark.asyncio
    async def test_docker_run_command_no_secrets(self, mock_settings, monkeypatch):
        """Preflight display command does NOT contain the actual NGC key."""
        result = await svc.run_preflight_checks(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_memory_minimum_gb=56,
            gpu_assignment="device=0",
            role="teacher",
            settings=mock_settings,
        )
        assert "test-ngc-key-123" not in (result.docker_run_command or "")
        assert "-e NGC_API_KEY" in (result.docker_run_command or "")
        assert "NGC_API_KEY=" not in (result.docker_run_command or "")


# ── Port Allocation ───────────────────────────────────────────────────────────


class TestPortAllocation:
    """Port resolution with fallback to next available."""

    @pytest.mark.asyncio
    async def test_preferred_port_available(self):
        # Use a high port that's very likely available
        port = await svc._resolve_port(49152)
        assert port == 49152

    @pytest.mark.asyncio
    async def test_queued_port_reservation_finds_next(self):
        """A queued container has not bound its port yet, but the next
        background deployment must still avoid that persisted reservation."""
        port = await svc._resolve_port(49160, {49160})
        assert port == 49161

    @pytest.mark.asyncio
    async def test_preferred_port_occupied_finds_next(self):
        import socket

        # Occupy the preferred port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 49200))
        sock.listen(1)
        try:
            port = await svc._resolve_port(49200)
            assert port == 49201
        finally:
            sock.close()

    @pytest.mark.asyncio
    async def test_multiple_ports_occupied(self):
        import socket

        socks = []
        try:
            for p in range(49300, 49303):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", p))
                s.listen(1)
                socks.append(s)

            port = await svc._resolve_port(49300)
            assert port == 49303
        finally:
            for s in socks:
                s.close()


# ── GPU Placement ─────────────────────────────────────────────────────────────


class TestGpuPlacement:
    """GPU placement policy for concurrent local services."""

    @pytest.mark.asyncio
    async def test_teacher_defaults_to_device_0(self, monkeypatch, tmp_workspace):
        # Mock the GPU probe (as the other auto-placement tests do) so the
        # test is hermetic on GPU-less hosts (CI runners have no nvidia-smi).
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="A100", memory_total_mb=81920)]),
        )

        result = await svc.resolve_gpu_placement(
            role="teacher",
            explicit_gpu=None,
            workspace_root=str(tmp_workspace),
        )
        assert result == "device=0"

    @pytest.mark.asyncio
    async def test_explicit_gpu_assignment_honored(self):
        result = await svc.resolve_gpu_placement(
            role="teacher",
            explicit_gpu="device=3",
            workspace_root="/nonexistent",
        )
        assert result == "device=3"

    @pytest.mark.asyncio
    async def test_explicit_same_gpu_allowed(self):
        """Operator can explicitly co-locate on the same GPU."""
        result = await svc.resolve_gpu_placement(
            role="embedding",
            explicit_gpu="device=0",
            workspace_root="/nonexistent",
        )
        assert result == "device=0"

    @pytest.mark.asyncio
    async def test_embedding_multi_gpu_uses_different(self, monkeypatch, tmp_workspace):
        """On multi-GPU host, embedding uses a different GPU than teacher."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        result = await svc.resolve_gpu_placement(
            role="embedding",
            explicit_gpu=None,
            workspace_root=str(tmp_workspace),
        )
        # No teacher deployments → should pick device=0 (first unused)
        assert result == "device=0"

    @pytest.mark.asyncio
    async def test_multi_gpu_deterministic_lowest_free_index(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """Multi-GPU placement is
        deterministic — a new deployment claims the lowest-indexed
        GPU whose ``LocalNimDeployment`` rows are all terminal.
        Teacher on device=0 → embedding goes to device=1, not
        device=0 again."""
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id="t-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-x",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        result = await svc.resolve_gpu_placement(
            role="embedding",
            explicit_gpu=None,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )
        assert result == "device=1"

    @pytest.mark.asyncio
    async def test_multi_gpu_blocks_on_any_active_role(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """The residence scan looks at ALL
        roles (teacher / embedding / student), not just teacher. An
        active embedding on device=0 blocks Teacher placement on
        device=0 just as a Teacher would block embedding placement.
        Catches a regression where the scan filters to
        ``role=='teacher'`` only."""
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id="emb-mc",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-x",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        result = await svc.resolve_gpu_placement(
            role="teacher",
            explicit_gpu=None,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )
        assert result == "device=1"

    @pytest.mark.asyncio
    async def test_single_gpu_all_occupied_raises_gpu_exhausted(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """On a
        single-GPU host with a resident NIM, auto-placement raises
        ``GpuExhaustedError`` (not the legacy ``ValueError``). The
        caller decides whether to opt into replace semantics
        (``replace_resident=true`` / Student lifecycle path)."""
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        # Add a running teacher deployment on device=0
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=generate_uuid4(),
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="running",
            )
            session.add(dep)
            session.commit()

        with pytest.raises(svc.GpuExhaustedError, match="one-NIM-per-GPU"):
            await svc.resolve_gpu_placement(
                role="embedding",
                explicit_gpu=None,
                workspace_root=mock_settings.WORKSPACE_ROOT,
            )

    @pytest.mark.asyncio
    async def test_embedding_floor_skips_free_gpu_below_minimum(
        self, monkeypatch, tmp_workspace
    ):
        """With a memory floor, the auto-placer skips a free GPU below it
        instead of blindly taking the lowest index: on a 16 GB + 80 GB
        host the embedding NIM lands on the 80 GB device, never the
        16 GB one it would OOM on."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="L4", memory_total_mb=16384),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        result = await svc.resolve_gpu_placement(
            role="embedding",
            explicit_gpu=None,
            workspace_root=str(tmp_workspace),
            min_gpu_memory_gb=24,
        )
        assert result == "device=1"

    @pytest.mark.asyncio
    async def test_embedding_floor_refuses_when_only_small_gpu_free(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """On an 80 GB + 16 GB host with the Teacher holding the 80 GB
        device, the only free GPU is below the embedding floor — the
        placer refuses with an actionable reason that states the real
        condition: the floor, and that the host's largest GPU meets it
        but is occupied (so replace semantics ARE the fix here), rather
        than starting a deploy that cannot fit."""
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="L4", memory_total_mb=16384),
                ]
            ),
        )

        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id="t-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-x",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        with pytest.raises(svc.GpuExhaustedError, match="memory floor") as exc_info:
            await svc.resolve_gpu_placement(
                role="embedding",
                explicit_gpu=None,
                workspace_root=mock_settings.WORKSPACE_ROOT,
                min_gpu_memory_gb=24,
            )
        message = str(exc_info.value)
        assert "24 GB" in message
        assert "80 GB" in message
        assert "replace_resident=true" in message

    @pytest.mark.asyncio
    async def test_embedding_floor_refusal_without_replace_hint_when_host_too_small(
        self, monkeypatch, tmp_workspace
    ):
        """When NO GPU on the host meets the floor, the refusal must not
        suggest replace semantics (displacing a resident cannot grow the
        GPU) — it points at a larger host or the hosted provider."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="L4", memory_total_mb=16384)]),
        )

        with pytest.raises(svc.GpuExhaustedError, match="memory floor") as exc_info:
            await svc.resolve_gpu_placement(
                role="embedding",
                explicit_gpu=None,
                workspace_root=str(tmp_workspace),
                min_gpu_memory_gb=24,
            )
        message = str(exc_info.value)
        assert "16 GB" in message
        assert "replace_resident" not in message
        assert "NVIDIA_API_KEY" in message

    @pytest.mark.asyncio
    async def test_teacher_placement_applies_no_memory_floor(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """Teacher placement semantics are unchanged: with no floor
        passed, the placer hands the Teacher the free 16 GB device —
        memory fit for Teacher deploys stays preflight's job on the
        resolved device."""
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="L4", memory_total_mb=16384),
                ]
            ),
        )

        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id="t-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-x",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        result = await svc.resolve_gpu_placement(
            role="teacher",
            explicit_gpu=None,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )
        assert result == "device=1"

    @pytest.mark.asyncio
    async def test_skips_corrupt_project_in_gpu_scan(
        self, monkeypatch, tmp_workspace, caplog
    ):
        """A single corrupt project DB MUST NOT block GPU placement
        for a healthy ``:deploy_nim``. The cross-project scan inside
        ``resolve_gpu_placement`` opens every project's DB; if one
        raises ``DatabaseMigrationError`` the scan must skip that
        project with a warning instead of failing the whole placement.
        """
        import logging

        from vlm_feedback_loop.db.engine import DatabaseMigrationError

        # Monkeypatch GPU inventory to two GPUs (so the scan is reached).
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="A100", memory_total_mb=81920),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        # Stage two project directories in tmp_workspace.
        broken_id = "00000000-0000-0000-0000-bad000000bad"
        healthy_id = "11111111-1111-1111-1111-111111111111"
        for pid in (broken_id, healthy_id):
            (tmp_workspace / "projects" / pid).mkdir(parents=True, exist_ok=True)
            (tmp_workspace / "projects" / pid / "project.db").touch()

        # Monkeypatch ``get_project_engine`` inside the service to raise
        # for the broken project. The healthy one returns None (no
        # rows), which short-circuits the per-project block cleanly.
        def _fake_get(project_id, _wr):
            if project_id == broken_id:
                raise DatabaseMigrationError(
                    "Migration failed for "
                    f"{tmp_workspace / 'projects' / project_id / 'project.db'}. "
                    "Error: (sqlite3.OperationalError) table projects already exists"
                )
            return None

        monkeypatch.setattr(svc, "get_project_engine", _fake_get)

        with caplog.at_level(
            logging.WARNING, logger="vlm_feedback_loop.services.local_nim_service"
        ):
            result = await svc.resolve_gpu_placement(
                role="embedding",
                explicit_gpu=None,
                workspace_root=str(tmp_workspace),
            )

        # MUST NOT raise; healthy device picked.
        assert result == "device=0"
        # Warning naming the broken project_id + exception class.
        warnings = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "Skipping project" in r.getMessage()
            and "GPU-placement scan" in r.getMessage()
        ]
        assert any(broken_id in m for m in warnings), (
            f"expected warning citing broken project {broken_id}; got: {warnings}"
        )
        assert any("DatabaseMigrationError" in m for m in warnings), (
            f"expected exception class name in warning; got: {warnings}"
        )


# ── Replace-target resolution ────────────────────────────


class TestResolveReplaceTarget:
    """``resolve_replace_target`` picks the device a replace_resident
    deploy displaces onto — occupancy no longer disqualifies, but the
    memory floor still does."""

    @pytest.mark.asyncio
    async def test_embedding_replace_target_honors_floor(self, monkeypatch):
        """On a 16 GB + 80 GB host the embedding replace target is the
        80 GB device — replace semantics must not land the embedding
        NIM on a device below its floor just because it is device 0."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="L4", memory_total_mb=16384),
                    GpuInfo(name="A100", memory_total_mb=81920),
                ]
            ),
        )

        result = await svc.resolve_replace_target(
            role="embedding", min_gpu_memory_gb=24
        )
        assert result == "device=1"

    @pytest.mark.asyncio
    async def test_embedding_replace_refused_when_no_gpu_meets_floor(self, monkeypatch):
        """A single-16 GB-GPU host refuses an embedding replace deploy:
        displacing the resident cannot grow the GPU, so the caller gets
        the same actionable floor error instead of a doomed deploy."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="L4", memory_total_mb=16384)]),
        )

        with pytest.raises(svc.GpuExhaustedError, match="memory floor") as exc_info:
            await svc.resolve_replace_target(role="embedding", min_gpu_memory_gb=24)
        message = str(exc_info.value)
        assert "24 GB" in message
        assert "16 GB" in message

    @pytest.mark.asyncio
    async def test_no_floor_returns_device_0(self, monkeypatch):
        """Teacher/Student callers pass no floor — the replace target is
        device=0, the Student lifecycle's single-GPU fall-back."""
        monkeypatch.setattr(
            svc,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="L4", memory_total_mb=16384)]),
        )

        result = await svc.resolve_replace_target(role="teacher")
        assert result == "device=0"


# ── Stop GPU residents ────────────────────────────


class TestStopGpuResidents:
    """``stop_gpu_residents`` enforces the one-NIM-per-GPU invariant by
    stopping active deployments on a target GPU before a new NIM
    starts. Diagnostic fields (``displaced_by_deployment_id``,
    ``displaced_at``) are persisted on each displaced row."""

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_empty_list(self, tmp_workspace):
        """Idempotent: no residents to stop → no-op returning []."""
        result = await svc.stop_gpu_residents(
            workspace_root=str(tmp_workspace),
            device_index="device=0",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_stops_active_resident_and_persists_displacement(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """A running deployment on the target GPU is transitioned to
        stopped + ``displaced_by_deployment_id`` is stamped + the
        underlying ``docker stop`` is invoked exactly once."""
        project_id, engine = project_with_db

        subprocess_mock = AsyncMock(return_value=(0, "", ""))
        monkeypatch.setattr(svc, "run_subprocess", subprocess_mock)

        resident_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=resident_id,
                    project_id=project_id,
                    model_config_id="t-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-victim",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        new_id = generate_uuid4()
        stopped = await svc.stop_gpu_residents(
            workspace_root=mock_settings.WORKSPACE_ROOT,
            device_index="device=0",
            displaced_by_deployment_id=new_id,
        )

        assert len(stopped) == 1
        assert stopped[0].local_nim_deployment_id == resident_id
        assert stopped[0].status == "stopped"
        assert stopped[0].displaced_by_deployment_id == new_id
        assert stopped[0].displaced_at is not None
        assert stopped[0].status_reason == "displaced_by_replace"

        # ``docker stop`` + ``docker rm`` were invoked via run_subprocess.
        assert subprocess_mock.await_count >= 1

    @pytest.mark.asyncio
    async def test_skips_residents_on_other_devices(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """Only deployments on the requested device are stopped. A
        deployment on device=1 is untouched when stop_gpu_residents
        targets device=0 (the multi-GPU isolation invariant)."""
        project_id, engine = project_with_db

        monkeypatch.setattr(svc, "run_subprocess", AsyncMock(return_value=(0, "", "")))

        keeper_id = generate_uuid4()
        victim_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=keeper_id,
                    project_id=project_id,
                    model_config_id="t-mc",
                    role="teacher",
                    nim_container_image="img",
                    container_name="vlm-teacher-keeper",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=1",
                    status="running",
                )
            )
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=victim_id,
                    project_id=project_id,
                    model_config_id="e-mc",
                    role="embedding",
                    nim_container_image="img",
                    container_name="vlm-embedding-victim",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        stopped = await svc.stop_gpu_residents(
            workspace_root=mock_settings.WORKSPACE_ROOT,
            device_index="device=0",
            displaced_by_deployment_id=generate_uuid4(),
        )

        stopped_ids = {d.local_nim_deployment_id for d in stopped}
        assert stopped_ids == {victim_id}, (
            "device=1 keeper must NOT be touched by a device=0 acquire"
        )

        with Session(engine) as session:
            keeper = session.get(LocalNimDeployment, keeper_id)
            assert keeper is not None
            assert keeper.status == "running"
            assert keeper.displaced_by_deployment_id is None

    @pytest.mark.asyncio
    async def test_displaced_embedding_resident_resets_config(
        self, monkeypatch, project_with_db, mock_settings
    ):
        """Displacement funnels through ``stop_local_nim``, so an
        embedding NIM displaced by a replace deploy resets the
        EmbeddingDeploymentConfig singleton — the environment response
        must not keep claiming a self-hosted provider whose container a
        Teacher just evicted."""
        project_id, engine = project_with_db
        monkeypatch.setattr(svc, "run_subprocess", AsyncMock(return_value=(0, "", "")))

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=generate_uuid4(),
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-victim",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        stopped = await svc.stop_gpu_residents(
            workspace_root=mock_settings.WORKSPACE_ROOT,
            device_index="device=0",
            displaced_by_deployment_id=generate_uuid4(),
        )

        assert len(stopped) == 1
        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "none"
        assert config.endpoint_url is None


# ── Deploy ────────────────────────────────────────────────────────────────────


class TestDeployLocalNim:
    """Container deployment lifecycle tests."""

    @pytest.fixture(autouse=True)
    def _patch_all(self, monkeypatch):
        self._subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", self._subprocess_mock)
        monkeypatch.setattr(
            svc, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        # Mock port resolution to use a high port
        monkeypatch.setattr(svc, "_resolve_port", AsyncMock(return_value=49500))

        # Mock background manager to avoid real task registration.
        # Close the coroutine arg to prevent RuntimeWarning.
        def _mock_register(_task_id: str, coro: object) -> None:
            if hasattr(coro, "close"):
                coro.close()

        monkeypatch.setattr(svc.background_manager, "register", _mock_register)

    @pytest.mark.asyncio
    async def test_successful_deploy_creates_record(
        self, project_with_db, mock_settings
    ):
        project_id, engine = project_with_db

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        dep = result["deployment"]
        assert dep.status == "starting"
        assert dep.project_id == project_id
        assert dep.role == "teacher"
        assert dep.gpu_assignment == "device=0"
        assert dep.container_id is not None

        # Verify record persisted in DB
        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep.local_nim_deployment_id)
            assert db_dep is not None
            assert db_dep.status == "starting"

    @pytest.mark.asyncio
    async def test_preflight_failure_creates_failed_record(
        self, project_with_db, mock_settings, monkeypatch
    ):
        project_id, engine = project_with_db
        monkeypatch.setattr(
            svc,
            "check_docker_available",
            AsyncMock(return_value=(False, "Docker not found")),
        )

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        dep = result["deployment"]
        assert dep.status == "failed"
        assert dep.status_reason is not None
        assert result["preflight"].all_passed is False

    @pytest.mark.asyncio
    async def test_background_deploy_returns_before_preflight_and_persists_failure(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """The HTTP-facing mode persists ``starting`` immediately; slow
        preflight runs only when the tracked background coroutine executes."""
        project_id, engine = project_with_db
        captured: dict[str, object] = {}

        def capture_register(task_id: str, coro: object) -> None:
            captured["task_id"] = task_id
            captured["coro"] = coro

        preflight = AsyncMock(return_value=(False, "Docker not found"))
        monkeypatch.setattr(svc.background_manager, "register", capture_register)
        monkeypatch.setattr(svc, "check_docker_available", preflight)

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
            background=True,
        )

        deployment_id = result["deployment"].local_nim_deployment_id
        assert result["deployment"].status == "starting"
        assert result["preflight"].checks[0].check_name == "deployment_queued"
        assert preflight.await_count == 0
        assert captured["task_id"] == f"local-nim-deploy-{deployment_id}"

        coro = captured["coro"]
        assert hasattr(coro, "__await__")
        await coro

        assert preflight.await_count == 1
        with Session(engine) as session:
            failed = session.get(LocalNimDeployment, deployment_id)
            assert failed is not None
            assert failed.status == "failed"
            assert "Docker not found" in (failed.status_reason or "")

    @pytest.mark.asyncio
    async def test_teacher_uses_uid_flag(self, project_with_db, mock_settings):
        """Teacher docker run includes -u flag."""
        project_id, _ = project_with_db
        calls = []
        original = self._subprocess_mock.side_effect

        async def capture(*args, **kwargs):
            calls.append(args)
            return await original(*args, **kwargs)

        self._subprocess_mock.side_effect = capture

        await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        # Find the docker run call
        docker_run_calls = [
            c
            for c in calls
            if len(c) > 2 and c[0] == "docker" and c[1] == "run" and "-d" in c
        ]
        assert len(docker_run_calls) >= 1
        docker_args = docker_run_calls[0]
        assert "-u" in docker_args

    @pytest.mark.asyncio
    async def test_embedding_docker_run_includes_uid_flag(
        self, project_with_db, mock_settings
    ):
        """Embedding NIM docker run MUST include -u $(id -u) so the
        container writes the model cache as the host user. Without it
        the container falls back to a hardcoded internal UID
        (1000:1000 on NeMo Retriever VL 1B v2) and ``Permission denied
        (os error 13)`` on the cache mount kills the model-weight
        download mid-startup.
        """
        project_id, _ = project_with_db
        calls = []
        original = self._subprocess_mock.side_effect

        async def capture(*args, **kwargs):
            calls.append(args)
            return await original(*args, **kwargs)

        self._subprocess_mock.side_effect = capture

        await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="embedding",
            role="embedding",
            nim_container_image=EMBEDDING_NIM_IMAGE,
            gpu_assignment="device=1",
            gpu_memory_minimum_gb=24,
            preferred_port=8001,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        docker_run_calls = [
            c
            for c in calls
            if len(c) > 2 and c[0] == "docker" and c[1] == "run" and "-d" in c
        ]
        assert len(docker_run_calls) >= 1
        docker_args = docker_run_calls[0]
        assert "-u" in docker_args
        uid_idx = docker_args.index("-u")
        # Value is the host UID as a string (whatever os.getuid() returned).
        assert docker_args[uid_idx + 1].isdigit()

    @pytest.mark.asyncio
    async def test_docker_run_args_contain_required_flags(
        self, project_with_db, mock_settings
    ):
        """Real deploy forwards NGC by name through the Docker child env."""
        project_id, _ = project_with_db
        sentinel = "SENTINEL_REAL_DEPLOY_NGC"
        mock_settings.NGC_API_KEY = sentinel

        await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        docker_run_calls = [
            c
            for c in self._subprocess_mock.await_args_list
            if len(c.args) > 2
            and c.args[0] == "docker"
            and c.args[1] == "run"
            and "-d" in c.args
        ]
        assert len(docker_run_calls) >= 1
        docker_call = docker_run_calls[0]
        docker_args = docker_call.args
        args_str = " ".join(str(a) for a in docker_args)
        assert "--runtime=nvidia" in args_str
        assert "--gpus" in args_str
        assert "--shm-size=32GB" in args_str
        assert "--name" in args_str
        assert "-d" in args_str
        ngc_index = docker_args.index("NGC_API_KEY")
        assert docker_args[ngc_index - 1] == "-e"
        assert "NGC_API_KEY=" not in args_str
        assert sentinel not in args_str
        assert docker_call.kwargs["secret_env"] == {"NGC_API_KEY": sentinel}
        assert "/opt/nim/.cache" in args_str

    @pytest.mark.asyncio
    async def test_extra_container_env_reaches_docker_run(
        self, project_with_db, mock_settings
    ):
        """An operator PATCHing extra_container_env onto the ModelConfig's
        local_deploy_metadata must see those vars on the actual docker run —
        the in-product path for live-validated NIM remediations."""
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        project_id, engine = project_with_db
        with Session(engine) as session:
            mc = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_name == "nvidia/cosmos-reason2-2b")
                .first()
            )
            mcid = mc.model_config_id
            mc.local_deploy_metadata = {
                "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                "nim_gpu_memory_minimum_gb": 36,
                "extra_container_env": {"NIM_DISABLE_CUDA_GRAPH": "1"},
            }
            session.commit()

        await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id=mcid,
            role="teacher",
            nim_container_image="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=36,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        docker_run_calls = [
            c
            for c in _docker_calls(self._subprocess_mock)
            if c[1] == "run" and "-d" in c
        ]
        assert len(docker_run_calls) >= 1
        assert "NIM_DISABLE_CUDA_GRAPH=1" in docker_run_calls[0]

    @pytest.mark.asyncio
    async def test_container_name_format(self, project_with_db, mock_settings):
        """Container name follows vlm-{role}-{project_id[:8]} format."""
        project_id, _ = project_with_db

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        dep = result["deployment"]
        assert dep.container_name == f"vlm-teacher-{project_id[:8]}"

    @pytest.mark.asyncio
    async def test_endpoint_url_uses_resolved_port(
        self, project_with_db, mock_settings
    ):
        """Endpoint URL uses the resolved port (not the preferred)."""
        project_id, _ = project_with_db

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        dep = result["deployment"]
        assert dep.endpoint_url == "http://localhost:49500/v1"
        assert dep.host_port == 49500


# ── Health Polling ────────────────────────────────────────────────────────────


class TestHealthPolling:
    """Health polling and auto-registration tests."""

    @pytest.mark.asyncio
    async def test_healthy_on_first_poll(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Immediate healthy response updates status to 'running'."""
        project_id, engine = project_with_db

        # Create a deployment record
        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="starting",
            )
            session.add(dep)
            session.commit()

        # Mock healthy response
        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        # Served-model verification is exercised by its own dedicated tests;
        # here we patch it to pass so the test stays host-cache-independent
        # and focuses on the running-transition behavior.
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id")),
        )

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8000/v1",
            timeout_s=600,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"
            assert db_dep.deployed_at is not None

    @pytest.mark.asyncio
    async def test_timeout_marks_failed(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """All polls fail while the container stays alive → 'failed' with
        the timeout reason (container death has its own immediate path),
        AND the still-running container is stopped and removed: a failed
        row is invisible to the one-NIM-per-GPU placement scan, so its
        container must not keep holding the GPU's VRAM and port."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="starting",
            )
            session.add(dep)
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="timeout", attempts=1
                )
            ),
        )
        # Container stays alive throughout — pins that liveness checking
        # does not short-circuit the timeout path.
        subprocess_mock = AsyncMock(return_value=(0, "true 0", ""))
        monkeypatch.setattr(svc, "run_subprocess", subprocess_mock)
        # Use a very short timeout for testing
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8000/v1",
            timeout_s=5,  # Very short for test
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "timed out" in db_dep.status_reason
        docker_calls = _docker_calls(subprocess_mock)
        assert ("docker", "stop", "vlm-teacher-test") in docker_calls
        assert ("docker", "rm", "-f", "vlm-teacher-test") in docker_calls

    @pytest.mark.asyncio
    async def test_container_death_fails_deployment_immediately(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A container that exits during startup fails the deployment at
        once — with the exit code and a log tail in the reason — instead
        of holding 'starting' for the full NIM_STARTUP_TIMEOUT_S with the
        GPU marked occupied and no diagnostic (observed live: a NIM whose
        selected profile could not build its attention kernels exited 0
        while the deployment sat in 'starting')."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-dead",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="connect", attempts=1
                )
            ),
        )

        async def fake_subprocess(*args, **kwargs):
            if args[1] == "inspect":
                return (0, "false 1", "")
            if args[1] == "logs":
                return (0, "", "RuntimeError: Engine core initialization failed.")
            return (0, "", "")

        monkeypatch.setattr(svc, "run_subprocess", fake_subprocess)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        # Long timeout on purpose: the death check, not the timeout, must
        # end the loop. wait_for guards against regression to a hang.
        await asyncio.wait_for(
            svc._poll_health(
                deployment_id=dep_id,
                endpoint_url="http://localhost:8000/v1",
                timeout_s=1200,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                workspace_root=mock_settings.WORKSPACE_ROOT,
                settings=mock_settings,
            ),
            timeout=10,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "exited during startup (exit code 1)" in db_dep.status_reason
            assert "Engine core initialization failed" in db_dep.status_reason

    @pytest.mark.asyncio
    async def test_container_removed_fails_deployment_immediately(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A container removed out-of-band (docker rm) during startup also
        fails the deployment immediately rather than timing out."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-gone",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="connect", attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "run_subprocess",
            AsyncMock(return_value=(1, "", "Error: No such object: vlm-teacher-gone")),
        )
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        await asyncio.wait_for(
            svc._poll_health(
                deployment_id=dep_id,
                endpoint_url="http://localhost:8000/v1",
                timeout_s=1200,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                workspace_root=mock_settings.WORKSPACE_ROOT,
                settings=mock_settings,
            ),
            timeout=10,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "removed during startup" in db_dep.status_reason

    @pytest.mark.asyncio
    async def test_embedding_healthy_stamps_config_and_resweeps(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A healthy embedding NIM stamps the EmbeddingDeploymentConfig
        singleton (provider + endpoint) and immediately resweeps
        embedding tasks, so open projects flip to the local provider and
        pending embeddings drain without a backend restart or ingest."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        resweep = AsyncMock()
        monkeypatch.setattr(svc, "resweep_embedding_tasks", resweep)

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8001/v1",
            timeout_s=600,
            project_id=project_id,
            model_config_id="embedding",
            role="embedding",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "self_hosted_nvclip"
        assert config.endpoint_url == "http://localhost:8001/v1"
        resweep.assert_awaited_once_with(mock_settings)

    @pytest.mark.asyncio
    async def test_teacher_healthy_does_not_resweep(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """The resweep hook is scoped to the embedding arm — a Teacher
        turning healthy must not re-probe embedding providers."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id")),
        )
        resweep = AsyncMock()
        monkeypatch.setattr(svc, "resweep_embedding_tasks", resweep)

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8000/v1",
            timeout_s=600,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        resweep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_embedding_timeout_resets_deployment_config(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A startup health timeout for the embedding role resets the
        singleton: a redeploy's ``docker rm -f`` may have silently
        killed the previously healthy container, so a stale
        self-hosted stamp must not survive the failed attempt. The
        reset resweeps immediately so open projects flip to the
        fallback provider without waiting for the next worker trigger."""
        project_id, engine = project_with_db

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="timeout", attempts=1
                )
            ),
        )
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        resweep = AsyncMock()
        monkeypatch.setattr(svc, "resweep_embedding_tasks", resweep)

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8001/v1",
            timeout_s=5,
            project_id=project_id,
            model_config_id="embedding",
            role="embedding",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "none"
        assert config.endpoint_url is None
        resweep.assert_awaited_once_with(mock_settings)

    @pytest.mark.asyncio
    async def test_late_healthy_does_not_resurrect_torn_down_deployment(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A deployment torn down concurrently (status='stopped') must NOT be
        promoted back to 'running' by a health poll that becomes healthy after
        the teardown — otherwise a benchmark/timeout watcher's stop is undone
        and a dead endpoint gets re-registered."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            # Simulate the state after a concurrent stop: not "starting".
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="stopped",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        register = AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id"))
        monkeypatch.setattr(svc, "create_nim_endpoint", register)

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8000/v1",
            timeout_s=600,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "stopped", "late healthy must not resurrect"
        register.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_teacher_auto_registers_endpoint(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """On healthy, teacher auto-creates NimEndpoint."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="starting",
            )
            session.add(dep)
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )

        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        mock_create_ep = AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id"))
        monkeypatch.setattr(svc, "create_nim_endpoint", mock_create_ep)

        await svc._poll_health(
            deployment_id=dep_id,
            endpoint_url="http://localhost:8000/v1",
            timeout_s=600,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        # Verify create_nim_endpoint was called with correct params
        mock_create_ep.assert_called_once()
        call_args = mock_create_ep.call_args
        assert call_args[1]["data"]["endpoint_mode"] == "local_system_managed"
        assert call_args[1]["data"]["source_kind"] == "auto_registered_local"


# ── Stop ──────────────────────────────────────────────────────────────────────


class TestStopLocalNim:
    """Container stop tests."""

    @pytest.fixture(autouse=True)
    def _patch_subprocess(self, monkeypatch):
        self._subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", self._subprocess_mock)

    @pytest.mark.asyncio
    async def test_stop_targets_container_id_not_reused_name(
        self, project_with_db, mock_settings
    ):
        """Container names are reused across deployment generations; the
        stop must target the generation-unique docker id so it can never
        stop a successor container that inherited the name."""
        project_id, engine = project_with_db
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    container_id="sha256-generation-1",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                    deployed_at=utc_now(),
                )
            )
            session.commit()
        await svc.stop_local_nim(dep_id, project_id, mock_settings.WORKSPACE_ROOT)
        stop_refs = [
            c[2] for c in _docker_calls(self._subprocess_mock) if c[1] == "stop"
        ]
        assert stop_refs == ["sha256-generation-1"]
        assert "vlm-teacher-test" not in stop_refs

    @pytest.mark.asyncio
    async def test_stop_on_failed_deployment_never_touches_docker(
        self, project_with_db, mock_settings
    ):
        """A failed deployment's container was already torn down on the
        failure transition; a user stop must not `docker stop` its
        (possibly reused) name and must not overwrite the failure."""
        project_id, engine = project_with_db
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="failed",
                    status_reason="engine_dead",
                    deployed_at=utc_now(),
                )
            )
            session.commit()
        result = await svc.stop_local_nim(
            dep_id, project_id, mock_settings.WORKSPACE_ROOT
        )
        assert result is not None
        assert result.status == "failed"
        assert result.status_reason == "engine_dead"
        assert _docker_calls(self._subprocess_mock) == []

    @pytest.mark.asyncio
    async def test_stop_running_container(self, project_with_db, mock_settings):
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="running",
                deployed_at=utc_now(),
            )
            session.add(dep)
            session.commit()

        result = await svc.stop_local_nim(
            dep_id, project_id, mock_settings.WORKSPACE_ROOT
        )
        assert result is not None
        assert result.status == "stopped"
        assert result.stopped_at is not None

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self, project_with_db, mock_settings):
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="stopped",
                stopped_at=utc_now(),
            )
            session.add(dep)
            session.commit()

        result = await svc.stop_local_nim(
            dep_id, project_id, mock_settings.WORKSPACE_ROOT
        )
        assert result is not None
        assert result.status == "stopped"

    @pytest.mark.asyncio
    async def test_stop_nonexistent_returns_none(self, project_with_db, mock_settings):
        project_id, _ = project_with_db
        result = await svc.stop_local_nim(
            "nonexistent-id", project_id, mock_settings.WORKSPACE_ROOT
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_stop_embedding_resets_deployment_config(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Stopping the local embedding NIM resets the
        EmbeddingDeploymentConfig singleton (provider "none", endpoint
        cleared), so ``GET /v1/environment`` and the probe cascade stop
        advertising a dead endpoint — and (given a Settings) resweeps
        immediately so open projects flip to the fallback provider."""
        project_id, engine = project_with_db

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                    deployed_at=utc_now(),
                )
            )
            session.commit()

        resweep = AsyncMock()
        monkeypatch.setattr(svc, "resweep_embedding_tasks", resweep)

        result = await svc.stop_local_nim(
            dep_id,
            project_id,
            mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )
        assert result is not None
        assert result.status == "stopped"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "none"
        assert config.endpoint_url is None
        resweep.assert_awaited_once_with(mock_settings)

    @pytest.mark.asyncio
    async def test_stop_embedding_preserves_config_stamped_by_successor(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """The reset is scoped to the dying deployment's endpoint:
        stopping a displaced embedding NIM after a replacement already
        re-stamped the config must not un-register the live successor
        (and must not resweep)."""
        project_id, engine = project_with_db

        # Config stamped by the successor at a different endpoint.
        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8002/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-old",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                    deployed_at=utc_now(),
                )
            )
            session.commit()

        resweep = AsyncMock()
        monkeypatch.setattr(svc, "resweep_embedding_tasks", resweep)

        result = await svc.stop_local_nim(
            dep_id,
            project_id,
            mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )
        assert result is not None
        assert result.status == "stopped"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "self_hosted_nvclip"
        assert config.endpoint_url == "http://localhost:8002/v1"
        resweep.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_teacher_preserves_embedding_config(
        self, project_with_db, mock_settings
    ):
        """The config reset is scoped to the embedding role — stopping a
        Teacher must not un-register a live local embedding provider."""
        project_id, engine = project_with_db

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="running",
                    deployed_at=utc_now(),
                )
            )
            session.commit()

        result = await svc.stop_local_nim(
            dep_id, project_id, mock_settings.WORKSPACE_ROOT
        )
        assert result is not None
        assert result.status == "stopped"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "self_hosted_nvclip"
        assert config.endpoint_url == "http://localhost:8001/v1"


# ── Restart Recovery ──────────────────────────────────────────────────────────


class TestRestartRecovery:
    """Restart recovery inspects only persisted container names."""

    @pytest.fixture(autouse=True)
    def _patch_subprocess(self, monkeypatch):
        self._subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", self._subprocess_mock)

    @pytest.mark.asyncio
    async def test_running_healthy_container_rebound(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Running + healthy container → status stays running, endpoint rebound."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="running",
                deployed_at=utc_now(),
            )
            session.add(dep)
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id")),
        )
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"

    @pytest.mark.asyncio
    async def test_starting_unready_container_resumes_health_poll(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A backend restart must not destroy a live NIM that is legitimately
        still starting. Recovery keeps the durable GPU reservation and resumes
        the normal health watcher, which can promote the deployment once the
        container becomes ready."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    container_id="c0ffee123456",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                side_effect=[
                    HttpResult(
                        status_code=None,
                        body=None,
                        error_class="timeout",
                        attempts=1,
                    ),
                    HttpResult(
                        status_code=200,
                        body={},
                        error_class=None,
                        attempts=1,
                    ),
                ]
            ),
        )
        on_healthy = AsyncMock()
        monkeypatch.setattr(svc, "_on_healthy", on_healthy)
        captured: dict[str, object] = {}

        def capture_register(task_id, coro):
            captured["task_id"] = task_id
            captured["coro"] = coro

        monkeypatch.setattr(svc.background_manager, "register", capture_register)

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            assert session.get(LocalNimDeployment, dep_id).status == "starting"
        assert captured["task_id"] == f"local-nim-health-{dep_id}"
        docker_calls = _docker_calls(self._subprocess_mock)
        assert ("docker", "stop", "c0ffee123456") not in docker_calls
        assert ("docker", "rm", "-f", "c0ffee123456") not in docker_calls

        coro = captured["coro"]
        assert hasattr(coro, "__await__")
        await coro
        on_healthy.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_running_unhealthy_stopped_and_container_torn_down(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A previously-ready deployment that is no longer health-ready after
        restart is stopped rather than misclassified as a continuing startup.
        Its terminal row cannot hide an unmanaged GPU resident."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                container_id="c0ffee123456",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="running",
            )
            session.add(dep)
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="timeout", attempts=1
                )
            ),
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "stopped"
            assert "unhealthy" in db_dep.status_reason
            assert db_dep.stopped_at is not None
        docker_calls = _docker_calls(self._subprocess_mock)
        # Teardown targets the persisted docker id: container names are
        # project+role scoped and reused across deployment generations.
        assert ("docker", "stop", "c0ffee123456") in docker_calls
        assert ("docker", "rm", "-f", "c0ffee123456") in docker_calls

    @pytest.mark.asyncio
    async def test_container_not_found_marked_stopped(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Container not running → status becomes stopped."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            dep = LocalNimDeployment(
                local_nim_deployment_id=dep_id,
                project_id=project_id,
                model_config_id="test-mc",
                role="teacher",
                nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                container_name="vlm-teacher-test",
                host_port=8000,
                endpoint_url="http://localhost:8000/v1",
                gpu_assignment="device=0",
                status="running",
            )
            session.add(dep)
            session.commit()

        # Mock docker inspect returning "not found"
        async def mock_subprocess(*args, **kwargs):
            if args[0] == "docker" and args[1] == "inspect":
                return (1, "", "No such container")
            return (0, "", "")

        monkeypatch.setattr(
            svc, "run_subprocess", AsyncMock(side_effect=mock_subprocess)
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "stopped"
            assert "not running" in db_dep.status_reason

    @pytest.mark.asyncio
    async def test_gone_embedding_container_resets_config_on_recovery(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """An embedding container that vanished while the backend was
        down must not leave the config advertising its dead endpoint:
        recovery marks the row stopped AND resets the singleton, so the
        startup embedding recovery that runs right after falls back to
        the hosted provider instead of stranding projects."""
        project_id, engine = project_with_db

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        async def mock_subprocess(*args, **kwargs):
            if args[0] == "docker" and args[1] == "inspect":
                return (1, "", "No such container")
            return (0, "", "")

        monkeypatch.setattr(
            svc, "run_subprocess", AsyncMock(side_effect=mock_subprocess)
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "stopped"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "none"
        assert config.endpoint_url is None

    @pytest.mark.asyncio
    async def test_unhealthy_embedding_container_resets_config_on_recovery(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A running-but-unhealthy embedding container after restart is
        stopped AND its config stamp is reset — same rationale as the
        gone-container arm: the endpoint cannot serve, so nothing may
        keep advertising it as the self-hosted provider."""
        project_id, engine = project_with_db

        svc.update_embedding_deployment_config(
            mock_settings.WORKSPACE_ROOT,
            {
                "provider": "self_hosted_nvclip",
                "endpoint_url": "http://localhost:8001/v1",
            },
        )

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()

        from vlm_feedback_loop.services.http_client import HttpResult

        # docker inspect says running (default mock), health check fails.
        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=None, body=None, error_class="timeout", attempts=1
                )
            ),
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "stopped"

        config = _read_embedding_config(mock_settings.WORKSPACE_ROOT)
        assert config.provider == "none"
        assert config.endpoint_url is None

    @pytest.mark.asyncio
    async def test_no_generic_orphan_discovery(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Recovery never calls 'docker ps' or scans arbitrary containers."""
        project_id, engine = project_with_db
        calls = []

        async def tracking_subprocess(*args, **kwargs):
            calls.append(args)
            return (0, "", "")

        monkeypatch.setattr(
            svc, "run_subprocess", AsyncMock(side_effect=tracking_subprocess)
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        # No docker ps, docker container ls, or similar
        for call in calls:
            cmd_str = " ".join(str(a) for a in call)
            assert "docker ps" not in cmd_str
            assert "docker container ls" not in cmd_str

    @pytest.mark.asyncio
    async def test_recovery_with_no_deployments(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """No active deployments → no-op."""
        calls = []

        async def tracking_subprocess(*args, **kwargs):
            calls.append(args)
            return (0, "", "")

        monkeypatch.setattr(
            svc, "run_subprocess", AsyncMock(side_effect=tracking_subprocess)
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        # No docker commands should have been issued
        docker_calls = [c for c in calls if c and c[0] == "docker"]
        assert len(docker_calls) == 0

    @pytest.mark.parametrize(
        "missing_error",
        [
            "Error response from daemon: No such container: abc123",
            "Error: No such object: abc123",
        ],
    )
    @pytest.mark.asyncio
    async def test_student_recovery_fails_student_and_restores_residents_once(
        self, project_with_db, mock_settings, monkeypatch, missing_error
    ):
        """Restart recovery tears down the Student before restoring residents.

        Student containers cannot be resumed after process state is lost, but
        their durable displacement links still own restoration of every
        Teacher or embedding they stopped. Re-running startup recovery must
        not enqueue those restorations again.
        """
        project_id, engine = project_with_db

        sid = generate_uuid4()
        dep_id = generate_uuid4()
        displaced_teacher_id = generate_uuid4()
        displaced_embedding_id = generate_uuid4()
        with Session(engine) as session:
            session.add_all(
                [
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
                        checkpoint_packaging_status="validated",
                        nim_checkpoint_ref="/tmp/ckpt",
                        quality_status="validated",
                        serving_status="pending",
                        nim_endpoint_url="http://localhost:8002/v1",
                        nim_container_id="abc123",
                    ),
                    LocalNimDeployment(
                        local_nim_deployment_id=dep_id,
                        project_id=project_id,
                        model_config_id="mc-base",
                        role="student",
                        nim_container_image="student-image",
                        container_name=f"vlm-student-{project_id[:8]}-{sid[:8]}",
                        container_id="abc123",
                        host_port=8002,
                        endpoint_url="http://localhost:8002/v1",
                        gpu_assignment="device=0",
                        status="running",
                        student_model_id=sid,
                        checkpoint_mount_path="/tmp/ckpt",
                        nim_served_model_name=f"student-{sid[:8]}",
                        nim_model_name_path="/opt/checkpoints/student",
                        precision_method=None,
                    ),
                    LocalNimDeployment(
                        local_nim_deployment_id=displaced_teacher_id,
                        project_id=project_id,
                        model_config_id="teacher-model",
                        role="teacher",
                        nim_container_image="teacher-image",
                        container_name="vlm-teacher-displaced",
                        host_port=8000,
                        endpoint_url="http://localhost:8000/v1",
                        gpu_assignment="device=0",
                        status="stopped",
                        displaced_by_deployment_id=dep_id,
                        displaced_at=utc_now(),
                    ),
                    LocalNimDeployment(
                        local_nim_deployment_id=displaced_embedding_id,
                        project_id=project_id,
                        model_config_id="embedding-model",
                        role="embedding",
                        nim_container_image="embedding-image",
                        container_name="vlm-embedding-displaced",
                        host_port=8001,
                        endpoint_url="http://localhost:8001/v1",
                        gpu_assignment="device=0",
                        status="stopped",
                        displaced_by_deployment_id=dep_id,
                        displaced_at=utc_now(),
                    ),
                ]
            )
            session.commit()

        observed_restore: dict[str, object] = {}

        async def record_restore(displaced, *, workspace_root, settings):
            assert workspace_root == mock_settings.WORKSPACE_ROOT
            assert settings is mock_settings
            with Session(engine) as session:
                recovered_dep = session.get(LocalNimDeployment, dep_id)
                recovered_student = session.get(StudentModel, sid)
                observed_restore["deployment_status"] = recovered_dep.status
                observed_restore["student_status"] = recovered_student.serving_status
            observed_restore["deployment_ids"] = {
                row.local_nim_deployment_id for row in displaced
            }

        restore = AsyncMock(side_effect=record_restore)
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)
        remove = AsyncMock(return_value=(1, "", missing_error))
        monkeypatch.setattr(svc, "run_subprocess", remove)

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        remove.assert_awaited_once_with("docker", "rm", "-f", "abc123", timeout_s=15.0)
        assert observed_restore == {
            "deployment_status": "stopped",
            "student_status": "failed",
            "deployment_ids": {
                displaced_teacher_id,
                displaced_embedding_id,
            },
        }

        with Session(engine) as session:
            updated_dep = session.get(LocalNimDeployment, dep_id)
            assert updated_dep.status == "stopped"
            assert updated_dep.status_reason == "student_recovery_no_resume"
            assert updated_dep.stopped_at is not None
            student = session.get(StudentModel, sid)
            assert student.serving_status == "failed"
            assert (
                student.nim_preflight_details.get("failure_stage")
                == "interrupted_by_restart"
            )
            assert student.nim_endpoint_url is None
            assert student.nim_container_id is None

        restore.assert_awaited_once()

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)
        restore.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_recovery_snapshots_projects_before_queuing_cross_project_restore(
        self, mock_settings, monkeypatch
    ):
        """A restore queued into a later project is not stale startup work.

        Recovery first snapshots the rows that predated this process. A
        Student in one project may restore a shared Teacher owned by another;
        the fresh ``starting`` row must be left to its newly registered
        background task, not consumed again by the same recovery sweep.
        """
        from vlm_feedback_loop.services.project_service import (
            create_project,
            get_project_engine,
        )

        projects = [
            create_project(name="Recovery A", description=None, settings=mock_settings),
            create_project(name="Recovery B", description=None, settings=mock_settings),
        ]
        student_project_id, owner_project_id = sorted(
            project.project_id for project in projects
        )
        student_engine = get_project_engine(
            student_project_id, mock_settings.WORKSPACE_ROOT
        )
        owner_engine = get_project_engine(
            owner_project_id, mock_settings.WORKSPACE_ROOT
        )
        assert student_engine is not None
        assert owner_engine is not None

        student_deployment_id = generate_uuid4()
        displaced_teacher_id = generate_uuid4()
        queued_restore_id = generate_uuid4()
        with Session(student_engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=student_deployment_id,
                    project_id=student_project_id,
                    model_config_id="student-model",
                    role="student",
                    nim_container_image="student-image",
                    container_name="vlm-student-recovery",
                    container_id="student-container",
                    host_port=8002,
                    endpoint_url="http://localhost:8002/v1",
                    gpu_assignment="device=0",
                    status="running",
                )
            )
            session.commit()
        with Session(owner_engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=displaced_teacher_id,
                    project_id=owner_project_id,
                    model_config_id="teacher-model",
                    role="teacher",
                    nim_container_image="teacher-image",
                    container_name="vlm-teacher-displaced",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="stopped",
                    displaced_by_deployment_id=student_deployment_id,
                    displaced_at=utc_now(),
                )
            )
            session.commit()

        async def queue_restore(displaced, *, workspace_root, settings):
            assert workspace_root == mock_settings.WORKSPACE_ROOT
            assert settings is mock_settings
            assert [row.local_nim_deployment_id for row in displaced] == [
                displaced_teacher_id
            ]
            with Session(owner_engine) as session:
                session.add(
                    LocalNimDeployment(
                        local_nim_deployment_id=queued_restore_id,
                        project_id=owner_project_id,
                        model_config_id="teacher-model",
                        role="teacher",
                        nim_container_image="teacher-image",
                        container_name="vlm-teacher-restoring",
                        host_port=8000,
                        endpoint_url="http://localhost:8000/v1",
                        gpu_assignment="device=0",
                        status="starting",
                    )
                )
                session.commit()

        restore = AsyncMock(side_effect=queue_restore)
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)

        async def missing_container(*args, **kwargs):
            if args[:3] == ("docker", "rm", "-f"):
                return (0, "", "")
            if args[:2] == ("docker", "inspect"):
                return (1, "", "Error: No such container")
            return (0, "", "")

        monkeypatch.setattr(
            svc, "run_subprocess", AsyncMock(side_effect=missing_container)
        )

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        restore.assert_awaited_once()
        with Session(owner_engine) as session:
            queued = session.get(LocalNimDeployment, queued_restore_id)
            assert queued is not None
            assert queued.status == "starting"

    @pytest.mark.asyncio
    async def test_student_recovery_retries_before_restore_when_remove_fails(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """An unconfirmed Student removal keeps its GPU reserved.

        Restoration may begin only after Docker confirms the Student is gone.
        A transient daemon failure therefore leaves the deployment active for
        the next recovery pass instead of risking two NIMs on one GPU.
        """
        project_id, engine = project_with_db
        student_deployment_id = generate_uuid4()
        displaced_teacher_id = generate_uuid4()
        with Session(engine) as session:
            session.add_all(
                [
                    LocalNimDeployment(
                        local_nim_deployment_id=student_deployment_id,
                        project_id=project_id,
                        model_config_id="student-model",
                        role="student",
                        nim_container_image="student-image",
                        container_name="vlm-student-recovery",
                        container_id="student-container-id",
                        host_port=8002,
                        endpoint_url="http://localhost:8002/v1",
                        gpu_assignment="device=0",
                        status="running",
                    ),
                    LocalNimDeployment(
                        local_nim_deployment_id=displaced_teacher_id,
                        project_id=project_id,
                        model_config_id="teacher-model",
                        role="teacher",
                        nim_container_image="teacher-image",
                        container_name="vlm-teacher-displaced",
                        host_port=8000,
                        endpoint_url="http://localhost:8000/v1",
                        gpu_assignment="device=0",
                        status="stopped",
                        displaced_by_deployment_id=student_deployment_id,
                        displaced_at=utc_now(),
                    ),
                ]
            )
            session.commit()

        remove = AsyncMock(return_value=(1, "", "Cannot connect to Docker daemon"))
        monkeypatch.setattr(svc, "run_subprocess", remove)
        restore = AsyncMock()
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            student_dep = session.get(LocalNimDeployment, student_deployment_id)
            assert student_dep.status == "running"
            assert student_dep.status_reason == "student_recovery_teardown_failed"
        assert svc.scan_active_residents_by_device(mock_settings.WORKSPACE_ROOT) == {
            "0": [(project_id, student_deployment_id)]
        }
        restore.assert_not_awaited()

        remove.return_value = (0, "", "")
        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            assert (
                session.get(LocalNimDeployment, student_deployment_id).status
                == "stopped"
            )
        restore.assert_awaited_once()


# ── Helpers ───────────────────────────────────────────────────────────────────


class TestHelpers:
    """Test internal helper functions."""

    def test_extract_device_index(self):
        assert svc.extract_device_index("device=0") == "0"
        assert svc.extract_device_index("device=3") == "3"
        assert svc.extract_device_index("0") == "0"

    def test_build_container_name(self):
        name = svc.build_container_name("teacher", "12345678-abcd-efgh")
        assert name == "vlm-teacher-12345678"

    def test_build_container_name_student_includes_id_suffix(self):
        """Student container names carry the Student id suffix so retries
        don't collide."""
        name = svc.build_container_name(
            "student",
            project_id="abcdef0123456789",
            student_model_id="0123456789abcdef0123456789abcdef",
        )
        assert "vlm-student-" in name
        assert "abcdef01" in name
        assert "01234567" in name

    def test_build_container_name_student_without_id_falls_back(self):
        name = svc.build_container_name("student", "abcdef0123456789", None)
        # Falls back to the role-only form
        assert name == "vlm-student-abcdef01"

    def test_build_endpoint_url(self):
        url = svc._build_endpoint_url(8000)
        assert url == "http://localhost:8000/v1"

    def test_docker_run_command_display_no_secrets(self):
        cmd = svc.docker_run_command_display(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
        )
        assert "-e NGC_API_KEY" in cmd
        assert "NGC_API_KEY=" not in cmd
        assert "nvapi-" not in cmd

    def test_docker_run_command_display_teacher_has_uid(self):
        cmd = svc.docker_run_command_display(
            nim_container_image="test:latest",
            container_name="test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
        )
        assert "-u $(id -u)" in cmd

    def test_omni_keeps_declared_container_user(self):
        """Omni's startup calls getpwuid, so an unknown host UID crashes it."""
        args = svc._build_docker_run_command(
            nim_container_image=NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            container_name="vlm-teacher-omni",
            gpu_assignment="device=0",
            host_port=8001,
            role="teacher",
        )
        assert "-u" not in args

        cmd = svc.docker_run_command_display(
            nim_container_image=NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            container_name="vlm-teacher-omni",
            gpu_assignment="device=0",
            host_port=8001,
            role="teacher",
        )
        assert "-u $(id -u)" not in cmd

    def test_omni_cache_preparation_opens_only_parent_directories(
        self, tmp_path, monkeypatch
    ):
        """The declared Omni user can create its model subtree without a
        recursive permission change to existing cached artifacts."""
        cache_root = tmp_path / "nim"
        existing = cache_root / "ngc" / "hub" / "existing-model" / "weights"
        existing.mkdir(parents=True)
        existing.chmod(0o700)
        monkeypatch.setattr(svc, "NIM_CACHE_ROOT", str(cache_root))

        svc._prepare_declared_user_cache(NEMOTRON_3_NANO_OMNI_NIM_IMAGE)

        assert cache_root.stat().st_mode & 0o777 == 0o777
        assert (cache_root / "ngc").stat().st_mode & 0o777 == 0o777
        assert (cache_root / "ngc" / "hub").stat().st_mode & 0o777 == 0o777
        assert (cache_root / "ngc" / "hub" / "tmp").stat().st_mode & 0o777 == 0o777
        assert (cache_root / "vllm" / "modelinfos").stat().st_mode & 0o777 == 0o777
        assert (cache_root / "flashinfer").stat().st_mode & 0o777 == 0o777
        assert existing.stat().st_mode & 0o777 == 0o700

    def test_docker_run_args_contain_nim_ulimit_flags(self):
        """NIM container startup hints recommend
        ``--ulimit memlock=-1 --ulimit stack=67108864`` for VLM NIMs.
        Without them, large multi-image batches can hit pinned-memory
        or thread-stack limits silently. The flags must be present in
        the argv list AND in the canonical display string (so the
        ``deployment_handoff`` Action Request stays byte-equivalent
        with what ``:deploy_nim`` actually executes).
        """
        for role in ("teacher", "student", "embedding"):
            args = svc._build_docker_run_command(
                nim_container_image="test:latest",
                container_name="test",
                gpu_assignment="device=0",
                host_port=8000,
                role=role,
            )
            # Flags appear as TWO consecutive argv items each (Docker CLI
            # accepts ``--ulimit memlock=-1`` as separate tokens or as
            # ``--ulimit=memlock=-1`` — we use the former).
            memlock_idx = args.index("memlock=-1")
            assert args[memlock_idx - 1] == "--ulimit", (
                f"role={role}: ``memlock=-1`` must be preceded by ``--ulimit``"
            )
            stack_idx = args.index("stack=67108864")
            assert args[stack_idx - 1] == "--ulimit", (
                f"role={role}: ``stack=67108864`` must be preceded by ``--ulimit``"
            )

    def test_docker_run_command_display_contains_nim_ulimit_flags(self):
        """The rendered command the operator copy-pastes (and that the
        deployment_handoff Action Request emits) MUST contain the same
        ulimit flags so it re-executes byte-equivalently with what
        :deploy_nim runs."""
        for role in ("teacher", "student", "embedding"):
            cmd = svc.docker_run_command_display(
                nim_container_image="test:latest",
                container_name="test",
                gpu_assignment="device=0",
                host_port=8000,
                role=role,
            )
            assert "--ulimit memlock=-1" in cmd, f"role={role}"
            assert "--ulimit stack=67108864" in cmd, f"role={role}"

    def test_display_is_generated_from_the_arg_builder(self):
        """Anti-drift contract: the display string is rendered
        FROM _build_docker_run_command's argv, so every real token (minus
        the numeric UID, which gets a portable placeholder)
        must appear verbatim in the display — for the fully-loaded student
        shape too, which exercises every conditional branch."""
        import os as _os

        kwargs = {
            "nim_container_image": "test:latest",
            "container_name": "student-test",
            "gpu_assignment": "device=1",
            "host_port": 8801,
            "role": "student",
            "checkpoint_mount": "/data/ckpt",
            "nim_model_name_path": "/opt/ckpt",
            "nim_served_model_name": "student-abc",
            "max_images_per_request": 12,
            "nim_model_size": "nano",
            "nim_model_profile": "prof-123",
        }
        args = svc._build_docker_run_command(**kwargs)
        cmd = svc.docker_run_command_display(**kwargs)
        for token in args:
            if token in ("docker", "run", "-d"):
                continue
            if token == str(_os.getuid()):
                continue  # rendered as $(id -u)
            assert token in cmd, f"arg token missing from display: {token!r}"
        assert "-e NGC_API_KEY" in cmd
        assert "NGC_API_KEY=" not in cmd
        assert "$(id -u)" in cmd

    def test_nim_max_images_per_prompt_env_emitted_when_capped(self):
        """When a per-prompt image cap is resolved, the docker run command
        pins it via ``-e NIM_MAX_IMAGES_PER_PROMPT=<n>`` so the NIM accepts
        exactly what the backend's ICL pruner will send. Without it, a cosmos
        :1.7.0 NIM keeps its silent profile default of 5 and HTTP-400s the
        moment ICL attaches a 6th image. Omitted when no cap is provided so
        the NIM keeps its profile default (pre-existing behavior)."""
        args = svc._build_docker_run_command(
            nim_container_image="test:latest",
            container_name="test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=33,
        )
        idx = args.index("NIM_MAX_IMAGES_PER_PROMPT=33")
        assert args[idx - 1] == "-e", "image cap env must be preceded by -e"

        # None ⇒ not emitted (NIM keeps its own default).
        args_none = svc._build_docker_run_command(
            nim_container_image="test:latest",
            container_name="test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
        )
        assert not any("NIM_MAX_IMAGES_PER_PROMPT" in a for a in args_none)

        # Display side stays byte-equivalent.
        cmd = svc.docker_run_command_display(
            nim_container_image="test:latest",
            container_name="test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=33,
        )
        assert "-e NIM_MAX_IMAGES_PER_PROMPT=33" in cmd

    def test_docker_run_command_display_embedding_includes_uid(self):
        """``-u $(id -u)`` MUST be present in the embedding display
        command: without the flag the NeMo Retriever VL 1B v2
        container runs as a hardcoded 1000:1000 UID and ``Permission
        denied (os error 13)`` on the ``~/.cache/nim`` mount kills the
        model-weight download.
        """
        cmd = svc.docker_run_command_display(
            nim_container_image="test:latest",
            container_name="test",
            gpu_assignment="device=0",
            host_port=8001,
            role="embedding",
        )
        assert "-u $(id -u)" in cmd

    def test_embedding_uses_validated_fp16_profile_and_persistent_model_cache(self):
        """Embedding deploys must avoid the broken SM120 automatic profile and
        keep downloaded weights across container replacement.

        NeMo Retriever 2.0.0's unset-precision path asks for a cuDNN plan
        directory that its own SM120 entrypoint skips creating.  The
        live-validated FP16 path serves the expected 2,048-dimensional image
        vector, and NIM_MODEL_PATH must live below the host cache mount rather
        than in the disposable container layer.
        """
        args = svc._build_docker_run_command(
            nim_container_image=EMBEDDING_NIM_IMAGE,
            container_name="vlm-embedding-test",
            gpu_assignment="device=0",
            host_port=8001,
            role="embedding",
        )
        assert "NIM_PRECISION=fp16" in args
        assert (
            "NIM_MODEL_PATH=/opt/nim/.cache/models/llama-nemotron-embed-vl-1b-v2"
        ) in args

        cmd = svc.docker_run_command_display(
            nim_container_image=EMBEDDING_NIM_IMAGE,
            container_name="vlm-embedding-test",
            gpu_assignment="device=0",
            host_port=8001,
            role="embedding",
        )
        assert "-e NIM_PRECISION=fp16" in cmd
        assert (
            "-e NIM_MODEL_PATH=/opt/nim/.cache/models/llama-nemotron-embed-vl-1b-v2"
        ) in cmd

    def test_teacher_does_not_inherit_embedding_runtime_profile(self):
        """Teacher images keep their own profile-selection behavior."""
        args = svc._build_docker_run_command(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
        )
        assert "NIM_PRECISION=fp16" not in args
        assert not any(token.startswith("NIM_MODEL_PATH=") for token in args)

    def test_unknown_embedding_image_does_not_inherit_retriever_profile(self):
        """Version-specific Retriever flags must not leak into another image."""
        args = svc._build_docker_run_command(
            nim_container_image="vendor/custom-embedding-nim:1.0",
            container_name="vlm-embedding-alternate",
            gpu_assignment="device=0",
            host_port=8001,
            role="embedding",
        )
        assert "NIM_PRECISION=fp16" not in args
        assert not any(token.startswith("NIM_MODEL_PATH=") for token in args)

    def test_student_docker_run_emits_checkpoint_mount(self):
        args = svc._build_docker_run_command(
            nim_container_image="nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
            container_name="vlm-student-abc-123",
            gpu_assignment="device=1",
            host_port=8002,
            role="student",
            checkpoint_mount="/tmp/ckpt/student-a",
            nim_model_name_path="/opt/checkpoints/student",
            nim_served_model_name="student-abcd1234",
        )
        # -v {host}:{container}:ro for the checkpoint mount
        assert "-v" in args
        assert "/tmp/ckpt/student-a:/opt/checkpoints/student:ro" in args
        # -e NIM_MODEL_NAME=...
        assert "NIM_MODEL_NAME=/opt/checkpoints/student" in args
        # -e NIM_SERVED_MODEL_NAME=...
        assert "NIM_SERVED_MODEL_NAME=student-abcd1234" in args
        # -u $(id -u) is included for student role
        assert "-u" in args

    def test_student_docker_run_without_checkpoint_omits_mount(self):
        # Student docker_run_command_display path with no checkpoint info
        # (e.g., display-only build) must not crash.
        args = svc._build_docker_run_command(
            nim_container_image="img",
            container_name="vlm-student-abc-123",
            gpu_assignment="device=0",
            host_port=8002,
            role="student",
            checkpoint_mount=None,
            nim_model_name_path=None,
            nim_served_model_name=None,
        )
        # No NIM_MODEL_NAME env var
        assert not any(
            isinstance(a, str) and a.startswith("NIM_MODEL_NAME=") for a in args
        )

    def test_teacher_docker_run_has_no_student_extras(self):
        args = svc._build_docker_run_command(
            nim_container_image="img",
            container_name="vlm-teacher-abc",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
        )
        assert not any(
            isinstance(a, str) and a.startswith("NIM_MODEL_NAME=") for a in args
        )
        assert not any(
            isinstance(a, str) and a.startswith("NIM_SERVED_MODEL_NAME=") for a in args
        )

    def test_docker_run_command_display_student_shape(self):
        text = svc.docker_run_command_display(
            nim_container_image="img",
            container_name="vlm-student-abc-123",
            gpu_assignment="device=0",
            host_port=8002,
            role="student",
            checkpoint_mount="/tmp/ckpt",
            nim_model_name_path="/opt/checkpoints/student",
            nim_served_model_name="student-foo",
        )
        assert "-e NGC_API_KEY" in text
        assert "NGC_API_KEY=" not in text
        # No literal nvapi-* secrets
        assert "nvapi-" not in text
        # The student-specific lines are present.
        assert "/tmp/ckpt:/opt/checkpoints/student:ro" in text
        assert "NIM_MODEL_NAME=/opt/checkpoints/student" in text
        assert "NIM_SERVED_MODEL_NAME=student-foo" in text


class TestCosmos3SharedImageDeployEnv:
    """The shared-image (cosmos3-reasoner) deploy selectors: ``NIM_MODEL_SIZE``,
    ``NIM_MODEL_PROFILE`` (footgun fix), and ``NIM_SERVED_MODEL_NAME`` (so
    /v1/models reports the size-specific name). Single-image teachers
    (cosmos-reason2-*) must NOT emit any of them — their docker command stays
    byte-for-byte unchanged."""

    def test_cr3_super_env(self):
        """CR3-Super: NIM_MODEL_SIZE=super + served name + cap=999, no profile
        pin (super auto-selects off its cached default weights)."""
        args = svc._build_docker_run_command(
            nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=999,
            nim_model_size="super",
            nim_served_model_name=COSMOS3_SUPER_REASONER,
        )
        assert "NIM_MODEL_SIZE=super" in args
        assert "NIM_SERVED_MODEL_NAME=nvidia/cosmos3-super-reasoner" in args
        assert "NIM_MAX_IMAGES_PER_PROMPT=999" in args
        assert not any("NIM_MODEL_PROFILE" in a for a in args)

    def test_cr3_nano_env_pins_profile(self):
        """CR3-Nano: NIM_MODEL_SIZE=nano + the pinned fp8 profile id + served
        name + cap=999. The pinned profile bypasses the fragile auto-selector
        that silently served super."""
        profile = "e2e00f3e555bb4fe0ef011faadd56a37441c7274e149d482cfeb67dbfb75b092"
        args = svc._build_docker_run_command(
            nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=999,
            nim_model_size="nano",
            nim_model_profile=profile,
            nim_served_model_name=COSMOS3_NANO_REASONER,
        )
        assert "NIM_MODEL_SIZE=nano" in args
        assert f"NIM_MODEL_PROFILE={profile}" in args
        assert "NIM_SERVED_MODEL_NAME=nvidia/cosmos3-nano-reasoner" in args
        assert "NIM_MAX_IMAGES_PER_PROMPT=999" in args

    def test_cr2_8b_env_unchanged(self):
        """CR2-8B (single-image teacher): only the image cap is set — no
        model-size, no profile, no served-name. This is the exact pre-existing
        behavior that MUST NOT regress."""
        args = svc._build_docker_run_command(
            nim_container_image="nvcr.io/nim/nvidia/cosmos-reason2-8b:1.7.0",
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=999,
        )
        assert "NIM_MAX_IMAGES_PER_PROMPT=999" in args
        assert not any("NIM_MODEL_SIZE" in a for a in args)
        assert not any("NIM_MODEL_PROFILE" in a for a in args)
        assert not any("NIM_SERVED_MODEL_NAME" in a for a in args)

    def test_display_mirrors_env(self):
        """The operator-facing display command emits the same selectors."""
        profile = "e2e00f3e555bb4fe0ef011faadd56a37441c7274e149d482cfeb67dbfb75b092"
        cmd = svc.docker_run_command_display(
            nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            max_images_per_request=999,
            nim_model_size="nano",
            nim_model_profile=profile,
            nim_served_model_name=COSMOS3_NANO_REASONER,
        )
        assert "-e NIM_MODEL_SIZE=nano" in cmd
        assert f"-e NIM_MODEL_PROFILE={profile}" in cmd
        assert "-e NIM_SERVED_MODEL_NAME=nvidia/cosmos3-nano-reasoner" in cmd


# ── Served-model verification (anti-silent-fallback) ──────────────────────────


def _make_cache(tmp_path, slug: str, *, weight_bytes: int | None) -> str:
    """Stage a NIM cache hub dir for *slug* under a fresh cache root.

    ``weight_bytes=None`` → config-only stub (the silent-fallback footgun:
    a ``config.json`` symlink and ZERO weight files). An int → one
    ``model.safetensors`` shard of that size symlinked from ``blobs/`` into
    a snapshot dir, mirroring the real HF-hub layout NIM writes.

    Returns the cache ROOT (the value passed as ``cache_root`` to the
    verification helpers).
    """
    root = tmp_path / "nimcache"
    model_dir = root / "ngc" / "hub" / f"models--nim--nvidia--{slug}"
    blobs = model_dir / "blobs"
    snap = model_dir / "snapshots" / "profile-x"
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)

    # config.json blob + symlink (present in BOTH the stub and the real case)
    cfg_blob = blobs / "cfgblob"
    cfg_blob.write_text('{"model_type": "x"}')
    (snap / "config.json").symlink_to(cfg_blob)

    if weight_bytes is not None:
        weight_blob = blobs / "weightblob"
        with open(weight_blob, "wb") as f:
            f.write(b"\0" * weight_bytes)
        (snap / "model.safetensors").symlink_to(weight_blob)

    return str(root)


class TestCountCachedWeightFiles:
    """Offline weight-file presence check — the direct detector for the
    '0 weight files for the requested size' silent-fallback root cause."""

    def test_real_shard_counts(self, tmp_path):
        root = _make_cache(
            tmp_path, "cosmos3-super-reasoner", weight_bytes=2 * 1024 * 1024
        )
        assert svc.count_cached_weight_files("cosmos3-super-reasoner", root) == 1

    def test_config_only_stub_counts_zero(self, tmp_path):
        """The nano footgun: 52K cache, only a config.json symlink, ZERO
        weight files. Must count 0 so verification fails the deploy."""
        root = _make_cache(tmp_path, "cosmos3-nano-reasoner", weight_bytes=None)
        assert svc.count_cached_weight_files("cosmos3-nano-reasoner", root) == 0

    def test_missing_model_dir_counts_zero(self, tmp_path):
        root = _make_cache(tmp_path, "cosmos3-super-reasoner", weight_bytes=None)
        # Asking for a model whose dir doesn't exist at all → 0.
        assert svc.count_cached_weight_files("cosmos-reason2-8b", root) == 0

    def test_tiny_weight_file_ignored_as_stub(self, tmp_path):
        """A sub-1MiB '.safetensors' is an index sidecar / truncated pull,
        not real weights — must not count."""
        root = _make_cache(tmp_path, "cosmos-reason2-2b", weight_bytes=1024)
        assert svc.count_cached_weight_files("cosmos-reason2-2b", root) == 0


class TestModelNameSlug:
    """Catalog model_name / NGC-URL → model-slug parsing the verification
    relies on. Verification keys on the size-specific catalog model_name (NOT
    the shared image slug), so a correct cosmos3-super deploy passes."""

    def test_model_name_slug(self):
        assert svc._model_name_slug(COSMOS3_NANO_REASONER) == "cosmos3-nano-reasoner"

    def test_ngc_url_slug(self):
        assert (
            svc._ngc_model_slug(
                "ngc://nim/nvidia/cosmos-reason2-2b:1208-fp8-dynamic-kv8"
            )
            == "cosmos-reason2-2b"
        )

    def test_empty_model_name_returns_none(self):
        # Empty name → None so the caller treats verification as inconclusive.
        assert svc._model_name_slug("") is None


class TestVerifyServedModel:
    """The healthy-path gate. Three cases:
    (a) healthy when served model matches / weights present,
    (b) FAILS on nano→super fallback (live metadata mismatch) and on
        0-weight-file cache,
    plus the inconclusive-never-fails policy."""

    @staticmethod
    def _patch_metadata(monkeypatch, *, loaded_slug: str | None, profile: str = "p1"):
        """Patch resilient_request so /v1/metadata returns a NIM-shaped
        body naming *loaded_slug* as the loaded model (or a non-dict body
        when loaded_slug is None to simulate no /v1/metadata)."""
        from vlm_feedback_loop.services.http_client import HttpResult

        if loaded_slug is None:
            body: object = "no metadata endpoint"
            status = 404
        else:
            body = {
                "modelInfo": [
                    {"modelUrl": f"ngc://nim/nvidia/{loaded_slug}:{profile}-tag"}
                ],
                "selectedModelProfileId": profile,
            }
            status = 200
        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(status_code=status, body=body, attempts=1)
            ),
        )

    @pytest.mark.asyncio
    async def test_passes_when_live_slug_matches(self, tmp_path, monkeypatch):
        root = _make_cache(tmp_path, "cosmos-reason2-2b", weight_bytes=2 * 1024 * 1024)
        self._patch_metadata(monkeypatch, loaded_slug="cosmos-reason2-2b")

        v = await svc.verify_served_model(
            expected_model_name=COSMOS_REASON2_2B,
            endpoint_url="http://localhost:8000/v1",
            cache_root=root,
        )
        assert v.ok is True
        assert v.expected_slug == "cosmos-reason2-2b"
        assert v.served_slug == "cosmos-reason2-2b"

    @pytest.mark.asyncio
    async def test_passes_when_weights_present_and_no_metadata(
        self, tmp_path, monkeypatch
    ):
        """Weights present + /v1/metadata unavailable → PASS (the live
        probe being unavailable must not fail a real deploy)."""
        root = _make_cache(tmp_path, "cosmos-reason2-8b", weight_bytes=2 * 1024 * 1024)
        self._patch_metadata(monkeypatch, loaded_slug=None)

        v = await svc.verify_served_model(
            expected_model_name=COSMOS_REASON2_8B,
            endpoint_url="http://localhost:8000/v1",
            cache_root=root,
        )
        assert v.ok is True
        assert v.weight_files_found == 1

    @pytest.mark.asyncio
    async def test_fails_on_nano_to_super_live_mismatch(self, tmp_path, monkeypatch):
        """The silent-fallback footgun: requested nano, NIM reports super
        loaded via /v1/metadata. MUST fail verification."""
        # Even if a partial nano cache somehow had a shard, the live
        # mismatch is authoritative and fails first.
        root = _make_cache(tmp_path, "cosmos3-nano-reasoner", weight_bytes=None)
        self._patch_metadata(monkeypatch, loaded_slug="cosmos3-super-reasoner")

        v = await svc.verify_served_model(
            expected_model_name=COSMOS3_NANO_REASONER,
            endpoint_url="http://localhost:8000/v1",
            cache_root=root,
        )
        assert v.ok is False
        assert v.expected_slug == "cosmos3-nano-reasoner"
        assert v.served_slug == "cosmos3-super-reasoner"
        assert "mismatch" in v.reason.lower()

    @pytest.mark.asyncio
    async def test_fails_on_zero_weight_files(self, tmp_path, monkeypatch):
        """Requested nano has a config-only cache (0 weight files) and
        /v1/metadata is unavailable. MUST fail — the requested weights
        aren't present, so a 'healthy' container is serving fallback."""
        root = _make_cache(tmp_path, "cosmos3-nano-reasoner", weight_bytes=None)
        self._patch_metadata(monkeypatch, loaded_slug=None)

        v = await svc.verify_served_model(
            expected_model_name=COSMOS3_NANO_REASONER,
            endpoint_url="http://localhost:8000/v1",
            cache_root=root,
        )
        assert v.ok is False
        assert v.weight_files_found == 0
        assert "no model weights" in v.reason.lower()

    @pytest.mark.asyncio
    async def test_unrecognised_model_name_is_inconclusive_pass(
        self, tmp_path, monkeypatch
    ):
        """A model name we can't parse a slug from must not fail a deploy —
        no signal disproved the request."""
        self._patch_metadata(monkeypatch, loaded_slug=None)
        v = await svc.verify_served_model(
            expected_model_name="",
            endpoint_url="http://localhost:8000/v1",
            cache_root=str(tmp_path / "empty"),
        )
        assert v.ok is True


class TestOnHealthyVerificationGate:
    """``_on_healthy`` must refuse to mark a deployment running (and must
    NOT auto-register an endpoint) when served-model verification fails,
    and must stop the container — a wrong-model NIM must not keep serving
    on a known port while its terminal row hides it from the placement
    scan."""

    @pytest.fixture(autouse=True)
    def _patch_subprocess(self, monkeypatch):
        self._subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", self._subprocess_mock)

    @pytest.mark.asyncio
    async def test_failed_verification_marks_failed_and_skips_register(
        self, project_with_db, mock_settings, monkeypatch
    ):
        project_id, engine = project_with_db
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image="nvcr.io/nim/nvidia/cosmos3-nano-reasoner:1.7.0",
                    container_name="vlm-teacher-test",
                    container_id="c0ffee123456",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(
                    ok=False,
                    reason="No model weights cached for requested model "
                    "'cosmos3-nano-reasoner'",
                    expected_slug="cosmos3-nano-reasoner",
                )
            ),
        )
        mock_create_ep = AsyncMock(return_value=MagicMock(endpoint_id="ep"))
        monkeypatch.setattr(svc, "create_nim_endpoint", mock_create_ep)

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "weights" in db_dep.status_reason.lower()
        # The wrong model must NEVER be auto-registered for use.
        mock_create_ep.assert_not_called()
        # ...and must NOT keep serving: the container is stopped+removed,
        # targeted by the persisted docker id (names are reused across
        # deployment generations).
        docker_calls = _docker_calls(self._subprocess_mock)
        assert ("docker", "stop", "c0ffee123456") in docker_calls
        assert ("docker", "rm", "-f", "c0ffee123456") in docker_calls

    @pytest.mark.asyncio
    async def test_student_role_is_not_gated(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Student deployments mount their own fine-tuned checkpoint
        (weights from the mount, not the NGC cache) and serve it under a
        custom name, so the NGC-size silent-fallback gate does NOT apply —
        running it would falsely fail legitimate student deploys. The
        lifecycle orchestrator owns the student verification (smoke
        inference). Lock in that ``_on_healthy`` never calls
        ``verify_served_model`` for a student."""
        project_id, engine = project_with_db
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="student",
                    nim_container_image="nvcr.io/nim/nvidia/cosmos-reason2-8b:1.7.0",
                    container_name="vlm-student-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                    student_model_id="sm-1",
                )
            )
            session.commit()

        verify_mock = AsyncMock(
            return_value=svc.ServedModelVerification(ok=False, reason="should not run")
        )
        monkeypatch.setattr(svc, "verify_served_model", verify_mock)

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id="test-mc",
            role="student",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        verify_mock.assert_not_called()
        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            # Student reaches running on health (verification is the
            # lifecycle's job, not _on_healthy's).
            assert db_dep.status == "running"


class TestInferenceProbeGate:
    """Teacher adoption (startup health poll and restart-recovery rebind)
    is gated on one minimal real completion after served-model
    verification: a NIM whose vLLM engine died keeps answering
    ``/v1/health/ready`` (and ``/v1/metadata``) 200 from the surviving
    HTTP front-end, so only the inference path proves the engine is
    alive."""

    def _seed_teacher(self, engine, project_id: str) -> str:
        """Deployment row plus the ModelConfig whose model_name the probe
        sends as the completion's model id."""
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                ModelConfig(
                    model_config_id="test-mc",
                    project_id=project_id,
                    endpoint_id="test-ep",
                    model_name=COSMOS_REASON2_8B,
                    context_window_tokens=131072,
                    eligible_roles=["teacher"],
                    supports_image_input=True,
                )
            )
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="test-mc",
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    container_id="c0ffee123456",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()
        return dep_id

    @staticmethod
    def _request_mock(chat_result):
        """resilient_request mock: health/metadata answer 200, the
        chat-completions probe answers *chat_result*."""
        from vlm_feedback_loop.services.http_client import HttpResult

        async def _fake_request(method, url, **kwargs):
            if url.endswith("/chat/completions"):
                return chat_result
            return HttpResult(status_code=200, body={}, error_class=None, attempts=1)

        return AsyncMock(side_effect=_fake_request)

    @pytest.mark.asyncio
    async def test_dead_engine_fails_adoption_and_stops_container(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Health 200 + verified model but the completion probe returns an
        endpoint error (the dead-engine signature: the front-end serves
        health while /chat/completions 500s) → deployment failed with an
        actionable reason, container stopped, endpoint never registered."""
        from vlm_feedback_loop.services.http_client import HttpResult

        project_id, engine = project_with_db
        dep_id = self._seed_teacher(engine, project_id)

        subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", subprocess_mock)
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc,
            "resilient_request",
            self._request_mock(
                HttpResult(
                    status_code=500,
                    body={"error": "EngineDeadError"},
                    error_class="endpoint_error",
                    error_detail="HTTP 500: EngineDeadError",
                    attempts=1,
                )
            ),
        )
        mock_create_ep = AsyncMock(return_value=MagicMock(endpoint_id="ep"))
        monkeypatch.setattr(svc, "create_nim_endpoint", mock_create_ep)

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "inference" in db_dep.status_reason.lower()
        mock_create_ep.assert_not_called()
        docker_calls = _docker_calls(subprocess_mock)
        assert ("docker", "stop", "c0ffee123456") in docker_calls
        assert ("docker", "rm", "-f", "c0ffee123456") in docker_calls

    @pytest.mark.asyncio
    async def test_probe_timeout_is_inconclusive_and_promotes(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """A probe timeout must NOT fail adoption: a busy-but-healthy NIM
        queues real completions for tens of seconds under load, and only
        an affirmative engine-level error may tear a deployment down."""
        from vlm_feedback_loop.services.http_client import HttpResult

        project_id, engine = project_with_db
        dep_id = self._seed_teacher(engine, project_id)

        monkeypatch.setattr(svc, "run_subprocess", _make_subprocess_mock())
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc,
            "resilient_request",
            self._request_mock(
                HttpResult(
                    status_code=None, body=None, error_class="timeout", attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id="test-ep-id")),
        )

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"

    @pytest.mark.asyncio
    async def test_recovery_readoption_gated_on_inference_probe(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Restart recovery must not re-adopt a dead engine behind a live
        HTTP server: docker-running + health 200 + failing completion
        probe → the row goes failed and the container is stopped instead
        of being rebound as a healthy Teacher."""
        from vlm_feedback_loop.services.http_client import HttpResult

        project_id, engine = project_with_db
        dep_id = self._seed_teacher(engine, project_id)
        with Session(engine) as session:
            d = session.get(LocalNimDeployment, dep_id)
            d.status = "running"
            session.commit()

        subprocess_mock = _make_subprocess_mock()
        monkeypatch.setattr(svc, "run_subprocess", subprocess_mock)
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc,
            "resilient_request",
            self._request_mock(
                HttpResult(
                    status_code=500,
                    body={"error": "EngineDeadError"},
                    error_class="endpoint_error",
                    error_detail="HTTP 500: EngineDeadError",
                    attempts=1,
                )
            ),
        )
        mock_create_ep = AsyncMock(return_value=MagicMock(endpoint_id="ep"))
        monkeypatch.setattr(svc, "create_nim_endpoint", mock_create_ep)

        await svc.recover_local_deployments(mock_settings.WORKSPACE_ROOT, mock_settings)

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "failed"
            assert "inference" in db_dep.status_reason.lower()
        mock_create_ep.assert_not_called()
        docker_calls = _docker_calls(subprocess_mock)
        assert ("docker", "stop", "c0ffee123456") in docker_calls
        assert ("docker", "rm", "-f", "c0ffee123456") in docker_calls

    @pytest.mark.asyncio
    async def test_embedding_adoption_sends_no_chat_probe(
        self, project_with_db, mock_settings, monkeypatch
    ):
        """Embedding NIMs serve /v1/embeddings, not /chat/completions — a
        completion probe would false-fail every embedding adoption, so the
        gate is teacher-only (embedding provider resolution live-verifies
        with a real embeddings call, covering that inference path)."""
        project_id, engine = project_with_db

        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id="embedding",
                    role="embedding",
                    nim_container_image=EMBEDDING_NIM_IMAGE,
                    container_name="vlm-embedding-test",
                    host_port=8001,
                    endpoint_url="http://localhost:8001/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        requested_urls: list[str] = []
        from vlm_feedback_loop.services.http_client import HttpResult

        async def _tracking_request(method, url, **kwargs):
            requested_urls.append(url)
            return HttpResult(status_code=200, body={}, error_class=None, attempts=1)

        monkeypatch.setattr(
            svc, "resilient_request", AsyncMock(side_effect=_tracking_request)
        )
        monkeypatch.setattr(svc, "resweep_embedding_tasks", AsyncMock())

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id="embedding",
            role="embedding",
            endpoint_url="http://localhost:8001/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"
        assert not [u for u in requested_urls if u.endswith("/chat/completions")]


class TestOnHealthyImageCapAutoSet:
    """On a verified-healthy teacher, the auto-registered endpoint's
    ``max_images_per_request`` is set from the served NIM's cap so deep
    ICL isn't silently truncated to the conservative per-model default."""

    @pytest.fixture(autouse=True)
    def _patch_subprocess(self, monkeypatch):
        monkeypatch.setattr(svc, "run_subprocess", _make_subprocess_mock())

    @pytest.mark.asyncio
    async def test_endpoint_cap_set_from_resolved_deploy_cap(
        self, project_with_db, mock_settings, monkeypatch
    ):
        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint

        project_id, engine = project_with_db

        # A real endpoint (the one _on_healthy "creates") + a ModelConfig
        # whose per-model cap is 32 (a deep-ICL-friendly local cap).
        endpoint_id = generate_uuid4()
        mc_id = generate_uuid4()
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                NimEndpoint(
                    endpoint_id=endpoint_id,
                    project_id=project_id,
                    display_name="Local Teacher",
                    endpoint_mode="local_system_managed",
                    base_url="http://localhost:8000/v1",
                    auth_mode="none",
                    source_kind="auto_registered_local",
                )
            )
            session.add(
                ModelConfig(
                    model_config_id=mc_id,
                    project_id=project_id,
                    endpoint_id=endpoint_id,
                    model_name=COSMOS_REASON2_8B,
                    context_window_tokens=131072,
                    eligible_roles=["teacher"],
                    supports_image_input=True,
                    max_images_per_request=32,
                )
            )
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id=mc_id,
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-test",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                )
            )
            session.commit()

        # Verification passes (tested separately); create_nim_endpoint
        # returns the pre-seeded endpoint id so _auto_set can load it.
        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        # Inference probe passes (gated behavior tested separately).
        from vlm_feedback_loop.services.http_client import HttpResult

        monkeypatch.setattr(
            svc,
            "resilient_request",
            AsyncMock(
                return_value=HttpResult(
                    status_code=200, body={}, error_class=None, attempts=1
                )
            ),
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id=endpoint_id)),
        )

        await svc._on_healthy(
            deployment_id=dep_id,
            project_id=project_id,
            model_config_id=mc_id,
            role="teacher",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            ep = session.get(NimEndpoint, endpoint_id)
            # The endpoint override now carries the deploy-time cap (32),
            # so image_cap_resolver lets the ICL pruner send up to 32 images
            # instead of falling back to a conservative per-model default.
            assert ep.max_images_per_request == 32
            db_dep = session.get(LocalNimDeployment, dep_id)
            assert db_dep.status == "running"


class TestVerifiedTeacherActivation:
    """A NIM Configuration choice activates only after adoption gates pass."""

    @pytest.mark.asyncio
    async def test_verified_healthy_deployment_selects_requested_teacher(
        self, project_with_db, mock_settings, monkeypatch
    ):
        project_id, engine = project_with_db
        endpoint_id = generate_uuid4()
        deployment_id = generate_uuid4()

        with Session(engine) as session:
            model_config = session.execute(
                select(ModelConfig).where(ModelConfig.model_name == COSMOS_REASON2_8B)
            ).scalar_one()
            model_config_id = model_config.model_config_id
            session.add(
                NimEndpoint(
                    endpoint_id=endpoint_id,
                    project_id=project_id,
                    display_name="Verified local Teacher",
                    endpoint_mode="local_system_managed",
                    base_url="http://localhost:8000/v1",
                    auth_mode="none",
                    source_kind="auto_registered_local",
                )
            )
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=deployment_id,
                    project_id=project_id,
                    model_config_id=model_config_id,
                    role="teacher",
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name="vlm-teacher-activation",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="starting",
                    activate_on_success=True,
                )
            )
            original_teacher_id = session.get(
                Project, project_id
            ).teacher_model_config_id
            assert original_teacher_id != model_config_id
            session.commit()

        monkeypatch.setattr(
            svc,
            "verify_served_model",
            AsyncMock(
                return_value=svc.ServedModelVerification(ok=True, reason="verified")
            ),
        )
        monkeypatch.setattr(
            svc, "_probe_inference_ready", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            svc,
            "create_nim_endpoint",
            AsyncMock(return_value=MagicMock(endpoint_id=endpoint_id)),
        )
        monkeypatch.setattr(svc, "_auto_set_endpoint_image_cap", AsyncMock())

        await svc._on_healthy(
            deployment_id=deployment_id,
            project_id=project_id,
            model_config_id=model_config_id,
            role="teacher",
            endpoint_url="http://localhost:8000/v1",
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

        with Session(engine) as session:
            assert session.get(LocalNimDeployment, deployment_id).status == "running"
            assert (
                session.get(Project, project_id).teacher_model_config_id
                == model_config_id
            )


class TestFailedReplacementRestore:
    """A failed replacement must not leave the previously occupied GPU empty."""

    @pytest.mark.asyncio
    async def test_background_task_registration_failure_restores_resident(
        self, project_with_db, mock_settings, monkeypatch
    ):
        project_id, _engine = project_with_db
        displaced = MagicMock(spec=LocalNimDeployment)
        displaced.project_id = "owner-project"
        displaced.local_nim_deployment_id = "old-deployment"
        monkeypatch.setattr(
            svc,
            "stop_gpu_residents",
            AsyncMock(return_value=[displaced]),
        )
        monkeypatch.setattr(
            svc,
            "_resolve_port",
            AsyncMock(return_value=49500),
        )

        def reject_registration(_task_id: str, coro: object) -> None:
            if hasattr(coro, "close"):
                coro.close()
            raise RuntimeError("background manager unavailable")

        monkeypatch.setattr(
            svc.background_manager,
            "register",
            reject_registration,
        )
        restore = AsyncMock()
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)

        result = await svc.deploy_local_nim(
            project_id=project_id,
            model_config_id="test-mc",
            role="teacher",
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=56,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
            replace_resident=True,
            background=True,
        )

        assert result["deployment"].status == "failed"
        restore.assert_awaited_once_with(
            [displaced],
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_background_preflight_failure_requeues_displaced_resident(
        self, mock_settings, monkeypatch
    ):
        displaced = MagicMock(spec=LocalNimDeployment)
        displaced.project_id = "owner-project"
        displaced.local_nim_deployment_id = "old-deployment"
        failed = MagicMock(spec=LocalNimDeployment)
        failed.status = "failed"
        monkeypatch.setattr(
            svc,
            "_deploy_local_nim_impl",
            AsyncMock(return_value={"deployment": failed, "displaced": [displaced]}),
        )
        restore = AsyncMock()
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)

        await svc._run_queued_local_nim_deploy(
            project_id="new-project",
            model_config_id="new-model",
            role="teacher",
            nim_container_image="new-image",
            gpu_assignment="device=0",
            gpu_memory_minimum_gb=80,
            preferred_port=8000,
            settings=mock_settings,
            workspace_root=mock_settings.WORKSPACE_ROOT,
            student_model_id=None,
            checkpoint_mount=None,
            nim_model_name_path=None,
            nim_served_model_name=None,
            precision_method=None,
            deployment_id="new-deployment",
            host_port=8000,
            displaced=[displaced],
            activate_on_success=True,
        )

        restore.assert_awaited_once_with(
            [displaced],
            workspace_root=mock_settings.WORKSPACE_ROOT,
            settings=mock_settings,
        )

    @pytest.mark.asyncio
    async def test_health_failure_uses_durable_displacement_audit_for_restore(
        self, project_with_db, mock_settings, monkeypatch
    ):
        project_id, engine = project_with_db
        replacement_id = generate_uuid4()
        displaced_id = generate_uuid4()
        with Session(engine) as session:
            session.add_all(
                [
                    LocalNimDeployment(
                        local_nim_deployment_id=replacement_id,
                        project_id=project_id,
                        model_config_id="replacement-model",
                        role="teacher",
                        nim_container_image="replacement-image",
                        container_name="replacement-container",
                        container_id="replacement-id",
                        host_port=8000,
                        endpoint_url="http://localhost:8000/v1",
                        gpu_assignment="device=0",
                        status="starting",
                    ),
                    LocalNimDeployment(
                        local_nim_deployment_id=displaced_id,
                        project_id=project_id,
                        model_config_id="old-model",
                        role="teacher",
                        nim_container_image="old-image",
                        container_name="old-container",
                        host_port=8001,
                        endpoint_url="http://localhost:8001/v1",
                        gpu_assignment="device=0",
                        status="stopped",
                        displaced_by_deployment_id=replacement_id,
                    ),
                ]
            )
            session.commit()

        monkeypatch.setattr(svc, "_teardown_deployment_container", AsyncMock())
        monkeypatch.setattr(svc, "disable_teacher_resident_endpoints", MagicMock())
        restore = AsyncMock()
        monkeypatch.setattr(svc, "restore_displaced_deployments", restore)

        await svc._fail_active_deployment(
            deployment_id=replacement_id,
            project_id=project_id,
            workspace_root=mock_settings.WORKSPACE_ROOT,
            endpoint_url="http://localhost:8000/v1",
            role="teacher",
            reason="verification failed",
            settings=mock_settings,
        )

        restored_rows = restore.await_args.args[0]
        assert [row.local_nim_deployment_id for row in restored_rows] == [displaced_id]


class TestResolveSharedImageDeployEnv:
    """Profile-pin forwarding for the deploy docker command (F-A remedial flow)."""

    def _mcid(self, engine, model_name):
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        with Session(engine) as session:
            mc = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_name == model_name)
                .first()
            )
            return mc.model_config_id

    def _set_metadata(self, engine, mcid, metadata):
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        with Session(engine) as session:
            mc = session.get(ModelConfig, mcid)
            mc.local_deploy_metadata = metadata
            session.commit()

    def test_single_image_model_without_pin_is_unchanged(self, project_with_db):
        """No pin ⇒ all three selectors None: the docker command for a
        single-image teacher stays byte-for-byte what it always was."""
        _pid, engine = project_with_db
        mcid = self._mcid(engine, "nvidia/cosmos-reason2-2b")
        assert svc._resolve_shared_image_deploy_env(engine, mcid) == (
            None,
            None,
            None,
        )

    def test_single_image_model_profile_pin_is_forwarded(self, project_with_db):
        """An operator-pinned profile on a single-image NIM must reach the
        container. The pin used to be silently dropped because forwarding was
        gated on ``nim_model_size`` — which made the documented remedial flow
        (pin a stable profile when the auto-selected one is unusable) a no-op
        exactly when it was needed: cosmos-reason2-2b:1.6.0's auto-selected
        fp8 profile crashes on H100 NVL and the bf16 pin could not be applied."""
        _pid, engine = project_with_db
        mcid = self._mcid(engine, "nvidia/cosmos-reason2-2b")
        self._set_metadata(
            engine,
            mcid,
            {
                "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                "nim_gpu_memory_minimum_gb": 36,
                "nim_model_profile": "prof-bf16-tp1",
            },
        )
        assert svc._resolve_shared_image_deploy_env(engine, mcid) == (
            None,
            "prof-bf16-tp1",
            None,
        )

    def test_shared_image_model_keeps_size_profile_and_served_name(
        self, project_with_db
    ):
        """Shared-image (size-selected) models still resolve all three
        selectors — the original cosmos3 nano/super behavior is unchanged."""
        _pid, engine = project_with_db
        mcid = self._mcid(engine, "nvidia/cosmos3-nano-reasoner")
        self._set_metadata(
            engine,
            mcid,
            {
                "nim_container_image": "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0",
                "nim_model_size": "nano",
                "nim_model_profile": "prof-nano-bf16",
                "nim_gpu_memory_minimum_gb": 56,
            },
        )
        assert svc._resolve_shared_image_deploy_env(engine, mcid) == (
            "nano",
            "prof-nano-bf16",
            "nvidia/cosmos3-nano-reasoner",
        )


class TestExtraContainerEnv:
    """Operator env passthrough: ``local_deploy_metadata.extra_container_env``
    rides the system-managed docker run, so live-validated NIM remediations
    (NIM_DISABLE_CUDA_GRAPH=1 for the CR2 Hopper XID-31 crash,
    NIM_MAX_MODEL_LEN context clamps) can be applied through the product
    instead of a manual docker run that forfeits lifecycle management."""

    def _mcid(self, engine, model_name):
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        with Session(engine) as session:
            mc = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_name == model_name)
                .first()
            )
            return mc.model_config_id

    def _set_metadata(self, engine, mcid, metadata):
        from vlm_feedback_loop.db.models.model_config import ModelConfig

        with Session(engine) as session:
            mc = session.get(ModelConfig, mcid)
            mc.local_deploy_metadata = metadata
            session.commit()

    def test_builder_appends_sorted_env_before_image(self):
        """With extra_env the command is the byte-identical base command
        plus sorted ``-e KEY=VALUE`` pairs directly before the image."""
        kwargs = {
            "nim_container_image": COSMOS_REASON2_8B_NIM_IMAGE,
            "container_name": "vlm-teacher-test",
            "gpu_assignment": "device=0",
            "host_port": 8000,
            "role": "teacher",
            "max_images_per_request": 999,
        }
        base = svc._build_docker_run_command(**kwargs)
        with_env = svc._build_docker_run_command(
            **kwargs,
            extra_env={"NIM_MAX_MODEL_LEN": "65536", "NIM_DISABLE_CUDA_GRAPH": "1"},
        )
        assert with_env == base[:-1] + [
            "-e",
            "NIM_DISABLE_CUDA_GRAPH=1",
            "-e",
            "NIM_MAX_MODEL_LEN=65536",
            COSMOS_REASON2_8B_NIM_IMAGE,
        ]

    def test_display_mirrors_extra_env(self):
        """The operator-facing display/handoff command carries the same env."""
        cmd = svc.docker_run_command_display(
            nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
            container_name="vlm-teacher-test",
            gpu_assignment="device=0",
            host_port=8000,
            role="teacher",
            extra_env={"NIM_DISABLE_CUDA_GRAPH": "1"},
        )
        assert "-e NIM_DISABLE_CUDA_GRAPH=1" in cmd

    def test_resolver_validates_keys_and_refuses_reserved(self, project_with_db):
        """Keys must be UPPER_SNAKE_CASE; builder-owned keys (NGC_API_KEY,
        the NIM_* selectors) are refused so metadata can't shadow the
        secret or the served-model identity; numbers coerce to strings."""
        _pid, engine = project_with_db
        mcid = self._mcid(engine, "nvidia/cosmos-reason2-2b")
        self._set_metadata(
            engine,
            mcid,
            {
                "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
                "nim_gpu_memory_minimum_gb": 36,
                "extra_container_env": {
                    "NIM_DISABLE_CUDA_GRAPH": "1",
                    "NIM_MAX_MODEL_LEN": 65536,
                    "NGC_API_KEY": "shadowed-secret",
                    "lower_case": "invalid",
                },
            },
        )
        assert svc._resolve_extra_container_env(engine, mcid) == {
            "NIM_DISABLE_CUDA_GRAPH": "1",
            "NIM_MAX_MODEL_LEN": "65536",
        }

    def test_resolver_empty_without_metadata_key(self, project_with_db):
        """Absent key ⇒ empty dict ⇒ the docker command stays byte-for-byte
        unchanged (the established F-L10 invariant)."""
        _pid, engine = project_with_db
        mcid = self._mcid(engine, "nvidia/cosmos-reason2-2b")
        assert svc._resolve_extra_container_env(engine, mcid) == {}


class TestCommitWithLockRetry:
    @pytest.mark.asyncio
    async def test_retries_lock_contention_then_succeeds(self, monkeypatch):
        """A transient 'database is locked' must not kill a state
        transition. WAL's busy_timeout does not cover snapshot-upgrade
        conflicts (read-then-write vs a concurrent writer) — observed
        live: a healthy NIM was reported failed because the promote
        commit died on first contention."""
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        calls = {"n": 0}

        from sqlalchemy.exc import OperationalError

        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise OperationalError("stmt", {}, Exception("database is locked"))

        await svc._commit_with_lock_retry(fn, what="test")
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_non_lock_errors_reraise_immediately(self, monkeypatch):
        from sqlalchemy.exc import OperationalError

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise OperationalError("stmt", {}, Exception("no such table: x"))

        with pytest.raises(OperationalError):
            await svc._commit_with_lock_retry(fn, what="test")
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_final_attempt(self, monkeypatch):
        from sqlalchemy.exc import OperationalError

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise OperationalError("stmt", {}, Exception("database is locked"))

        with pytest.raises(OperationalError):
            await svc._commit_with_lock_retry(fn, what="test", attempts=3)
        assert calls["n"] == 3


# ── Cross-project Teacher resident reuse ─────────────────────────────────────


class TestTeacherResidentReuse:
    """A running Teacher is host infrastructure and can serve another project."""

    def test_matching_teacher_is_attached_and_all_consumers_disable_on_stop(
        self, project_with_db, mock_settings
    ):
        owner_id, owner_engine = project_with_db
        from vlm_feedback_loop.services.project_service import (
            create_project,
            get_project_engine,
        )

        target = create_project(
            name="Target project",
            description=None,
            settings=mock_settings,
        )
        target_engine = get_project_engine(
            target.project_id, mock_settings.WORKSPACE_ROOT
        )
        assert target_engine is not None

        with Session(owner_engine) as session:
            owner_mc = session.execute(
                select(ModelConfig).where(
                    ModelConfig.model_name == COSMOS3_SUPER_REASONER
                )
            ).scalar_one()
            owner_mc_id = owner_mc.model_config_id
        with Session(target_engine) as session:
            target_mc = session.execute(
                select(ModelConfig).where(
                    ModelConfig.model_name == COSMOS3_SUPER_REASONER
                )
            ).scalar_one()
            target_mc_id = target_mc.model_config_id

        deployment_id = generate_uuid4()
        endpoint_url = "http://localhost:8001/v1"
        with Session(owner_engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=deployment_id,
                    project_id=owner_id,
                    model_config_id=owner_mc_id,
                    role="teacher",
                    nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
                    container_name=f"vlm-teacher-{owner_id[:8]}",
                    container_id="container-1",
                    host_port=8001,
                    endpoint_url=endpoint_url,
                    gpu_assignment="device=0",
                    status="running",
                    deployed_at=utc_now(),
                )
            )
            owner_endpoint = NimEndpoint(
                endpoint_id=generate_uuid4(),
                project_id=owner_id,
                display_name="Owner local Teacher",
                endpoint_mode="local_system_managed",
                base_url=endpoint_url,
                api_format="openai_compatible",
                auth_mode="none",
                models_path="/models",
                health_ready_path="/health/ready",
                health_live_path="/health/live",
                metrics_path="/metrics",
                is_enabled=True,
                last_probe_status="healthy",
                source_kind="auto_registered_local",
                local_nim_deployment_id=deployment_id,
                max_images_per_request=999,
                image_cap_support="supported",
            )
            session.add(owner_endpoint)
            owner_mc = session.get(ModelConfig, owner_mc_id)
            assert owner_mc is not None
            owner_mc.endpoint_id = owner_endpoint.endpoint_id
            session.commit()

        reuse_result = svc.reuse_first_compatible_running_teacher_for_project(
            project_id=target.project_id,
            workspace_root=mock_settings.WORKSPACE_ROOT,
        )

        assert reuse_result is not None
        selected_model_config_id, reused = reuse_result
        assert selected_model_config_id == target_mc_id
        assert reused is not None
        assert reused.deployment_id == deployment_id
        assert reused.project_name == "Test Project"
        assert reused.model_name == COSMOS3_SUPER_REASONER

        with Session(target_engine) as session:
            attached_mc = session.get(ModelConfig, target_mc_id)
            assert attached_mc is not None
            attached_endpoint = session.get(NimEndpoint, attached_mc.endpoint_id)
            assert attached_endpoint is not None
            assert attached_endpoint.base_url == endpoint_url
            assert attached_endpoint.local_nim_deployment_id == deployment_id
            assert attached_endpoint.max_images_per_request == 999
            assert attached_endpoint.is_enabled is True

        # The normal fresh-project path performs the same exact reuse and
        # returns with that local model already selected.
        fresh = create_project(
            name="Fresh project",
            description=None,
            settings=mock_settings,
        )
        fresh_engine = get_project_engine(
            fresh.project_id, mock_settings.WORKSPACE_ROOT
        )
        assert fresh_engine is not None
        with Session(fresh_engine) as session:
            selected_mc = session.get(ModelConfig, fresh.teacher_model_config_id)
            assert selected_mc is not None
            assert selected_mc.model_name == COSMOS3_SUPER_REASONER
            selected_endpoint = session.get(NimEndpoint, selected_mc.endpoint_id)
            assert selected_endpoint is not None
            assert selected_endpoint.local_nim_deployment_id == deployment_id

        assert (
            svc.disable_teacher_resident_endpoints(
                mock_settings.WORKSPACE_ROOT, deployment_id
            )
            == 3
        )
        for engine in (owner_engine, target_engine, fresh_engine):
            with Session(engine) as session:
                endpoints = session.execute(
                    select(NimEndpoint).where(
                        NimEndpoint.local_nim_deployment_id == deployment_id
                    )
                ).scalars()
                assert all(endpoint.is_enabled is False for endpoint in endpoints)


# ── Stale-failure detection (matches_active_role_config) ─────────────────────


class TestMatchesActiveRoleConfig:
    """A failed teacher deploy is stale evidence once the project's active
    Teacher no longer references its model config — the SME switched
    Teachers and labeling works, so the failure banner must not outlive
    its truth. Non-teacher roles never suppress on this axis."""

    def _insert_deployment(self, engine, project_id, *, role, model_config_id):
        dep_id = generate_uuid4()
        with Session(engine) as session:
            session.add(
                LocalNimDeployment(
                    local_nim_deployment_id=dep_id,
                    project_id=project_id,
                    model_config_id=model_config_id,
                    role=role,
                    nim_container_image=COSMOS_REASON2_8B_NIM_IMAGE,
                    container_name=f"vlm-{role}-{dep_id[:8]}",
                    host_port=8000,
                    endpoint_url="http://localhost:8000/v1",
                    gpu_assignment="device=0",
                    status="failed",
                    status_reason="Health check timed out after 1200s",
                )
            )
            session.commit()
        return dep_id

    def test_teacher_failure_stale_when_active_teacher_moved_on(
        self, project_with_db, mock_settings
    ):
        """The GT-project case: a 17-hour-old cosmos deploy failure while
        the active Teacher is a different (hosted/self-hosted) config."""
        project_id, engine = project_with_db
        dep_id = self._insert_deployment(
            engine, project_id, role="teacher", model_config_id="orphaned-mc"
        )

        deployments = svc.list_local_deployments(
            project_id, mock_settings.WORKSPACE_ROOT
        )
        flags = svc.matches_active_role_config(
            project_id, mock_settings.WORKSPACE_ROOT, deployments
        )
        assert flags[dep_id] is False

    def test_teacher_failure_current_when_active_teacher_is_that_config(
        self, project_with_db, mock_settings
    ):
        """The fresh-FTUE case: the failed deploy's config IS the active
        Teacher (repointing only happens on deploy success), so the
        failure is live and must stay visible."""
        project_id, engine = project_with_db
        from vlm_feedback_loop.db.models.project import Project

        with Session(engine) as session:
            project = session.get(Project, project_id)
            assert project is not None
            active_mc = project.teacher_model_config_id
        assert active_mc is not None
        dep_id = self._insert_deployment(
            engine, project_id, role="teacher", model_config_id=active_mc
        )

        deployments = svc.list_local_deployments(
            project_id, mock_settings.WORKSPACE_ROOT
        )
        flags = svc.matches_active_role_config(
            project_id, mock_settings.WORKSPACE_ROOT, deployments
        )
        assert flags[dep_id] is True

    def test_embedding_role_never_suppressed(self, project_with_db, mock_settings):
        """Embedding NIM config is deployment-scoped, not a project role
        config — its failures always map True (visible)."""
        project_id, engine = project_with_db
        dep_id = self._insert_deployment(
            engine, project_id, role="embedding", model_config_id="emb-mc"
        )

        deployments = svc.list_local_deployments(
            project_id, mock_settings.WORKSPACE_ROOT
        )
        flags = svc.matches_active_role_config(
            project_id, mock_settings.WORKSPACE_ROOT, deployments
        )
        assert flags[dep_id] is True
