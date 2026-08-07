# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the environment assessment service (services/environment.py)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from vlm_feedback_loop.model_catalog_constants import (
    COSMOS3_NANO_REASONER,
    COSMOS3_REASONER_NIM_IMAGE,
    COSMOS3_SUPER_REASONER,
    COSMOS_REASON2_2B,
    COSMOS_REASON2_2B_NIM_IMAGE,
    COSMOS_REASON2_8B,
    COSMOS_REASON2_8B_NIM_IMAGE,
    EMBEDDING_MODEL_ID,
    EMBEDDING_NIM_GPU_MIN_GB,
    EMBEDDING_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN,
    NEMOTRON_3_NANO_OMNI_GPU_MIN_GB,
    NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
    NEMOTRON_3_NANO_OMNI_REASONING,
)
from vlm_feedback_loop.services.environment import (
    GpuInfo,
    _assess_local_deployable_models,
    _build_missing_prerequisites,
    assess_environment,
    gpu_memory_meets_floor,
    probe_gpu_inventory,
    run_subprocess,
)
from vlm_feedback_loop.services.local_nim_service import ActiveNimResident

_RETIRED_MISTRAL_LARGE_3 = "mistralai/mistral-large-3-675b-instruct-2512"

# ── _assess_local_deployable_models ─────────────────────────────────────────


class TestAssessLocalDeployableModels:
    def test_nominal_gpu_capacity_tolerates_nvidia_smi_reporting_gap(self):
        assert gpu_memory_meets_floor(79.6, 80) is True
        assert gpu_memory_meets_floor(79.0, 80) is False

    def test_large_gpu_fits_all(self):
        # 96 GB fits every locally-deployable model, including the largest —
        # Cosmos 3 Super-Reasoner at 88 GB. (An 80 GB A100 would NOT fit Super;
        # see test_super_reasoner_needs_more_than_80gb below.)
        gpus = [
            GpuInfo(
                name="RTX PRO 6000",
                memory_total_mb=98304,
                compute_capability=12.0,
            )
        ]  # 96 GB
        models = _assess_local_deployable_models(gpus)
        # Four Cosmos variants plus the specialized Omni Teacher NIM.
        assert len(models) >= 5
        for m in models:
            assert m.fits is True

    def test_super_reasoner_needs_more_than_80gb(self):
        # Cosmos 3 Super-Reasoner (88 GB minimum) does NOT fit an 80 GB A100,
        # while the ~56 GB-class models do.
        gpus = [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]  # 80 GB
        fits = {m.model_name: m.fits for m in _assess_local_deployable_models(gpus)}
        assert fits.get(COSMOS3_SUPER_REASONER) is False
        assert fits.get(COSMOS3_NANO_REASONER) is True
        assert fits.get(COSMOS_REASON2_8B) is True
        assert fits.get(NEMOTRON_3_NANO_OMNI_REASONING) is False

    def test_small_gpu_fits_none(self):
        gpus = [GpuInfo(name="RTX 3060", memory_total_mb=12288)]  # 12 GB
        models = _assess_local_deployable_models(gpus)
        for m in models:
            assert m.fits is False

    def test_medium_gpu_partial_fit(self):
        gpus = [GpuInfo(name="RTX 4090", memory_total_mb=24576)]  # 24 GB
        models = _assess_local_deployable_models(gpus)
        fits_map = {m.model_name: m.fits for m in models}
        # 24 GB < 36 GB minimum for nvidia/cosmos-reason2-2b
        assert fits_map.get(COSMOS_REASON2_2B) is False
        # 24 GB < 56 GB minimum for nvidia/cosmos-reason2-8b
        assert fits_map.get(COSMOS_REASON2_8B) is False

    def test_no_gpus_nothing_fits(self):
        models = _assess_local_deployable_models([])
        for m in models:
            assert m.fits is False

    def test_only_models_with_local_deploy_metadata(self):
        models = _assess_local_deployable_models(
            [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]
        )
        # Hosted-only entries, including the retired Mistral Large identity,
        # are excluded from the result.
        names = {m.model_name for m in models}
        assert COSMOS_REASON2_8B in names
        assert COSMOS_REASON2_2B in names
        assert _RETIRED_MISTRAL_LARGE_3 not in names

    def test_custom_catalog(self):
        gpus = [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]
        catalog = [
            {
                "model_name": "test-model",
                "local_deploy_metadata": {
                    "nim_container_image": "nvcr.io/test:1.0",
                    "nim_gpu_memory_minimum_gb": 40,
                    "preferred_host_port": 8000,
                },
            },
            {
                "model_name": "no-local",
                "local_deploy_metadata": None,
            },
        ]
        models = _assess_local_deployable_models(gpus, catalog)
        assert len(models) == 1
        assert models[0].model_name == "test-model"
        assert models[0].fits is True


# ── _build_missing_prerequisites ────────────────────────────────────────────


class TestBuildMissingPrerequisites:
    def test_all_available(self):
        result = _build_missing_prerequisites(
            docker_available=True,
            nvidia_toolkit_available=True,
            nvidia_api_key_configured=True,
            ngc_api_key_configured=True,
            containerized=False,
        )
        assert result == []

    def test_docker_missing(self):
        result = _build_missing_prerequisites(
            docker_available=False,
            nvidia_toolkit_available=False,
            nvidia_api_key_configured=True,
            ngc_api_key_configured=True,
            containerized=False,
        )
        checks = [p.check for p in result]
        assert "Docker" in checks
        # Toolkit check only fires when Docker is present
        assert "NVIDIA Container Toolkit" not in checks

    def test_docker_missing_hint_points_at_real_script(self):
        """The install hint must reference scripts/setup-local.sh — the
        path that exists in the repo (a bare ./setup-local.sh shipped for
        months and pointed at nothing)."""
        result = _build_missing_prerequisites(
            docker_available=False,
            nvidia_toolkit_available=False,
            nvidia_api_key_configured=True,
            ngc_api_key_configured=True,
            containerized=False,
        )
        docker = next(p for p in result if p.check == "Docker")
        assert "./scripts/setup-local.sh" in docker.install_hint

    def test_docker_missing_inside_container_gets_honest_hint(self):
        """Containerized mode has no docker CLI/socket by design — the hint
        must say local NIMs need local-source mode, not 'install Docker'
        on a host that is already running Docker."""
        result = _build_missing_prerequisites(
            docker_available=False,
            nvidia_toolkit_available=False,
            nvidia_api_key_configured=True,
            ngc_api_key_configured=True,
            containerized=True,
        )
        docker = next(p for p in result if p.check == "Docker")
        assert "local-source mode" in docker.install_hint
        assert "Install Docker" not in docker.install_hint

    def test_toolkit_missing_with_docker(self):
        result = _build_missing_prerequisites(
            docker_available=True,
            nvidia_toolkit_available=False,
            nvidia_api_key_configured=True,
            ngc_api_key_configured=True,
            containerized=False,
        )
        checks = [p.check for p in result]
        assert "NVIDIA Container Toolkit" in checks

    def test_no_credentials(self):
        result = _build_missing_prerequisites(
            docker_available=True,
            nvidia_toolkit_available=True,
            nvidia_api_key_configured=False,
            ngc_api_key_configured=False,
            containerized=False,
        )
        checks = [p.check for p in result]
        assert "API credentials" in checks


class TestIsContainerized:
    def test_reflects_dockerenv_marker(self, tmp_path, monkeypatch):
        from vlm_feedback_loop.services import environment

        marker = tmp_path / ".dockerenv"
        monkeypatch.setattr(environment, "_DOCKERENV_PATH", marker)
        assert environment.is_containerized() is False
        marker.touch()
        assert environment.is_containerized() is True


class TestRunSubprocessEnvironment:
    @pytest.mark.asyncio
    async def test_timeout_uses_shared_kill_and_reap_boundary(self, monkeypatch):
        from vlm_feedback_loop.services import environment

        class FakeProcess:
            returncode = None

        async def fake_create(*_args, **_kwargs):
            return FakeProcess()

        communication = AsyncMock(side_effect=TimeoutError)
        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create)
        monkeypatch.setattr(environment, "communicate_with_timeout", communication)

        result = await environment.run_subprocess("docker", timeout_s=2.5)

        assert result == (-1, "", "Command timed out after 2.5s")
        communication.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_private_overrides_reach_only_child_environment(self, monkeypatch):
        """Private overrides merge with inheritance without mutating the parent."""
        sentinel = "SENTINEL_CHILD_ONLY_NGC"
        monkeypatch.setenv("HARMLESS_INHERITED", "present")
        parent_before = dict(os.environ)
        captured: dict[str, object] = {}

        class FakeProcess:
            returncode = 0

            async def communicate(self, input=None):
                captured["input"] = input
                return b"ok", b""

        async def fake_create(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return FakeProcess()

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            fake_create,
        )
        result = await run_subprocess(
            "docker",
            "run",
            "-e",
            "NGC_API_KEY",
            secret_env={"NGC_API_KEY": sentinel},
        )

        assert result == (0, "ok", "")
        assert sentinel not in captured["args"]
        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        child_env = kwargs["env"]
        assert child_env["NGC_API_KEY"] == sentinel
        assert child_env["HARMLESS_INHERITED"] == "present"
        assert dict(os.environ) == parent_before

    @pytest.mark.asyncio
    async def test_private_values_are_redacted_from_output_and_spawn_error(
        self, monkeypatch
    ):
        sentinel = "SENTINEL_CHILD_DIAGNOSTIC"

        class EchoProcess:
            returncode = 7

            async def communicate(self, input=None):
                return (
                    f"stdout around {sentinel}".encode(),
                    f"stderr around {sentinel}".encode(),
                )

        async def echo_create(*args, **kwargs):
            return EchoProcess()

        monkeypatch.setattr("asyncio.create_subprocess_exec", echo_create)
        rc, stdout, stderr = await run_subprocess(
            "child",
            secret_env={"PRIVATE_TOKEN": sentinel},
        )
        assert rc == 7
        assert stdout == "stdout around [REDACTED]"
        assert stderr == "stderr around [REDACTED]"

        async def failing_create(*args, **kwargs):
            raise OSError(f"spawn failed around {sentinel}")

        monkeypatch.setattr("asyncio.create_subprocess_exec", failing_create)
        rc, stdout, stderr = await run_subprocess(
            "child",
            secret_env={"PRIVATE_TOKEN": sentinel},
        )
        assert (rc, stdout) == (-1, "")
        assert stderr == "spawn failed around [REDACTED]"

    @pytest.mark.asyncio
    async def test_private_stdin_is_redacted_if_child_echoes_it(self, monkeypatch):
        """Password-stdin transport must not make an echoing child diagnostic unsafe."""
        sentinel = "SENTINEL_PRIVATE_STDIN"

        class EchoProcess:
            returncode = 1

            async def communicate(self, input=None):
                assert input == sentinel.encode()
                return b"", f"login rejected {sentinel}".encode()

        async def echo_create(*args, **kwargs):
            return EchoProcess()

        monkeypatch.setattr("asyncio.create_subprocess_exec", echo_create)
        rc, stdout, stderr = await run_subprocess(
            "docker",
            "login",
            "--password-stdin",
            stdin_input=sentinel,
        )

        assert (rc, stdout) == (1, "")
        assert stderr == "login rejected [REDACTED]"


class TestProbeGpuInventory:
    @pytest.mark.asyncio
    async def test_parses_memory_and_compute_capability_from_nvidia_smi(
        self, monkeypatch
    ):
        """The inventory preserves every hardware field used by placement."""
        probe = AsyncMock(
            return_value=(
                0,
                "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 12.0\n"
                "NVIDIA A100-SXM4-80GB, 81920, 8.0",
                "",
            )
        )
        monkeypatch.setattr(
            "vlm_feedback_loop.services.environment.run_subprocess", probe
        )

        gpus = await probe_gpu_inventory()

        assert gpus == [
            GpuInfo(
                name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                memory_total_mb=97887,
                compute_capability=12.0,
            ),
            GpuInfo(
                name="NVIDIA A100-SXM4-80GB",
                memory_total_mb=81920,
                compute_capability=8.0,
            ),
        ]
        probe.assert_awaited_once_with(
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
            timeout_s=10.0,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("returncode", "stdout"),
        [
            (1, "driver communication failed"),
            (0, ""),
            (0, "malformed inventory line"),
        ],
    )
    async def test_returns_empty_inventory_when_nvidia_smi_is_unusable(
        self, monkeypatch, returncode, stdout
    ):
        """A missing, failed, empty, or malformed probe cannot invent GPUs."""
        monkeypatch.setattr(
            "vlm_feedback_loop.services.environment.run_subprocess",
            AsyncMock(return_value=(returncode, stdout, "probe error")),
        )

        assert await probe_gpu_inventory() == []


# ── assess_environment ──────────────────────────────────────────────────────


class TestAssessEnvironment:
    """Tests for the full assessment orchestrator.

    Monkeypatches subprocess helpers to avoid real system calls.
    """

    @pytest.fixture()
    def mock_env(self, monkeypatch, tmp_workspace):
        """Set up mock environment with configurable state."""
        from conftest import make_settings
        from vlm_feedback_loop.config import Settings
        from vlm_feedback_loop.db.engine import init_deployment_db

        # Ensure deployment.db exists
        init_deployment_db(tmp_workspace)

        def _make_settings(
            nvidia_api_key: str | None = None,
            ngc_api_key: str | None = None,
        ) -> Settings:
            return make_settings(
                tmp_workspace,
                NVIDIA_API_KEY=nvidia_api_key,
                NGC_API_KEY=ngc_api_key,
            )

        return _make_settings

    @pytest.fixture(autouse=True)
    def _mock_subprocess(self, monkeypatch):
        """Default: no Docker, no GPU."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod,
            "check_docker_available",
            AsyncMock(return_value=(False, "Docker not available")),
        )
        monkeypatch.setattr(
            env_mod,
            "check_nvidia_toolkit",
            AsyncMock(return_value=(False, "Toolkit not available")),
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(return_value=[]),
        )

    @pytest.mark.asyncio
    async def test_api_key_only_recommends_hosted(self, mock_env):
        settings = mock_env(nvidia_api_key="nvapi-test")
        result = await assess_environment(settings)

        assert result["hosted_nim_available"] is True
        assert result["nvidia_api_key_configured"] is True
        assert result["local_deploy_available"] is False
        assert result["recommended_teacher_mode"] == "hosted"
        assert result["recommended_embedding_mode"] == "hosted"

    @pytest.mark.asyncio
    async def test_explicitly_disabled_embeddings_recommend_phash(self, mock_env):
        """An operator who disables embedding NIMs gets the supported pHash
        path, even if an unrelated hosted credential remains configured."""
        settings = mock_env(nvidia_api_key="nvapi-test")
        settings.EMBEDDING_PROVIDER = "none"

        result = await assess_environment(settings)

        assert result["recommended_embedding_mode"] == "none"

    @pytest.mark.asyncio
    async def test_exposes_default_teacher_model_verbatim(self, mock_env):
        """The environment surfaces the effective DEFAULT_TEACHER_MODEL so the
        Confirm Defaults screen preselects it without hardcoding a model name.
        Guards the single-source contract: a config override or a default
        reseat must reach the UI through this field, not a frontend copy."""
        settings = mock_env(nvidia_api_key="nvapi-test")
        result = await assess_environment(settings)

        assert result["default_teacher_model_name"] == settings.DEFAULT_TEACHER_MODEL

    @pytest.mark.asyncio
    async def test_no_credentials_recommends_hosted(self, mock_env):
        """The recommendation surface never falls back to ``"none"``.
        With no keys and no GPU, the recommendation is hosted-only
        (the SME needs an NVIDIA_API_KEY, and the NIM Configuration
        screen prompts for it).
        """
        settings = mock_env()
        result = await assess_environment(settings)

        assert result["hosted_nim_available"] is False
        assert result["local_deploy_available"] is False
        # No "none" — always recommend the ideal forward path.
        assert result["recommended_teacher_mode"] == "hosted"
        assert result["recommended_embedding_mode"] == "hosted"

    @pytest.mark.asyncio
    async def test_full_local_recommends_local_and_hosted(self, mock_env, monkeypatch):
        """API key + Docker + GPU + NGC → hosted teacher, local embedding."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod,
            "check_docker_available",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "check_nvidia_toolkit",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)
                ]
            ),
        )

        settings = mock_env(nvidia_api_key="nvapi-test", ngc_api_key="ngc-test")
        result = await assess_environment(settings)

        assert result["local_deploy_available"] is True
        assert result["docker_available"] is True
        assert result["nvidia_toolkit_available"] is True
        assert len(result["gpus"]) == 1
        assert result["gpus"][0]["name"] == "NVIDIA A100-SXM4-80GB"
        assert result["recommended_teacher_mode"] == "hosted"
        assert result["recommended_embedding_mode"] == "local"

    @pytest.mark.asyncio
    async def test_gpu_only_no_api_key(self, mock_env, monkeypatch):
        """Docker + GPU +
        NGC but no NVIDIA_API_KEY → recommend LOCAL Teacher (Cosmos
        Reason2, sized to the GPU): when NVIDIA_API_KEY is not configured
        and local deployment is available, recommend the local Teacher.
        An always-hosted recommendation would hide the local path behind
        a tiny text link on the FTUE Teacher setup screen; this branch
        makes Cosmos Reason2 a first-class peer on a GPU box with no API
        key."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod,
            "check_docker_available",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "check_nvidia_toolkit",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)
                ]
            ),
        )

        settings = mock_env(ngc_api_key="ngc-test")
        result = await assess_environment(settings)

        # Teacher: local (no API key + GPU fits the ≥56 GB tier at
        # 81920 MiB ≈ 80 GB, well over the 56 GB minimum).
        assert result["recommended_teacher_mode"] == "local"
        # At the 56 GB tier CR3-Nano is the default (preference rank > CR2-8B);
        # both have a 56 GB minimum and fit on an 80 GB GPU.
        assert result["recommended_local_teacher_model_name"] == COSMOS3_NANO_REASONER
        assert result["recommended_local_teacher_gpu_memory_minimum_gb"] == 56
        # Embedding: hosted — the recommendation is placement-aware, and
        # the planned local Teacher reserves the host's only GPU
        # (one-NIM-per-GPU), leaving no device for the embedding NIM.
        assert result["recommended_embedding_mode"] == "hosted"
        assert result["embedding_deployment"]["fits"] is False

    @pytest.mark.asyncio
    async def test_gpu_only_no_keys_recommends_local(self, mock_env, monkeypatch):
        """No NVIDIA_API_KEY
        AND no NGC key, but Docker + GPU present → still recommend local
        Teacher. NGC is collected on the FTUE key step; the env
        recommendation does NOT gate on NGC because the SME may walk
        back and add it."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod,
            "check_docker_available",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "check_nvidia_toolkit",
            AsyncMock(return_value=(True, None)),
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)
                ]
            ),
        )

        result = await assess_environment(mock_env())

        assert result["recommended_teacher_mode"] == "local"
        assert result["recommended_local_teacher_model_name"] == COSMOS3_NANO_REASONER

    @pytest.mark.asyncio
    async def test_gpu_fits_2b_only_picks_2b(self, mock_env, monkeypatch):
        """A GPU with 40 GB fits Cosmos
        Reason2 2B (36 GB min) but not 8B (56 GB min) → pick 2B."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="L40S", memory_total_mb=40960)]),
        )

        result = await assess_environment(mock_env())

        assert result["recommended_teacher_mode"] == "local"
        assert result["recommended_local_teacher_model_name"] == COSMOS_REASON2_2B
        assert result["recommended_local_teacher_gpu_memory_minimum_gb"] == 36

    @pytest.mark.asyncio
    async def test_embedding_only_gpu_uses_local_embedding(self, mock_env, monkeypatch):
        """A GPU with 24 GB fits neither
        Cosmos variant → recommended_teacher_mode falls back to "hosted"
        (forward-looking — the FTUE Teacher setup screen prompts for an
        API key). Local
        teacher fields stay null so the frontend collapses the local
        peer card.

        This is the small-GPU host class: every GPU is below every
        Teacher floor but at the 24 GB embedding floor, so the GPU goes
        to the embedding NIM (hosted Teacher + local embeddings) — no
        planned local Teacher reserves it."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(return_value=[GpuInfo(name="NVIDIA L4", memory_total_mb=24576)]),
        )

        result = await assess_environment(mock_env())

        # No teacher fits: fall back to forward-looking hosted recommendation.
        assert result["recommended_teacher_mode"] == "hosted"
        assert result["recommended_local_teacher_model_name"] is None
        assert result["recommended_local_teacher_image"] is None
        assert result["recommended_local_teacher_gpu_memory_minimum_gb"] is None
        # The supported 24 GB eligibility floor fits → local.
        assert result["embedding_deployment"]["fits"] is True
        assert result["recommended_embedding_mode"] == "local"

    @pytest.mark.asyncio
    async def test_unvalidated_24gb_gpu_does_not_recommend_embedding_nim(
        self, mock_env, monkeypatch
    ):
        """Memory alone must not present an unvalidated GPU as supported."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA GeForce RTX 4090", memory_total_mb=24576)
                ]
            ),
        )

        result = await assess_environment(mock_env())

        assert result["embedding_deployment"]["fits"] is False
        assert result["recommended_embedding_mode"] == "hosted"

    @pytest.mark.asyncio
    async def test_key_set_with_gpu_still_populates_local_fields(
        self, mock_env, monkeypatch
    ):
        """When NVIDIA_API_KEY is
        configured AND a GPU fits Cosmos Reason2, the teacher
        recommendation stays "hosted" (Mistral Large 3 is instant) but
        the local teacher fields ARE populated so the FTUE Teacher setup
        screen can render the hybrid peer card ("Also deploy Cosmos
        Reason2 locally?")."""
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)
                ]
            ),
        )

        result = await assess_environment(
            mock_env(nvidia_api_key="nvapi-test", ngc_api_key="ngc-test")
        )

        # Key set → hosted recommendation (instant), local stays available
        # for the hybrid peer card. CR3-Nano is the big-GPU default.
        assert result["recommended_teacher_mode"] == "hosted"
        assert result["recommended_local_teacher_model_name"] == COSMOS3_NANO_REASONER
        assert result["recommended_local_teacher_image"] == (COSMOS3_REASONER_NIM_IMAGE)

    @pytest.mark.asyncio
    async def test_nonpreferred_running_teacher_remains_explicit_choice(
        self, mock_env, monkeypatch
    ):
        """A resident does not override a better current recommendation.

        Super remains visible as the GPU conflict, while Omni is recommended
        and setup asks whether to keep or replace the resident.
        """
        import vlm_feedback_loop.services.environment as env_mod
        import vlm_feedback_loop.services.local_nim_service as local_nim_module

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(
                        name="RTX PRO 6000 Blackwell",
                        memory_total_mb=98304,
                        compute_capability=12.0,
                    )
                ]
            ),
        )
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Trash",
            deployment_id="deployment-1",
            model_config_id="owner-super-config",
            role="teacher",
            model_name=COSMOS3_SUPER_REASONER,
            nim_container_image=COSMOS3_REASONER_NIM_IMAGE,
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8001/v1",
            host_port=8001,
            status="running",
            nim_model_size="super",
            nim_model_profile=None,
            extra_container_env=(),
        )
        monkeypatch.setattr(
            local_nim_module,
            "list_active_nim_residents",
            lambda workspace_root: [resident],
        )
        monkeypatch.setattr(
            local_nim_module,
            "scan_active_resident_roles_by_device",
            lambda workspace_root: {"0": {"teacher"}},
        )

        result = await assess_environment(
            mock_env(nvidia_api_key="nvapi-test", ngc_api_key="ngc-test")
        )

        assert result["recommended_teacher_mode"] == "hosted"
        assert (
            result["recommended_local_teacher_model_name"]
            == NEMOTRON_3_NANO_OMNI_REASONING
        )
        assert (
            result["recommended_local_teacher_image"] == NEMOTRON_3_NANO_OMNI_NIM_IMAGE
        )
        assert result["active_local_nim_residents"][0]["project_name"] == "Trash"

    @pytest.mark.asyncio
    async def test_preferred_running_omni_outranks_hosted_key(
        self, mock_env, monkeypatch
    ):
        """The exact quality-default resident is reused even with a hosted key."""
        import vlm_feedback_loop.services.environment as env_mod
        import vlm_feedback_loop.services.local_nim_service as local_nim_module

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(
                        name="RTX PRO 6000 Blackwell",
                        memory_total_mb=98304,
                        compute_capability=12.0,
                    )
                ]
            ),
        )
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Trash",
            deployment_id="deployment-1",
            model_config_id="owner-omni-config",
            role="teacher",
            model_name=NEMOTRON_3_NANO_OMNI_REASONING,
            nim_container_image=NEMOTRON_3_NANO_OMNI_NIM_IMAGE,
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8001/v1",
            host_port=8001,
            status="running",
            nim_model_size=None,
            nim_model_profile=None,
            extra_container_env=(),
        )
        monkeypatch.setattr(
            local_nim_module,
            "list_active_nim_residents",
            lambda workspace_root: [resident],
        )
        monkeypatch.setattr(
            local_nim_module,
            "scan_active_resident_roles_by_device",
            lambda workspace_root: {"0": {"teacher"}},
        )

        result = await assess_environment(
            mock_env(nvidia_api_key="nvapi-test", ngc_api_key="ngc-test")
        )

        assert result["recommended_teacher_mode"] == "local"
        assert (
            result["recommended_local_teacher_model_name"]
            == NEMOTRON_3_NANO_OMNI_REASONING
        )

    def test_running_teacher_match_ignores_env_docker_would_reject(self):
        """The FTUE reuses a resident built from the catalog's effective env."""
        import vlm_feedback_loop.services.environment as env_mod

        image = "nvcr.io/nim/nvidia/example:1.0"
        entry = {
            "model_name": "nvidia/example",
            "eligible_roles": ["teacher"],
            "local_deploy_metadata": {
                "nim_container_image": image,
                "extra_container_env": {
                    "NIM_DISABLE_CUDA_GRAPH": "1",
                    "NGC_API_KEY": "reserved",
                    "lower_case": "invalid",
                    "BOOL_VALUE": True,
                },
            },
        }
        resident = ActiveNimResident(
            project_id="owner-project",
            project_name="Example",
            deployment_id="deployment-1",
            model_config_id="example-config",
            role="teacher",
            model_name="nvidia/example",
            nim_container_image=image,
            gpu_assignment="device=0",
            endpoint_url="http://localhost:8001/v1",
            host_port=8001,
            status="running",
            nim_model_size=None,
            nim_model_profile=None,
            extra_container_env=(("NIM_DISABLE_CUDA_GRAPH", "1"),),
        )

        assert (
            env_mod._pick_running_teacher_resident_entry([resident], [entry]) is entry
        )

    # ── Placement-aware embedding recommendation ─────────────────────────

    def _patch_gpu_host(
        self,
        monkeypatch,
        gpus: list[GpuInfo],
        residents: dict[str, set[str]] | None = None,
    ) -> None:
        """Docker + toolkit available, the given GPU inventory, and
        (optionally) active NIM roles by device index."""
        import vlm_feedback_loop.services.environment as env_mod
        import vlm_feedback_loop.services.local_nim_service as local_nim_module

        monkeypatch.setattr(
            env_mod, "check_docker_available", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "check_nvidia_toolkit", AsyncMock(return_value=(True, None))
        )
        monkeypatch.setattr(
            env_mod, "probe_gpu_inventory", AsyncMock(return_value=gpus)
        )
        monkeypatch.setattr(
            local_nim_module,
            "scan_active_resident_roles_by_device",
            lambda workspace_root: residents or {},
        )

    @pytest.mark.asyncio
    async def test_teacher_resident_on_one_of_two_gpus_keeps_local_embedding(
        self, mock_env, monkeypatch
    ):
        """A resident Teacher excludes only its own device: the second,
        free 80 GB GPU can take the embedding NIM, so the recommendation
        stays local."""
        self._patch_gpu_host(
            monkeypatch,
            [
                GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920),
                GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920),
            ],
            residents={"0": {"teacher"}},
        )

        result = await assess_environment(mock_env(nvidia_api_key="nvapi-test"))

        assert result["embedding_deployment"]["fits"] is True
        assert result["recommended_embedding_mode"] == "local"

    @pytest.mark.asyncio
    async def test_single_gpu_teacher_resident_recommends_hosted_embedding(
        self, mock_env, monkeypatch
    ):
        """With a running Teacher on the host's only GPU, the embedding
        NIM would get no device at all — a "local" recommendation would
        promise a deploy the one-NIM-per-GPU gate refuses."""
        self._patch_gpu_host(
            monkeypatch,
            [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)],
            residents={"0": {"teacher"}},
        )

        result = await assess_environment(mock_env(nvidia_api_key="nvapi-test"))

        assert result["embedding_deployment"]["fits"] is False
        assert result["recommended_embedding_mode"] == "hosted"

    @pytest.mark.asyncio
    async def test_free_gpu_below_embedding_floor_recommends_hosted(
        self, mock_env, monkeypatch
    ):
        """Heterogeneous host: the Teacher holds the 80 GB device and
        the free device has 8 GB — below the 24 GB embedding floor.
        ``fits`` must reflect the GPU the embedding NIM would actually
        get, not host-wide max memory."""
        self._patch_gpu_host(
            monkeypatch,
            [
                GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920),
                GpuInfo(name="RTX 3070", memory_total_mb=8192),
            ],
            residents={"0": {"teacher"}},
        )

        result = await assess_environment(mock_env(nvidia_api_key="nvapi-test"))

        assert result["embedding_deployment"]["fits"] is False
        assert result["recommended_embedding_mode"] == "hosted"

    @pytest.mark.asyncio
    async def test_embedding_resident_device_stays_local(self, mock_env, monkeypatch):
        """A device already running the embedding NIM stays a candidate —
        the local provider is live there, so the recommendation must not
        flip to hosted just because the device is occupied."""
        self._patch_gpu_host(
            monkeypatch,
            [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)],
            residents={"0": {"embedding"}},
        )

        result = await assess_environment(mock_env(nvidia_api_key="nvapi-test"))

        assert result["embedding_deployment"]["fits"] is True
        assert result["recommended_embedding_mode"] == "local"

    @pytest.mark.asyncio
    async def test_planned_local_teacher_reserves_one_gpu_of_two(
        self, mock_env, monkeypatch
    ):
        """Keyless 2-GPU box: the recommended (not yet deployed) local
        Teacher reserves the auto-placer's pick; the embedding NIM still
        fits on the remaining GPU, so both recommendations are local."""
        self._patch_gpu_host(
            monkeypatch,
            [
                GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920),
                GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920),
            ],
        )

        result = await assess_environment(mock_env())

        assert result["recommended_teacher_mode"] == "local"
        assert result["embedding_deployment"]["fits"] is True
        assert result["recommended_embedding_mode"] == "local"

    def test_pick_local_teacher_sort_is_stable_regardless_of_catalog_order(self):
        """Regression guard for ``_pick_local_teacher_recommendation``:
        even when the catalog declares 2B before 8B (or vice-versa),
        the 8B variant wins on an 80 GB GPU because the sort is by
        ``nim_gpu_memory_minimum_gb`` descending, not by declaration
        order."""
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        gpus = [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]

        # Synthetic catalog with 2B declared first, 8B second.
        flipped_catalog = [
            {
                "model_name": COSMOS_REASON2_2B,
                "eligible_roles": ["teacher", "student_base"],
                "local_deploy_metadata": {
                    "nim_container_image": COSMOS_REASON2_2B_NIM_IMAGE,
                    "nim_gpu_memory_minimum_gb": 36,
                },
            },
            {
                "model_name": COSMOS_REASON2_8B,
                "eligible_roles": ["teacher", "student_base"],
                "local_deploy_metadata": {
                    "nim_container_image": COSMOS_REASON2_8B_NIM_IMAGE,
                    "nim_gpu_memory_minimum_gb": 56,
                },
            },
        ]

        pick = _pick_local_teacher_recommendation(gpus, flipped_catalog)
        assert pick is not None
        assert pick["model_name"] == COSMOS_REASON2_8B

    def test_pick_local_teacher_prefers_cr3_nano_at_56gb_tier(self):
        """CR3-Nano default amendment: CR3-Nano and CR2-8B share
        the 56 GB minimum, but CR3-Nano is the recommended big-GPU default via
        ``_LOCAL_TEACHER_PREFERENCE_RANK`` — regardless of catalog order."""
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        gpus = [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]

        # CR2-8B declared FIRST so a naive stable sort would pick it; the
        # preference rank must override declaration order.
        tie_catalog = [
            {
                "model_name": COSMOS_REASON2_8B,
                "eligible_roles": ["teacher", "student_base"],
                "local_deploy_metadata": {
                    "nim_container_image": COSMOS_REASON2_8B_NIM_IMAGE,
                    "nim_gpu_memory_minimum_gb": 56,
                },
            },
            {
                "model_name": COSMOS3_NANO_REASONER,
                "eligible_roles": ["teacher", "student_base"],
                "local_deploy_metadata": {
                    "nim_container_image": COSMOS3_REASONER_NIM_IMAGE,
                    "nim_gpu_memory_minimum_gb": 56,
                },
            },
        ]

        pick = _pick_local_teacher_recommendation(gpus, tie_catalog)
        assert pick is not None
        assert pick["model_name"] == COSMOS3_NANO_REASONER

    def test_pick_local_teacher_uses_quality_first_on_supported_96gb_gpu(self):
        """Omni is the quality default when both of its hardware floors pass."""
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        pick = _pick_local_teacher_recommendation(
            [
                GpuInfo(
                    name="RTX PRO 6000 Blackwell",
                    memory_total_mb=98304,
                    compute_capability=12.0,
                )
            ]
        )

        assert pick is not None
        assert pick["model_name"] == NEMOTRON_3_NANO_OMNI_REASONING
        metadata = pick["local_deploy_metadata"]
        assert metadata["nim_gpu_memory_minimum_gb"] == (
            NEMOTRON_3_NANO_OMNI_GPU_MIN_GB
        )
        assert metadata["nim_compute_capability_minimum"] == (
            NEMOTRON_3_NANO_OMNI_COMPUTE_CAPABILITY_MIN
        )

    def test_pick_local_teacher_falls_back_when_omni_architecture_is_unsupported(
        self,
    ):
        """Memory alone cannot make the cc>=9.0 Omni NIM eligible."""
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        pick = _pick_local_teacher_recommendation(
            [
                GpuInfo(
                    name="A100",
                    memory_total_mb=98304,
                    compute_capability=8.0,
                )
            ]
        )

        assert pick is not None
        assert pick["model_name"] == COSMOS3_NANO_REASONER

    def test_pick_local_teacher_skips_non_teacher_entries(self):
        """Regression guard: the embedding NIM has local_deploy_metadata
        but no ``"teacher"`` role; the picker MUST skip it so the
        recommendation doesn't accidentally name the embedding model as
        the Teacher."""
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        gpus = [GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)]
        embedding_only_catalog = [
            {
                "model_name": EMBEDDING_MODEL_ID,
                "eligible_roles": [],  # not a teacher entry in the catalog
                "local_deploy_metadata": {
                    "nim_container_image": (EMBEDDING_NIM_IMAGE),
                    "nim_gpu_memory_minimum_gb": EMBEDDING_NIM_GPU_MIN_GB,
                },
            }
        ]

        assert _pick_local_teacher_recommendation(gpus, embedding_only_catalog) is None

    def test_pick_local_teacher_no_gpu_returns_none(self):
        from vlm_feedback_loop.services.environment import (
            _pick_local_teacher_recommendation,
        )

        assert _pick_local_teacher_recommendation([]) is None

    @pytest.mark.asyncio
    async def test_no_secrets_in_response(self, mock_env):
        settings = mock_env(nvidia_api_key="nvapi-fake-key-123")
        result = await assess_environment(settings)

        # Flatten all values and check no secrets leak
        flat = str(result)
        assert "nvapi-fake-key-123" not in flat
        # Boolean flags only, not values
        assert result["nvidia_api_key_configured"] is True

    @pytest.mark.asyncio
    async def test_embedding_deployment_from_config(self, mock_env):
        settings = mock_env()
        result = await assess_environment(settings)

        emb = result["embedding_deployment"]
        assert emb["model_name"] == EMBEDDING_MODEL_ID
        assert emb["nim_container_image"] == EMBEDDING_NIM_IMAGE
        assert emb["gpu_memory_minimum_gb"] == EMBEDDING_NIM_GPU_MIN_GB
        assert emb["provider"] == "none"  # default seeded value

    @pytest.mark.asyncio
    async def test_local_deployable_models_only_with_metadata(
        self, mock_env, monkeypatch
    ):
        import vlm_feedback_loop.services.environment as env_mod

        monkeypatch.setattr(
            env_mod,
            "probe_gpu_inventory",
            AsyncMock(
                return_value=[
                    GpuInfo(name="NVIDIA A100-SXM4-80GB", memory_total_mb=81920)
                ]
            ),
        )

        settings = mock_env()
        result = await assess_environment(settings)

        names = {m["model_name"] for m in result["local_deployable_models"]}
        assert COSMOS_REASON2_8B in names
        assert COSMOS_REASON2_2B in names
        assert _RETIRED_MISTRAL_LARGE_3 not in names

    @pytest.mark.asyncio
    async def test_missing_prerequisites_populated(self, mock_env):
        settings = mock_env()  # no Docker, no GPU, no keys
        result = await assess_environment(settings)

        checks = [p["check"] for p in result["missing_prerequisites"]]
        assert "Docker" in checks
        assert "API credentials" in checks
