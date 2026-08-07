# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-model availability computation and env-probe caching.

Exercises ``compute_availability`` (pure logic, no DB) and the cache
behavior of ``get_cached_environment``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vlm_feedback_loop.services import environment as env_mod
from vlm_feedback_loop.services.model_config_service import compute_availability

# ── compute_availability ────────────────────────────────────────────────────


def _mc(*, hosted_compatible: bool = True) -> SimpleNamespace:
    """Lightweight stand-in for a ModelConfig ORM row."""
    return SimpleNamespace(hosted_compatible=hosted_compatible)


def _endpoint(
    *,
    endpoint_mode: str,
    last_probe_status: str = "unknown",
    is_enabled: bool = True,
) -> SimpleNamespace:
    """Lightweight stand-in for a NimEndpoint ORM row."""
    return SimpleNamespace(
        endpoint_mode=endpoint_mode,
        last_probe_status=last_probe_status,
        is_enabled=is_enabled,
    )


def _env(*, nvidia_api_key_configured: bool = True) -> dict:
    return {"nvidia_api_key_configured": nvidia_api_key_configured}


class TestComputeAvailability:
    def test_hosted_compatible_with_key_is_available(self):
        # Mistral / Qwen / Nemotron on hosted with a valid key — the
        # labeling dropdown's happy path.
        result = compute_availability(
            _mc(hosted_compatible=True),
            _endpoint(endpoint_mode="hosted"),
            _env(nvidia_api_key_configured=True),
        )
        assert result == {"available": True, "reason": None}

    def test_hosted_compatible_without_key_blocks(self):
        result = compute_availability(
            _mc(hosted_compatible=True),
            _endpoint(endpoint_mode="hosted"),
            _env(nvidia_api_key_configured=False),
        )
        assert result == {"available": False, "reason": "no_nvidia_api_key"}

    def test_hosted_incompatible_with_key_blocks(self):
        # Cosmos on hosted returns 404 even with a valid key (NVCF gate);
        # compute_availability hides it before invocation.
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(endpoint_mode="hosted"),
            _env(nvidia_api_key_configured=True),
        )
        assert result == {"available": False, "reason": "hosted_not_compatible"}

    def test_local_healthy_is_available(self):
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(
                endpoint_mode="local_system_managed", last_probe_status="healthy"
            ),
            _env(),
        )
        assert result == {"available": True, "reason": None}

    def test_local_unknown_is_available_pre_probe(self):
        # Newly deployed local NIMs report "unknown" until the first probe
        # completes. Treating that as available avoids a flicker right
        # after the deploy completes.
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(
                endpoint_mode="local_system_managed", last_probe_status="unknown"
            ),
            _env(),
        )
        assert result == {"available": True, "reason": None}

    def test_local_unhealthy_blocks(self):
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(
                endpoint_mode="local_system_managed", last_probe_status="unhealthy"
            ),
            _env(),
        )
        assert result == {"available": False, "reason": "endpoint_unhealthy"}

    def test_disabled_endpoint_blocks_even_when_probe_was_healthy(self):
        """Lifecycle-disabled shared endpoints cannot dispatch by stale URL."""
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(
                endpoint_mode="local_system_managed",
                last_probe_status="healthy",
                is_enabled=False,
            ),
            _env(),
        )
        assert result == {"available": False, "reason": "endpoint_unhealthy"}

    def test_self_hosted_trusted(self):
        # Operators bringing their own NIM are trusted — if they pointed
        # it at this endpoint, they know it serves the model.
        result = compute_availability(
            _mc(hosted_compatible=False),
            _endpoint(endpoint_mode="self_hosted", last_probe_status="unknown"),
            _env(),
        )
        assert result == {"available": True, "reason": None}

    def test_self_hosted_unhealthy_blocks(self):
        result = compute_availability(
            _mc(),
            _endpoint(endpoint_mode="self_hosted", last_probe_status="unreachable"),
            _env(),
        )
        assert result == {"available": False, "reason": "endpoint_unhealthy"}

    def test_endpoint_missing(self):
        # Defensive: orphaned reference (endpoint deleted under us).
        result = compute_availability(_mc(), None, _env())
        assert result == {"available": False, "reason": "endpoint_missing"}

    def test_unknown_endpoint_mode(self):
        result = compute_availability(
            _mc(), _endpoint(endpoint_mode="some_future_mode"), _env()
        )
        assert result == {"available": False, "reason": "unknown_endpoint_mode"}


# ── get_cached_environment TTL ──────────────────────────────────────────────


class TestEnvCache:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        env_mod.invalidate_machine_assessment_cache()
        yield
        env_mod.invalidate_machine_assessment_cache()

    @pytest.mark.asyncio
    async def test_second_call_reuses_machine_probe_but_recomposes_state(
        self, monkeypatch
    ):
        machine = env_mod.MachineAssessment(
            docker_available=True,
            nvidia_toolkit_available=True,
            gpu_inventory=(),
        )
        probe = AsyncMock(return_value=machine)
        compose = Mock(
            side_effect=[
                {"nvidia_api_key_configured": False},
                {"nvidia_api_key_configured": True},
            ]
        )
        monkeypatch.setattr(env_mod, "_probe_machine_assessment", probe)
        monkeypatch.setattr(env_mod, "_compose_environment", compose)

        result1 = await env_mod.get_cached_environment(SimpleNamespace())
        result2 = await env_mod.get_cached_environment(SimpleNamespace())

        assert result1["nvidia_api_key_configured"] is False
        assert result2["nvidia_api_key_configured"] is True
        assert probe.call_count == 1
        assert compose.call_count == 2

    @pytest.mark.asyncio
    async def test_explicit_invalidation_reprobes_machine(self, monkeypatch):
        probe = AsyncMock(
            side_effect=[
                env_mod.MachineAssessment(False, False, ()),
                env_mod.MachineAssessment(True, True, ()),
            ]
        )
        compose = Mock(
            side_effect=lambda _settings, machine: {
                "docker_available": machine.docker_available
            }
        )
        monkeypatch.setattr(env_mod, "_probe_machine_assessment", probe)
        monkeypatch.setattr(env_mod, "_compose_environment", compose)

        result1 = await env_mod.get_cached_environment(SimpleNamespace())
        env_mod.invalidate_machine_assessment_cache()
        result2 = await env_mod.get_cached_environment(SimpleNamespace())

        assert probe.call_count == 2
        assert result1["docker_available"] is False
        assert result2["docker_available"] is True
