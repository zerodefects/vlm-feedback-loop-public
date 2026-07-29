# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-model availability computation and env-probe caching.

Exercises ``compute_availability`` (pure logic, no DB) and the cache
behavior of ``get_cached_environment``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
) -> SimpleNamespace:
    """Lightweight stand-in for a NimEndpoint ORM row."""
    return SimpleNamespace(
        endpoint_mode=endpoint_mode,
        last_probe_status=last_probe_status,
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
        env_mod.invalidate_env_cache()
        yield
        env_mod.invalidate_env_cache()

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_reuses_cached_result(self, monkeypatch):
        probe = AsyncMock(return_value={"nvidia_api_key_configured": True})
        monkeypatch.setattr(env_mod, "assess_environment", probe)

        result1 = await env_mod.get_cached_environment(SimpleNamespace())
        result2 = await env_mod.get_cached_environment(SimpleNamespace())

        # Same dict identity → cache hit, single probe.
        assert result1 is result2
        assert probe.call_count == 1

    @pytest.mark.asyncio
    async def test_call_after_ttl_reprobes(self, monkeypatch):
        probe = AsyncMock(
            side_effect=[
                {"nvidia_api_key_configured": False},
                {"nvidia_api_key_configured": True},
            ]
        )
        monkeypatch.setattr(env_mod, "assess_environment", probe)
        monkeypatch.setattr(env_mod, "_ENV_CACHE_TTL_S", 0.01)

        result1 = await env_mod.get_cached_environment(SimpleNamespace())
        time.sleep(0.02)
        result2 = await env_mod.get_cached_environment(SimpleNamespace())

        assert probe.call_count == 2
        assert result1["nvidia_api_key_configured"] is False
        assert result2["nvidia_api_key_configured"] is True

    @pytest.mark.asyncio
    async def test_invalidate_forces_reprobe(self, monkeypatch):
        probe = AsyncMock(return_value={"nvidia_api_key_configured": True})
        monkeypatch.setattr(env_mod, "assess_environment", probe)

        await env_mod.get_cached_environment(SimpleNamespace())
        env_mod.invalidate_env_cache()
        await env_mod.get_cached_environment(SimpleNamespace())

        assert probe.call_count == 2
