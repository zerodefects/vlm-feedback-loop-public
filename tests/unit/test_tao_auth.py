# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO bearer-token resolver with JWT auto-exchange.

Covered scenarios:
- NGC-form TAO_API_KEY triggers /login and caches the returned JWT.
- Second call for the same settings returns the cached JWT (no /login).
- Non-NGC TAO_API_KEY (pre-exchanged JWT) passes through unchanged.
- ``invalidate_tao_bearer`` drops the cache so the next call re-exchanges.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from conftest import make_settings
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.services import tao_auth


def _settings_with(tmp_path: Path, tao_api_key: str, **overrides: Any) -> Settings:
    return make_settings(
        tmp_path / "workspace",
        TAO_API_BASE_URL="https://tao.example/api/v2",
        TAO_ORG_NAME="my-org",
        TAO_API_KEY=tao_api_key,
        **overrides,
    )


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    """Clear the module-level JWT cache before every test."""
    tao_auth.reset_tao_bearer_cache()
    yield
    tao_auth.reset_tao_bearer_cache()


class _LoginRecorder:
    """Records every call to the mocked login_tao so tests assert
    invocation counts without relying on unittest.mock internals."""

    def __init__(self, token: str = "header.payload.signature") -> None:
        self.calls: list[dict[str, Any]] = []
        self.token = token
        self.next_error: str | None = None

    async def login(
        self,
        tao_api_base_url: str,
        ngc_api_key: str,
        org_name: str,
        deadline_s: float = 30.0,  # noqa: ARG002 — matches real signature
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "base_url": tao_api_base_url,
                "ngc_api_key": ngc_api_key,
                "org": org_name,
            }
        )
        if self.next_error:
            err = self.next_error
            self.next_error = None
            return {"success": False, "token": None, "error": err}
        return {"success": True, "token": self.token, "error": None}


class TestNgcKeyExchange:
    @pytest.mark.asyncio
    async def test_ngc_key_triggers_login_and_returns_jwt(self, tmp_path, monkeypatch):
        settings = _settings_with(tmp_path, "nvapi-abc123")
        recorder = _LoginRecorder(token="jwt-xyz")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", recorder.login
        )

        token = await tao_auth.get_tao_bearer(settings)

        assert token == "jwt-xyz"
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["ngc_api_key"] == "nvapi-abc123"
        assert recorder.calls[0]["base_url"] == "https://tao.example/api/v2"
        assert recorder.calls[0]["org"] == "my-org"

    @pytest.mark.asyncio
    async def test_second_call_uses_cache_without_re_login(self, tmp_path, monkeypatch):
        settings = _settings_with(tmp_path, "nvapi-cached")
        recorder = _LoginRecorder(token="jwt-first")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", recorder.login
        )

        t1 = await tao_auth.get_tao_bearer(settings)
        t2 = await tao_auth.get_tao_bearer(settings)

        assert t1 == t2 == "jwt-first"
        # Exactly one /login call — the second get_tao_bearer hit cache.
        assert len(recorder.calls) == 1

    @pytest.mark.asyncio
    async def test_login_failure_raises(self, tmp_path, monkeypatch):
        settings = _settings_with(tmp_path, "nvapi-bad")
        recorder = _LoginRecorder()
        recorder.next_error = "NGC key rejected"
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", recorder.login
        )

        with pytest.raises(RuntimeError, match="NGC key rejected"):
            await tao_auth.get_tao_bearer(settings)

        # Failed exchanges are NOT cached — the next call should retry.
        assert len(recorder.calls) == 1
        recorder.next_error = None
        recorder.token = "jwt-after-retry"
        token = await tao_auth.get_tao_bearer(settings)
        assert token == "jwt-after-retry"
        assert len(recorder.calls) == 2


class TestPreExchangedJwt:
    @pytest.mark.asyncio
    async def test_non_nvapi_value_passes_through_without_login(
        self, tmp_path, monkeypatch
    ):
        # Realistic JWT shape: three dot-separated segments.
        jwt_value = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJibHVlcHJpbnQifQ.sig"
        settings = _settings_with(tmp_path, jwt_value)
        recorder = _LoginRecorder()
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", recorder.login
        )

        token = await tao_auth.get_tao_bearer(settings)

        assert token == jwt_value
        # /login was never called — backward-compat path for pre-exchanged JWTs.
        assert recorder.calls == []


class TestMissingKey:
    @pytest.mark.asyncio
    async def test_missing_key_raises(self, tmp_path, monkeypatch):
        # Build settings WITHOUT TAO_API_KEY override, forcing the default
        # (None). The _settings_with helper always sets it, so construct
        # directly here.
        settings = make_settings(
            tmp_path / "workspace",
            TAO_API_BASE_URL="https://tao.example/api/v2",
            TAO_ORG_NAME="my-org",
            # TAO_API_KEY left as default (None)
        )

        with pytest.raises(RuntimeError, match="TAO_API_KEY is not configured"):
            await tao_auth.get_tao_bearer(settings)


class TestInvalidateBearer:
    @pytest.mark.asyncio
    async def test_invalidate_forces_re_exchange(self, tmp_path, monkeypatch):
        settings = _settings_with(tmp_path, "nvapi-refreshable")
        recorder = _LoginRecorder(token="jwt-v1")
        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", recorder.login
        )

        # First exchange caches jwt-v1.
        t1 = await tao_auth.get_tao_bearer(settings)
        assert t1 == "jwt-v1"
        assert len(recorder.calls) == 1

        # Simulate a TAO service observing a 401 and invalidating.
        tao_auth.invalidate_tao_bearer(settings)
        recorder.token = "jwt-v2"

        t2 = await tao_auth.get_tao_bearer(settings)
        assert t2 == "jwt-v2"
        assert len(recorder.calls) == 2


class TestCacheScoping:
    @pytest.mark.asyncio
    async def test_different_keys_do_not_share_cache(self, tmp_path, monkeypatch):
        # Two Settings instances with different NGC keys — each should
        # trigger its own /login exchange.
        s1 = _settings_with(tmp_path, "nvapi-alpha")
        s2 = _settings_with(tmp_path, "nvapi-beta")

        tokens = iter(["jwt-alpha", "jwt-beta"])

        async def fake_login(*args, **kwargs):  # noqa: ARG001
            return {"success": True, "token": next(tokens), "error": None}

        monkeypatch.setattr(
            "vlm_feedback_loop.services.tao_client.login_tao", fake_login
        )

        t1 = await tao_auth.get_tao_bearer(s1)
        t2 = await tao_auth.get_tao_bearer(s2)

        assert t1 == "jwt-alpha"
        assert t2 == "jwt-beta"
        assert t1 != t2

        # Re-querying each returns the cached value, so only 2 exchanges total.
        assert await tao_auth.get_tao_bearer(s1) == "jwt-alpha"
        assert await tao_auth.get_tao_bearer(s2) == "jwt-beta"
