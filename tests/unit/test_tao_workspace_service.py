# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TAO workspace + base-experiment service.

Covers:
- create happy path
- idempotent re-call (GET only, no POST)
- adoption paths (persisted id, by name, cloud_type mismatch)
- workspace unreachable surfaces a structured error
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session

from conftest import make_settings
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db.deployment_models import TAODeploymentConfig
from vlm_feedback_loop.db.engine import init_deployment_db
from vlm_feedback_loop.services.tao_workspace_service import (
    create_or_get_workspace,
    get_workspace,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _settings_for(tmp_workspace: Path, **overrides: Any) -> Settings:
    return make_settings(
        tmp_workspace,
        TAO_API_BASE_URL="https://tao.example/api/v2",
        TAO_API_KEY="jwt-token",
        TAO_ORG_NAME="my-org",
        **overrides,
    )


_WORKSPACE_ENDPOINTS = {
    "endpoint_url_external": "http://127.0.0.1:8333",
    "endpoint_url_internal": "http://seaweedfs-s3:8333",
}


class _Recorder:
    """Collects every request dispatched through MockTransport."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def record(self, request: httpx.Request) -> None:
        self.requests.append(request)

    @property
    def methods_and_paths(self) -> list[tuple[str, str]]:
        return [(r.method, r.url.path) for r in self.requests]


def _mock_transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
def _noop_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speed up tests by dropping HTTP retry backoff."""

    async def _noop(_attempt_index: int) -> None:
        pass

    monkeypatch.setattr("vlm_feedback_loop.services.http_client._backoff", _noop)


# ── W1: create happy path ────────────────────────────────────────────────────


class TestCreateWorkspaceHappyPath:
    @pytest.mark.asyncio
    async def test_create_sends_ftms_6263_workspace_shape(self, tmp_workspace):
        """Workspace creation uses TAO's nested discriminator and TAO-visible
        S3 endpoint; the Blueprint-visible endpoint is local metadata only."""
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)
        posted_body: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"workspaces": []})
            posted_body.update(json.loads(request.content))
            return httpx.Response(201, json={"id": "ws-ftms-6263"})

        result = await create_or_get_workspace(
            settings,
            workspace_name="my-workspace",
            cloud_type="seaweedfs",
            bucket="vlm-bucket",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert posted_body == {
            "name": "my-workspace",
            "cloud_type": "seaweedfs",
            "cloud_specific_details": {
                "cloud_type": "seaweedfs",
                "cloud_bucket_name": "vlm-bucket",
                "cloud_region": "us-east-1",
                "endpoint_url": "http://seaweedfs-s3:8333",
                "access_key": "ak",
                "secret_key": "sk",
            },
        }
        assert "http://127.0.0.1:8333" not in str(posted_body)

    @pytest.mark.asyncio
    async def test_create_persists_identity_on_singleton(self, tmp_workspace):
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            if request.method == "GET":
                # Name-adoption probe finds nothing on a fresh FTMS.
                return httpx.Response(200, json={"workspaces": []})
            return httpx.Response(
                201,
                json={"id": "ws-abc-123", "name": "my-workspace"},
            )

        result = await create_or_get_workspace(
            settings,
            workspace_name="my-workspace",
            cloud_type="seaweedfs",
            bucket="vlm-bucket",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.workspace_id == "ws-abc-123"
        assert result.already_provisioned is False

        # Name-adoption listing first, then exactly one POST.
        assert recorder.methods_and_paths == [
            ("GET", "/api/v2/orgs/my-org/workspaces"),
            ("POST", "/api/v2/orgs/my-org/workspaces"),
        ]

        # Singleton updated
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            assert cfg.tao_workspace_id == "ws-abc-123"
            assert cfg.tao_workspace_name == "my-workspace"
            assert cfg.tao_workspace_cloud_type == "seaweedfs"
            assert cfg.tao_workspace_bucket == "vlm-bucket"
            assert cfg.tao_workspace_s3_access_key_ref == "TAO_WORKSPACE_S3_ACCESS_KEY"
            assert cfg.tao_workspace_s3_secret_key_ref == "TAO_WORKSPACE_S3_SECRET_KEY"
            # bootstrap_status remains "not_bootstrapped" until the CLI sets it.


# ── W2: idempotent re-call (GET only, no POST) ───────────────────────────────


class TestIdempotentReCall:
    @pytest.mark.asyncio
    async def test_bootstrapped_returns_get_only(self, tmp_workspace):
        # Pre-populate the singleton as if bootstrap already ran.
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            cfg.tao_workspace_id = "ws-existing"
            cfg.bootstrap_status = "bootstrapped"
            session.commit()

        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            assert request.method == "GET"
            return httpx.Response(200, json={"id": "ws-existing", "name": "existing"})

        result = await create_or_get_workspace(
            settings,
            workspace_name="unused-on-idempotent-path",
            cloud_type="seaweedfs",
            bucket="unused",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.already_provisioned is True
        assert result.workspace_id == "ws-existing"
        # Only a GET was issued — no POST.
        assert recorder.methods_and_paths == [
            ("GET", "/api/v2/orgs/my-org/workspaces/ws-existing")
        ]


# ── W2b: adoption paths (id regardless of status; by name; guards) ──────────


def _seed_workspace_id(tmp_workspace: Path, *, status: str) -> None:
    engine = init_deployment_db(tmp_workspace)
    with Session(engine) as session:
        cfg = session.query(TAODeploymentConfig).first()
        assert cfg is not None
        cfg.tao_workspace_id = "ws-existing"
        cfg.bootstrap_status = status
        session.commit()


class TestAdoptionPaths:
    @pytest.mark.asyncio
    async def test_persisted_id_adopts_regardless_of_bootstrap_status(
        self, tmp_workspace
    ):
        """A persisted workspace id must adopt even when the CLI has already
        stamped bootstrap_status='in_progress' — the status gate made every
        CLI re-run POST a duplicate (FTMS 400 name-conflict, found live)."""
        _seed_workspace_id(tmp_workspace, status="in_progress")
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            return httpx.Response(200, json={"id": "ws-existing", "name": "existing"})

        result = await create_or_get_workspace(
            settings,
            workspace_name="existing",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.already_provisioned is True
        assert recorder.methods_and_paths == [
            ("GET", "/api/v2/orgs/my-org/workspaces/ws-existing")
        ]

    @pytest.mark.asyncio
    async def test_persisted_id_transient_error_never_creates(self, tmp_workspace):
        """When the id-confirm GET fails for any reason other than 404, the
        call must surface the failure — creating on an unconfirmed error
        risks a duplicate workspace once TAO recovers."""
        _seed_workspace_id(tmp_workspace, status="failed")
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            return httpx.Response(500, json={"error": "boom"})

        result = await create_or_get_workspace(
            settings,
            workspace_name="existing",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is False
        assert all(m == "GET" for m, _ in recorder.methods_and_paths)

    @pytest.mark.asyncio
    async def test_persisted_id_404_falls_through_to_name_adoption(self, tmp_workspace):
        """A stale persisted id (workspace deleted FTMS-side) falls through
        to name adoption; the singleton is re-pointed at the matched id."""
        _seed_workspace_id(tmp_workspace, status="bootstrapped")
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            if request.url.path.endswith("/workspaces/ws-existing"):
                return httpx.Response(404, json={"error": "gone"})
            return httpx.Response(
                200,
                json={
                    "workspaces": [
                        {
                            "id": "ws-renewed",
                            "name": "existing",
                            "cloud_type": "seaweedfs",
                        }
                    ]
                },
            )

        result = await create_or_get_workspace(
            settings,
            workspace_name="existing",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.already_provisioned is True
        assert result.workspace_id == "ws-renewed"
        assert ("POST", "/api/v2/orgs/my-org/workspaces") not in (
            recorder.methods_and_paths
        )
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            assert cfg.tao_workspace_id == "ws-renewed"

    @pytest.mark.asyncio
    async def test_fresh_db_adopts_existing_workspace_by_name(self, tmp_workspace):
        """Fresh deployment.db + admin-provisioned FTMS workspace of the same
        name → adopt (no POST) and persist identity, exactly as a create
        would have."""
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            assert request.method == "GET"
            return httpx.Response(
                200,
                json={
                    "workspaces": [
                        {
                            "id": "ws-admin",
                            "name": "blueprint-ws",
                            "cloud_type": "seaweedfs",
                        }
                    ]
                },
            )

        result = await create_or_get_workspace(
            settings,
            workspace_name="blueprint-ws",
            cloud_type="seaweedfs",
            bucket="blueprint",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.already_provisioned is True
        assert result.workspace_id == "ws-admin"
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            assert cfg.tao_workspace_id == "ws-admin"
            assert cfg.tao_workspace_name == "blueprint-ws"
            assert cfg.tao_workspace_bucket == "blueprint"

    @pytest.mark.asyncio
    async def test_name_match_with_cloud_type_mismatch_is_an_error(self, tmp_workspace):
        """Adopting a same-name workspace whose cloud_type differs would
        silently misroute uploads — surface an actionable error instead."""
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "workspaces": [
                        {"id": "ws-aws", "name": "blueprint-ws", "cloud_type": "aws"}
                    ]
                },
            )

        result = await create_or_get_workspace(
            settings,
            workspace_name="blueprint-ws",
            cloud_type="seaweedfs",
            bucket="blueprint",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is False
        assert result.error is not None and "cloud_type" in result.error

    @pytest.mark.asyncio
    async def test_listing_failure_falls_through_to_create(self, tmp_workspace):
        """If the name-adoption listing is unavailable, the create path must
        proceed (pre-listing behavior) rather than block bootstrap."""
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)
        recorder = _Recorder()

        def handler(request: httpx.Request) -> httpx.Response:
            recorder.record(request)
            if request.method == "GET":
                return httpx.Response(500, json={"error": "listing down"})
            return httpx.Response(201, json={"id": "ws-new", "name": "fresh"})

        result = await create_or_get_workspace(
            settings,
            workspace_name="fresh",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is True
        assert result.workspace_id == "ws-new"
        assert result.already_provisioned is False
        assert recorder.methods_and_paths[-1] == (
            "POST",
            "/api/v2/orgs/my-org/workspaces",
        )


# ── W5: workspace unreachable surfaces a structured error ────────────────────


class TestWorkspaceUnreachable:
    @pytest.mark.asyncio
    async def test_create_error_redacts_before_shared_detail_truncation(
        self, tmp_workspace
    ):
        """A provider-echoed S3 secret cannot survive as a truncated prefix."""
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)
        sentinel = "SENTINEL_WORKSPACE_BOUNDARY_" + ("X" * 40)
        padding = "p" * (400 - (len(sentinel) - 1))

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(200, json={"workspaces": []})
            return httpx.Response(
                400,
                json={"error": {"message": padding + sentinel}},
            )

        result = await create_or_get_workspace(
            settings,
            workspace_name="ws",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="access-key",
            secret_key=sentinel,
            _transport=_mock_transport(handler),
        )

        assert result.success is False
        assert result.error is not None
        assert sentinel[:-1] not in result.error

    @pytest.mark.asyncio
    async def test_503_after_retry_returns_structured_error(self, tmp_workspace):
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        result = await create_or_get_workspace(
            settings,
            workspace_name="ws",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
            _transport=_mock_transport(handler),
        )

        assert result.success is False
        assert "Workspace creation failed" in (result.error or "")
        # resilient_request returns status_code=None on exhausted retries,
        # with error_detail "Exhausted N retries. Last: HTTP 503".
        assert result.status_code is None
        assert "503" in (result.error or "")

        # Singleton remains unmodified
        engine = init_deployment_db(tmp_workspace)
        with Session(engine) as session:
            cfg = session.query(TAODeploymentConfig).first()
            assert cfg is not None
            assert cfg.tao_workspace_id is None
            assert cfg.bootstrap_status == "not_bootstrapped"

    @pytest.mark.asyncio
    async def test_get_workspace_404_surfaces_not_found(self, tmp_workspace):
        init_deployment_db(tmp_workspace)
        settings = _settings_for(tmp_workspace)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="Not Found")

        result = await get_workspace(
            settings,
            workspace_id="ws-missing",
            _transport=_mock_transport(handler),
        )

        assert result.success is False
        assert result.status_code == 404
        assert "not found" in (result.error or "").lower()


# ── Guardrails: settings missing ─────────────────────────────────────────────


class TestSettingsMissing:
    @pytest.mark.asyncio
    async def test_missing_tao_settings_is_friendly(self, tmp_workspace):
        settings = make_settings(tmp_workspace)  # no TAO_* configured

        result = await create_or_get_workspace(
            settings,
            workspace_name="ws",
            cloud_type="seaweedfs",
            bucket="b",
            **_WORKSPACE_ENDPOINTS,
            access_key="ak",
            secret_key="sk",
        )

        assert result.success is False
        assert "TAO_API_BASE_URL" in (result.error or "")
