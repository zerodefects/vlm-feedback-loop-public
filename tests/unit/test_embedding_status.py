# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authoritative embedding-status surface and worker recovery.

Covers:

  * ``clip_embedding_service.get_embedding_status`` — counts, worker activity,
    and the ``complete`` rule, including the embeddings-disabled short-circuit.
  * ``GET .../examples:embedding_status`` — 200 shape + 404.
  * the dead-drain self-heal — a status poll re-arms the embedding worker
    when examples are pending, no worker is running, and embeddings are
    enabled (rate-limited per project).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from conftest import create_project_via_api, make_settings


def _make_client_and_settings(tmp_path: Path, **overrides: Any):
    from vlm_feedback_loop.main import app
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services import project_service

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    settings = make_settings(workspace, **overrides)
    app.dependency_overrides[get_current_settings] = lambda: settings
    project_service.clear_engine_cache()
    client = TestClient(app, raise_server_exceptions=False)
    return client, settings, workspace


def _create_project(client: TestClient, name: str = "embed-status") -> str:
    return create_project_via_api(client, name)["project_id"]


def _add_examples(workspace: Path, pid: str, n_examples: int, n_embedded: int) -> None:
    """Insert ``n_examples`` Examples and embeddings for the first ``n_embedded``."""
    from vlm_feedback_loop.db.models.clip_embedding import ClipEmbedding
    from vlm_feedback_loop.db.models.example import Example
    from vlm_feedback_loop.services.clip_embedding_service import serialize_vector
    from vlm_feedback_loop.services.project_service import get_project_engine

    engine = get_project_engine(pid, str(workspace))
    assert engine is not None
    dim = 4
    blob = serialize_vector([1.0, 2.0, 3.0, 4.0], dim)
    with Session(engine) as session:
        for i in range(n_examples):
            session.add(
                Example(
                    example_key=f"k{i}",
                    project_id=pid,
                    storage_ref=f"/fake/{i}.jpg",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                )
            )
        # Ingestion commits parents before the background embedding pass.
        session.flush()
        for i in range(n_embedded):
            session.add(
                ClipEmbedding(
                    project_id=pid,
                    example_key=f"k{i}",
                    embedding_provider="hosted_nvclip",
                    clip_embedding_model_id="nvidia/llama-nemotron-embed-vl-1b-v2",
                    clip_embedding_dim=dim,
                    vector_blob_f32=blob,
                )
            )
        session.commit()


# ── get_embedding_status (service) ──────────────────────────────────────────


class TestGetEmbeddingStatus:
    def test_complete_when_all_embedded(self, tmp_path: Path) -> None:
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        client, settings, workspace = _make_client_and_settings(tmp_path)
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=3)

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["total_examples"] == 3
        assert status["embedded"] == 3
        assert status["pending"] == 0
        assert status["worker_active"] is False
        assert status["complete"] is True

    def test_incomplete_when_pending(self, tmp_path: Path) -> None:
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        client, settings, workspace = _make_client_and_settings(tmp_path)
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=1)

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["embedded"] == 1
        assert status["pending"] == 2
        # auto-compute on, provider unprobed, examples still pending → not done
        assert status["complete"] is False

    def test_empty_project_is_trivially_complete(self, tmp_path: Path) -> None:
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        client, settings, workspace = _make_client_and_settings(tmp_path)
        pid = _create_project(client)

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["total_examples"] == 0
        assert status["complete"] is True  # nothing to wait for

    def test_complete_when_auto_compute_off(self, tmp_path: Path) -> None:
        """Auto-compute off ⇒ complete even with pending examples (no hang)."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        client, settings, workspace = _make_client_and_settings(
            tmp_path, EMBEDDINGS_AUTO_COMPUTE=False
        )
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=0)

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["embedded"] == 0
        assert status["pending"] == 3
        assert status["auto_compute"] is False
        assert status["complete"] is True

    def test_complete_when_provider_setting_none(self, tmp_path: Path) -> None:
        """EMBEDDING_PROVIDER=none ⇒ disabled by operator intent (no hang)."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        client, settings, workspace = _make_client_and_settings(
            tmp_path, EMBEDDING_PROVIDER="none"
        )
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=0)

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["complete"] is True

    def test_pending_not_complete_when_provider_unprobed(self, tmp_path: Path) -> None:
        """A fresh project row has provider 'none' until the worker probes it;
        with EMBEDDING_PROVIDER='auto' (default) the barrier must NOT treat that
        transient state as disabled — pending examples ⇒ not complete."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )
        from vlm_feedback_loop.services.project_service import get_project_engine

        client, settings, workspace = _make_client_and_settings(tmp_path)
        assert settings.EMBEDDING_PROVIDER == "auto"
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=1)

        # confirm the project row really is the transient 'none'
        from vlm_feedback_loop.db.models.project import Project

        engine = get_project_engine(pid, str(workspace))
        assert engine is not None
        with Session(engine) as session:
            row = session.query(Project).filter_by(project_id=pid).first()
            assert row is not None and row.embedding_provider == "none"

        status = get_embedding_status(pid, str(workspace), settings)
        assert status is not None
        assert status["complete"] is False  # must wait, not short-circuit

    def test_missing_project_returns_none(self, tmp_path: Path) -> None:
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
        )

        _, settings, workspace = _make_client_and_settings(tmp_path)
        assert get_embedding_status("nope-123", str(workspace), settings) is None


# ── GET .../examples:embedding_status (router) ──────────────────────────────


class TestEmbeddingStatusEndpoint:
    def test_endpoint_returns_status_shape(self, tmp_path: Path) -> None:
        client, _settings, workspace = _make_client_and_settings(tmp_path)
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=2, n_embedded=2)

        r = client.get(f"/v1/projects/{pid}/examples:embedding_status")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total_examples"] == 2
        assert body["embedded"] == 2
        assert body["complete"] is True
        assert body["model_id"] is None or isinstance(body["model_id"], str)

    def test_endpoint_404_for_missing_project(self, tmp_path: Path) -> None:
        client, _settings, _workspace = _make_client_and_settings(tmp_path)
        r = client.get("/v1/projects/missing-xyz/examples:embedding_status")
        assert r.status_code == 404
        assert r.json()["detail"] == "Project not found"


# ── Dead-drain self-heal (status poll re-arms the worker) ───────────────────


_TRIGGER_PATH = (
    "vlm_feedback_loop.services.clip_embedding_service.trigger_embedding_computation"
)


class TestEmbeddingStatusSelfHeal:
    """A drain that died mid-flight (pending > 0, no worker) is re-armed by
    status polls.

    The embedding worker treats one hard provider failure as terminal for
    the whole drain, and every other recovery trigger is event-coupled
    (backend restart, new ingest, local-NIM lifecycle) — polling
    ``:embedding_status`` is the recovery path that needs no unrelated
    event.
    """

    def test_endpoint_rekicks_dead_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET :embedding_status with pending examples and no worker must
        re-trigger the embedding worker — ``autorun --wait-embeddings``
        polls this endpoint, so a dead drain becomes a retry loop instead
        of an infinite hang."""
        mock_trigger = MagicMock()
        monkeypatch.setattr(_TRIGGER_PATH, mock_trigger)

        client, settings, workspace = _make_client_and_settings(tmp_path)
        pid = _create_project(client)
        _add_examples(workspace, pid, n_examples=3, n_embedded=1)

        r = client.get(f"/v1/projects/{pid}/examples:embedding_status")
        assert r.status_code == 200, r.text
        assert r.json()["pending"] == 2
        mock_trigger.assert_called_once_with(pid, settings.WORKSPACE_ROOT, settings)

    @pytest.mark.asyncio
    async def test_rekick_registers_worker_on_running_loop(
        self, tmp_path: Path
    ) -> None:
        """The self-heal registers a real worker task on the event loop —
        the reason the status route is ``async def``: without a running
        loop, ``try_register`` silently no-ops."""
        from vlm_feedback_loop.db.engine import open_project_db
        from vlm_feedback_loop.db.models.example import Example
        from vlm_feedback_loop.db.models.project import Project
        from vlm_feedback_loop.services.background import background_manager
        from vlm_feedback_loop.services.clip_embedding_service import (
            get_embedding_status,
            maybe_restart_dead_embedding_worker,
        )
        from vlm_feedback_loop.services.project_service import set_project_engine

        workspace = tmp_path / "workspace"
        project_dir = workspace / "projects" / "heal-proj"
        project_dir.mkdir(parents=True)
        engine = open_project_db(project_dir)
        set_project_engine("heal-proj", engine)
        with Session(engine) as session:
            session.add(
                Project(
                    project_id="heal-proj",
                    name="heal",
                    project_dir=str(project_dir),
                    embedding_provider="hosted_nvclip",
                    embedding_dim=4,
                )
            )
            session.add(
                Example(
                    example_key="k1",
                    project_id="heal-proj",
                    storage_ref="/fake/k1.jpg",
                    ingested_at="2026-01-01T00:00:00Z",
                    source_metadata={},
                    state="Unlabeled",
                )
            )
            session.commit()

        settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test")
        status = get_embedding_status("heal-proj", str(workspace), settings)
        assert status is not None
        assert status["pending"] == 1
        assert status["worker_active"] is False

        ran: list[str] = []

        async def _stub_worker(pid: str, _ws: str, _st: Any) -> None:
            ran.append(pid)

        with patch(
            "vlm_feedback_loop.services.clip_embedding_service._embedding_worker",
            new=_stub_worker,
        ):
            kicked = maybe_restart_dead_embedding_worker(
                "heal-proj", str(workspace), settings, status
            )
            assert kicked is True
            assert "clip-embed-heal-proj" in background_manager.active_task_ids
            # Let the stub run to completion so no task leaks across tests.
            await asyncio.sleep(0)
        assert ran == ["heal-proj"]

    def test_no_rekick_when_worker_active_or_nothing_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No restart while a worker is already running, and none when
        nothing is pending — the self-heal only targets a dead drain."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            maybe_restart_dead_embedding_worker,
        )

        mock_trigger = MagicMock()
        monkeypatch.setattr(_TRIGGER_PATH, mock_trigger)
        settings = make_settings(tmp_path / "workspace")

        for status in (
            {"pending": 5, "worker_active": True},
            {"pending": 0, "worker_active": False},
        ):
            assert (
                maybe_restart_dead_embedding_worker(
                    "gate-proj", str(tmp_path), settings, status
                )
                is False
            )
        mock_trigger.assert_not_called()

    @pytest.mark.parametrize(
        "overrides",
        [{"EMBEDDINGS_AUTO_COMPUTE": False}, {"EMBEDDING_PROVIDER": "none"}],
    )
    def test_no_rekick_when_operator_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        overrides: dict[str, Any],
    ) -> None:
        """An explicit operator disable is never second-guessed — the same
        intent test as startup recovery."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            maybe_restart_dead_embedding_worker,
        )

        mock_trigger = MagicMock()
        monkeypatch.setattr(_TRIGGER_PATH, mock_trigger)
        settings = make_settings(tmp_path / "workspace", **overrides)

        assert (
            maybe_restart_dead_embedding_worker(
                "disabled-proj",
                str(tmp_path),
                settings,
                {"pending": 3, "worker_active": False},
            )
            is False
        )
        mock_trigger.assert_not_called()

    def test_cooldown_caps_restart_rate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Back-to-back polls against a still-dead drain trigger one restart
        per cooldown window — a permanently failing provider is retried
        politely, not hot-looped on every poll."""
        from vlm_feedback_loop.services.clip_embedding_service import (
            maybe_restart_dead_embedding_worker,
        )

        mock_trigger = MagicMock()
        monkeypatch.setattr(_TRIGGER_PATH, mock_trigger)
        settings = make_settings(tmp_path / "workspace")
        status = {"pending": 3, "worker_active": False}

        assert (
            maybe_restart_dead_embedding_worker(
                "cool-proj", str(tmp_path), settings, status
            )
            is True
        )
        assert (
            maybe_restart_dead_embedding_worker(
                "cool-proj", str(tmp_path), settings, status
            )
            is False
        )
        mock_trigger.assert_called_once()
