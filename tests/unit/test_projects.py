# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Project CRUD, seeded model catalog, and project locking."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from sqlalchemy import text

from conftest import create_project_via_api

UUID4_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

EXPECTED_TABLES = {
    "alembic_version",
    "audit_events",
    "clip_embeddings",
    "dataset_exports",
    "examples",
    "guidances",
    "labels",
    "local_nim_deployments",
    "model_configs",
    "nim_endpoints",
    "operation_records",
    "pools",
    "projects",
    "run_records",
    "student_models",
    "tao_jobs",
}


# ── POST creates directory with subdirs ────────────────────────────────


class TestCreateProjectDirectory:
    def test_creates_directory_with_subdirs(self, test_app_client, tmp_path):
        data = create_project_via_api(test_app_client)
        project_dir = Path(data["project_dir"])
        assert project_dir.exists()
        assert (project_dir / "exports").is_dir()
        assert (project_dir / "artifacts").is_dir()
        assert (project_dir / "logs").is_dir()
        assert (project_dir / "logs" / "operations").is_dir()
        assert (project_dir / "logs" / "runs").is_dir()


# ── POST initializes project.db with all table schemas ────────────────


class TestCreateProjectDatabase:
    def test_initializes_db_with_all_tables(self, test_app_client):
        data = create_project_via_api(test_app_client)
        project_dir = Path(data["project_dir"])
        db_path = project_dir / "project.db"
        assert db_path.exists()

        from vlm_feedback_loop.db.engine import open_project_db

        engine = open_project_db(project_dir)
        with engine.connect() as conn:
            tables = set(
                conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
                .scalars()
                .all()
            )
        assert EXPECTED_TABLES.issubset(tables)


# ── POST returns full Project record with all defaults ─────────────────


class TestCreateProjectDefaults:
    def test_returns_all_defaults(self, test_app_client):
        data = create_project_via_api(test_app_client)

        assert data["name"] == "Test"
        assert UUID4_RE.match(data["project_id"])
        assert data["labeling_generation_preset_key"] == "precise"
        assert data["thinking_default_on"] is True
        assert data["visual_budget_preset_key"] == "high_detail"
        assert data["structured_generation_mode_default"] == "auto"
        assert data["rationale_anti_anchoring"] is True
        assert data["auto_evaluate_enabled"] is False
        assert data["export_field_mode"] == "all"
        assert data["test_pool_fraction"] == pytest.approx(0.40)
        assert data["phash_algorithm"] == "dct_phash_64"
        assert data["scaleup_exact_match_threshold"] == pytest.approx(0.80)
        assert data["scaleup_per_field_match_threshold"] == pytest.approx(0.80)
        assert data["scaleup_min_per_value_f1_threshold"] == pytest.approx(0.80)
        assert data["scaleup_accept_rate_threshold"] == pytest.approx(0.80)
        assert data["scaleup_accept_rate_window"] == 50
        assert data["scaleup_min_test_pool_size"] == 60
        assert data["embedding_provider"] == "none"

        # Teacher selection is a non-null UUID
        assert UUID4_RE.match(data["teacher_model_config_id"])

        # Nullable pointers are null on creation
        assert data["active_guidance_id"] is None
        assert data["active_student_model_config_id"] is None

    def test_selects_teacher_reused_during_project_creation(
        self, test_app_client, monkeypatch
    ):
        """The POST response already selects a matching running local Teacher."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.model_catalog_constants import (
            NEMOTRON_3_NANO_OMNI_REASONING,
        )
        from vlm_feedback_loop.services import environment as environment_service
        from vlm_feedback_loop.services import local_nim_service
        from vlm_feedback_loop.services.environment import GpuInfo
        from vlm_feedback_loop.services.project_service import get_project_engine

        monkeypatch.setattr(
            environment_service,
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

        def reuse_seeded_omni(
            *,
            project_id: str,
            workspace_root: str,
            preferred_model_name: str | None,
        ):
            assert preferred_model_name == NEMOTRON_3_NANO_OMNI_REASONING
            engine = get_project_engine(project_id, workspace_root)
            assert engine is not None
            with Session(engine) as session:
                model_config_id = session.execute(
                    select(ModelConfig.model_config_id).where(
                        ModelConfig.model_name == NEMOTRON_3_NANO_OMNI_REASONING
                    )
                ).scalar_one()
            return (
                model_config_id,
                SimpleNamespace(
                    model_name=NEMOTRON_3_NANO_OMNI_REASONING,
                    project_id="owner-project",
                    deployment_id="deployment-1",
                ),
            )

        monkeypatch.setattr(
            local_nim_service,
            "reuse_first_compatible_running_teacher_for_project",
            reuse_seeded_omni,
        )

        data = create_project_via_api(test_app_client)
        assert UUID4_RE.match(data["teacher_model_config_id"])

        engine = get_project_engine(
            data["project_id"],
            str(Path(data["project_dir"]).parents[1]),
        )
        assert engine is not None
        with Session(engine) as session:
            selected = session.get(ModelConfig, data["teacher_model_config_id"])
            assert selected is not None
            assert selected.model_name == NEMOTRON_3_NANO_OMNI_REASONING

    def test_thinking_default_seeded_from_settings(self, tmp_path):
        """An operator's THINKING_DEFAULT_ON=false reaches new projects.

        The process-level knob seeds the per-project column at create time,
        so a client that configures it off gets projects whose Thinking
        toggle starts OFF.
        """
        from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
        from vlm_feedback_loop.config import Settings
        from vlm_feedback_loop.main import app
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services import project_service

        non_secret = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
        non_secret["THINKING_DEFAULT_ON"] = False
        settings = Settings(WORKSPACE_ROOT=str(tmp_path / "workspace"), **non_secret)
        app.dependency_overrides[get_current_settings] = lambda: settings
        project_service.clear_engine_cache()
        try:
            from fastapi.testclient import TestClient

            client = TestClient(app, raise_server_exceptions=False)
            data = create_project_via_api(client)
            assert data["thinking_default_on"] is False
        finally:
            app.dependency_overrides.clear()
            project_service.clear_engine_cache()


# ── POST with missing name → 422 ──────────────────────────────────────


class TestCreateProjectValidation:
    def test_missing_name_returns_422(self, test_app_client):
        resp = test_app_client.post("/v1/projects", json={})
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("name" in err.get("loc", []) for err in errors), errors


# ── GET non-existent → 404 ────────────────────────────────────────────


class TestGetProjectNotFound:
    def test_returns_404(self, test_app_client):
        resp = test_app_client.get("/v1/projects/nonexistent-uuid")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"


# ── GET list with counts ──────────────────────────────────────────────


class TestListProjects:
    def test_list_with_counts_zero(self, test_app_client):
        create_project_via_api(test_app_client, name="Project A")
        create_project_via_api(test_app_client, name="Project B")

        resp = test_app_client.get("/v1/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        for item in data["items"]:
            assert item["counts"]["verified"] == 0
            assert item["counts"]["unlabeled"] == 0
            assert item["counts"]["auto_labeled"] == 0
            assert item["counts"]["omitted"] == 0
            assert item["counts"]["pending_relabel"] == 0

    def test_list_reflects_setup_completed_at(self, test_app_client):
        """list_projects populates setup_completed_at (it was always null,
        despite the schema/TS type documenting it as meaningful — the UI uses
        it to decide whether onboarding is done)."""
        created = create_project_via_api(test_app_client, name="P")
        pid = created["project_id"]

        # Freshly created → not completed → null in the list.
        before = test_app_client.get("/v1/projects").json()["items"][0]
        assert before["setup_completed_at"] is None

        # Complete setup, then it must appear non-null in the list.
        done = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed",
            json={
                "auto_skip": False,
                "teacher_mode": "hosted",
                "embedding_mode": "hosted",
                "embedding_provider": "none",
            },
        )
        assert done.status_code == 200, done.text
        after = test_app_client.get("/v1/projects").json()["items"][0]
        assert after["setup_completed_at"] is not None


# ── Per-project list isolation on the LIST endpoint ─────────────────────────


class TestListProjectsIsolatesBrokenProject:
    """``GET /v1/projects`` must NOT 500 when one project DB is in
    inconsistent Alembic state (e.g., an empty ``alembic_version`` table
    alongside already-created tables makes ``_run_migrations`` raise
    ``DatabaseMigrationError``). ``services/project_service.py::list_projects``
    skips broken projects with a logged warning so healthy siblings still
    render.
    """

    def test_list_skips_broken_and_returns_healthy(
        self, test_app_client, monkeypatch, caplog
    ):
        import logging

        from vlm_feedback_loop.db.engine import DatabaseMigrationError
        from vlm_feedback_loop.services import project_service

        # Create one healthy project through the proper init path.
        healthy = create_project_via_api(test_app_client, name="Healthy")
        healthy_pid = healthy["project_id"]

        # Stage a second project directory with a stub project.db so the
        # iteration's ``project.db.exists()`` short-circuit doesn't skip
        # it. The DB is opened by ``get_project_engine``, which we patch
        # to raise on this specific project_id.
        broken_pid = "00000000-0000-0000-0000-000000000bad"
        workspace_root = test_app_client.app.dependency_overrides[
            list(test_app_client.app.dependency_overrides)[0]
        ]().WORKSPACE_ROOT
        from pathlib import Path as _Path

        broken_dir = _Path(workspace_root) / "projects" / broken_pid
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "project.db").touch()

        # Monkey-patch ``get_project_engine`` to raise for the broken
        # project and pass through for the healthy one.
        original = project_service.get_project_engine

        def _fakeget_project_engine(project_id, workspace_root):
            if project_id == broken_pid:
                raise DatabaseMigrationError(
                    "Migration failed for "
                    f"{broken_dir / 'project.db'}. Error: "
                    "(sqlite3.OperationalError) table projects already exists"
                )
            return original(project_id, workspace_root)

        monkeypatch.setattr(
            project_service, "get_project_engine", _fakeget_project_engine
        )

        with caplog.at_level(
            logging.WARNING, logger="vlm_feedback_loop.services.project"
        ):
            resp = test_app_client.get("/v1/projects")

        # Whole endpoint MUST NOT 500.
        assert resp.status_code == 200, resp.text
        body = resp.json()
        item_ids = {item["project_id"] for item in body["items"]}

        # Healthy project still rendered.
        assert healthy_pid in item_ids
        # Broken project was skipped — does NOT appear in items.
        assert broken_pid not in item_ids

        # A warning naming the broken project_id and the exception class
        # was logged so an operator immediately sees which DB to fix.
        warning_msgs = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and "Skipping project" in r.getMessage()
        ]
        assert any(broken_pid in m for m in warning_msgs), (
            f"expected a warning citing the broken project {broken_pid}; "
            f"got: {warning_msgs}"
        )
        assert any("DatabaseMigrationError" in m for m in warning_msgs), (
            f"expected exception class name in the warning; got: {warning_msgs}"
        )

    def test_list_skips_broken_and_handles_empty_message_exception(
        self, test_app_client, monkeypatch, caplog
    ):
        """Empty-string exceptions (e.g., ``httpx.ReadError("")``)
        must still produce a non-empty warning so the operator sees
        actionable signal — same pattern as ``services/background.py
        _on_task_done``.
        """
        import logging

        from vlm_feedback_loop.services import project_service

        create_project_via_api(test_app_client, name="Healthy")
        broken_pid = "00000000-0000-0000-0000-000000000eee"
        workspace_root = test_app_client.app.dependency_overrides[
            list(test_app_client.app.dependency_overrides)[0]
        ]().WORKSPACE_ROOT
        from pathlib import Path as _Path

        broken_dir = _Path(workspace_root) / "projects" / broken_pid
        broken_dir.mkdir(parents=True, exist_ok=True)
        (broken_dir / "project.db").touch()

        original = project_service.get_project_engine

        def _fakeget_project_engine(project_id, workspace_root):
            if project_id == broken_pid:
                raise RuntimeError("")  # empty-message
            return original(project_id, workspace_root)

        monkeypatch.setattr(
            project_service, "get_project_engine", _fakeget_project_engine
        )

        with caplog.at_level(
            logging.WARNING, logger="vlm_feedback_loop.services.project"
        ):
            resp = test_app_client.get("/v1/projects")

        assert resp.status_code == 200
        # Warning must include the "(no message)" placeholder since the
        # exception's message is empty.
        warning_msgs = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING and "Skipping project" in r.getMessage()
        ]
        assert any("(no message)" in m for m in warning_msgs), (
            f"expected '(no message)' fallback in warning; got: {warning_msgs}"
        )


# ── counts included on GET /v1/projects/{id} ────────────────────────────────


class TestProjectResponseIncludesCounts:
    """``GET /v1/projects/{id}`` returns ``counts`` so the Student Training
    screen's Training Data card can render verified + auto_labeled totals
    in one request.
    """

    def test_read_project_includes_counts_zero(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        # Create response should already include counts
        assert data["counts"] == {
            "verified": 0,
            "unlabeled": 0,
            "auto_labeled": 0,
            "omitted": 0,
            "pending_relabel": 0,
            "prior_relabeled": 0,
        }

        resp = test_app_client.get(f"/v1/projects/{pid}")
        assert resp.status_code == 200
        body = resp.json()
        assert "counts" in body
        assert body["counts"]["verified"] == 0
        assert body["counts"]["auto_labeled"] == 0
        assert body["counts"]["unlabeled"] == 0
        assert body["counts"]["omitted"] == 0
        assert body["counts"]["pending_relabel"] == 0

    def test_patch_project_response_includes_counts(self, test_app_client):
        pid = create_project_via_api(test_app_client)["project_id"]
        resp = test_app_client.patch(f"/v1/projects/{pid}", json={"name": "X"})
        assert resp.status_code == 200
        assert "counts" in resp.json()
        assert resp.json()["counts"]["verified"] == 0

    def test_patch_no_updates_includes_counts(self, test_app_client):
        pid = create_project_via_api(test_app_client)["project_id"]
        resp = test_app_client.patch(f"/v1/projects/{pid}", json={})
        assert resp.status_code == 200
        assert "counts" in resp.json()


# ── PATCH accepts valid, rejects unknown ──────────────────────────────


class TestUpdateProject:
    def test_accepts_valid_fields(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"name": "New Name", "description": "Updated"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "New Name"
        assert resp.json()["description"] == "Updated"

    def test_rejects_unknown_fields(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"name": "OK", "totally_bogus_field": "nope"},
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("totally_bogus_field" in err.get("loc", []) for err in errors), (
            errors
        )

    def test_active_guidance_id_activates_once_then_cannot_switch(
        self, test_app_client
    ):
        """The active_guidance_id pointer is set once (FTUE first activation);
        after that it only moves through the guidance edit endpoint, which
        cancels in-flight runs and re-points existing labels. A direct switch
        via PATCH would orphan the corpus, so it is rejected."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.guidance import Guidance
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            for gid, ver in (("g1", 1), ("g2", 2)):
                session.add(
                    Guidance(
                        guidance_id=gid,
                        project_id=pid,
                        version_number=ver,
                        description="g",
                        schema={"fields": []},
                    )
                )
            session.commit()

        # First activation (FTUE): allowed.
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"active_guidance_id": "g1"}
        )
        assert resp.status_code == 200
        assert resp.json()["active_guidance_id"] == "g1"

        # Direct switch to a different version: rejected.
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"active_guidance_id": "g2"}
        )
        assert resp.status_code == 400
        assert "switched or cleared directly" in resp.json()["detail"]

        # Explicit null is rejected too: clearing the pointer would orphan
        # the corpus outright, and clear-then-set would sidestep the switch
        # guard in two PATCHes.
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"active_guidance_id": None}
        )
        assert resp.status_code == 400
        assert "switched or cleared directly" in resp.json()["detail"]

        # Setting it to the current value is an idempotent no-op, not a switch.
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"active_guidance_id": "g1"}
        )
        assert resp.status_code == 200

        resp = test_app_client.get(f"/v1/projects/{pid}")
        assert resp.json()["active_guidance_id"] == "g1"

    def test_rejects_out_of_range_test_pool_fraction(self, test_app_client):
        """A fraction outside [0, 1] would corrupt pool routing; ProjectUpdate
        now bounds it (was applied via blind setattr with no validation)."""
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]

        for bad in (2.0, -0.5):
            resp = test_app_client.patch(
                f"/v1/projects/{pid}", json={"test_pool_fraction": bad}
            )
            assert resp.status_code == 422, f"{bad} should be rejected"
            errors = resp.json()["detail"]
            assert any("test_pool_fraction" in err.get("loc", []) for err in errors), (
                errors
            )

    def test_rejects_out_of_range_scaleup_threshold(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"scaleup_exact_match_threshold": 1.5}
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any(
            "scaleup_exact_match_threshold" in err.get("loc", []) for err in errors
        ), errors

    def test_rejects_invalid_export_field_mode(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"export_field_mode": "not_a_mode"}
        )
        assert resp.status_code == 422
        errors = resp.json()["detail"]
        assert any("export_field_mode" in err.get("loc", []) for err in errors), errors

    def test_rejects_schema_refinement_reminders_dismissed(self, test_app_client):
        """Reminder dismissal is owned by ``POST .../guidance:dismiss_reminder``
        the counter is not client-patchable, so ``extra="forbid"``
        rejects it like any unknown field. Guards against the field quietly
        re-growing a second writer path."""
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]

        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"schema_refinement_reminders_dismissed": 1},
        )
        assert resp.status_code == 422, resp.text
        errors = resp.json()["detail"]
        assert any(
            "schema_refinement_reminders_dismissed" in err.get("loc", [])
            for err in errors
        ), errors


# ── PATCH rejects cross-project ID references ─────────────────────────


class TestUpdateCrossProjectRef:
    def test_rejects_cross_project_ref(self, test_app_client):
        proj_a = create_project_via_api(test_app_client, name="A")
        proj_b = create_project_via_api(test_app_client, name="B")

        # Get project A's teacher_model_config_id (which belongs to A)
        a_teacher = proj_a["teacher_model_config_id"]

        # Try to set project B's teacher to project A's model config
        resp = test_app_client.patch(
            f"/v1/projects/{proj_b['project_id']}",
            json={"teacher_model_config_id": a_teacher},
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "teacher_model_config_id" in detail, detail
        assert "does not exist in project" in detail, detail


# ── Top-bar Teacher picker — role + vision validation ─────────


class TestUpdateTeacherRoleValidation:
    """A model assigned as
    ``teacher_model_config_id`` MUST have the ``teacher`` role and
    ``supports_image_input=true``.

    The top-bar Teacher picker filters the dropdown to teacher-role
    entries client-side, but the backend is the single source of truth
    and MUST reject invalid assignments independently.
    """

    def test_accepts_teacher_role_peer_switch(self, test_app_client):
        """Switch between two teacher-eligible entries in the same project."""
        proj = create_project_via_api(test_app_client)
        pid = proj["project_id"]

        # List catalog entries and pick a different teacher-role entry.
        catalog = test_app_client.get(
            f"/v1/projects/{pid}/model_configs?eligible_role=teacher",
        ).json()["items"]
        assert len(catalog) >= 2
        current = proj["teacher_model_config_id"]
        peer = next(mc for mc in catalog if mc["model_config_id"] != current)

        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"teacher_model_config_id": peer["model_config_id"]},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["teacher_model_config_id"] == peer["model_config_id"]


# ── WORKSPACE_ROOT auto-creates ────────────────────────────────────────


class TestInvalidCursorHandler:
    """Malformed pagination cursors are a client error (400) on every
    list endpoint — some list endpoints used to 500 because only the
    examples router caught InvalidCursorError."""

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/projects/{pid}/student_models?cursor=%%%garbage",
            "/v1/projects/{pid}/evaluation_runs?cursor=%%%garbage",
            "/v1/projects/{pid}/examples?cursor=%%%garbage",
        ],
    )
    def test_garbage_cursor_returns_400(self, test_app_client, path):
        create = test_app_client.post(
            "/v1/projects", json={"name": "cursor-test", "description": ""}
        )
        pid = create.json()["project_id"]
        resp = test_app_client.get(path.format(pid=pid))
        assert resp.status_code == 400, resp.text
        assert "cursor" in resp.json()["detail"].lower()


class TestProjectLockedHandler:
    """The app maps ProjectLockedError to a 409 with the exact
    already-open message — exercised through the REAL registered handler,
    not a source grep."""

    def test_locked_project_returns_409_with_spec_message(
        self, test_app_client, monkeypatch
    ):
        """The 409 detail is the fixed already-open message regardless of
        what text the raised ProjectLockedError carries."""
        from vlm_feedback_loop.services import project_service
        from vlm_feedback_loop.services.locks import ProjectLockedError

        create_resp = test_app_client.post(
            "/v1/projects", json={"name": "Lock Message Test"}
        )
        project_id = create_resp.json()["project_id"]
        project_service.clear_engine_cache()

        def fake_lock(project_dir):
            raise ProjectLockedError(str(project_dir))

        monkeypatch.setattr(
            "vlm_feedback_loop.services.project_service.acquire_project_lock",
            fake_lock,
        )

        resp = test_app_client.get(f"/v1/projects/{project_id}")
        assert resp.status_code == 409
        assert resp.json()["detail"] == (
            "This project is already open in another process."
        )


class TestProbeLocalTeacher:
    """The create-time local-Teacher health probe rides the repo's one HTTP
    library and carries the Blueprint source header."""

    def test_healthy_probe_returns_true_and_carries_source_header(self, monkeypatch):
        import httpx

        from vlm_feedback_loop.services import project_service

        seen: dict = {}

        def fake_get(url, timeout, headers):
            seen["url"] = url
            seen["headers"] = headers
            return httpx.Response(200, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "get", fake_get)
        assert project_service._probe_local_teacher("http://localhost:8801/v1")
        assert seen["url"] == "http://localhost:8801/v1/health/ready"
        assert seen["headers"]["source"] == "vlm-feedback-loop"

    def test_unreachable_probe_returns_false(self, monkeypatch):
        import httpx

        from vlm_feedback_loop.services import project_service

        def fake_get(url, timeout, headers):
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(httpx, "get", fake_get)
        assert not project_service._probe_local_teacher("http://localhost:8801/v1")


class TestWorkspaceAutoCreate:
    def test_auto_creates_workspace(self, test_app_client, tmp_path):
        """The workspace is auto-created by the create_project service."""
        data = create_project_via_api(test_app_client)
        workspace = Path(data["project_dir"]).parent.parent
        assert workspace.exists()


# ── Project scoping ──────────────────────────────────────────────────


class TestProjectScoping:
    def test_catalog_scoped_to_project(self, test_app_client):
        proj_a = create_project_via_api(test_app_client, name="A")
        proj_b = create_project_via_api(test_app_client, name="B")

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        # Get settings workspace from project_dir
        ws = str(Path(proj_a["project_dir"]).parent.parent)

        engine_a = get_project_engine(proj_a["project_id"], ws)
        from sqlalchemy.orm import Session

        with Session(engine_a) as s:
            configs_a = s.query(ModelConfig).all()
            assert len(configs_a) == 10
            for mc in configs_a:
                assert mc.project_id == proj_a["project_id"]

        engine_b = get_project_engine(proj_b["project_id"], ws)
        with Session(engine_b) as s:
            configs_b = s.query(ModelConfig).all()
            assert len(configs_b) == 10
            for mc in configs_b:
                assert mc.project_id == proj_b["project_id"]


# ── feature_flags accepts the known flags ───────────────────────


class TestFeatureFlags:
    def test_accepts_spec_flags(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]

        flags = {
            "REVIEW_SELECTION_MODE": "auto",
            "BATCH_LABEL_ENABLED": True,
            "STUDENT_TRAINING_ENABLED": False,
            "EMBEDDINGS_AUTO_COMPUTE": True,
            "CLIP_SWITCHOVER_MIN_COUNT": 100,
        }
        resp = test_app_client.patch(
            f"/v1/projects/{pid}",
            json={"feature_flags": flags},
        )
        assert resp.status_code == 200
        assert resp.json()["feature_flags"] == flags


# ── seeded catalog entries with correct fields ──────────────────────


class TestSeededCatalog:
    def test_seeded_catalog_entries_with_correct_fields(self, test_app_client):
        data = create_project_via_api(test_app_client)
        ws = str(Path(data["project_dir"]).parent.parent)

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(data["project_id"], ws)
        with Session(engine) as s:
            configs = s.query(ModelConfig).order_by(ModelConfig.model_name).all()
            assert len(configs) == 10

            by_name = {mc.model_name: mc for mc in configs}

            # Nemotron 3 Nano Omni — hosted or system-managed local Teacher.
            # The NIM supports the Qwen thinking switch; it intentionally
            # remains Teacher-only while Cosmos supplies Student bases.
            nemo_omni = by_name["nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"]
            assert nemo_omni.context_window_tokens == 128000
            assert set(nemo_omni.eligible_roles) == {"teacher"}
            assert nemo_omni.supports_image_input is True
            assert nemo_omni.thinking_toggle_mode == "qwen_enable_thinking"
            assert nemo_omni.thinking_toggle_support == "supported"
            assert nemo_omni.visual_budget_mode == "none"
            assert nemo_omni.default_icl_max_examples == 4
            assert nemo_omni.local_deploy_metadata is not None
            assert (
                nemo_omni.local_deploy_metadata["nim_container_image"]
                == "nvcr.io/nim/nvidia/"
                "nemotron-3-nano-omni-30b-a3b-reasoning:1.7.0-variant"
            )
            assert nemo_omni.local_deploy_metadata["nim_gpu_memory_minimum_gb"] == 80
            assert nemo_omni.local_deploy_metadata[
                "nim_compute_capability_minimum"
            ] == pytest.approx(9.0)

            # nvidia/cosmos-reason2-8b
            c8b = by_name["nvidia/cosmos-reason2-8b"]
            assert c8b.context_window_tokens == 256000
            assert set(c8b.eligible_roles) == {"teacher", "student_base"}
            assert c8b.supports_image_input is True
            assert c8b.thinking_toggle_mode == "qwen_enable_thinking"
            assert c8b.visual_budget_mode == "mm_processor_size"
            assert c8b.local_deploy_metadata is not None
            assert (
                c8b.local_deploy_metadata["nim_container_image"]
                == "nvcr.io/nim/nvidia/cosmos-reason2-8b:1.6.0"
            )
            assert c8b.local_deploy_metadata["nim_gpu_memory_minimum_gb"] == 56

            # nvidia/cosmos-reason2-2b
            c2b = by_name["nvidia/cosmos-reason2-2b"]
            assert c2b.context_window_tokens == 256000
            assert c2b.local_deploy_metadata["nim_gpu_memory_minimum_gb"] == 36

            # Cosmos 3 reasoning-tower student_base entries. The trainable
            # ``-Reasoner`` VLMs — proven end-to-end on TAO FTMS.
            cr3_nano = by_name["nvidia/cosmos3-nano-reasoner"]
            assert "student_base" in cr3_nano.eligible_roles
            assert cr3_nano.supports_image_input is True
            assert cr3_nano.hosted_compatible is False
            assert (
                cr3_nano.local_deploy_metadata["nim_container_image"]
                == "nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0"
            )
            assert cr3_nano.local_deploy_metadata["nim_model_size"] == "nano"

            cr3_super = by_name["nvidia/cosmos3-super-reasoner"]
            assert "student_base" in cr3_super.eligible_roles
            assert cr3_super.local_deploy_metadata["nim_model_size"] == "super"

            # mistral
            mistral = by_name["mistralai/mistral-large-3-675b-instruct-2512"]
            assert mistral.context_window_tokens == 262144
            assert set(mistral.eligible_roles) == {"teacher"}
            assert mistral.thinking_toggle_mode == "none"
            assert mistral.visual_budget_mode == "none"
            assert mistral.local_deploy_metadata is None

            # nemotron
            nemo = by_name["nvidia/nemotron-nano-12b-v2-vl"]
            assert nemo.visual_budget_mode == "mm_processor_tiles"

            # stepfun-ai/step-3.7-flash — 2026-07-21 pv-campaign addition.
            # Reasoning-by-default with no working toggle (enable_thinking
            # is accepted-but-ignored, live-probed), so it carries the
            # always_on_reasoning headroom contract like Omni.
            step = by_name["stepfun-ai/step-3.7-flash"]
            assert step.context_window_tokens == 262144
            assert set(step.eligible_roles) == {"teacher"}
            assert step.supports_image_input is True
            assert step.thinking_toggle_mode == "always_on_reasoning"
            assert step.thinking_toggle_support == "unsupported"
            assert step.visual_budget_mode == "none"
            assert step.max_images_per_request == 8
            assert step.image_cap_support == "supported"
            assert step.local_deploy_metadata is None

            # minimaxai/minimax-m3 — deep-ICL hosted Teacher: no image-count
            # boundary found through 33 (the ~5 MB body cap bites first),
            # seeded at the pv-validated 32 with cap support "unknown".
            minimax = by_name["minimaxai/minimax-m3"]
            assert minimax.context_window_tokens == 500000
            assert set(minimax.eligible_roles) == {"teacher"}
            assert minimax.thinking_toggle_mode == "none"
            assert minimax.max_images_per_request == 32
            assert minimax.image_cap_support == "unknown"
            assert minimax.default_icl_max_examples == 8
            assert minimax.local_deploy_metadata is None

            # mistralai/mistral-medium-3.5-128b — near-ceiling Mistral-family
            # alternate; same no-thinking contract as Large, live-probed
            # 10-image cap.
            medium = by_name["mistralai/mistral-medium-3.5-128b"]
            assert medium.context_window_tokens == 262144
            assert set(medium.eligible_roles) == {"teacher"}
            assert medium.thinking_toggle_mode == "none"
            assert medium.visual_budget_mode == "none"
            assert medium.max_images_per_request == 10
            assert medium.local_deploy_metadata is None

    def test_capability_support_preseeded_never_unknown(self, test_app_client):
        """Preseeded models ship with resolved capability
        support, not `unknown`. Values come from live probes, vendor
        docs, or definitional (mode=none ⇒ unsupported). The `unknown`
        sentinel is reserved for non-seeded entries an SME adds later."""
        data = create_project_via_api(test_app_client)
        ws = str(Path(data["project_dir"]).parent.parent)

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(data["project_id"], ws)
        with Session(engine) as s:
            configs = s.query(ModelConfig).all()
            for mc in configs:
                assert mc.structured_generation_support in (
                    "supported",
                    "unsupported",
                ), f"{mc.model_name} sg={mc.structured_generation_support}"
                assert mc.thinking_toggle_support in (
                    "supported",
                    "unsupported",
                ), f"{mc.model_name} tt={mc.thinking_toggle_support}"
                assert mc.visual_budget_support in (
                    "supported",
                    "unsupported",
                ), f"{mc.model_name} vb={mc.visual_budget_support}"

            by_name = {mc.model_name: mc for mc in configs}

            # Values come from live probes against the hosted NIM.
            nemo = by_name["nvidia/nemotron-nano-12b-v2-vl"]
            assert nemo.visual_budget_support == "supported"

            # Intrinsic capability (Cosmos hosted endpoint is NVCF-gated;
            # intended deployment is local NIM where both SG and TT work).
            c8b = by_name["nvidia/cosmos-reason2-8b"]
            assert c8b.structured_generation_support == "supported"
            assert c8b.thinking_toggle_support == "supported"
            assert c8b.visual_budget_support == "supported"


# ── The Teacher selection points to a real entry ──────────────────────


class TestDefaultSelections:
    def test_teacher_references_real_entry(self, test_app_client):
        data = create_project_via_api(test_app_client)
        ws = str(Path(data["project_dir"]).parent.parent)

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(data["project_id"], ws)
        with Session(engine) as s:
            teacher = (
                s.query(ModelConfig)
                .filter_by(model_config_id=data["teacher_model_config_id"])
                .first()
            )
            assert teacher is not None
            # Default Teacher is MiniMax-M3, superseding Step 3.7 Flash after
            # its hosted endpoint's reasoning-length regression
            # tripled interactive latency): best-fit reachable hosted Teacher —
            # top of the certified accuracy field, deep-ICL capable, and lower
            # measured interactive latency.
            assert teacher.model_name == "minimaxai/minimax-m3"


# ── All seeded entries reference the hosted NIM endpoint ──────────────


class TestHostedEndpoint:
    def test_all_entries_reference_hosted_endpoint(self, test_app_client):
        data = create_project_via_api(test_app_client)
        ws = str(Path(data["project_dir"]).parent.parent)

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
        from vlm_feedback_loop.services.project_service import get_project_engine

        engine = get_project_engine(data["project_id"], ws)
        with Session(engine) as s:
            configs = s.query(ModelConfig).all()
            endpoint_ids = {mc.endpoint_id for mc in configs}
            assert len(endpoint_ids) == 1  # all share the same endpoint

            ep = s.query(NimEndpoint).filter_by(endpoint_id=endpoint_ids.pop()).first()
            assert ep is not None
            assert ep.display_name == "NVIDIA Hosted NIM"
            assert ep.base_url == "https://integrate.api.nvidia.com/v1"
            assert ep.endpoint_mode == "hosted"
            assert ep.source_kind == "seeded_hosted"
            assert ep.auth_mode == "bearer"


# ─── Archive (soft-archive feature) ────────────────────────────────


class TestArchiveProject:
    """Soft-archive sets archived_at, writes the .archived sentinel marker,
    writes a project_archived AuditEvent, and excludes the project from the
    default list response."""

    def test_archive_sets_archived_at_and_marker(self, test_app_client):
        data = create_project_via_api(test_app_client, name="To Archive")
        pid = data["project_id"]
        project_dir = Path(data["project_dir"])

        resp = test_app_client.post(f"/v1/projects/{pid}:archive")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["archived_at"] is not None
        assert body["archived_at"].endswith("Z")
        assert (project_dir / ".archived").exists()

    def test_archive_writes_audit_event(self, test_app_client):
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client, name="Audit Archive")
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        resp = test_app_client.post(f"/v1/projects/{pid}:archive")
        assert resp.status_code == 200

        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            events = (
                session.query(AuditEvent)
                .filter_by(project_id=pid, event_type="project_archived")
                .all()
            )
        assert len(events) == 1
        assert events[0].event_data.get("archived_at")

    def test_archive_already_archived_returns_409(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        assert test_app_client.post(f"/v1/projects/{pid}:archive").status_code == 200

        resp2 = test_app_client.post(f"/v1/projects/{pid}:archive")
        assert resp2.status_code == 409
        assert resp2.json().get("code") == "already_archived"

    def test_list_excludes_archived_by_default(self, test_app_client):
        a = create_project_via_api(test_app_client, name="Active")
        b = create_project_via_api(test_app_client, name="Archived")
        assert (
            test_app_client.post(f"/v1/projects/{b['project_id']}:archive").status_code
            == 200
        )

        resp = test_app_client.get("/v1/projects")
        ids = [p["project_id"] for p in resp.json()["items"]]
        assert a["project_id"] in ids
        assert b["project_id"] not in ids

    def test_list_include_archived_includes_both(self, test_app_client):
        a = create_project_via_api(test_app_client, name="Active")
        b = create_project_via_api(test_app_client, name="Archived")
        assert (
            test_app_client.post(f"/v1/projects/{b['project_id']}:archive").status_code
            == 200
        )

        resp = test_app_client.get("/v1/projects?include_archived=true")
        rows = {p["project_id"]: p for p in resp.json()["items"]}
        assert a["project_id"] in rows
        assert b["project_id"] in rows
        assert rows[a["project_id"]]["archived_at"] is None
        assert rows[b["project_id"]]["archived_at"] is not None

    def test_busy_gate_blocks_archive_with_inflight_run(self, test_app_client):
        """A non-terminal RunRecord makes the project 'busy' — archive 409s."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.run import RunRecord
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        engine = get_project_engine(pid, ws)

        # Insert a fake "running" RunRecord directly.
        with Session(engine) as session:
            session.add(
                RunRecord(
                    project_id=pid,
                    run_type="evaluation_run",
                    status="running",
                )
            )
            session.commit()

        resp = test_app_client.post(f"/v1/projects/{pid}:archive")
        assert resp.status_code == 409
        body = resp.json()
        assert body.get("code") == "project_busy"
        assert any("evaluation/batch" in r for r in body.get("reasons", []))

    def test_busy_gate_blocks_archive_with_running_export(self, test_app_client):
        """A running dataset export (its own table since the async export
        change, not a RunRecord) makes the project busy — archive 409s,
        so recovery's archived-skip cannot strand the export as running
        forever."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.dataset_export import DatasetExport
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            session.add(
                DatasetExport(
                    dataset_export_id="de-running",
                    project_id=pid,
                    dataset_intent="training",
                    label_tier_filter="verified_only",
                    export_field_mode="all",
                    guidance_id="g",
                    example_count=0,
                    selection_definition_snapshot={},
                    status="running",
                )
            )
            session.commit()

        resp = test_app_client.post(f"/v1/projects/{pid}:archive")
        assert resp.status_code == 409
        assert resp.json().get("code") == "project_busy"
        assert any("export" in r for r in resp.json().get("reasons", []))


class TestHasArchivedFlag:
    """The list response carries a workspace-global ``has_archived`` flag,
    computed from the ``.archived`` marker files alone (never opening a
    project DB), so the UI can decide whether to offer "Show archived"
    without a second archived-inclusive fetch."""

    def test_false_when_nothing_archived(self, test_app_client):
        create_project_via_api(test_app_client, name="Only Active")
        resp = test_app_client.get("/v1/projects")
        assert resp.status_code == 200
        assert resp.json()["has_archived"] is False

    def test_true_after_archiving_even_on_default_list(self, test_app_client):
        create_project_via_api(test_app_client, name="Active")
        b = create_project_via_api(test_app_client, name="Archived")
        assert (
            test_app_client.post(f"/v1/projects/{b['project_id']}:archive").status_code
            == 200
        )

        # Default (active-only) list still reports the workspace-global flag.
        resp = test_app_client.get("/v1/projects")
        body = resp.json()
        assert body["has_archived"] is True
        assert b["project_id"] not in [p["project_id"] for p in body["items"]]

    def test_missing_marker_is_healed_by_list_scan(self, test_app_client):
        """Marker lost (crash/manual) while the DB column says archived: the
        list scan trusts the DB and rewrites the marker, so the flag stays
        true and later scans stop paying a DB open for this project."""
        data = create_project_via_api(test_app_client, name="Drifted")
        pid = data["project_id"]
        marker = Path(data["project_dir"]) / ".archived"
        assert test_app_client.post(f"/v1/projects/{pid}:archive").status_code == 200
        marker.unlink()

        resp = test_app_client.get("/v1/projects")
        assert marker.exists(), "list scan should rewrite the lost marker"
        assert resp.json()["has_archived"] is True

    def test_stale_marker_is_removed_by_archived_scan(self, test_app_client):
        """Marker present on a project whose DB column says NOT archived:
        the archived-inclusive scan removes the stale marker and keeps the
        project listed as active."""
        data = create_project_via_api(test_app_client, name="Stale Marker")
        pid = data["project_id"]
        marker = Path(data["project_dir"]) / ".archived"
        marker.write_text("2026-01-01T00:00:00Z")

        resp = test_app_client.get("/v1/projects?include_archived=true")
        rows = {p["project_id"]: p for p in resp.json()["items"]}
        assert rows[pid]["archived_at"] is None
        assert not marker.exists(), "archived scan should drop the stale marker"


class TestUnarchiveProject:
    def test_unarchive_clears_column_and_marker(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        project_dir = Path(data["project_dir"])
        assert test_app_client.post(f"/v1/projects/{pid}:archive").status_code == 200
        assert (project_dir / ".archived").exists()

        resp = test_app_client.post(f"/v1/projects/{pid}:unarchive")
        assert resp.status_code == 200, resp.text
        assert resp.json()["archived_at"] is None
        assert not (project_dir / ".archived").exists()

    def test_unarchive_when_not_archived_returns_409(self, test_app_client):
        data = create_project_via_api(test_app_client)
        resp = test_app_client.post(f"/v1/projects/{data['project_id']}:unarchive")
        assert resp.status_code == 409
        assert resp.json().get("code") == "not_archived"


class TestArchivedProjectMutationGuard:
    """PATCH and the three start-runs endpoints reject mutations on archived
    projects with 409 ``project_archived`` via the ``require_not_archived``
    dependency."""

    def test_patch_archived_returns_409(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        assert test_app_client.post(f"/v1/projects/{pid}:archive").status_code == 200

        resp = test_app_client.patch(
            f"/v1/projects/{pid}", json={"thinking_default_on": False}
        )
        assert resp.status_code == 409
        assert resp.json().get("code") == "project_archived"

    def test_evaluation_run_on_archived_returns_409(self, test_app_client):
        data = create_project_via_api(test_app_client)
        pid = data["project_id"]
        assert test_app_client.post(f"/v1/projects/{pid}:archive").status_code == 200

        resp = test_app_client.post(f"/v1/projects/{pid}/evaluation_runs", json={})
        assert resp.status_code == 409
        assert resp.json().get("code") == "project_archived"


# ── FTU setup_completed_at ───────────────────────────────────────────────────


class TestSetupCompletedAt:
    """``POST /v1/projects/{id}:mark_setup_completed``."""

    def test_new_project_has_null_setup_completed_at(self, test_app_client):
        """A freshly-created project has setup_completed_at=null so
        ProjectIndexRedirect routes the SME through the setup screens."""
        data = create_project_via_api(test_app_client, "FTU")
        assert data["setup_completed_at"] is None

    def test_mark_setup_completed_first_call_stamps_and_emits_audit(
        self, test_app_client
    ):
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client, "FTU")
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        resp = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed",
            json={
                "auto_skip": True,
                "teacher_mode": "hosted",
                "embedding_mode": "hosted",
                "embedding_provider": "hosted_nvclip",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["transitioned"] is True
        assert body["project"]["setup_completed_at"] is not None

        # AuditEvent persisted with the recommendation context.
        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            events = (
                session.query(AuditEvent).filter_by(event_type="setup_completed").all()
            )
            assert len(events) == 1
            assert events[0].event_data["auto_skip"] is True
            assert events[0].event_data["teacher_mode"] == "hosted"
            assert events[0].event_data["embedding_provider"] == "hosted_nvclip"

    def test_mark_setup_completed_is_idempotent(self, test_app_client):
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client, "FTU")
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)
        body_payload = {
            "auto_skip": False,
            "teacher_mode": "hosted",
            "embedding_mode": "hosted",
            "embedding_provider": "hosted_nvclip",
        }
        first = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed", json=body_payload
        )
        assert first.status_code == 200
        assert first.json()["transitioned"] is True
        first_timestamp = first.json()["project"]["setup_completed_at"]

        # Second call: no-op, no double-stamp, no second AuditEvent.
        second = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed", json=body_payload
        )
        assert second.status_code == 200
        assert second.json()["transitioned"] is False
        # Timestamp unchanged — proves the service-layer guard fired.
        assert second.json()["project"]["setup_completed_at"] == first_timestamp

        # Exactly one AuditEvent total.
        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            events = (
                session.query(AuditEvent).filter_by(event_type="setup_completed").all()
            )
            assert len(events) == 1

    def test_mark_setup_completed_not_found(self, test_app_client):
        resp = test_app_client.post(
            "/v1/projects/00000000-0000-4000-8000-000000000000:mark_setup_completed",
            json={
                "auto_skip": False,
                "teacher_mode": "hosted",
                "embedding_mode": "hosted",
                "embedding_provider": "hosted_nvclip",
            },
        )
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Project not found"

    def test_mark_setup_completed_persists_local_deploy_queued(self, test_app_client):
        """The FTUE local-deploy confirm step
        calls :mark_setup_completed with the list of local NIM models
        it just kicked off (Cosmos Reason2 + embedding NIM).
        That list MUST flow into the ``setup_completed`` AuditEvent so
        the forensic record captures which deploys were queued."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client, "FTU-local-queued")
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)

        resp = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed",
            json={
                "auto_skip": False,
                "teacher_mode": "local",
                "embedding_mode": "local",
                "embedding_provider": "self_hosted_nvclip",
                "local_deploy_queued": [
                    "nvidia/cosmos-reason2-8b",
                    "nvidia/llama-nemotron-embed-vl-1b-v2",
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["transitioned"] is True

        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            event = (
                session.query(AuditEvent).filter_by(event_type="setup_completed").one()
            )
            assert event.event_data["local_deploy_queued"] == [
                "nvidia/cosmos-reason2-8b",
                "nvidia/llama-nemotron-embed-vl-1b-v2",
            ]
            assert event.event_data["teacher_mode"] == "local"

    def test_mark_setup_completed_defaults_local_deploy_queued_to_empty(
        self, test_app_client
    ):
        """Existing callers that don't pass ``local_deploy_queued`` get
        an empty list in the AuditEvent (backwards-compatible default
        on ``MarkSetupCompletedRequest``)."""
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        data = create_project_via_api(test_app_client, "FTU-no-queued")
        pid = data["project_id"]
        ws = str(Path(data["project_dir"]).parent.parent)

        resp = test_app_client.post(
            f"/v1/projects/{pid}:mark_setup_completed",
            json={
                "auto_skip": True,
                "teacher_mode": "hosted",
                "embedding_mode": "hosted",
                "embedding_provider": "hosted_nvclip",
            },
        )
        assert resp.status_code == 200

        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            event = (
                session.query(AuditEvent).filter_by(event_type="setup_completed").one()
            )
            assert event.event_data["local_deploy_queued"] == []
