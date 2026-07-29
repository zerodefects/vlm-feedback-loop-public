# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit-suite-wide fixtures and canonical test factories.

Complements ``tests/conftest.py`` (shared with the integration suite) with
fixtures that must run around *every* unit test, plus the canonical
helpers that replace the per-file ``_make_settings`` /
``_create_project`` / ``_setup_project_db`` copies.

Test modules import the helpers directly (``from conftest import
make_settings``): ``tests/unit/`` is not a package, so pytest's default
prepend import mode puts this directory on ``sys.path`` and the import
resolves to this very module object.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from vlm_feedback_loop._defaults import DEFAULTS, SECRET_KEYS
from vlm_feedback_loop.config import Settings
from vlm_feedback_loop.db import engine as db_engine_module
from vlm_feedback_loop.db.base import generate_uuid4, utc_now
from vlm_feedback_loop.db.engine import open_project_db
from vlm_feedback_loop.db.models.dataset_export import DatasetExport
from vlm_feedback_loop.db.models.example import Example
from vlm_feedback_loop.db.models.guidance import Guidance
from vlm_feedback_loop.db.models.model_config import ModelConfig
from vlm_feedback_loop.db.models.nim_endpoint import NimEndpoint
from vlm_feedback_loop.db.models.project import Project
from vlm_feedback_loop.db.models.tao_job import TAOJob
from vlm_feedback_loop.services.project_service import (
    clear_engine_cache,
    set_project_engine,
)
from vlm_feedback_loop.services.prompt_service import TeacherInvocationResult


@pytest.fixture(autouse=True)
def _clear_project_engine_cache():
    """Reset process-global state after every test: the project-engine
    cache and the FastAPI app's dependency overrides.

    yield-first teardown, so it runs even when the test body fails —
    replacing the old per-test trailing ``_cleanup()`` idiom, which leaked
    cached engines (and lock state) into the next test whenever an
    assertion fired before the trailing call.

    The ``dependency_overrides.clear()`` covers the many helpers that
    install a Settings override on the shared ``main.app`` without their
    own teardown; without it, a tmp-path Settings leaks into any later
    test in the same process that doesn't set its own. Fixtures that
    already clear (e.g. ``test_app_client``) just make this a no-op.

    Function-scoped, so it is correct under ``pytest-xdist --dist
    loadfile``: each worker process holds its own engine cache and clears
    it after each of its own tests.
    """
    yield
    from vlm_feedback_loop.main import app

    app.dependency_overrides.clear()
    clear_engine_cache()


# ── Template project DB ──────────────────────────────────────────────────────
# The suite creates hundreds of project databases, so `_fast_project_db`
# replaces repeated Alembic baseline execution with a file copy of a
# once-per-worker template built by the real migration runner. Marker
# `real_migrations` opts a test out; `test_project_db_template.py` pins
# template ≡ baseline equivalence and that the fast path is engaged.

_REAL_RUN_MIGRATIONS = db_engine_module._run_migrations


class ProjectDbTemplate:
    """Lazy, per-xdist-worker, current-revision ``project.db`` template."""

    def __init__(self, factory: pytest.TempPathFactory) -> None:
        self._factory = factory
        self.path: Path | None = None
        self.head_rev: str | None = None
        self.fast_copies = 0  # observability for the engaged-canary test

    def ensure_built(self) -> Path:
        if self.path is not None:
            return self.path
        tmpl_path = self._factory.mktemp("project-db-template") / "project.db"
        engine = db_engine_module._create_engine(tmpl_path)
        try:
            db_engine_module._ensure_wal_mode(engine)
            _REAL_RUN_MIGRATIONS(engine, tmpl_path)  # the real chain, once
            with engine.connect() as conn:
                # Fold the WAL into the main file so it is a complete,
                # self-contained snapshot safe to copy.
                conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
                self.head_rev = conn.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar()
        finally:
            engine.dispose()
        self.path = tmpl_path
        return tmpl_path


@pytest.fixture(scope="session")
def project_db_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> ProjectDbTemplate:
    """Session-scoped (= per xdist worker) template holder; built lazily so
    workers that never open a project DB pay nothing."""
    return ProjectDbTemplate(tmp_path_factory)


@pytest.fixture(autouse=True)
def _fast_project_db(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    project_db_template: ProjectDbTemplate,
):
    """Swap the fresh-DB migration replay for a template file copy.

    Only a *pristine* file (no alembic stamp, zero objects) takes the
    copy path; a DB already at the template's head short-circuits like
    the real runner; anything else — hand-crafted stamps, mid-chain
    states, corruption tests — falls through to the real chain, so tests
    that exercise migration behavior stay honest without opting out.
    """
    if request.node.get_closest_marker("real_migrations") or os.environ.get(
        "VLM_TEST_REAL_MIGRATIONS"
    ):
        # Marker: this test validates migration behavior itself.
        # Env var: suite-wide escape hatch (e.g. timing the true chain,
        # or ruling the fast path out while debugging a DB-shaped failure).
        yield
        return
    template = project_db_template

    def fast_run_migrations(engine: Engine, db_path: Path) -> None:
        with engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
            objects = conn.exec_driver_sql(
                "SELECT count(*) FROM sqlite_master"
            ).scalar()
        if current is None and objects == 0:
            template.ensure_built()
            # The pool holds a connection to the (empty) file from the
            # open_project_db preamble; overwriting under a live pooled
            # connection risks stale page-cache reads. Dispose first —
            # the engine object stays valid and reconnects on next use.
            engine.dispose()
            Path(f"{db_path}-wal").unlink(missing_ok=True)
            Path(f"{db_path}-shm").unlink(missing_ok=True)
            assert template.path is not None
            shutil.copyfile(template.path, db_path)
            template.fast_copies += 1
            return
        if template.head_rev is not None and current == template.head_rev:
            return  # reopen of a template-built DB — already at head
        _REAL_RUN_MIGRATIONS(engine, db_path)

    monkeypatch.setattr(db_engine_module, "_run_migrations", fast_run_migrations)
    yield


# ── Canonical fixture ids and schema ─────────────────────────────────────────
# The standard three-field damage-classification project the service suites
# share: one aux free-text field plus two core fields (enum + boolean).

PID = "test-proj"
GID = "guid-001"
MCID = "mc-001"
EID = "ep-001"

FIXTURE_FIELDS = [
    {
        "field_id": "f0",
        "field_name": "rationale_note",
        "type": "string",
        "role": "aux",
        "display_order": -1,
    },
    {
        "field_id": "f1",
        "field_name": "severity",
        "type": "enum",
        "role": "core",
        "display_order": 1,
        "allowed_values": ["low", "medium", "high"],
    },
    {
        "field_id": "f2",
        "field_name": "damaged",
        "type": "boolean",
        "role": "core",
        "display_order": 2,
    },
]

FIXTURE_SCHEMA = {
    "fields": FIXTURE_FIELDS,
    "generation_order": ["rationale_note", "severity", "damaged"],
    "derived_json_schema": {},
    "schema_hash": "test",
}

# FIXTURE_SCHEMA with a real derived_json_schema, for suites that exercise
# validation. Kept separate from FIXTURE_SCHEMA: several suites depend on its
# empty derived_json_schema.
FIXTURE_SCHEMA_ENVELOPE = {
    "fields": FIXTURE_FIELDS,
    "generation_order": ["rationale_note", "severity", "damaged"],
    "derived_json_schema": {
        "type": "object",
        "properties": {
            "rationale_note": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "damaged": {"type": "boolean"},
        },
        "required": ["severity", "damaged"],
        "additionalProperties": False,
    },
    "schema_hash": "test_hash",
}

VALID_PROPOSAL_JSON = (
    '{"rationale_note": "visible dent", "severity": "high", "damaged": true}'
)
INVALID_PROPOSAL_JSON = (
    '{"rationale_note": "ok", "severity": "invalid_val", "damaged": true}'
)


# ── Canonical factories ──────────────────────────────────────────────────────


def make_settings(workspace: Path | str, **overrides: Any) -> Settings:
    """Workspace-rooted ``Settings``: non-secret defaults + overrides.

    The canonical replacement for the per-file ``_make_settings`` copies.
    ``overrides`` win over everything, including ``WORKSPACE_ROOT``.
    """
    vals: dict[str, Any] = {k: v for k, v in DEFAULTS.items() if k not in SECRET_KEYS}
    vals["WORKSPACE_ROOT"] = str(workspace)
    vals.update(overrides)
    return Settings(**vals)


def make_stub_settings(**overrides: Any) -> Settings:
    """``make_settings`` rooted at a throwaway ``/tmp/test`` workspace.

    For service calls that need a ``Settings`` value but never touch the
    workspace path (or override ``WORKSPACE_ROOT`` explicitly).
    """
    return make_settings("/tmp/test", **overrides)


def create_project_via_api(
    client: TestClient, name: str = "Test", **fields: Any
) -> dict:
    """TestClient-level project factory: ``POST /v1/projects`` → response JSON.

    Callers needing only the id take ``["project_id"]`` on the result.
    (Named ``_via_api`` to stay distinct from the backend's
    ``project_service.create_project``, which several tests import.)
    """
    resp = client.post("/v1/projects", json={"name": name, **fields})
    assert resp.status_code == 201, resp.text
    return resp.json()


def make_api_client(
    tmp_path: Path, settings: Settings | None = None, **settings_overrides: Any
) -> TestClient:
    """TestClient on the real app with a temp-workspace Settings override.

    Builds ``make_settings(tmp_path / "workspace", **settings_overrides)``
    unless an explicit ``settings`` is passed, installs it as the
    ``get_current_settings`` dependency override, and clears the
    project-engine cache. Both are reset after the test by the autouse
    ``_clear_project_engine_cache`` fixture.
    """
    from vlm_feedback_loop.main import app
    from vlm_feedback_loop.routers.projects import get_current_settings

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    if settings is None:
        settings = make_settings(workspace, **settings_overrides)

    app.dependency_overrides[get_current_settings] = lambda: settings
    clear_engine_cache()

    return TestClient(app, raise_server_exceptions=False)


def make_test_image(
    path: Path,
    fmt: str = "JPEG",
    width: int = 100,
    height: int = 100,
    color: tuple[int, int, int] = (128, 64, 32),
) -> Path:
    """Write a real Pillow-rendered image at ``path`` (parents created)."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=color).save(str(path), format=fmt)
    return path


def open_project_workspace(
    tmp_path: Path,
    project_id: str = "test-proj",
    *,
    register_engine: bool = False,
    subdirs: tuple[str, ...] = ("artifacts",),
) -> tuple[Engine, Path, Path]:
    """Service-level project workspace factory.

    Creates ``{tmp_path}/workspace/projects/{project_id}`` (plus
    ``subdirs``), opens + migrates the project DB, and optionally
    registers the engine in the project-engine cache. Returns
    ``(engine, project_dir, workspace)``.
    """
    workspace = tmp_path / "workspace"
    project_dir = workspace / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    for sub in subdirs:
        (project_dir / sub).mkdir(exist_ok=True)
    engine = open_project_db(project_dir)
    if register_engine:
        set_project_engine(project_id, engine)
    return engine, project_dir, workspace


def setup_project_db(
    tmp_path: Path,
    project_id: str = PID,
    *,
    subdirs: tuple[str, ...] = ("artifacts",),
) -> tuple[Engine, str]:
    """``open_project_workspace`` with engine registration, returned as the
    ``(engine, str(project_dir))`` pair the service suites consume."""
    engine, project_dir, _ = open_project_workspace(
        tmp_path, project_id, register_engine=True, subdirs=subdirs
    )
    return engine, str(project_dir)


def make_teacher_result(
    inference_invocation_id: str = "fake-invocation-id",
    content: str | None = VALID_PROPOSAL_JSON,
    **overrides: Any,
) -> TeacherInvocationResult:
    """Successful ``TeacherInvocationResult`` for the three-field fixture.

    The canonical replacement for the per-file ``_fake_teacher_result``
    copies; ``overrides`` flip individual fields for failure scenarios.
    """
    defaults: dict[str, Any] = {
        "inference_invocation_id": inference_invocation_id,
        "content": content,
        "finish_reason": "stop",
        "invocation_status": "success",
        "latency_ms": 150,
        "usage": {
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "total_tokens": 550,
        },
        "icl_example_keys_used": [],
        "prompt_hash": "abc123",
        "structured_generation_attempted": True,
        "structured_generation_fallback_used": False,
        "generation_preset_key": "precise",
        "sampling_params_effective": {"temperature": 0.0, "top_p": 1.0},
        "thinking_mode_effective": "on",
        "thinking_request_fields_effective": None,
        "max_tokens_effective": 256,
        "reasoning_headroom_tokens_effective": None,
        "visual_budget_preset_key": "balanced",
        "visual_budget_params_effective": {
            "mm_processor_kwargs": {"size": {"shortest_edge": 672}},
        },
        "seed_effective": None,
        "image_transport_mode": "base64_inline",
        "image_format_transmitted": "image/jpeg",
        "error": None,
    }
    defaults.update(overrides)
    return TeacherInvocationResult(**defaults)


def make_tao_settings(workspace: Path | str, **overrides: Any) -> Settings:
    """``make_settings`` plus the TAO endpoint trio the TAO suites share."""
    tao = {
        "TAO_API_BASE_URL": "http://tao.test/api/v2",
        "TAO_API_KEY": "jwt-test-token",
        "TAO_ORG_NAME": "example-org",
    }
    tao.update(overrides)
    return make_settings(workspace, **tao)


# ── Canonical DB-row factories ───────────────────────────────────────────────
# One source for the fixture rows the service suites seed. Values are the
# suite-wide canonical shapes the per-file copies had converged on; pass
# ``**overrides`` for scenario-specific fields (pointers, capabilities,
# states) rather than forking the dict again.

_PROJECT_ROW_DEFAULTS: dict[str, Any] = {
    "name": "Test",
    "labeling_generation_preset_key": "precise",
    "thinking_default_on": True,
    "visual_budget_preset_key": "balanced",
    "structured_generation_mode_default": "auto",
    "rationale_anti_anchoring": True,
    "auto_evaluate_enabled": False,
    "icl_recommendation_dismissed_at_count": 0,
    "export_field_mode": "all",
    "phash_algorithm": "dct_phash_64",
    "test_pool_fraction": 0.40,
    "scaleup_exact_match_threshold": 0.80,
    "scaleup_per_field_match_threshold": 0.80,
    "scaleup_min_per_value_f1_threshold": 0.80,
    "scaleup_accept_rate_threshold": 0.80,
    "scaleup_accept_rate_window": 50,
    "scaleup_min_test_pool_size": 20,
    "schema_refinement_reminders_dismissed": 0,
}


def add_project_row(
    session: Any, project_id: str, project_dir: str, **overrides: Any
) -> None:
    """Insert a Project row with the suite's canonical defaults."""
    row: dict[str, Any] = {
        "project_id": project_id,
        "project_dir": project_dir,
        **_PROJECT_ROW_DEFAULTS,
    }
    row.update(overrides)
    session.add(Project(**row))


def add_guidance_row(
    session: Any,
    project_id: str,
    guidance_id: str,
    schema: str,
    **overrides: Any,
) -> None:
    """Insert a Guidance row (version 1, empty rules) for ``schema``."""
    row: dict[str, Any] = {
        "guidance_id": guidance_id,
        "project_id": project_id,
        "version_number": 1,
        "description": "Test",
        "rules": "",
        "schema": schema,
    }
    row.update(overrides)
    session.add(Guidance(**row))


def add_endpoint_row(
    session: Any, project_id: str, endpoint_id: str, **overrides: Any
) -> None:
    """Insert a hosted NimEndpoint row."""
    row: dict[str, Any] = {
        "endpoint_id": endpoint_id,
        "project_id": project_id,
        "display_name": "Test",
        "base_url": "https://test.nvidia.com/v1",
        "endpoint_mode": "hosted",
        "api_format": "openai_compatible",
        "auth_mode": "bearer",
        "source_kind": "seeded_hosted",
    }
    row.update(overrides)
    session.add(NimEndpoint(**row))


def add_model_config_row(
    session: Any,
    project_id: str,
    model_config_id: str,
    endpoint_id: str,
    **overrides: Any,
) -> None:
    """Insert a teacher-eligible ModelConfig row with no optional capabilities."""
    row: dict[str, Any] = {
        "model_config_id": model_config_id,
        "project_id": project_id,
        "endpoint_id": endpoint_id,
        "model_name": "test-model",
        "context_window_tokens": 256000,
        "eligible_roles": json.dumps(["teacher"]),
        "supports_image_input": True,
        "structured_generation_support": "supported",
        "thinking_toggle_mode": "none",
        "thinking_toggle_support": "unsupported",
        "visual_budget_mode": "none",
        "visual_budget_support": "unsupported",
    }
    row.update(overrides)
    session.add(ModelConfig(**row))


def add_standard_project_row(
    session: Any, project_id: str, project_dir: str, **overrides: Any
) -> None:
    """``add_project_row`` wired to the canonical guidance/teacher ids.

    The three-field-fixture suites all point their project at guidance
    ``GID`` and teacher ``MCID``; overrides still win (e.g.
    ``active_guidance_id=None`` for the unconfigured-project cases).
    """
    add_project_row(
        session,
        project_id,
        project_dir,
        **{"active_guidance_id": GID, "teacher_model_config_id": MCID, **overrides},
    )


def add_fixture_guidance_row(
    session: Any, project_id: str = PID, guidance_id: str = GID, **overrides: Any
) -> None:
    """``add_guidance_row`` carrying the canonical three-field ``FIXTURE_SCHEMA``."""
    add_guidance_row(session, project_id, guidance_id, FIXTURE_SCHEMA, **overrides)


def add_endpoint_and_model_rows(
    session: Any,
    project_id: str = PID,
    *,
    endpoint_id: str = EID,
    model_config_id: str = MCID,
) -> None:
    """Hosted endpoint + teacher ModelConfig pair under the canonical ids."""
    add_endpoint_row(session, project_id, endpoint_id)
    add_model_config_row(session, project_id, model_config_id, endpoint_id)


def add_dataset_export_row(
    session: Any,
    project_id: str,
    dataset_export_id: str,
    **overrides: Any,
) -> None:
    """Insert a DatasetExport row (training intent, all fields, verified-only)."""
    row: dict[str, Any] = {
        "dataset_export_id": dataset_export_id,
        "project_id": project_id,
        "dataset_intent": "training",
        "export_field_mode": "all",
        "guidance_id": generate_uuid4(),
        "label_tier_filter": "verified_only",
        "selection_definition_snapshot": {},
        "artifact_refs": {"archive_path": f"/tmp/{dataset_export_id}.tar.gz"},
        "manifest_ref": f"/tmp/{dataset_export_id}.manifest.json",
        "example_count": 1,
    }
    row.update(overrides)
    session.add(DatasetExport(**row))


def add_tao_job_row(
    session: Any,
    project_id: str,
    tao_job_id: str,
    *,
    action: str = "train",
    **overrides: Any,
) -> None:
    """Insert a TAOJob row (cosmos-rl backend, not_started, minimal payload)."""
    row: dict[str, Any] = {
        "tao_job_id": tao_job_id,
        "project_id": project_id,
        "student_base_model_config_id": "mc-1",
        "dataset_export_ids": [],
        "action": action,
        "status": "not_started",
        "training_backend": "cosmos_rl_tao_vlm",
        "training_policy_type": "sft" if action == "train" else None,
        "job_config": {"training_preset": "standard"},
        "tao_create_job_request": {
            "kind": "experiment",
            "action": action,
            "specs": {},
        },
    }
    row.update(overrides)
    session.add(TAOJob(**row))


def seed_tao_chain_project(
    session: Any,
    project_id: str,
    project_dir: str,
    *,
    guidance_id: str | None = None,
    train_export_id: str = "de-train",
    eval_export_id: str | None = "de-eval",
    train_export_mode: str = "all",
    eval_export_mode: str = "all",
) -> str:
    """Seed the shared TAO-chain project scaffold and return the guidance_id.

    Project + hosted endpoint ``ep-1`` + student_base ModelConfig ``mc-1``
    (Cosmos Reason2 8B) + a training DatasetExport (and, unless
    ``eval_export_id=None``, an evaluation DatasetExport) — the fixture
    base every TAO-family suite builds chains on. ``*_export_mode``
    parameterizes ``export_field_mode`` per intent for the lineage-guard
    tests; the returned guidance_id links both exports.
    """
    add_project_row(session, project_id, project_dir)
    add_endpoint_row(session, project_id, "ep-1")
    add_model_config_row(
        session,
        project_id,
        "mc-1",
        "ep-1",
        model_name="nvidia/cosmos-reason2-8b",
        eligible_roles=json.dumps(["student_base"]),
    )
    gid = guidance_id or generate_uuid4()
    add_dataset_export_row(
        session,
        project_id,
        train_export_id,
        dataset_intent="training",
        export_field_mode=train_export_mode,
        guidance_id=gid,
    )
    if eval_export_id is not None:
        add_dataset_export_row(
            session,
            project_id,
            eval_export_id,
            dataset_intent="evaluation",
            export_field_mode=eval_export_mode,
            guidance_id=gid,
        )
    return gid


def seed_proposal_project(
    tmp_path: Path, num_examples: int = 1
) -> tuple[Engine, str, str, str, list[str], Settings]:
    """Seed the shared proposal-path project: fully-capable teacher, active
    guidance on ``FIXTURE_SCHEMA_ENVELOPE``, ``num_examples`` Unlabeled
    examples.

    Returns ``(engine, project_id, guidance_id, model_config_id,
    example_keys, settings)``.
    """
    project_id = "test-proj"
    guidance_id = generate_uuid4()
    model_config_id = generate_uuid4()
    endpoint_id = generate_uuid4()
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, project_id, register_engine=True
    )

    with Session(engine) as session:
        add_project_row(
            session,
            project_id,
            str(project_dir),
            active_guidance_id=guidance_id,
            teacher_model_config_id=model_config_id,
        )
        add_endpoint_row(session, project_id, endpoint_id, display_name="Test NIM")
        add_model_config_row(
            session,
            project_id,
            model_config_id,
            endpoint_id,
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["teacher"]),
            thinking_toggle_mode="qwen_enable_thinking",
            thinking_toggle_support="supported",
            visual_budget_mode="mm_processor_size",
            visual_budget_support="supported",
        )
        add_guidance_row(
            session,
            project_id,
            guidance_id,
            FIXTURE_SCHEMA_ENVELOPE,
            description="Classify damage severity.",
            rules="Focus on visible defects.",
        )
        example_keys = []
        for i in range(num_examples):
            key = f"img_{i:03d}"
            add_example_row(session, project_id, key)
            example_keys.append(key)
        session.commit()

    settings = make_settings(workspace, NVIDIA_API_KEY="nvapi-test-key")
    return engine, project_id, guidance_id, model_config_id, example_keys, settings


def add_example_row(
    session: Any, project_id: str, example_key: str, **overrides: Any
) -> None:
    """Insert an Example row (Unlabeled, fake storage ref, constant pHash)."""
    row: dict[str, Any] = {
        "example_key": example_key,
        "project_id": project_id,
        "storage_ref": f"/fake/{example_key}.jpg",
        "ingested_at": utc_now(),
        "source_metadata": {},
        "state": "Unlabeled",
        "phash": "a" * 16,
    }
    row.update(overrides)
    session.add(Example(**row))


def close_coro_register(task_id: str, coro: Any) -> None:
    """Test-side replacement for ``background_manager.register``.

    Callers create the background coroutine eagerly before registering it;
    a plain MagicMock would discard it and trip
    ``RuntimeWarning: coroutine ... was never awaited`` at garbage
    collection. Closing it disposes the coroutine cleanly without running
    it, leaving call-count assertions intact.
    """
    del task_id
    coro.close()


@contextmanager
def patched_register(service_module: ModuleType):
    """Patch ``service_module.background_manager.register`` for one block.

    The dispatched background coroutine is closed instead of run; the
    yielded mock supports assertions on registration calls.
    """
    with patch.object(
        service_module.background_manager,
        "register",
        side_effect=close_coro_register,
    ) as mock_register:
        yield mock_register
