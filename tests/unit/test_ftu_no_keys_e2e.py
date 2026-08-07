# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FTU no-keys end-to-end integration test.

Verifies the spec-mandated FTU contract: an SME who creates a project
with no NIM credentials lands in a degraded but correct state, with all
FTU plumbing in place to recover without a backend restart.

Drives the full FTU contract:

1. Create project (no NVIDIA_API_KEY / NGC_API_KEY).
2. Assert ``setup_completed_at`` is null on creation — the routing gate
   sends fresh projects through the setup screens.
3. Assert ``embedding_provider`` is "none" — the probe runs cleanly with
   no keys and writes the honest result rather than lying about
   embeddings; the ingestion-summary copy renders whatever this says.
4. Stamp setup via ``POST :mark_setup_completed`` (mirrors the auto-skip
   or manual Continue path in NIMConnectionPage).
5. Assert ``setup_completed_at`` is now non-null.
6. Assert a ``setup_completed`` AuditEvent was emitted with the
   recommendation context (auto_skip, modes, provider).
7. Second mark_setup_completed is idempotent — returns
   ``transitioned=false`` and no second AuditEvent.
8. Apply a runtime secret via ``POST /v1/secrets:set``.
9. Assert ``GET /v1/environment.nvidia_api_key_configured`` flips to
   true and ``allow_secret_persist`` is exposed — proves the runtime
   override is honored by the env assessment.
10. Persist the secret to disk via ``persist=true``; the disk file
    is created with user-only 0600 permissions.

This end-to-end regression guards against the defect classes the FTU
contract exists to prevent: silent setup-screen bypass, a hardcoded
"embeddings computing" lie, restart-required-to-add-keys, and a missing
audit trail for onboarding acknowledgment.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from starlette.testclient import TestClient

from support import build_test_settings
from vlm_feedback_loop.db.models.audit_event import AuditEvent
from vlm_feedback_loop.main import app
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.services import project_service, runtime_secrets

# In-process e2e over the real proposal/save/eval pipeline: fast standalone,
# but the bare 30s pytest-timeout ceiling is too tight under full-suite
# xdist with coverage instrumentation — and --timeout-method=thread
# hard-exits the whole worker on overrun, hanging the xdist controller.
# Restore the generous ceiling this file had from the integration conftest.
pytestmark = pytest.mark.timeout(120)


@pytest.fixture()
def no_keys_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """TestClient with isolated workspace + NO NIM credentials in
    Settings. Simulates the FTU cold-start environment."""
    import vlm_feedback_loop.config as cfg

    workspace = tmp_path / "workspace"
    # Construct Settings without any *_API_KEY values.
    settings = build_test_settings(workspace)
    # ALLOW_UI_SECRET_PERSIST defaults to True; explicit set for
    # documentation purposes (controls the secret-persist gate).
    settings.ALLOW_UI_SECRET_PERSIST = True

    # Direct the .env-persist target into the temp workspace so the
    # test never touches the developer's actual ~/.vlm_feedback_loop/.env.
    env_file = tmp_path / "test.env"
    monkeypatch.setenv("VLM_FEEDBACK_LOOP_ENV_FILE", str(env_file))
    # Any earlier test in this worker that ran load_settings() left the
    # process-global ``_active_env_file`` pointing at ITS env file, and
    # get_canonical_env_file_path() prefers that over the env var — sending
    # this test's persisted write to a stale tmp path. Clear it so the
    # env-var override applies (this test simulates a process that loaded
    # with VLM_FEEDBACK_LOOP_ENV_FILE set from the start).
    monkeypatch.setattr(cfg, "_active_env_file", None)
    monkeypatch.setattr(runtime_secrets, "_runtime_overrides", {})

    app.dependency_overrides[get_current_settings] = lambda: settings
    project_service.close_project_resources()

    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client, settings, env_file
    finally:
        app.dependency_overrides.clear()
        project_service.close_project_resources()
        # The persist step calls init_settings() against the temp .env,
        # replacing the process-wide Settings singleton with one rooted in
        # this test's (soon-deleted) tmp dir — clear it so later tests in
        # the worker never see the leaked "nvapi-persisted" credential.
        cfg.reset_settings()


def test_ftu_no_keys_end_to_end(no_keys_client):
    client, settings, env_file = no_keys_client

    # Step 1: create project with no keys.
    create_resp = client.post("/v1/projects", json={"name": "Cold Start"})
    assert create_resp.status_code == 201, create_resp.text
    project = create_resp.json()
    pid = project["project_id"]
    project_dir = Path(project["project_dir"])
    workspace_root = str(project_dir.parent.parent)

    # Step 2: setup_completed_at is null — the routing gate will send
    # the SME through the setup screens.
    assert project["setup_completed_at"] is None, (
        "Fresh project must have setup_completed_at=null so "
        "ProjectIndexRedirect routes to /setup"
    )

    # Step 3: embedding_provider is "none" — the project-creation probe
    # cleanly recognized "no key, no provider" rather than crashing or
    # lying. This is the premise the IngestionSummary truth-telling
    # relies on.
    assert project["embedding_provider"] == "none", (
        "embedding_provider must be 'none' when no key is configured — "
        "the IngestionSummary copy renders this value verbatim"
    )

    # Step 4: stamp setup_completed_at via the named verb endpoint.
    stamp_payload = {
        "auto_skip": False,
        "teacher_mode": "none",
        "embedding_mode": "none",
        "embedding_provider": "none",
    }
    stamp_resp = client.post(
        f"/v1/projects/{pid}:mark_setup_completed", json=stamp_payload
    )
    assert stamp_resp.status_code == 200, stamp_resp.text
    stamp_body = stamp_resp.json()

    # Step 5: setup_completed_at is now non-null.
    assert stamp_body["transitioned"] is True
    assert stamp_body["project"]["setup_completed_at"] is not None
    first_timestamp = stamp_body["project"]["setup_completed_at"]

    # Step 6: AuditEvent emitted with the recommendation context.
    engine = project_service.get_project_engine(pid, workspace_root)
    with Session(engine) as session:
        events = session.query(AuditEvent).filter_by(event_type="setup_completed").all()
        assert len(events) == 1
        assert events[0].event_data["auto_skip"] is False
        assert events[0].event_data["teacher_mode"] == "none"
        assert events[0].event_data["embedding_provider"] == "none"

    # Step 7: idempotent re-stamp.
    second_resp = client.post(
        f"/v1/projects/{pid}:mark_setup_completed", json=stamp_payload
    )
    assert second_resp.status_code == 200
    assert second_resp.json()["transitioned"] is False
    assert second_resp.json()["project"]["setup_completed_at"] == first_timestamp
    # No second AuditEvent.
    with Session(engine) as session:
        events = session.query(AuditEvent).filter_by(event_type="setup_completed").all()
        assert len(events) == 1, "second mark_setup_completed must not double-stamp"

    # Step 8: apply a runtime secret (session-only, persist=false).
    env_resp = client.get("/v1/environment")
    assert env_resp.status_code == 200
    assert env_resp.json()["nvidia_api_key_configured"] is False, (
        "no key configured pre-apply"
    )
    assert env_resp.json()["allow_secret_persist"] is True, (
        "env response exposes allow_secret_persist so the UI knows "
        "whether it may offer to persist pasted keys"
    )

    apply_resp = client.post(
        "/v1/secrets:set",
        json={
            "name": "NVIDIA_API_KEY",
            "value": "nvapi-runtime-test",
            "persist": False,
        },
    )
    assert apply_resp.status_code == 200, apply_resp.text
    assert apply_resp.json()["effective"] is True
    assert apply_resp.json()["persisted"] is False

    # Step 9: env response now reports the key as configured — proves
    # the runtime override is honored by the recommendation logic, no
    # backend restart required.
    env_after = client.get("/v1/environment").json()
    assert env_after["nvidia_api_key_configured"] is True, (
        "Runtime override must flip env.nvidia_api_key_configured to true "
        "without a backend restart"
    )

    # Step 10: persist to disk (.env) — file created with 0600 perms.
    persist_resp = client.post(
        "/v1/secrets:set",
        json={
            "name": "NVIDIA_API_KEY",
            "value": "nvapi-persisted",
            "persist": True,
        },
    )
    assert persist_resp.status_code == 200, persist_resp.text
    persist_body = persist_resp.json()
    assert persist_body["persisted"] is True
    assert persist_body["env_path"] == str(env_file)
    assert env_file.exists()
    # User-only file permissions.
    file_perms = stat.S_IMODE(os.stat(env_file).st_mode)
    assert file_perms == 0o600, f"expected 600, got {file_perms:o}"
    # .env content has the new line.
    content = env_file.read_text()
    assert "NVIDIA_API_KEY=nvapi-persisted" in content
