# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Action Request endpoints."""

from __future__ import annotations

import re

from sqlalchemy.orm import Session

from conftest import create_project_via_api

ISO8601_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _create_project(client, name="AR Test"):
    return create_project_via_api(client, name)


class TestGenerateFullShape:
    """Generate returns the full response shape."""

    def test_returns_all_fields(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["request_type"] == "nim_setup"
        assert ISO8601_Z_RE.match(data["generated_at"])
        assert data["project_name"] == "AR Test"
        assert isinstance(data["technical_requirements"], dict)
        assert isinstance(data["current_environment"], dict)
        assert isinstance(data["rendered_text"], str)
        assert len(data["rendered_text"]) > 0


class TestRenderedTextNoSecrets:
    """rendered_text does not contain secrets."""

    def test_no_secrets_in_rendered_text(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        data = resp.json()
        text = data["rendered_text"]

        # Should not contain secret-like patterns
        assert "nvapi-" not in text
        assert "Bearer " not in text


class TestLogCopyCreatesAuditEvent:
    """log_copy creates an AuditEvent."""

    def test_creates_audit_event(self, test_app_client):
        from pathlib import Path

        from vlm_feedback_loop.db.models.audit_event import AuditEvent
        from vlm_feedback_loop.services.project_service import get_project_engine

        proj = _create_project(test_app_client)
        pid = proj["project_id"]
        ws = str(Path(proj["project_dir"]).parent.parent)

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:log_copy",
            json={
                "request_type": "nim_setup",
                "rendered_text": "Test rendered text for clipboard",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "audit_event_id" in data

        # Verify AuditEvent in DB
        engine = get_project_engine(pid, ws)
        with Session(engine) as session:
            event = (
                session.query(AuditEvent)
                .filter_by(audit_event_id=data["audit_event_id"])
                .first()
            )
            assert event is not None
            assert event.event_type == "action_request_copied"
            assert (
                event.event_data["rendered_text"] == "Test rendered text for clipboard"
            )
            assert event.event_data["request_type"] == "nim_setup"


class TestUnknownRequestType:
    """Unknown request_type returns an error."""

    def test_unknown_type_returns_400(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={"request_type": "totally_bogus_type"},
        )
        assert resp.status_code == 400
        assert "totally_bogus_type" in resp.json()["detail"]


class TestContextOptional:
    """Context object is optional with nullable fields."""

    def test_without_context(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={"request_type": "nim_setup"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_type"] == "nim_setup"
        assert data["rendered_text"].strip()

    def test_with_null_context(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={"request_type": "nim_setup", "context": None},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_type"] == "nim_setup"
        assert data["rendered_text"].strip()

    def test_with_partial_context(self, test_app_client):
        proj = _create_project(test_app_client)
        pid = proj["project_id"]

        resp = test_app_client.post(
            f"/v1/projects/{pid}/action_requests:generate",
            json={
                "request_type": "tao_setup",
                "context": {
                    "student_model_id": "some-id",
                    "example_keys": None,
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_type"] == "tao_setup"
        assert data["rendered_text"].strip()
