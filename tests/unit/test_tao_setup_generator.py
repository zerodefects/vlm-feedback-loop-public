# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tao_setup Action Request generator (services/tao_setup_generator.py)."""

from __future__ import annotations

from starlette.testclient import TestClient

from conftest import create_project_via_api


def _create_project(client: TestClient) -> str:
    return create_project_via_api(client, name="TAO Test Project")["project_id"]


class TestTaoSetupGenerator:
    def test_generates_with_config_fields(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["request_type"] == "tao_setup"
        assert data["project_name"] == "TAO Test Project"

        text = data["rendered_text"]
        assert "TAO_API_BASE_URL" in text
        assert "TAO_API_KEY" in text
        assert "TAO_ORG_NAME" in text

    def test_contains_connection_test_endpoint(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        text = resp.json()["rendered_text"]
        assert "/orgs/" in text
        assert "/jobs" in text

    def test_contains_student_base_models(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        text = resp.json()["rendered_text"]
        assert "nvidia/cosmos-reason2-8b" in text
        assert "nvidia/cosmos-reason2-2b" in text

    def test_contains_gpu_requirements(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        text = resp.json()["rendered_text"]
        assert "A100" in text or "GPU" in text

    def test_contains_auth_instructions(self, test_app_client):
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        text = resp.json()["rendered_text"]
        # Should mention the two-step auth flow
        assert "login" in text.lower() or "JWT" in text


# ── Workspace + bootstrap CLI references ───────────────────────────


class TestTaoSetupGeneratorWorkspaceReferences:
    """Workspace + bootstrap CLI requirements."""

    def test_rendered_text_includes_workspace_and_bootstrap_section(
        self, test_app_client
    ):
        """Rendered text references the workspace + bootstrap CLI and
        matches the config contract: workspace identity lives in
        deployment.db, so only the API values and the two S3 credential
        secrets may be advertised as .env configuration."""
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        text = resp.json()["rendered_text"]
        # Workspace + base-experiment references
        assert "TAO workspace" in text
        assert "base experiment" in text.lower() or "base_experiment" in text
        # The rendered text must point at the bootstrap CLI.
        assert "vlm-feedback-loop tao-bootstrap" in text
        # Workspace identity is deployment.db-persisted, never .env keys.
        assert "deployment.db" in text
        assert "TAO_WORKSPACE_ID" not in text
        assert "TAO_WORKSPACE_S3_BUCKET" not in text
        # The S3 credential secrets DO live in .env.
        assert "TAO_WORKSPACE_S3_ACCESS_KEY" in text
        assert "TAO_WORKSPACE_S3_SECRET_KEY" in text
        for retired_flag in (
            "--s3-access-key",
            "--s3-secret-key",
            "--ngc-key",
            "--hf-token",
        ):
            assert retired_flag not in text
        assert "does not persist them" in text
        assert "persisted by `tao-bootstrap`" not in text
        assert "persists them to .env" not in text

    def test_technical_requirements_include_tao_workspace_block(self, test_app_client):
        """technical_requirements carries the tao_workspace block."""
        project_id = _create_project(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{project_id}/action_requests:generate",
            json={"request_type": "tao_setup"},
        )
        tech = resp.json()["technical_requirements"]
        assert "tao_workspace" in tech
        ws_block = tech["tao_workspace"]
        assert "description" in ws_block
        assert ws_block["bootstrap_cli"] == "vlm-feedback-loop tao-bootstrap"
        assert "required_base_experiments" in ws_block
        # Must list both Cosmos Reason2 student bases.
        assert any(
            "nvidia/cosmos-reason2-8b" in m
            for m in ws_block["required_base_experiments"]
        )
        assert any(
            "nvidia/cosmos-reason2-2b" in m
            for m in ws_block["required_base_experiments"]
        )
        # required_config_fields lists exactly the .env-resident values:
        # the three TAO_API_* operator inputs plus the two S3 credential
        # secrets. Workspace identity fields are deployment.db-persisted
        # and must not be advertised as env configuration.
        required = tech["required_config_fields"]
        assert "TAO_API_BASE_URL" in required
        assert "TAO_API_KEY" in required
        assert "TAO_ORG_NAME" in required
        assert "TAO_WORKSPACE_S3_ACCESS_KEY" in required
        assert "TAO_WORKSPACE_S3_SECRET_KEY" in required
        assert "TAO_WORKSPACE_ID" not in required
        assert "TAO_WORKSPACE_S3_BUCKET" not in required
