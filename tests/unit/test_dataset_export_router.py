# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset_exports router and schema-invalid manifest endpoint.

Router-level HTTP tests using TestClient, mocking the service layer.
"""

from __future__ import annotations

from unittest.mock import patch

from vlm_feedback_loop.db.base import utc_now

# Service-level patch targets
_SVC = "vlm_feedback_loop.services.dataset_export_service"

PID = "proj-001"
DEID = "de-001"
RID = "run-001"


# The create response reflects the background build: the full record exists
# in ``status="running"`` with null artifact refs until the build completes.
_SAMPLE_CREATE_RESPONSE = {
    "dataset_export_id": DEID,
    "project_id": PID,
    "dataset_intent": "training",
    "export_field_mode": "all",
    "label_tier_filter": "verified_only",
    "guidance_id": "g-1",
    "selection_definition_snapshot": {"dataset_intent": "training"},
    "example_count": 10,
    "status": "running",
    "status_reason": None,
    "progress": {"images_written": 0, "images_total": 10},
    "started_at": utc_now(),
    "completed_at": None,
    "artifact_refs": None,
    "manifest_ref": None,
    "created_at": utc_now(),
}

_SAMPLE_EXPORT_DETAIL = {
    "dataset_export_id": DEID,
    "project_id": PID,
    "dataset_intent": "training",
    "export_field_mode": "all",
    "guidance_id": "g-1",
    "label_tier_filter": "verified_only",
    "selection_definition_snapshot": {"dataset_intent": "training"},
    "example_count": 10,
    "status": "completed",
    "status_reason": None,
    "progress": {"images_written": 10, "images_total": 10},
    "started_at": utc_now(),
    "completed_at": utc_now(),
    "artifact_refs": {
        "archive_path": "/tmp/test.tar.gz",
        "checksum_sha256": "abc123" * 10 + "abcd",
    },
    "manifest_ref": "/tmp/manifest.json",
    "created_at": utc_now(),
}

_SAMPLE_MANIFEST = {
    "batch_label_run_id": RID,
    "schema_invalid_examples": [
        {
            "example_key": "ex_001",
            "validation_errors_core": ["severity: bad value"],
            "inference_invocation_id": "inv-001",
        }
    ],
    "total_count": 1,
}


# ══════════════════════════════════════════════════════════════════════════════
# POST /dataset_exports
# ══════════════════════════════════════════════════════════════════════════════


class TestPostCreate:
    def test_create_returns_201_running(self, test_app_client):
        """The create endpoint dispatches the background build and returns
        the running record — null artifact refs until the build completes."""
        with patch(f"{_SVC}.start_dataset_export") as mock_start:
            mock_start.return_value = _SAMPLE_CREATE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/dataset_exports",
                json={
                    "dataset_intent": "training",
                    "label_tier_filter": "verified_only",
                    "export_field_mode": "all",
                },
            )
        assert resp.status_code == 201
        body = resp.json()
        assert body["dataset_export_id"] == DEID
        assert body["example_count"] == 10
        assert body["status"] == "running"
        assert body["artifact_refs"] is None

    def test_create_no_guidance_returns_400(self, test_app_client):
        with patch(f"{_SVC}.start_dataset_export") as mock_start:
            mock_start.return_value = "No active Guidance configured"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/dataset_exports",
                json={"dataset_intent": "training"},
            )
        assert resp.status_code == 400
        assert "No active Guidance" in resp.json()["detail"]

    def test_create_forwards_batch_run_filter(self, test_app_client):
        """The route must not discard the optional Batch lineage filter."""
        with patch(f"{_SVC}.start_dataset_export") as mock_start:
            mock_start.return_value = _SAMPLE_CREATE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/dataset_exports",
                json={
                    "dataset_intent": "training",
                    "label_tier_filter": "auto_labeled_only",
                    "export_field_mode": "core_only",
                    "batch_label_run_id": RID,
                },
            )

        assert resp.status_code == 201
        assert mock_start.call_args.kwargs["batch_label_run_id"] == RID

    def test_create_rejects_empty_batch_run_filter(self, test_app_client):
        """An empty run ID cannot silently widen a supposedly scoped export."""
        with patch(f"{_SVC}.start_dataset_export") as mock_start:
            mock_start.return_value = _SAMPLE_CREATE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/dataset_exports",
                json={
                    "dataset_intent": "training",
                    "label_tier_filter": "auto_labeled_only",
                    "batch_label_run_id": "",
                },
            )

        assert resp.status_code == 422
        mock_start.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# GET /dataset_exports/{id}
# ══════════════════════════════════════════════════════════════════════════════


class TestGetExport:
    def test_get_returns_200(self, test_app_client):
        with patch(f"{_SVC}.get_dataset_export") as mock_get:
            mock_get.return_value = _SAMPLE_EXPORT_DETAIL
            resp = test_app_client.get(f"/v1/projects/{PID}/dataset_exports/{DEID}")
        assert resp.status_code == 200
        assert resp.json()["dataset_export_id"] == DEID

    def test_get_nonexistent_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_dataset_export") as mock_get:
            mock_get.return_value = "not found: Dataset export no-such"
            resp = test_app_client.get(f"/v1/projects/{PID}/dataset_exports/no-such")
        assert resp.status_code == 404
        assert "Dataset export no-such" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# GET /dataset_exports
# ══════════════════════════════════════════════════════════════════════════════


class TestListExports:
    def test_list_returns_200(self, test_app_client):
        with patch(f"{_SVC}.list_dataset_exports") as mock_list:
            mock_list.return_value = ([_SAMPLE_EXPORT_DETAIL], None)
            resp = test_app_client.get(f"/v1/projects/{PID}/dataset_exports")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 1
        assert body["next_cursor"] is None


# ══════════════════════════════════════════════════════════════════════════════
# GET /batch_label_runs/{run_id}/schema_invalid_manifest
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemaInvalidManifest:
    def test_manifest_returns_200(self, test_app_client):
        with patch(f"{_SVC}.get_schema_invalid_manifest") as mock_manifest:
            mock_manifest.return_value = _SAMPLE_MANIFEST
            resp = test_app_client.get(
                f"/v1/projects/{PID}/batch_label_runs/{RID}/schema_invalid_manifest",
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_count"] == 1
        assert body["schema_invalid_examples"][0]["example_key"] == "ex_001"

    def test_manifest_nonexistent_run_returns_404(self, test_app_client):
        with patch(f"{_SVC}.get_schema_invalid_manifest") as mock_manifest:
            mock_manifest.return_value = "not found: Batch label run no-such"
            resp = test_app_client.get(
                f"/v1/projects/{PID}/batch_label_runs/no-such/schema_invalid_manifest",
            )
        assert resp.status_code == 404
        assert "Batch label run no-such" in resp.json()["detail"]
