# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP-level tests for the labels router.

Service behavior is covered by ``test_label_service``; this file exercises
the router's error-mapping contract — the conversion of service-returned
error strings to HTTP status codes (404 / 409 / 422 / 400). Without these
tests, the router's branch logic stays dark even though the service layer
has comprehensive coverage.

Mocks the service layer at the import boundary; asserts the response code,
detail, and (for success cases) the body shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

# Service-level patch targets
_LABEL_SVC = "vlm_feedback_loop.services.label_service"
_RATIONALE_SVC = "vlm_feedback_loop.services.rationale_service"

PID = "proj-router-test"
EK = "ex_000"
INV_ID = "inv-router-001"


# ══════════════════════════════════════════════════════════════════════════════
# POST /labels — Save
# ══════════════════════════════════════════════════════════════════════════════


class TestSaveLabelEndpoint:
    """Map service results to 200 / 404 / 409 / 422 per the spec."""

    _SAMPLE_REQUEST = {
        "example_key": EK,
        "inference_invocation_id": INV_ID,
        "label_json": {
            "rationale_note": "ok",
            "category": "a",
        },
        "rationale_source": "teacher_proposal",
    }

    _SAMPLE_RESPONSE = {
        "example_key": EK,
        "verified_outcome": "Accept",
        "verified_at": "2026-04-27T12:00:00Z",
        "edited_core_fields": [],
        "edited_aux_fields": [],
        "label_status": "verified",
        "pool_assignment": None,
    }

    def test_save_returns_200_on_success(self, test_app_client):
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = self._SAMPLE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels", json=self._SAMPLE_REQUEST
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["example_key"] == EK
        assert body["verified_outcome"] == "Accept"
        assert body["label_status"] == "verified"

    def test_save_allows_rationale_metadata_to_be_omitted(self, test_app_client):
        """The API accepts the default rationale-disabled labeling request."""
        request = {
            "example_key": EK,
            "inference_invocation_id": INV_ID,
            "label_json": {"category": "a"},
        }
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = self._SAMPLE_RESPONSE
            resp = test_app_client.post(f"/v1/projects/{PID}/labels", json=request)

        assert resp.status_code == 200
        assert mock.call_args.kwargs["rationale_source"] is None
        assert mock.call_args.kwargs["rationale_regeneration_invocation_id"] is None

    def test_save_joins_project_write_queue(self, test_app_client):
        """Interactive label saves serialize ahead of later background writes."""
        from vlm_feedback_loop.services.project_db_locks import (
            get_project_write_lock,
        )

        lock_seen = False

        def result_while_locked(*_args, **_kwargs):
            nonlocal lock_seen
            lock_seen = get_project_write_lock(PID).locked()
            return self._SAMPLE_RESPONSE

        with patch(
            f"{_LABEL_SVC}.save_label",
            side_effect=result_while_locked,
        ):
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels",
                json=self._SAMPLE_REQUEST,
            )

        assert resp.status_code == 200
        assert lock_seen

    def test_save_invocation_not_found_returns_404(self, test_app_client):
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = f"Invocation not found: {INV_ID}"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels", json=self._SAMPLE_REQUEST
            )
        assert resp.status_code == 404
        assert "Invocation not found" in resp.json()["detail"]

    def test_save_project_not_found_returns_404(self, test_app_client):
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = f"Project not found: {PID}"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels", json=self._SAMPLE_REQUEST
            )
        assert resp.status_code == 404
        assert "Project not found" in resp.json()["detail"]

    def test_save_stale_proposal_returns_409(self, test_app_client):
        # Service emits "Stale proposal conflict: ... has been superseded by ..."
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = (
                "Stale proposal conflict: invocation X has been superseded by Y"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels", json=self._SAMPLE_REQUEST
            )
        assert resp.status_code == 409
        assert "superseded" in resp.json()["detail"]

    def test_save_invalid_rationale_returns_400(self, test_app_client):
        # Anything not matching "not found"/"conflict"/"superseded" is mapped
        # to ``validation_failed`` (HTTP 400 — see ``services/errors.py``).
        # Distinct from FastAPI's automatic 422 for Pydantic schema validation.
        with patch(f"{_LABEL_SVC}.save_label") as mock:
            mock.return_value = (
                "Edited label requires a reviewed rationale; got teacher_proposal"
            )
            resp = test_app_client.post(
                f"/v1/projects/{PID}/labels", json=self._SAMPLE_REQUEST
            )
        assert resp.status_code == 400
        assert "reviewed rationale" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# POST /examples/{example_key}:skip
# ══════════════════════════════════════════════════════════════════════════════


class TestSkipEndpoint:
    _SAMPLE_RESPONSE = {
        "example_key": EK,
        "state": "Omitted",
        "omitted_at": "2026-04-27T12:00:00Z",
    }

    def test_skip_returns_200_on_success(self, test_app_client):
        with patch(f"{_LABEL_SVC}.skip_example") as mock:
            mock.return_value = self._SAMPLE_RESPONSE
            resp = test_app_client.post(f"/v1/projects/{PID}/examples/{EK}:skip")
        assert resp.status_code == 200
        assert resp.json()["state"] == "Omitted"

    def test_skip_example_not_found_returns_404(self, test_app_client):
        with patch(f"{_LABEL_SVC}.skip_example") as mock:
            mock.return_value = f"Example not found: {EK}"
            resp = test_app_client.post(f"/v1/projects/{PID}/examples/{EK}:skip")
        assert resp.status_code == 404
        assert "Example not found" in resp.json()["detail"]

    def test_skip_already_verified_returns_400(self, test_app_client):
        with patch(f"{_LABEL_SVC}.skip_example") as mock:
            mock.return_value = "Cannot skip a Verified example"
            resp = test_app_client.post(f"/v1/projects/{PID}/examples/{EK}:skip")
        assert resp.status_code == 400
        assert "Cannot skip a Verified" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# POST /examples:restore_omitted
# ══════════════════════════════════════════════════════════════════════════════


class TestRestoreOmittedEndpoint:
    def test_restore_returns_200_with_count(self, test_app_client):
        with patch(f"{_LABEL_SVC}.restore_omitted") as mock:
            mock.return_value = {"restored_count": 4}
            resp = test_app_client.post(f"/v1/projects/{PID}/examples:restore_omitted")
        assert resp.status_code == 200
        assert resp.json()["restored_count"] == 4

    def test_restore_project_not_found_returns_404(self, test_app_client):
        with patch(f"{_LABEL_SVC}.restore_omitted") as mock:
            mock.return_value = f"Project not found: {PID}"
            resp = test_app_client.post(f"/v1/projects/{PID}/examples:restore_omitted")
        assert resp.status_code == 404
        assert "Project not found" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════════
# POST /examples/{example_key}:regenerate_rationale
# ══════════════════════════════════════════════════════════════════════════════


class TestRegenerateRationaleEndpoint:
    _SAMPLE_REQUEST = {
        "teacher_model_config_id": "mc-1",
    }

    _SAMPLE_RESPONSE = {
        "inference_invocation_id": "rrg-1",
        "rationale_note": "regenerated rationale text",
        "invocation_status": "success",
    }

    def test_regenerate_returns_200_on_success(self, test_app_client):
        with patch(
            f"{_RATIONALE_SVC}.regenerate_rationale",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = self._SAMPLE_RESPONSE
            resp = test_app_client.post(
                f"/v1/projects/{PID}/examples/{EK}:regenerate_rationale",
                json=self._SAMPLE_REQUEST,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["rationale_note"] == "regenerated rationale text"
        assert body["inference_invocation_id"] == "rrg-1"
        assert body["invocation_status"] == "success"

    def test_regenerate_example_not_found_returns_404(self, test_app_client):
        with patch(
            f"{_RATIONALE_SVC}.regenerate_rationale",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = f"Example not found: {EK}"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/examples/{EK}:regenerate_rationale",
                json=self._SAMPLE_REQUEST,
            )
        assert resp.status_code == 404
        assert "Example not found" in resp.json()["detail"]

    def test_regenerate_teacher_unreachable_returns_400(self, test_app_client):
        with patch(
            f"{_RATIONALE_SVC}.regenerate_rationale",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = "Teacher endpoint unreachable: connection refused"
            resp = test_app_client.post(
                f"/v1/projects/{PID}/examples/{EK}:regenerate_rationale",
                json=self._SAMPLE_REQUEST,
            )
        assert resp.status_code == 400
        assert "Teacher endpoint unreachable" in resp.json()["detail"]
