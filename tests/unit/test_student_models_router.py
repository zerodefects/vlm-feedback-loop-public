# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router tests for StudentModel list + detail.

Uses ``test_app_client`` with real SQLite DBs so seeding StudentModel
rows via the service produces records visible through the router.
"""

from __future__ import annotations

import io
import tarfile

from conftest import create_project_via_api


def _create_project(client):
    return create_project_via_api(
        client, name="student-router-test", description="Router test"
    )["project_id"]


def _insert_students(client, project_id: str, count: int = 3):
    """Insert StudentModel rows directly via the service layer.

    Each row gets a unique created_at / student_model_id so ordering
    tests are deterministic.
    """

    from sqlalchemy.orm import Session

    from vlm_feedback_loop.db.base import generate_uuid4
    from vlm_feedback_loop.db.models.student_model import StudentModel

    # The test_app_client fixture has already overridden get_current_settings;
    # call it to recover the settings object.
    from vlm_feedback_loop.main import app
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services.project_service import get_project_engine

    settings = app.dependency_overrides[get_current_settings]()
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None

    ids = []
    with Session(engine) as s:
        for i in range(count):
            sid = generate_uuid4()
            ids.append(sid)
            s.add(
                StudentModel(
                    student_model_id=sid,
                    project_id=project_id,
                    student_base_model_config_id="mc-base",
                    tao_job_id=f"tao-{i}",
                    guidance_id="g-x",
                    dataset_export_ids=[f"de-{i}"],
                    training_preset="standard",
                    lora_config={"enable_lora": True},
                    created_at=f"2026-04-{15 + i:02d}T00:00:00Z",
                    checkpoint_packaging_status="validated",
                    nim_checkpoint_ref=f"/tmp/ckpt-{i}",
                    quality_status="pending" if i == 0 else "validated",
                    quality_evaluation_run_id=None,
                    serving_status="not_attempted",
                    serving_evaluation_run_id=None,
                    quantization_method=None if i == 0 else "FP8_DYNAMIC",
                    quantize_tao_job_id=None if i == 0 else f"tao-q-{i}",
                )
            )
        s.commit()
    return ids


# ══════════════════════════════════════════════════════════════════════════
# GET /student_models
# ══════════════════════════════════════════════════════════════════════════


class TestList:
    def test_empty_project_returns_empty_list(self, test_app_client):
        pid = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{pid}/student_models")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"items": [], "next_cursor": None}

    def test_returns_full_records(self, test_app_client):
        pid = _create_project(test_app_client)
        _insert_students(test_app_client, pid, count=2)
        resp = test_app_client.get(f"/v1/projects/{pid}/student_models")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["items"]) == 2
        # Verify the full record shape on the first item
        first = body["items"][0]
        for field in [
            "student_model_id",
            "project_id",
            "training_suite_id",
            "student_base_model_config_id",
            "tao_job_id",
            "guidance_id",
            "dataset_export_ids",
            "training_preset",
            "lora_config",
            "created_at",
            "checkpoint_packaging_status",
            "nim_checkpoint_ref",
            "quality_status",
            "quality_evaluation_run_id",
            "serving_status",
            "serving_evaluation_run_id",
            "serving_benchmark_current",
            "serving_benchmark_blocker",
            "nim_preflight_status",
            "nim_preflight_details",
            "nim_preflight_at",
            "nim_deployment_mode",
            "nim_container_id",
            "nim_endpoint_url",
            "nim_vlm_release_version",
            "nim_model_profile_requested",
            "nim_model_profile_selected",
            "nim_profile_metadata",
            "gpu_type",
            "gpu_count",
            "quantization_method",
            "quantize_tao_job_id",
        ]:
            assert field in first, f"missing {field} on StudentModel response"

    def test_newest_first_ordering(self, test_app_client):
        pid = _create_project(test_app_client)
        _insert_students(test_app_client, pid, count=3)
        resp = test_app_client.get(f"/v1/projects/{pid}/student_models")
        items = resp.json()["items"]
        # Strictly descending by created_at
        dates = [it["created_at"] for it in items]
        assert dates == sorted(dates, reverse=True)

    def test_pagination_with_cursor(self, test_app_client):
        pid = _create_project(test_app_client)
        _insert_students(test_app_client, pid, count=5)
        page1 = test_app_client.get(f"/v1/projects/{pid}/student_models?limit=2").json()
        assert len(page1["items"]) == 2
        assert page1["next_cursor"] is not None

        page2 = test_app_client.get(
            f"/v1/projects/{pid}/student_models?limit=2&cursor={page1['next_cursor']}"
        ).json()
        assert len(page2["items"]) <= 2
        # Pages must not share any student_model_ids
        first_ids = {i["student_model_id"] for i in page1["items"]}
        second_ids = {i["student_model_id"] for i in page2["items"]}
        assert first_ids.isdisjoint(second_ids)

    def test_cross_project_isolation(self, test_app_client):
        pid_a = _create_project(test_app_client)
        resp_b = test_app_client.post(
            "/v1/projects",
            json={"name": "project-b", "description": None},
        )
        pid_b = resp_b.json()["project_id"]
        _insert_students(test_app_client, pid_a, count=2)
        resp = test_app_client.get(f"/v1/projects/{pid_b}/student_models")
        assert resp.status_code == 200
        assert resp.json()["items"] == []


# ══════════════════════════════════════════════════════════════════════════
# GET /student_models/{id}
# ══════════════════════════════════════════════════════════════════════════


class TestGet:
    def test_get_by_id_returns_record(self, test_app_client):
        pid = _create_project(test_app_client)
        ids = _insert_students(test_app_client, pid, count=1)
        resp = test_app_client.get(f"/v1/projects/{pid}/student_models/{ids[0]}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["student_model_id"] == ids[0]
        assert body["checkpoint_packaging_status"] == "validated"
        assert body["quality_status"] == "pending"
        assert body["serving_status"] == "not_attempted"
        assert body["serving_benchmark_current"] is False
        assert body["serving_benchmark_blocker"] == "serving_status_not_attempted"

    def test_get_unknown_id_returns_404(self, test_app_client):
        pid = _create_project(test_app_client)
        resp = test_app_client.get(f"/v1/projects/{pid}/student_models/nonexistent-id")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Student model not found"

    def test_cross_project_isolation(self, test_app_client):
        pid_a = _create_project(test_app_client)
        pid_b = test_app_client.post(
            "/v1/projects", json={"name": "b", "description": None}
        ).json()["project_id"]
        ids_a = _insert_students(test_app_client, pid_a, count=1)
        resp = test_app_client.get(f"/v1/projects/{pid_b}/student_models/{ids_a[0]}")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Student model not found"


class TestDeploymentBundle:
    def test_streams_attachment_without_buffering_through_json(
        self, test_app_client, tmp_path, monkeypatch
    ):
        """The public route returns a named tar attachment with the checkpoint."""
        from vlm_feedback_loop.services import deployment_bundle_service as bundles

        pid = _create_project(test_app_client)
        sid = "student-bundle-test"
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        model = checkpoint / "model.safetensors"
        model.write_bytes(b"weights")
        plan = bundles.DeploymentBundlePlan(
            project_id=pid,
            student_model_id=sid,
            archive_filename="vlm-student-student--nim.tar",
            archive_root="vlm-student-student--nim",
            checkpoint_root=checkpoint,
            checkpoint_files=(
                bundles.DeploymentBundleFile(
                    source=model,
                    relative_path=model.relative_to(checkpoint),
                    size=model.stat().st_size,
                    mode=0o644,
                ),
            ),
            handoff={
                "generated_at": "2026-08-03T00:00:00Z",
                "project_name": "bundle route",
                "technical_requirements": {
                    "nim_container_image": "nvcr.io/nim/test:1.0.0",
                    "nim_release_version": "1.0.0",
                    "nim_served_model_name": "student-test",
                    "nim_model_name_path": "/opt/checkpoints/student",
                    "nim_model_profile_recommended": None,
                    "nim_env_vars_recommended": {
                        "NGC_API_KEY": "$NGC_API_KEY",
                        "NIM_MODEL_NAME": "/opt/checkpoints/student",
                    },
                },
                "current_environment": {
                    "quality_status": "validated",
                    "serving_status": "validated",
                    "checkpoint_packaging_status": "validated",
                },
            },
            verification_request={
                "model": "student-test",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "__VLM_IMAGE_DATA_URL__"},
                            }
                        ],
                    }
                ],
            },
            verification_prompt_hash="prompt-sha",
            evaluated_prompt_hash="prompt-sha",
            guidance_id="guidance-test",
            guidance_schema_hash="schema-sha",
        )
        monkeypatch.setattr(bundles, "prepare_deployment_bundle", lambda **_: plan)

        response = test_app_client.get(
            f"/v1/projects/{pid}/student_models/{sid}/deployment_bundle"
        )

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/x-tar"
        assert response.headers["content-disposition"] == (
            'attachment; filename="vlm-student-student--nim.tar"'
        )
        assert response.headers["x-content-type-options"] == "nosniff"
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:") as archive:
            assert (
                "vlm-student-student--nim/checkpoint/model.safetensors"
                in archive.getnames()
            )

    def test_maps_bundle_gate_failure_to_conflict(self, test_app_client, monkeypatch):
        """The download route preserves production handoff gate failures."""
        from vlm_feedback_loop.services import deployment_bundle_service as bundles

        pid = _create_project(test_app_client)
        monkeypatch.setattr(
            bundles,
            "prepare_deployment_bundle",
            lambda **_: "conflict: serving_status_not_validated",
        )
        response = test_app_client.get(
            f"/v1/projects/{pid}/student_models/student/deployment_bundle"
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "conflict: serving_status_not_validated"


# ══════════════════════════════════════════════════════════════════════════
# :deploy_nim endpoint coverage lives in
# tests/unit/test_student_nim_deploy_router.py.
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# POST /student_models/{id}:repackage
# ══════════════════════════════════════════════════════════════════════════


class TestRepackage:
    def test_refuses_non_failed_packaging(self, test_app_client):
        """Repackage must never re-materialize a validated checkpoint
        underneath a live serving/eval consumer — 409 unless packaging is
        currently 'failed'."""
        pid = _create_project(test_app_client)
        ids = _insert_students(test_app_client, pid, count=1)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{ids[0]}:repackage"
        )
        assert resp.status_code == 409
        assert "not 'failed'" in resp.json()["detail"]

    def test_replays_packaging_for_failed_student(self, test_app_client, monkeypatch):
        """A packaging-failed Student (canonically: adapter-only LoRA
        checkpoint with the merge interpreter unprovisioned) is
        replayed in place through the idempotent registration path after
        the operator fixes the environment — no training-chain re-run."""
        from unittest.mock import AsyncMock

        from sqlalchemy.orm import Session

        from vlm_feedback_loop.main import app
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services import student_model_service as sms
        from vlm_feedback_loop.services.project_service import get_project_engine

        pid = _create_project(test_app_client)
        ids = _insert_students(test_app_client, pid, count=1)
        settings = app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        with Session(engine) as s:
            from vlm_feedback_loop.db.models.student_model import StudentModel

            row = s.get(StudentModel, ids[0])
            row.checkpoint_packaging_status = "failed"
            s.commit()

        replay = AsyncMock(return_value=ids[0])
        monkeypatch.setattr(sms, "register_from_tao_terminal", replay)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{ids[0]}:repackage"
        )
        assert resp.status_code == 200
        assert resp.json()["student_model_id"] == ids[0]
        replay.assert_awaited_once()
        # bf16 baseline: replay targets the TRAIN job (no quantize job).
        assert replay.await_args.args == (pid, "tao-0")

    def test_404_for_unknown_student(self, test_app_client):
        pid = _create_project(test_app_client)
        resp = test_app_client.post(f"/v1/projects/{pid}/student_models/nope:repackage")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Student model not found"

    def test_quantized_artifact_refresh_failure_is_actionable(
        self, test_app_client, monkeypatch
    ):
        """TAO/S3 refresh failure is a retryable upstream error, not 200."""
        from unittest.mock import AsyncMock

        from vlm_feedback_loop.services import student_model_service as sms

        pid = _create_project(test_app_client)
        ids = _insert_students(test_app_client, pid, count=1)
        monkeypatch.setattr(
            sms,
            "repackage_student_model",
            AsyncMock(
                return_value={
                    "error": "artifact_refresh_failed",
                    "student_model_id": ids[0],
                }
            ),
        )

        resp = test_app_client.post(
            f"/v1/projects/{pid}/student_models/{ids[0]}:repackage"
        )

        assert resp.status_code == 502
        assert "TAO/S3 reachability" in resp.json()["detail"]
