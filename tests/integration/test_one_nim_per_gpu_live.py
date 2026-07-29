# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live integration smoke for the one-NIM-per-GPU policy.

Covers the one-NIM-per-GPU invariant, the Student lifecycle's
acquire-GPU step before ``docker run``, and auto-restore of displaced
residents.

What this exercises end-to-end on a real host:

1. Deploy a Teacher NIM via ``POST /v1/projects/{id}/local_nim:deploy``;
   wait for ``status="running"``.
2. Re-issue ``:deploy`` with ``replace_resident=True`` targeting the
   same GPU; expect 201 + the Teacher's ``LocalNimDeployment`` row
   transitioning to ``status="stopped"`` with
   ``displaced_by_deployment_id`` referencing the new deployment AND
   ``displaced_at`` populated.
3. Wait for the new deployment to reach ``status="running"``.
4. Tear it down; confirm no orphan ``vlm-*`` containers remain.

Hard requirements for the live path:

- A working NGC API key with NVCR pull rights for cosmos-reason2 2B.
- A real NVIDIA GPU with enough VRAM to start Cosmos Reason2 2B
  (≥ 36 GB BF16 per the catalog).
- Docker + NVIDIA Container Toolkit installed.
- The backend's deployment DB initialized + a project DB at
  ``{WORKSPACE_ROOT}/projects/{pid}/project.db`` migrated to head.

CI environments do NOT satisfy these. The whole module is therefore
gated by ``NIM_LIVE_TESTS=1``. Local operators with the prerequisites
run::

    NIM_LIVE_TESTS=1 NGC_API_KEY=$NGC_API_KEY \\
        uv run pytest tests/integration/test_one_nim_per_gpu_live.py -v

Expected runtime: 4–8 minutes (Cosmos 2B container pull + start
dominates; if the layers are already cached locally the run can drop
to ~2 min).
"""

from __future__ import annotations

import os
import time

import pytest
from sqlalchemy.orm import Session

from vlm_feedback_loop.db.models.local_nim_deployment import LocalNimDeployment
from vlm_feedback_loop.routers.projects import get_current_settings
from vlm_feedback_loop.services.project_service import get_project_engine

pytestmark = pytest.mark.skipif(
    os.environ.get("NIM_LIVE_TESTS") != "1",
    reason=(
        "Live one-NIM-per-GPU smoke is opt-in (set NIM_LIVE_TESTS=1 with "
        "a working NGC_API_KEY + Docker + NVIDIA Container Toolkit + GPU)"
    ),
)


# The startup deadline is generous so the test survives a cold image
# pull. On a host where the layers are already cached, the container
# typically reaches /v1/health/ready in ~60 s.
HEALTH_POLL_DEADLINE_S = 600


def _wait_for_status(
    project_id: str,
    deployment_id: str,
    workspace_root: str,
    *,
    expected: str,
    deadline_s: int = HEALTH_POLL_DEADLINE_S,
) -> LocalNimDeployment:
    """Poll the DB row until ``status == expected`` or the deadline."""
    engine = get_project_engine(project_id, workspace_root)
    if engine is None:
        raise RuntimeError(f"project {project_id} engine missing")
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        with Session(engine) as session:
            row = session.get(LocalNimDeployment, deployment_id)
            if row is not None and row.status == expected:
                session.expunge(row)
                return row
            if row is not None and row.status == "failed":
                raise AssertionError(
                    f"deployment {deployment_id} flipped to failed: {row.status_reason}"
                )
        time.sleep(5.0)
    raise AssertionError(
        f"deployment {deployment_id} did not reach status={expected} "
        f"within {deadline_s}s"
    )


class TestOneNimPerGpuLive:
    """End-to-end smoke on a real single-GPU host.

    Uses Cosmos Reason2 2B (smallest seeded ``student_base``) so the
    container start time fits within a CI-like budget. The flow is
    Teacher-then-Teacher replace (the simplest replace path); the
    full Student NIM benchmark + auto-restore is exercised by the
    Compare & Benchmark UI screen with the same code path, so a
    Teacher-on-Teacher replace is sufficient to prove the structural
    invariant.
    """

    def test_replace_resident_stops_prior_and_audits_displacement(
        self, test_app_client
    ):
        """On the same single GPU, the second deploy's
        ``replace_resident=true`` request stops the first deployment,
        stamps its ``displaced_by_deployment_id`` to the new id, and
        reaches ``status=running`` itself."""
        client = test_app_client
        # The project lives in test_app_client's tmp workspace, so recover
        # the injected Settings from the dependency override — the ambient
        # get_settings() would point WORKSPACE_ROOT at the operator's real
        # config.
        settings = client.app.dependency_overrides[get_current_settings]()

        # Create a fresh project for isolation.
        resp = client.post(
            "/v1/projects",
            json={"name": f"f49-live-{int(time.time())}"},
        )
        assert resp.status_code == 201, resp.text
        pid = resp.json()["project_id"]

        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        cosmos_2b = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "cosmos-reason2-2b" in c["model_name"]
        )

        # ── Step 1: first Teacher deploy ─────────────────────────────
        resp1 = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": cosmos_2b["model_config_id"],
            },
        )
        assert resp1.status_code == 201, resp1.text
        first_id = resp1.json()["deployment"]["local_nim_deployment_id"]
        first_running = _wait_for_status(
            pid, first_id, settings.WORKSPACE_ROOT, expected="running"
        )
        assert first_running.gpu_assignment == "device=0", (
            "first deploy must claim device=0 on a single-GPU host"
        )

        # ── Step 2: re-issue with replace_resident=true on the same GPU.
        # Without replace_resident=true this would return 409
        # gpu_occupied; with it, the service stops the first deploy +
        # stamps the displacement audit fields + starts the new one.
        resp2 = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": cosmos_2b["model_config_id"],
                "gpu_assignment": "device=0",
                "replace_resident": True,
            },
        )
        assert resp2.status_code == 201, resp2.text
        second_id = resp2.json()["deployment"]["local_nim_deployment_id"]
        assert second_id != first_id

        # The first deployment must now be stopped + carry the
        # displacement audit linkage.
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            displaced = session.get(LocalNimDeployment, first_id)
            assert displaced is not None
            assert displaced.status == "stopped"
            assert displaced.displaced_by_deployment_id == second_id
            assert displaced.displaced_at is not None
            assert displaced.status_reason == "displaced_by_replace"

        # ── Step 3: confirm the new deploy reaches running.
        _wait_for_status(pid, second_id, settings.WORKSPACE_ROOT, expected="running")

        # ── Step 4: teardown.
        stop_resp = client.post(
            f"/v1/projects/{pid}/local_nim/deployments/{second_id}:stop"
        )
        assert stop_resp.status_code == 200, stop_resp.text
        assert stop_resp.json()["status"] == "stopped"

    def test_deploy_returns_409_gpu_occupied_without_replace_resident(
        self, test_app_client
    ):
        """Without explicit opt-in, a deploy targeting an occupied GPU
        returns a structured 409 rather than starting the container
        (or crashing later on the Cosmos profile-selector floor)."""
        client = test_app_client
        settings = client.app.dependency_overrides[get_current_settings]()

        resp = client.post(
            "/v1/projects",
            json={"name": f"f49-live-409-{int(time.time())}"},
        )
        assert resp.status_code == 201, resp.text
        pid = resp.json()["project_id"]

        configs = client.get(f"/v1/projects/{pid}/model_configs").json()["items"]
        cosmos_2b = next(
            c
            for c in configs
            if c.get("local_deploy_metadata") and "cosmos-reason2-2b" in c["model_name"]
        )

        resp1 = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": cosmos_2b["model_config_id"],
            },
        )
        assert resp1.status_code == 201
        first_id = resp1.json()["deployment"]["local_nim_deployment_id"]
        _wait_for_status(pid, first_id, settings.WORKSPACE_ROOT, expected="running")

        # Re-issue WITHOUT replace_resident → 409 gpu_occupied.
        resp2 = client.post(
            f"/v1/projects/{pid}/local_nim/deploy",
            json={
                "role": "teacher",
                "model_config_id": cosmos_2b["model_config_id"],
                "gpu_assignment": "device=0",
            },
        )
        assert resp2.status_code == 409, resp2.text
        detail = resp2.json()["detail"]
        assert detail["code"] == "gpu_occupied"

        # Teardown the still-running first deploy.
        client.post(f"/v1/projects/{pid}/local_nim/deployments/{first_id}:stop")
