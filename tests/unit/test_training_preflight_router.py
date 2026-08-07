# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the training preflight router and service.

Covers the preflight checks themselves plus the Profile D preflight's
differentiation from the NIM deployment preflight.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from vlm_feedback_loop.db.base import generate_uuid4
from vlm_feedback_loop.model_catalog_constants import COSMOS3_SUPER_REASONER

_SVC = "vlm_feedback_loop.services.training_preflight_service"


PID = "proj-preflight-test"


@pytest.fixture(autouse=True)
def _merge_runtime_ready(monkeypatch):
    """Keep non-runtime checks independent of packages on the test host."""
    from vlm_feedback_loop.services import student_model_service

    monkeypatch.setattr(
        student_model_service,
        "check_lora_merge_readiness",
        AsyncMock(return_value=(True, "LoRA merge runtime ready.")),
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _seed_project_with_catalog(client):
    """Create a project via the API so the catalog seeds automatically."""
    resp = client.post(
        "/v1/projects",
        json={"name": "preflight-test", "description": "Training preflight test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["project_id"]


def _patch_probe_success():
    return patch(
        f"{_SVC}.probe_tao_connection",
        new=AsyncMock(
            return_value={
                "success": True,
                "error": None,
                "status_code": 200,
                "job_timeout_supported": True,
                "job_timeout_error": None,
            }
        ),
    )


def _patch_probe_failure(error: str = "Connection to TAO timed out"):
    return patch(
        f"{_SVC}.probe_tao_connection",
        new=AsyncMock(
            return_value={"success": False, "error": error, "status_code": None}
        ),
    )


def _patch_workspace_check_pass():
    """Return a context manager that makes the ``tao_workspace_reachable`` check pass."""
    return patch(
        f"{_SVC}._check_tao_workspace_reachable",
        new=AsyncMock(
            return_value={
                "check_name": "tao_workspace_reachable",
                "passed": True,
                "message": "TAO workspace reachable.",
                "model_config_id": None,
            }
        ),
    )


def _patch_base_experiment_pass_for(model_config_ids: list[str]):
    """Make ``tao_base_experiment_ready`` pass for every listed model id."""
    return patch(
        f"{_SVC}._check_tao_base_experiment_ready",
        new=lambda *, project_id, student_base_model_config_ids, workspace_root: [
            {
                "check_name": "tao_base_experiment_ready",
                "passed": mcid in model_config_ids,
                "message": (
                    f"Base experiment ready for {mcid!r}."
                    if mcid in model_config_ids
                    else f"Base experiment not ready for {mcid!r}."
                ),
                "model_config_id": mcid,
            }
            for mcid in student_base_model_config_ids
        ],
    )


def _patch_train_examples_pass():
    """Make both dataset checks pass for tests focused on other checks."""
    return patch.multiple(
        _SVC,
        _check_verified_train_examples=lambda *, project_id, workspace_root: {
            "check_name": "verified_train_examples",
            "passed": True,
            "message": "1 Verified training example available (Test Pool excluded).",
            "model_config_id": None,
        },
        _check_min_test_pool_size=lambda data_summary: {
            "check_name": "min_test_pool_size",
            "passed": True,
            "message": "Test Pool has 60 held-out evaluation examples (need 60).",
            "model_config_id": None,
        },
    )


def _patch_test_pool_pass():
    """Keep Verified-training selection tests focused on that one rule."""
    return patch(
        f"{_SVC}._check_min_test_pool_size",
        new=lambda data_summary: {
            "check_name": "min_test_pool_size",
            "passed": True,
            "message": "Test Pool has 60 held-out evaluation examples (need 60).",
            "model_config_id": None,
        },
    )


def _set_hf_token(client, value: str | None) -> None:
    """Set the deployment-scoped HF token on the injected test settings."""
    from vlm_feedback_loop.routers.projects import get_current_settings

    settings = client.app.dependency_overrides[get_current_settings]()
    settings.HF_TOKEN = value


def _seed_active_guidance(client, project_id: str) -> str:
    """Create + activate a minimal guidance; returns its guidance_id."""
    resp = client.post(
        f"/v1/projects/{project_id}/guidance",
        json={
            "description": "Classify the subject.",
            "rules": "",
            "schema": [
                {
                    "field_name": "rationale_note",
                    "type": "string",
                    "role": "aux",
                    "display_order": 0,
                },
                {
                    "field_name": "category",
                    "type": "enum",
                    "role": "core",
                    "allowed_values": ["a", "b"],
                    "display_order": 1,
                },
            ],
        },
    )
    assert resp.status_code in (200, 201), resp.text
    gid = resp.json()["guidance_id"]
    patch_resp = client.patch(
        f"/v1/projects/{project_id}", json={"active_guidance_id": gid}
    )
    assert patch_resp.status_code == 200, patch_resp.text
    return gid


def _insert_verified_label(
    client, project_id: str, guidance_id: str, *, pool_assignment: str | None
) -> None:
    """Insert one verified Label row directly (no public API writes labels)."""
    from sqlalchemy.orm import Session

    from vlm_feedback_loop.db.models.label import Label
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services.project_service import get_project_engine

    settings = client.app.dependency_overrides[get_current_settings]()
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None
    with Session(engine) as session:
        session.add(
            Label(
                label_id=generate_uuid4(),
                project_id=project_id,
                example_key=f"ex-{generate_uuid4()[:8]}",
                label_status="verified",
                guidance_id=guidance_id,
                inference_invocation_id=generate_uuid4(),
                label_json={"category": "a", "rationale_note": "seed"},
                labeled_at="2026-07-07T00:00:00Z",
                verified_outcome="Accept",
                verified_at="2026-07-07T00:00:00Z",
                pool_assignment=pool_assignment,
            )
        )
        session.commit()


def _insert_auto_labeled_example(client, project_id: str, guidance_id: str) -> None:
    """Insert one export-eligible Auto-Labeled example and label."""
    from sqlalchemy.orm import Session

    from vlm_feedback_loop.db.models.example import Example
    from vlm_feedback_loop.db.models.label import Label
    from vlm_feedback_loop.routers.projects import get_current_settings
    from vlm_feedback_loop.services.project_service import get_project_engine

    settings = client.app.dependency_overrides[get_current_settings]()
    engine = get_project_engine(project_id, settings.WORKSPACE_ROOT)
    assert engine is not None
    example_key = f"auto-{generate_uuid4()[:8]}"
    with Session(engine) as session:
        session.add(
            Example(
                example_key=example_key,
                project_id=project_id,
                storage_ref=f"/tmp/{example_key}.png",
                ingested_at="2026-07-29T00:00:00Z",
                source_metadata={},
                state="Auto-Labeled",
            )
        )
        session.add(
            Label(
                label_id=generate_uuid4(),
                project_id=project_id,
                example_key=example_key,
                label_status="auto_labeled",
                guidance_id=guidance_id,
                inference_invocation_id=generate_uuid4(),
                label_json={"category": "a", "rationale_note": "auto"},
                labeled_at="2026-07-29T00:00:00Z",
                batch_label_run_id=generate_uuid4(),
            )
        )
        session.commit()


def _student_base_model_id(client, project_id: str) -> str:
    """Return the seeded default-compatible 2B student base."""
    resp = client.get(
        f"/v1/projects/{project_id}/model_configs",
        params={"eligible_role": "student_base"},
    )
    assert resp.status_code == 200, resp.text
    items = resp.json()["items"]
    match = next(
        (item for item in items if item["model_name"] == "nvidia/cosmos-reason2-2b"),
        None,
    )
    assert match is not None, "seed catalog should include Cosmos Reason2 2B"
    return match["model_config_id"]


def _student_base_model_id_by_name(client, project_id: str, model_name: str) -> str:
    """Return the seeded student-base row with the exact model identity."""
    resp = client.get(
        f"/v1/projects/{project_id}/model_configs",
        params={"eligible_role": "student_base"},
    )
    assert resp.status_code == 200, resp.text
    matches = [
        item["model_config_id"]
        for item in resp.json()["items"]
        if item["model_name"] == model_name
    ]
    assert len(matches) == 1, f"expected one seeded {model_name!r} row"
    return matches[0]


def _teacher_only_model_id(client, project_id: str) -> str:
    """Return a catalog entry that has ``teacher`` but NOT ``student_base``.

    The seed catalog contains hosted Teachers without the ``student_base``
    role, which exercise the negative role path without pinning a provider.
    """
    resp = client.get(f"/v1/projects/{project_id}/model_configs")
    assert resp.status_code == 200, resp.text
    for item in resp.json()["items"]:
        if (
            "teacher" in item["eligible_roles"]
            and "student_base" not in item["eligible_roles"]
        ):
            return item["model_config_id"]
    raise AssertionError("seed catalog should contain a non-student_base teacher entry")


# ── Tests ───────────────────────────────────────────────────────────────────


class TestPreflightRouter:
    def test_all_pass_returns_passed(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test")
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "passed"
        check_names = [c["check_name"] for c in body["checks"]]
        assert "tao_reachable" in check_names
        assert "tao_job_timeout_supported" in check_names
        assert "tao_workspace_reachable" in check_names
        assert "tao_base_experiment_ready" in check_names
        assert "hf_token_configured" in check_names
        assert "lora_merge_runtime" in check_names
        assert "student_base_role" in check_names
        assert "training_mode_compatible" in check_names
        assert "quantization_compatible" in check_names
        assert "verified_train_examples" in check_names
        assert "min_test_pool_size" in check_names
        for c in body["checks"]:
            assert c["passed"] is True

    def test_cosmos3_super_lora_requires_full_weight_on_qualified_runtime(
        self, test_app_client
    ):
        """Readiness must stop the proven Super LoRA/TP crash before TAO spend."""
        pid = _seed_project_with_catalog(test_app_client)
        super_id = _student_base_model_id_by_name(
            test_app_client, pid, COSMOS3_SUPER_REASONER
        )
        _set_hf_token(test_app_client, "hf_test")

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([super_id]),
            _patch_train_examples_pass(),
        ):
            lora_resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [super_id],
                    "enable_lora": True,
                    "quantization_schemes": [],
                },
            )

        assert lora_resp.status_code == 200, lora_resp.text
        lora_body = lora_resp.json()
        mode_check = next(
            check
            for check in lora_body["checks"]
            if check["check_name"] == "training_mode_compatible"
        )
        assert lora_body["status"] == "failed"
        assert mode_check["passed"] is False
        assert "tensor-parallel" in mode_check["message"]
        assert "Full-weight" in mode_check["remediation"]

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([super_id]),
            _patch_train_examples_pass(),
        ):
            full_weight_resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [super_id],
                    "enable_lora": False,
                    "quantization_schemes": [],
                },
            )

        assert full_weight_resp.status_code == 200, full_weight_resp.text
        full_weight_body = full_weight_resp.json()
        full_weight_check = next(
            check
            for check in full_weight_body["checks"]
            if check["check_name"] == "training_mode_compatible"
        )
        assert full_weight_body["status"] == "passed"
        assert full_weight_check["passed"] is True
        assert "Full-weight" in full_weight_check["message"]

    def test_cosmos3_super_quantization_requires_baseline_only(self, test_app_client):
        """The proven Super baseline stays usable while quantization fails closed."""
        pid = _seed_project_with_catalog(test_app_client)
        super_id = _student_base_model_id_by_name(
            test_app_client, pid, COSMOS3_SUPER_REASONER
        )

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([super_id]),
            _patch_train_examples_pass(),
        ):
            quantized = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [super_id],
                    "enable_lora": False,
                    "quantization_schemes": ["FP8_DYNAMIC"],
                },
            )

        assert quantized.status_code == 200, quantized.text
        quantized_body = quantized.json()
        quantized_check = next(
            check
            for check in quantized_body["checks"]
            if check["check_name"] == "quantization_compatible"
        )
        assert quantized_body["status"] == "failed"
        assert quantized_check["passed"] is False
        assert "not currently supported" in quantized_check["message"]
        assert "full-precision Super baseline" in quantized_check["message"]

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([super_id]),
            _patch_train_examples_pass(),
        ):
            baseline = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [super_id],
                    "enable_lora": False,
                    "quantization_schemes": [],
                },
            )

        assert baseline.status_code == 200, baseline.text
        baseline_body = baseline.json()
        baseline_check = next(
            check
            for check in baseline_body["checks"]
            if check["check_name"] == "quantization_compatible"
        )
        assert baseline_body["status"] == "passed"
        assert baseline_check["passed"] is True
        assert "Baseline-only" in baseline_check["message"]

    def test_tao_unreachable_returns_failed(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        with (
            _patch_probe_failure("TAO_API_BASE_URL is not configured"),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        tao_check = next(
            c for c in body["checks"] if c["check_name"] == "tao_reachable"
        )
        assert tao_check["passed"] is False
        assert "TAO_API_BASE_URL" in tao_check["message"]

    def test_tao_without_safe_job_timeout_support_returns_failed(self, test_app_client):
        """Training stays gated when FTMS would silently apply its unsafe
        60-minute stale-heartbeat default."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        probe = AsyncMock(
            return_value={
                "success": True,
                "error": None,
                "status_code": 200,
                "job_timeout_supported": False,
                "job_timeout_error": (
                    "TAO does not declare `timeout_minutes` on its v2 "
                    "ExperimentJobReq schema."
                ),
            }
        )
        with (
            patch(f"{_SVC}.probe_tao_connection", new=probe),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        check = next(
            c for c in body["checks"] if c["check_name"] == "tao_job_timeout_supported"
        )
        assert check["passed"] is False
        assert "timeout_minutes" in check["message"]
        assert "docs/tao-ftms-install.md" in check["remediation"]

    def test_non_student_base_model_returns_failed(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        teacher_id = _teacher_only_model_id(test_app_client, pid)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([teacher_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [teacher_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        role_check = next(
            c
            for c in body["checks"]
            if c["check_name"] == "student_base_role"
            and c["model_config_id"] == teacher_id
        )
        assert role_check["passed"] is False
        assert "student_base" in role_check["message"]

    def test_missing_model_id_returns_failed_with_not_found_message(
        self, test_app_client
    ):
        pid = _seed_project_with_catalog(test_app_client)
        ghost_id = generate_uuid4()
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([ghost_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [ghost_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        role_check = next(
            c for c in body["checks"] if c["check_name"] == "student_base_role"
        )
        assert role_check["passed"] is False
        assert "not found" in role_check["message"].lower()
        assert role_check["model_config_id"] == ghost_id

    def test_multiple_models_report_per_model_detail(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        teacher_id = _teacher_only_model_id(test_app_client, pid)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id, teacher_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id, teacher_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        role_checks = [
            c for c in body["checks"] if c["check_name"] == "student_base_role"
        ]
        assert len(role_checks) == 2
        sb_check = next(c for c in role_checks if c["model_config_id"] == sb_id)
        teacher_check = next(
            c for c in role_checks if c["model_config_id"] == teacher_id
        )
        assert sb_check["passed"] is True
        assert teacher_check["passed"] is False

    def test_response_carries_server_resolved_presets(self, test_app_client):
        """The Advanced expander renders the patches from THIS
        response — the backend resolver is the one source of truth (the
        UI's former TS mirror drifted: max_keep 8 vs the real 1). Pin the
        drift-sensitive values so a re-diverging mirror can't come back
        silently."""
        from vlm_feedback_loop.services.training_preset import (
            resolve_training_preset,
        )

        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200, resp.text
        presets = resp.json()["resolved_presets"]
        assert sb_id in presets
        patches = presets[sb_id]
        assert set(patches) == {"quick", "standard", "high_quality", "max_quality"}
        # Drift-sensitive: the checkpoint retention bound (the TS mirror said 8).
        assert patches["standard"]["train"]["ckpt"]["max_keep"] == 1
        # And each patch must be byte-equal to the canonical resolver.
        model_name = next(
            it["model_name"]
            for it in test_app_client.get(
                f"/v1/projects/{pid}/model_configs",
                params={"eligible_role": "student_base"},
            ).json()["items"]
            if it["model_config_id"] == sb_id
        )
        for preset_key, resolved_patch in patches.items():
            assert resolved_patch == resolve_training_preset(preset_key, model_name)

    def test_empty_model_list_returns_422(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/training_preflight",
            json={"student_base_model_config_ids": []},
        )
        assert resp.status_code == 422, resp.text
        errors = resp.json()["detail"]
        assert any(
            "student_base_model_config_ids" in err.get("loc", []) for err in errors
        ), errors

    def test_missing_body_returns_422(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        resp = test_app_client.post(
            f"/v1/projects/{pid}/training_preflight",
            json={},
        )
        assert resp.status_code == 422, resp.text
        errors = resp.json()["detail"]
        assert any(
            "student_base_model_config_ids" in err.get("loc", []) for err in errors
        ), errors


# ── C5 — Profile D preflight differentiation ────────────────────────────────


class TestProfileDDifferentiation:
    """Training preflight (Profile D) MUST NOT run the NIM deploy checks."""

    def test_no_docker_or_gpu_or_ngc_checks(self, test_app_client):
        """Asserts the preflight response contains only Profile D checks —
        check_names — no docker / gpu / ngc / container-toolkit checks.
        """
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        allowed = {
            "tao_reachable",
            "tao_job_timeout_supported",
            "tao_workspace_reachable",
            "tao_base_experiment_ready",
            "hf_token_configured",
            "lora_merge_runtime",
            "student_base_role",
            "training_mode_compatible",
            "quantization_compatible",
            "verified_train_examples",
            "min_test_pool_size",
        }
        observed = {c["check_name"] for c in body["checks"]}
        leaked = observed - allowed
        assert not leaked, (
            f"training preflight leaked NIM-preflight checks: {leaked}. "
            "The NIM deployment preflight must "
            "never run here."
        )

    def test_probe_function_is_tao_client_not_docker(self, test_app_client):
        """Confirm the preflight service calls ``probe_tao_connection``
        (TAO client) and NOT any Docker / NVIDIA Container Toolkit probe.

        We patch the *real* ``probe_tao_connection`` and assert it was called.
        """
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        mock = AsyncMock(
            return_value={
                "success": True,
                "error": None,
                "status_code": 200,
                "job_timeout_supported": True,
                "job_timeout_error": None,
            }
        )
        with (
            patch(f"{_SVC}.probe_tao_connection", new=mock),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
            assert resp.status_code == 200, resp.text
        mock.assert_awaited_once()
        # The probe outcome must surface in the response body — if the
        # endpoint stops reporting the TAO probe result, this fails even
        # though the probe itself was still called.
        tao_check = next(
            c for c in resp.json()["checks"] if c["check_name"] == "tao_reachable"
        )
        assert tao_check["passed"] is True


# ── Service-layer unit ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestPreflightServiceDirect:
    """Service-layer tests: bypass FastAPI to validate behavior directly."""

    async def test_independent_checks_on_tao_failure(
        self, test_app_client, monkeypatch
    ):
        """A failure in the TAO probe MUST NOT prevent the role checks from
        executing. Mirrors the capability-probe independence requirement.
        """
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)

        # Override the probe to fail
        from vlm_feedback_loop.services import training_preflight_service

        monkeypatch.setattr(
            training_preflight_service,
            "probe_tao_connection",
            AsyncMock(
                return_value={
                    "success": False,
                    "error": "Connection refused",
                    "status_code": None,
                }
            ),
        )

        # Use the Settings instance injected by test_app_client (not the
        # process-global get_settings(), which loads the developer's
        # canonical config.yaml and hard-exits on hosts without one).
        from vlm_feedback_loop.routers.projects import get_current_settings

        settings = test_app_client.app.dependency_overrides[get_current_settings]()

        with _patch_workspace_check_pass(), _patch_base_experiment_pass_for([sb_id]):
            result = await training_preflight_service.run_training_preflight(
                project_id=pid,
                student_base_model_config_ids=[sb_id],
                settings=settings,
            )
        # Role check ran despite TAO failure
        role_checks = [
            c for c in result["checks"] if c["check_name"] == "student_base_role"
        ]
        assert role_checks
        assert role_checks[0]["passed"] is True
        # Overall still failed because TAO check failed
        assert result["status"] == "failed"

    async def test_default_settings_no_tao_url_returns_clear_message(
        self, test_app_client, tmp_path
    ):
        """When TAO settings are unset, the probe returns a clear error.

        This test deliberately constructs a Settings object with TAO
        fields blanked so it is robust to the host's global ``.env``
        state (which may or may not carry live TAO credentials).
        """
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)

        from conftest import make_settings
        from vlm_feedback_loop.services import training_preflight_service

        unset_tao_settings = make_settings(
            tmp_path / "workspace",
            TAO_API_BASE_URL=None,
            TAO_API_KEY=None,
            TAO_ORG_NAME=None,
        )

        result = await training_preflight_service.run_training_preflight(
            project_id=pid,
            student_base_model_config_ids=[sb_id],
            settings=unset_tao_settings,
        )
        tao_check = next(
            c for c in result["checks"] if c["check_name"] == "tao_reachable"
        )
        assert tao_check["passed"] is False
        # Message must be non-empty, plain text.
        assert tao_check["message"].strip()


# ── P1-P4: workspace + base-experiment check correctness ────────────────────


class TestPreflightWorkspaceAndBaseExperimentChecks:
    """Acceptance items P1-P4 — workspace + base-experiment checks."""

    def test_p1_workspace_pass_when_reachable(self, test_app_client):
        """P1: tao_workspace_reachable passes when bootstrapped + 200."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        ws = next(
            c for c in body["checks"] if c["check_name"] == "tao_workspace_reachable"
        )
        assert ws["passed"] is True

    def test_p2_workspace_fail_surfaces_bootstrap_guidance(self, test_app_client):
        """P2: when not bootstrapped, the message references the CLI."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        # No workspace-check patch → real helper runs and fails because the
        # freshly-created deployment.db has bootstrap_status='not_bootstrapped'.
        with _patch_probe_success(), _patch_base_experiment_pass_for([sb_id]):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        ws = next(
            c for c in body["checks"] if c["check_name"] == "tao_workspace_reachable"
        )
        assert ws["passed"] is False
        assert "vlm-feedback-loop tao-bootstrap" in ws["message"]
        assert body["status"] == "failed"

    def test_p3_base_experiment_ready_pass_when_pull_complete(self, test_app_client):
        """P3: pass when tao_base_experiment_id + pull_complete are set."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)

        # Patch the student_base row to simulate a completed pull.
        from sqlalchemy.orm import Session

        from vlm_feedback_loop.db.models.model_config import ModelConfig
        from vlm_feedback_loop.routers.projects import get_current_settings
        from vlm_feedback_loop.services.project_service import get_project_engine

        # Use the Settings instance injected by test_app_client (not the
        # process-global get_settings(), which loads the developer's
        # canonical config.yaml and hard-exits on hosts without one).
        settings = test_app_client.app.dependency_overrides[get_current_settings]()
        engine = get_project_engine(pid, settings.WORKSPACE_ROOT)
        assert engine is not None
        with Session(engine) as session:
            row = (
                session.query(ModelConfig)
                .filter(ModelConfig.model_config_id == sb_id)
                .one()
            )
            row.tao_base_experiment_id = "exp-abc"
            row.tao_base_experiment_pull_status = "pull_complete"
            session.commit()

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        be = next(
            c
            for c in body["checks"]
            if c["check_name"] == "tao_base_experiment_ready"
            and c["model_config_id"] == sb_id
        )
        assert be["passed"] is True, be
        assert "exp-abc" in be["message"]

    def test_p2_workspace_fail_message_names_admin_handoff_doc(self, test_app_client):
        """Workspace failure references both provisioning paths.

        The failure message MUST point to both the
        Blueprint CLI (`vlm-feedback-loop tao-bootstrap`) and the admin
        handoff reference doc so air-gapped operators see the escape
        hatch without reading the spec first.
        """
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        with _patch_probe_success(), _patch_base_experiment_pass_for([sb_id]):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        ws = next(
            c for c in body["checks"] if c["check_name"] == "tao_workspace_reachable"
        )
        assert ws["passed"] is False
        assert "vlm-feedback-loop tao-bootstrap" in ws["message"]
        assert "docs/tao-ftms-install.md" in ws["message"]

    def test_missing_base_is_non_blocking_and_names_model(self, test_app_client):
        """A selected missing base is explicit first-use work, not a blocker."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test_provisioning")
        # Look up the model_name for assertions.
        resp = test_app_client.get(
            f"/v1/projects/{pid}/model_configs",
            params={"eligible_role": "student_base"},
        )
        assert resp.status_code == 200
        model_name = next(
            m["model_name"]
            for m in resp.json()["items"]
            if m["model_config_id"] == sb_id
        )

        # Don't patch the base-experiment check — the seeded row has NULL
        # tao_base_experiment_id, so Start Training must plan provisioning.
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        be = next(
            c
            for c in body["checks"]
            if c["check_name"] == "tao_base_experiment_ready"
            and c["model_config_id"] == sb_id
        )
        assert be["passed"] is True
        assert be["provisioning_required"] is True
        assert model_name in be["message"]
        assert "Start Training will provision it automatically" in be["message"]
        assert body["status"] == "passed"

    def test_missing_base_needs_no_manual_cli_remediation(self, test_app_client):
        """The normal SME path no longer asks for a separate provisioning CLI."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test_provisioning")
        with _patch_probe_success(), _patch_workspace_check_pass():
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )
        assert resp.status_code == 200
        body = resp.json()
        be = next(
            c
            for c in body["checks"]
            if c["check_name"] == "tao_base_experiment_ready"
            and c["model_config_id"] == sb_id
        )
        assert be["passed"] is True
        assert be["provisioning_required"] is True
        assert be["remediation"] is None
        assert "tao-pull-base-experiments" not in be["message"]

    def test_missing_base_without_hf_token_fails_before_go(self, test_app_client):
        """Gated-base provisioning credentials are part of readiness."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, None)

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )

        assert resp.status_code == 200
        body = resp.json()
        hf = next(c for c in body["checks"] if c["check_name"] == "hf_token_configured")
        assert hf["passed"] is False
        assert "HF_TOKEN is required" in hf["message"]
        assert "~/.vlm_feedback_loop/.env" in hf["remediation"]
        assert body["status"] == "failed"

    def test_ready_lora_base_still_requires_hf_token(self, test_app_client):
        """Local adapter merge must load the gated base even when TAO has it."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, None)

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )

        assert resp.status_code == 200
        body = resp.json()
        hf = next(c for c in body["checks"] if c["check_name"] == "hf_token_configured")
        assert hf["passed"] is False
        assert "HF_TOKEN is required" in hf["message"]
        assert body["status"] == "failed"

    def test_ready_full_weight_base_does_not_require_hf_token(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, None)
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [sb_id],
                    "enable_lora": False,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        hf = next(c for c in body["checks"] if c["check_name"] == "hf_token_configured")
        assert hf["passed"] is True
        assert "not required" in hf["message"]

    def test_every_missing_selected_base_is_marked_for_provisioning(
        self, test_app_client
    ):
        """Each missing selection carries its own first-use provisioning flag."""
        pid = _seed_project_with_catalog(test_app_client)
        resp = test_app_client.get(
            f"/v1/projects/{pid}/model_configs",
            params={"eligible_role": "student_base"},
        )
        assert resp.status_code == 200
        sb_ids = [m["model_config_id"] for m in resp.json()["items"]][:2]
        assert len(sb_ids) == 2, "seed catalog should have >= 2 student_base entries"
        _set_hf_token(test_app_client, "hf_test_provisioning")

        # No base-experiment patch — seeded rows have NULL ids.
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_train_examples_pass(),
        ):
            resp = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": sb_ids,
                    "enable_lora": False,
                    "quantization_schemes": [],
                },
            )
        assert resp.status_code == 200
        be_checks = [
            c
            for c in resp.json()["checks"]
            if c["check_name"] == "tao_base_experiment_ready"
        ]
        assert len(be_checks) == 2
        assert all(c["passed"] for c in be_checks)
        assert all(c["provisioning_required"] for c in be_checks)
        assert len({c["message"] for c in be_checks}) == 2
        assert resp.json()["status"] == "passed"


# ── verified_train_examples — training-data availability check ─────────────


class TestVerifiedTrainExamplesCheck:
    """The preflight must fail — with SME-facing copy — when the
    training-export selection is empty, and count only Verified labels
    under the ACTIVE guidance outside the Test Pool."""

    def _post(self, client, pid: str, sb_id: str):
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
            _patch_test_pool_pass(),
        ):
            return client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )

    def test_fails_on_project_without_training_data(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        resp = self._post(test_app_client, pid, sb_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "failed"
        check = next(
            c for c in body["checks"] if c["check_name"] == "verified_train_examples"
        )
        assert check["passed"] is False
        assert check["message"] == (
            "No Verified training examples yet. Continue labeling."
        )
        assert check["model_config_id"] is None

    def test_passes_with_verified_non_pool_label(self, test_app_client):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test")
        gid = _seed_active_guidance(test_app_client, pid)
        _insert_verified_label(test_app_client, pid, gid, pool_assignment=None)
        resp = self._post(test_app_client, pid, sb_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        check = next(
            c for c in body["checks"] if c["check_name"] == "verified_train_examples"
        )
        assert check["passed"] is True, check
        assert "1 Verified training example" in check["message"]
        assert body["status"] == "passed"

    def test_pool_only_labels_do_not_count(self, test_app_client):
        """Test Pool members are evaluation-only — a project whose
        every verified label sits in the pool still cannot train."""
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        gid = _seed_active_guidance(test_app_client, pid)
        _insert_verified_label(test_app_client, pid, gid, pool_assignment="test_pool")
        resp = self._post(test_app_client, pid, sb_id)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        check = next(
            c for c in body["checks"] if c["check_name"] == "verified_train_examples"
        )
        assert check["passed"] is False
        assert body["status"] == "failed"

    def test_data_summary_matches_export_eligibility_and_selection(
        self, test_app_client
    ):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        gid = _seed_active_guidance(test_app_client, pid)
        _insert_verified_label(test_app_client, pid, gid, pool_assignment=None)
        _insert_verified_label(test_app_client, pid, gid, pool_assignment="test_pool")
        _insert_auto_labeled_example(test_app_client, pid, gid)

        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            included = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [sb_id],
                    "include_auto_labeled": True,
                },
            )
            excluded = test_app_client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={
                    "student_base_model_config_ids": [sb_id],
                    "include_auto_labeled": False,
                },
            )

        assert included.status_code == 200, included.text
        assert included.json()["data_summary"] == {
            "verified_training_count": 1,
            "test_pool_count": 1,
            "required_test_pool_count": 60,
            "auto_labeled_eligible_count": 1,
            "auto_labeled_included_count": 1,
            "excluded_test_pool_count": 1,
            "excluded_auto_labeled_count": 0,
            "usable_training_count": 2,
        }
        assert excluded.status_code == 200, excluded.text
        assert excluded.json()["data_summary"] == {
            "verified_training_count": 1,
            "test_pool_count": 1,
            "required_test_pool_count": 60,
            "auto_labeled_eligible_count": 1,
            "auto_labeled_included_count": 0,
            "excluded_test_pool_count": 1,
            "excluded_auto_labeled_count": 1,
            "usable_training_count": 1,
        }


class TestMinimumTestPoolSizeCheck:
    """Student evaluation needs the project's configured held-out minimum."""

    def _post(self, client, pid: str, sb_id: str):
        with (
            _patch_probe_success(),
            _patch_workspace_check_pass(),
            _patch_base_experiment_pass_for([sb_id]),
        ):
            return client.post(
                f"/v1/projects/{pid}/training_preflight",
                json={"student_base_model_config_ids": [sb_id]},
            )

    def test_blocks_until_configured_test_pool_minimum_is_reached(
        self, test_app_client
    ):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test")
        gid = _seed_active_guidance(test_app_client, pid)
        update = test_app_client.patch(
            f"/v1/projects/{pid}", json={"scaleup_min_test_pool_size": 2}
        )
        assert update.status_code == 200, update.text
        _insert_verified_label(test_app_client, pid, gid, pool_assignment=None)
        _insert_verified_label(test_app_client, pid, gid, pool_assignment="test_pool")

        blocked = self._post(test_app_client, pid, sb_id)
        assert blocked.status_code == 200, blocked.text
        blocked_body = blocked.json()
        check = next(
            c for c in blocked_body["checks"] if c["check_name"] == "min_test_pool_size"
        )
        assert check["passed"] is False
        assert check["message"] == (
            "Test Pool has 1 of 2 required held-out evaluation examples. "
            "Continue labeling to grow the pool."
        )
        assert blocked_body["data_summary"]["required_test_pool_count"] == 2
        assert blocked_body["status"] == "failed"

        _insert_verified_label(test_app_client, pid, gid, pool_assignment="test_pool")
        ready = self._post(test_app_client, pid, sb_id)
        assert ready.status_code == 200, ready.text
        ready_body = ready.json()
        check = next(
            c for c in ready_body["checks"] if c["check_name"] == "min_test_pool_size"
        )
        assert check["passed"] is True
        assert ready_body["status"] == "passed"

    def test_configured_zero_still_requires_nonempty_evaluation_set(
        self, test_app_client
    ):
        pid = _seed_project_with_catalog(test_app_client)
        sb_id = _student_base_model_id(test_app_client, pid)
        _set_hf_token(test_app_client, "hf_test")
        gid = _seed_active_guidance(test_app_client, pid)
        update = test_app_client.patch(
            f"/v1/projects/{pid}", json={"scaleup_min_test_pool_size": 0}
        )
        assert update.status_code == 200, update.text
        _insert_verified_label(test_app_client, pid, gid, pool_assignment=None)

        blocked = self._post(test_app_client, pid, sb_id)
        assert blocked.status_code == 200, blocked.text
        body = blocked.json()
        check = next(
            c for c in body["checks"] if c["check_name"] == "min_test_pool_size"
        )
        assert check["passed"] is False
        assert body["data_summary"]["required_test_pool_count"] == 1


# ── Regression — TAO auth failures render gracefully (not as HTTP 500) ─────


class TestWorkspaceCheckCatchesTaoAuthError:
    """TAO auth errors from ``get_workspace`` must not propagate to the HTTP
    handler as a 500. They must be caught and surfaced as a structured
    ``tao_workspace_reachable`` failure so the Student Training screen
    can render its "Cannot reach TAO" state cleanly.
    Mirrors the pattern already in ``probe_tao_connection`` (tao_client.py)."""

    @pytest.mark.asyncio
    async def test_tao_auth_error_becomes_structured_failure(self):
        from types import SimpleNamespace

        from vlm_feedback_loop.services.tao_auth import TaoAuthError
        from vlm_feedback_loop.services.training_preflight_service import (
            _check_tao_workspace_reachable,
        )

        # Minimal "bootstrapped" deployment config so we reach get_workspace.
        tao_config = SimpleNamespace(
            bootstrap_status="bootstrapped",
            tao_workspace_id="ws-abc-123",
        )
        settings = SimpleNamespace(
            TAO_API_KEY="nvapi-test", TAO_API_BASE_URL="http://x", TAO_ORG_NAME="o"
        )

        auth_error = TaoAuthError(
            "TAO /login exchange failed: All connection attempts failed"
        )
        with patch(
            f"{_SVC}.get_workspace",
            new=AsyncMock(side_effect=auth_error),
        ):
            result = await _check_tao_workspace_reachable(
                settings=settings, tao_config=tao_config
            )

        assert result["check_name"] == "tao_workspace_reachable"
        assert result["passed"] is False
        assert "Cannot reach TAO" in result["message"]
        # The underlying error detail is preserved so an operator sees
        # the root cause without digging into logs.
        assert "All connection attempts failed" in result["message"]
