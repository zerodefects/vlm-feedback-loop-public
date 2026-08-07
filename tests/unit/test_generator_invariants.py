# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-generator Action Request invariants.

Two contracts hold for every Action Request generator, pinned once here
(parametrized per generator) instead of copy-pasted into each suite:

  1. ``rendered_text`` reaches the SME's clipboard with no secret
     material — templates never embed key-shaped literals, and a
     secret-shaped string arriving via ``context`` is redacted by the
     framework wrapper before rendering.
  2. ``technical_requirements`` exposes the generator's documented
     keys/values — the machine-readable half of the Action Request.

Generator-specific behavior (fallback lookups, per-preflight-check copy,
redaction of DB-persisted errors) stays in each generator's own suite.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

# Side-effect imports: each module registers its generator.
import vlm_feedback_loop.services.local_nim_deploy_generator  # noqa: F401
import vlm_feedback_loop.services.missing_files_generator  # noqa: F401
import vlm_feedback_loop.services.nim_issue_generator  # noqa: F401
import vlm_feedback_loop.services.tao_issue_generator  # noqa: F401
import vlm_feedback_loop.services.tao_setup_generator  # noqa: F401
from conftest import (
    add_endpoint_row,
    add_model_config_row,
    add_project_row,
    add_tao_job_row,
    make_tao_settings,
    open_project_workspace,
)
from vlm_feedback_loop.services.action_requests import generate_action_request

PID = "test-proj"

# Key-shaped token planted in a context field the generator renders; the
# framework redact is the last line of defense before the clipboard.
SECRET = "nvapi-fake-INJECTEDSECRET123456"


@dataclass(frozen=True)
class GeneratorCase:
    request_type: str
    # Representative invocation context; where the generator renders context
    # fields, one of them carries ``SECRET``.
    context: dict[str, Any]
    # Generator-specific technical_requirements assertions.
    check_tech: Callable[[dict[str, Any]], None]
    # Extra environment (project DB, settings patch) for DB-backed generators.
    setup: Callable[[Path, pytest.MonkeyPatch], None] | None = None
    # Strings that must survive redaction into rendered_text.
    must_render: tuple[str, ...] = ()


# ── missing_files ────────────────────────────────────────────────────────────


def _check_missing_files_tech(tech: dict[str, Any]) -> None:
    assert "missing_paths" in tech
    assert tech["affected_example_count"] == 2
    assert "remap_endpoint" in tech


MISSING_FILES = GeneratorCase(
    request_type="missing_files",
    context={
        "missing_paths": [
            f"/data/{SECRET}/img_001.jpg",
            "/data/images/img_002.jpg",
        ]
    },
    check_tech=_check_missing_files_tech,
)


# ── nim_issue ────────────────────────────────────────────────────────────────
# base_url + model_name are both provided so the generator never falls back
# to the project-DB Teacher lookup (covered in test_nim_issue_generator.py).


def _check_nim_issue_tech(tech: dict[str, Any]) -> None:
    assert tech["endpoint_base_url"] == "https://nim.example.com/v1"
    assert tech["model_name"] == "nvidia/cosmos-reason2-8b"
    assert tech["diagnostic_endpoint"] == "GET https://nim.example.com/v1/models"


NIM_ISSUE = GeneratorCase(
    request_type="nim_issue",
    context={
        "base_url": "https://nim.example.com/v1",
        "model_name": "nvidia/cosmos-reason2-8b",
        "error": f"401 Unauthorized: key {SECRET} rejected",
        "error_timestamp": "2026-04-14T12:00:00Z",
    },
    check_tech=_check_nim_issue_tech,
)


# ── tao_setup ────────────────────────────────────────────────────────────────
# The generator ignores context (uniform signature), so no secret can be
# injected — its no-secrets run pins pure template hygiene.


def _check_tao_setup_tech(tech: dict[str, Any]) -> None:
    assert "required_config_fields" in tech
    assert "connection_test" in tech
    # 2 Cosmos Reason2 + 2 Cosmos 3 reasoner student_base entries.
    assert len(tech["student_base_models"]) == 4


TAO_SETUP = GeneratorCase(
    request_type="tao_setup",
    context={},
    check_tech=_check_tao_setup_tech,
)


# ── tao_issue ────────────────────────────────────────────────────────────────


def _tao_issue_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the project DB with a failed TAO train job and point the
    generator's ``get_settings`` at the temp workspace."""
    engine, project_dir, workspace = open_project_workspace(
        tmp_path, PID, register_engine=True
    )
    settings = make_tao_settings(workspace)
    with Session(engine) as s:
        add_project_row(s, PID, str(project_dir))
        add_endpoint_row(s, PID, "ep-1")
        add_model_config_row(
            s,
            PID,
            "mc-1",
            "ep-1",
            model_name="nvidia/cosmos-reason2-8b",
            eligible_roles=json.dumps(["student_base"]),
        )
        add_tao_job_row(
            s,
            PID,
            "tao-abc",
            status="failed",
            error_ref="CUDA out of memory on GPU 3",
            tao_external_job_id="ext-xyz-123",
            dataset_export_ids=["de-a", "de-b"],
            job_config={
                "training_preset": "standard",
                "hyperparameters": {"train": {"epoch": 3}},
                "tao_release_version": "6.26.3",
                "cosmos_rl_container_tag": "6.26.3-cosmos-rl",
            },
        )
        s.commit()
    monkeypatch.setattr(
        "vlm_feedback_loop.services.tao_issue_generator.get_settings",
        lambda: settings,
    )


def _check_tao_issue_tech(tech: dict[str, Any]) -> None:
    assert tech["tao_endpoint"] == "http://tao.test/api/v2"
    assert tech["tao_org"] == "example-org"
    assert tech["tao_job_id"] == "tao-abc"
    assert tech["tao_external_job_id"] == "ext-xyz-123"
    assert tech["action"] == "train"
    assert tech["base_model_name"] == "nvidia/cosmos-reason2-8b"
    assert tech["training_preset"] == "standard"
    assert tech["status"] == "failed"
    assert tech["error"] == "CUDA out of memory on GPU 3"
    assert tech["dataset_export_ids"] == ["de-a", "de-b"]
    assert tech["tao_release_version"] == "6.26.3"
    assert tech["cosmos_rl_container_tag"] == "6.26.3-cosmos-rl"
    assert tech["diagnostic_endpoint"] == (
        "GET http://tao.test/api/v2/orgs/example-org/jobs/ext-xyz-123:logs"
    )


TAO_ISSUE = GeneratorCase(
    request_type="tao_issue",
    context={"tao_job_id": "tao-abc"},
    check_tech=_check_tao_issue_tech,
    setup=_tao_issue_setup,
)


# ── student_nim_deploy ───────────────────────────────────────────────────────


def _check_student_nim_deploy_tech(tech: dict[str, Any]) -> None:
    student = tech.get("student_deployment")
    assert student is not None
    assert student["student_model_id"] == "0123456789abcdef0123456789abcdef"
    assert student["nim_checkpoint_ref"] == "/tmp/ckpt"
    assert student["quantization_method"] == "FP8_DYNAMIC"
    assert student["nim_served_model_name"] == "student-abc12345"
    assert student["nim_model_name_path"] == "/opt/checkpoints/student"
    assert student["nim_release_version"] == "1.6.0"
    # Standard (non-Student) fields still present.
    assert "docker_run_command" in tech
    assert "preflight_checks" in tech
    assert "host_prerequisites" in tech
    assert "health_check" in tech
    assert tech["role"] == "student"


STUDENT_NIM_DEPLOY = GeneratorCase(
    request_type="student_nim_deploy",
    context={
        "docker_run_command": (
            "docker run -d -e NGC_API_KEY nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0"
        ),
        "preflight_checks": [
            {
                "check_name": "ngc_api_key",
                "passed": False,
                "diagnostic": f"NGC key rejected: {SECRET}",
            }
        ],
        "role": "student",
        "nim_container_image": "nvcr.io/nim/nvidia/cosmos-reason2-2b:1.6.0",
        "gpu_assignment": "device=0",
        "gpu_memory_minimum_gb": 24,
        "host_port": 8002,
        "student_model_id": "0123456789abcdef0123456789abcdef",
        "nim_checkpoint_ref": "/tmp/ckpt",
        "quantization_method": "FP8_DYNAMIC",
        "nim_served_model_name": "student-abc12345",
        "nim_model_name_path": "/opt/checkpoints/student",
        "nim_release_version": "1.6.0",
    },
    check_tech=_check_student_nim_deploy_tech,
    # Docker references the exported key by name only.
    must_render=("-e NGC_API_KEY",),
)


CASES = [MISSING_FILES, NIM_ISSUE, TAO_SETUP, TAO_ISSUE, STUDENT_NIM_DEPLOY]
# tao_issue's secret path is DB-persisted error_ref, pinned by its own
# dedicated redaction test (test_tao_issue_generator.py::TestNoSecrets).
NO_SECRET_CASES = [c for c in CASES if c.request_type != "tao_issue"]


def _ids(case: GeneratorCase) -> str:
    return case.request_type


@pytest.mark.parametrize("case", NO_SECRET_CASES, ids=_ids)
def test_rendered_text_carries_no_secret_material(
    case: GeneratorCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A secret-shaped token fed through context never reaches rendered_text,
    and no generator template embeds key-shaped literals — the text is safe
    to copy to a clipboard / support ticket verbatim."""
    if case.setup is not None:
        case.setup(tmp_path, monkeypatch)
    result = generate_action_request(
        case.request_type, "Test Project", PID, case.context
    )
    text = result["rendered_text"]
    assert SECRET not in text
    assert "nvapi-" not in text
    assert "Bearer " not in text
    for needle in case.must_render:
        assert needle in text
    # The structured halves ship through the same API response — the
    # injected token must not survive there either (recursive scrub).
    structured = json.dumps(
        {
            "technical_requirements": result["technical_requirements"],
            "current_environment": result["current_environment"],
        }
    )
    assert SECRET not in structured


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_technical_requirements_expose_documented_structure(
    case: GeneratorCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each generator's technical_requirements block carries its documented
    keys/values so the UI's structured half of the Action Request stays
    populated."""
    if case.setup is not None:
        case.setup(tmp_path, monkeypatch)
    result = generate_action_request(
        case.request_type, "Test Project", PID, case.context
    )
    case.check_tech(result["technical_requirements"])
